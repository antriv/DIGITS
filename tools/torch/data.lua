-- Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
require 'torch' -- torch
require 'nn' -- provides a normalization operator
require 'utils' -- various utility functions
require 'hdf5' -- import HDF5 now as it is unsafe to do it from a worker thread

require 'unsup' -- for zca whitening

local threads = require 'threads' -- for multi-threaded data loader
check_require 'image' -- for color transforms

package.path = debug.getinfo(1, "S").source:match[[^@?(.*[\/])[^\/]-$]] .."?.lua;".. package.path

require 'logmessage'
local ffi = require 'ffi'
local tds = check_require 'tds'

local tdsIsInstalled, tds = pcall(function() return check_require 'tds' end)

-- enable shared serialization to speed up Tensor passing between threads
threads.Threads.serialization('threads.sharedserialize')

----------------------------------------------------------------------

function copy (t) -- shallow-copy a table
    if type(t) ~= "table" then return t end
    local meta = getmetatable(t)
    local target = {}
    for k, v in pairs(t) do target[k] = v end
    setmetatable(target, meta)
    return target
end

local function all_keys(cursor_,key_,op_)
    return coroutine.wrap(
        function()
            local k = key_
            local v
            repeat
                k,v = cursor_:get(k,op_ or MDB.NEXT)
                if k then
                    coroutine.yield(k,v)
                end
            until (not k)
        end)
end



-- Augmentation to grayscale, rips out one channel
local augmentGrayscale = function(im)
    return image.rgb2y(im)
end

-- Local Contrast Normalization, 1 or 3 channels.
local augmentLCN = function(im)
    -- This function retains the input size and pads with zeros, although LCS does not do that natively
    local kernel_size = 9
    local kernel_size_offset = (kernel_size-1)/2+1

    -- Allocate the return variable
    im_target = torch.Tensor(im:size()):fill(0)

    -- Note the LCS does not natively pad, we need to take that into account (lcs image will be smaller)
    im_lcn = torch.Tensor(im:size(1),im:size(2)-kernel_size+1,im:size(3)-kernel_size+1)
    for c=1,im:size(1) do
        im_lcn[c] = image.lcn(im[c], image.gaussian({size=kernel_size,sigma=1.591/kernel_size,normalize=true}))
    end

    -- Return to the original size
    im_target:narrow(2, kernel_size_offset, im_target:size(2)-kernel_size+1):narrow(3, kernel_size_offset, im_target:size(3)-kernel_size+1):copy(im_lcn)
    return im_target
end

-- HSV Augmentation
local augmentHSV = function(augHSV)
    -- input parameter augHSV should contain 'H' 'S' and 'V' keys with a float as stddev value
    -- Fair stddev values are [H,S,V] = [0.01, 0.05, 0.05], a more noticable change is [0.02, 0.08, 0.08]
    local applyHsvAug = function(Data)
        Data = image.rgb2hsv(Data:double())
        if augHSV.H >0.001 then
            -- We do not need to account for overflow because every number wraps around (1.1=2.1=3.1,etc)
            local rng_h = torch.normal(0, augHSV.H) -- We add a round value to prevent an underflow bug (<0 becomes black bug)
            Data[1] = torch.add(Data[1], torch.Tensor(1,Data:size(2),Data:size(2)):fill(rng_h))
        end
        if augHSV.S >0.001 then
            local rng_s = torch.normal(0, augHSV.S) 
            Data[2] = torch.add(Data[2], torch.Tensor(1,Data:size(2),Data:size(3)):fill(rng_s))
        end
        if augHSV.V >0.001 then
            local rng_v = torch.normal(0, augHSV.V)
            Data[3] = torch.add(Data[3], torch.Tensor(1,Data:size(2),Data:size(3)):fill(rng_v))
        end
        Data.image.saturate(Data) -- bound between 0 and 1
        return image.hsv2rgb(Data):float()
    end
    return applyHsvAug
end

---- Scale augmentation, rescales but returns the same size
--local reSampleScale = function(scaleRange)
--    local applyReSample = function(Data)
--        local sizeImg = Data:size()
--        local szx = torch.random(math.ceil(sizeImg[3]*scaleRange)) -- maximum extra size x
--        local szy = torch.random(math.ceil(sizeImg[2]*scaleRange)) -- maximum extra size y
--        local startx = torch.random(szx)
--        local starty = torch.random(szy)
--        return image.scale(Data:narrow(2, starty, sizeImg[2]-szy):narrow(3, startx, sizeImg[3]-szx), sizeImg[3], sizeImg[2], 'bilinear')
--    end
--    return applyReSample
--end
--
---- Arbitrary rotation augmentation
--local rotate = function(angleRange)
--    local applyRot = function(Data)
--        local angle = torch.randn(1)[1]*angleRange
--        local rot = image.rotate(Data,math.rad(angle),'bilinear')
--        return rot
--    end
--    return applyRot
--end


local warp = function(im, augRot, augScale)
    -- A nice function of scale is 0.05 (stddev of scale change), 
    -- and a nice value for ration is a few degrees or more if your dataset allows for it

    local width = im:size()[3] 
    local height = im:size()[2]
    local nchan = im:size()[1]

    -- Scale <0=zoom in(+rand crop), >0=zoom out
    local scale_x = 0
    if augScale > 0.001 then
        local scale_x = -torch.normal(0, augScale) -- @TODO: WHY MINUS?
    end

    local scale_y = scale_x -- keep aspect ratio the same

    -- Angle of rotation
    local rot_angle = torch.randn(1)[1] * augRot --deg


    -- Given a zoom in or out, we move around our canvas.
    local move_x = torch.uniform(-scale_x,scale_x)
    local move_y = torch.uniform(-scale_y,scale_y)

    -- x/y grids
    local grid_y = torch.ger( torch.linspace(-1-scale_y,1+scale_y,height), torch.ones(width) )
    local grid_x = torch.ger( torch.ones(height), torch.linspace(-1-scale_x,1+scale_x,width) )

    local flow = torch.FloatTensor()
    flow:resize(2,height,width)
    flow:zero()

    -- Apply scale
    flow_scale = torch.FloatTensor()
    flow_scale:resize(2,height,width)
    flow_scale[1] = grid_y
    flow_scale[2] = grid_x
    flow_scale[1]:add(1-move_y):mul(0.5) -- 0 to 1
    flow_scale[2]:add(1-move_x):mul(0.5) -- 0 to 1
    flow_scale[1]:mul(height)
    flow_scale[2]:mul(width)
    flow:add(flow_scale)

    if (rot_angle>0.01 or rot_angle <0.01) then
        -- Apply rotation
        local flow_rot = torch.FloatTensor()
        flow_rot:resize(2,height,width)
        flow_rot[1] = grid_y * ((height-1)/2) * -1
        flow_rot[2] = grid_x * ((width-1)/2) * -1
        view = flow_rot:reshape(2,height*width)
        function rmat(deg)
          local r = deg/180*math.pi
          return torch.FloatTensor{{math.cos(r), -math.sin(r)}, {math.sin(r), math.cos(r)}}
        end

        local rotmat = rmat(rot_angle)
        local flow_rotr = torch.mm(rotmat, view)
        flow_rot = flow_rot - flow_rotr:reshape( 2, height, width )
        flow:add(flow_rot)
    end

    return image.warp(im, flow, 'bilinear', false)
end



-- Quadrilateral rotation (options {0 1 2 3}={0,rot90,rot270,rot180})
-- Note rotating like this in 90 degree steps is extremely efficient
local rot90 = function(rotFlag)
    local applyRot90 = function(Data)
        local rot
        if rotFlag == 2 then
            rot = Data:transpose(2,3) --switch X and Y dimentions @TODO: TEST ME
            rot = image.vflip(rot)
            return rot
        elseif rotFlag == 3 then
            rot = Data:transpose(2,3) --switch X and Y dimentions @TODO: TEST ME
            rot = image.hflip(rot)
            return rot
        elseif rotFlag == 4 then -- vflip+hflip=180 deg rotation
            rot = image.hflip(Data)
            rot = image.vflip(rot)
            return rot
        end
        -- Any other flag than {2,3,4} will return the original data
        return Data
    end
    return applyRot90
end

--PreProcess will be called per single image
local PreProcess = function(y, train, meanTensor, augOpt)
    -- augOpt.augFlip
    -- augOpt.augQuadRot
    -- augOpt.augRot
    -- augOpt.augScale
    -- augOpt.augHSV = {H=0, S=0, V=0}
    -- augOpt.ConvertColor
    -- augOpt.crop = {use=false, Y=-1, X=-1, len=-1}




    

    --print(y)

    if meanTensor then
        for i=1,meanTensor:size(1) do
            y[i]:add(-meanTensor[i])
        end
    else -- Not using mean tensor:
        -- normalize to [-1:1]
        --y:div(127.5) 
        --y:add(-1)

        -- normalize to [0:1]
        y:div(255) 
        --y:add(-1)
    end

    --image.savePNG( '/home/brainstorm/temp/' .. os.date("%H%I%M%S_A") .. '.png', y)


    -- augFlip {none, fliplr, flipud, fliplrud}
    if (augOpt.augFlip) then
        if (augOpt.augFlip == 'fliplr' or  augOpt.augFlip == 'fliplrud') and torch.random(2)==1 then 
            y = image.hflip(y)
        end

        if (augOpt.augFlip == 'flipud' or  augOpt.augFlip == 'fliplrud') and torch.random(2)==1 then
            y = image.vflip(y)
        end
    end

    -- augQuadRot {none, rot90, rot180, rotall}
    if augOpt.augQuadRot then
        if augOpt.augQuadRot == 'rot90' then 
            y = rot90(torch.random(3))(y)
        elseif augOpt.augQuadRot == 'rot180' and torch.random(2)==1 then
            y = rot90(3)(y)
        elseif augOpt.augQuadRot == 'rotall' then
            local random_rot = torch.random(4)
            y = rot90(random_rot)(y) -- Randomly choose one of four rotations
        end
    end

    --if augOpt.augRot and (augOpt.augRot >0.01) then
    --    y = rotate(augOpt.augRot)(y)
    --end
    --if augOpt.augScale and (augOpt.augScale > 0.01) then
    --   y = reSampleScale(augOpt.augScale)(y)
    --end

    if augOpt.augRot and augOpt.augScale and ((augOpt.augRot >0.01) or (augOpt.augScale > 0.01)) then
        y = warp(y, augOpt.augRot, augOpt.augScale)
    end



    if augOpt.augHSV then
        if (augOpt.augHSV.H > 0.001) or (augOpt.augHSV.S > 0.001)  or (augOpt.augHSV.V > 0.001) then
            assert(y:size(1)==3, 'Cannot mix HSV augmentation without the image having 3 channels.')
            -- Augments H,S,V values, but returns back to RGB.
            y = augmentHSV(augOpt.augHSV)(y)
            -- Note: an RGB image is returned (as was the input)
        end
    end

    if augOpt.ConvertColor then 
        if augOpt.ConvertColor == 'grayscale' then
            -- Converts from a RGB torch 1 channel grayscale (Y)
            y = augmentGrayscale(y)
            assert(not meanTensor, 'Mixing mean subtraction and using Grayscale is not allowed.')
        elseif augOpt.ConvertColor == 'LCN' then
            -- Converts all image channels with LCN
            y = augmentLCN(y)
            assert(not meanTensor, 'Mixing mean subtraction and using LCN is not allowed.')
        end
    end

    if augOpt.crop and augOpt.crop.use then
        if train == true then
            --During training we will crop randomly
            local valueY = math.ceil(torch.FloatTensor.torch.uniform() * augOpt.crop.Y)
            local valueX = math.ceil(torch.FloatTensor.torch.uniform() * augOpt.crop.X)
            y = image.crop(y, valueX-1, valueY-1, valueX + augOpt.crop.len-1, valueY + augOpt.crop.len-1)
        else
            --for validation we will crop at center
            y = image.crop(y, augOpt.crop.X-1, augOpt.crop.Y-1, augOpt.crop.X + augOpt.crop.len-1, augOpt.crop.Y + augOpt.crop.len-1)
        end
    end

    --image.savePNG( '/home/brainstorm/temp/' .. os.date("%H%I%M%S_B") ..'.png', y)
    --os.execute("sleep 2")

    return y
end

--Loading label definitions

local loadLabels = function(labels_file)
    local Classes = {}
    i=0

    local file = io.open(labels_file)
    if file then
        for line in file:lines() do
            i=i+1
            -- labels file might contain Windows line endings
            -- or other whitespaces => trim leading and trailing
            -- spaces here
            Classes[i] = trim(line)
        end
        return Classes
    else
        return nil -- nil indicates that file not present
    end
end

--loading mean tensor
local loadMean = function(mean_file, use_mean_pixel)
    local meanTensor
    local mean_im = image.load(mean_file,nil,'byte'):type('torch.FloatTensor'):contiguous()
    if use_mean_pixel then
        mean_of_mean = torch.FloatTensor(mean_im:size(1))
        for i=1,mean_im:size(1) do
            mean_of_mean[i] = mean_im[i]:mean()
        end
        meanTensor = mean_of_mean
    else
        meanTensor = mean_im
    end
    return meanTensor
end

local function pt(t)
    for k,v in pairs(t) do
        print(k,v)
    end
end

-- Meta class
LMDBSource = {e=nil, t=nil, d=nil, c=nil}

function LMDBSource:new(lighningmdb, path)
    local paths = require('paths')
    assert(paths.dirp(path), 'DB Directory '.. path .. ' does not exist')
    logmessage.display(0,'opening LMDB database: ' .. path)
    local self = copy(LMDBSource)
    self.lightningmdb = lighningmdb
    self.e = self.lightningmdb.env_create()
    local LMDB_MAP_SIZE = 1099511627776 -- 1 TB
    self.e:set_mapsize(LMDB_MAP_SIZE)
    local flags = lightningmdb.MDB_RDONLY + lightningmdb.MDB_NOTLS
    local db, err = self.e:open(path, flags, 0664)
    if not db then
        -- unable to open Database => this might be due to a permission error on
        -- the lock file so we will try again with MDB_NOLOCK. MDB_NOLOCK is safe
        -- in this process as we are opening the database in read-only mode.
        -- However if another process is writing into the database we might have a
        -- concurrency issue - note that this shouldn't happen in DIGITS since the
        -- database is written only once during dataset creation
        logmessage.display(0,'opening LMDB database failed with error: "' .. err .. '". Trying with MDB_NOLOCK')
        flags = bit.bor(flags, lighningmdb.MDB_NOLOCK)
        -- we need to close/re-open the LMDB environment
        self.e:close()
        self.e = self.lightningmdb.env_create()
        self.e:set_mapsize(LMDB_MAP_SIZE)
        db, err = self.e:open(path, flags ,0664)
        if not db then
            error('opening LMDB database failed with error: ' .. err)
        end
    end
    self.total = self.e:stat().ms_entries
    self.t = self.e:txn_begin(nil, self.lightningmdb.MDB_RDONLY)
    self.d = self.t:dbi_open(nil,0)
    self.c = self.t:cursor_open(self.d)
    self.path = path
    return self, self.total
end

function LMDBSource:close()
    logmessage.display(0,'closing LMDB database: ' .. self.path)
    self.total = 0
    self.c:close()
    self.e:dbi_close(self.d)
    self.t:abort()
    self.e:close()
end

function LMDBSource:get(key)
    if key then
        v = self.t:get(self.d,key,self.lightningmdb.MDB_FIRST)
    else
        k,v = self.c:get(nil,self.lightningmdb.MDB_NEXT)
    end
    return v
end

-- Meta class
DBSource = {mean = nil, ImageChannels = 0, ImageSizeY = 0, ImageSizeX = 0, total=0,
            -- mirror=false, crop=false, croplen=0, cropY=0, cropX=0, 
            augOpt={},
            subtractMean=true, train=false, classification=false}

-- Derived class method new
-- Creates a new instance of a database
-- Parameters:
--  backend (string): 'lmdb' or 'hdf5'
--  db_path (string): path to database
--  labels_db_path (string): path to labels database, or nil
--  mirror (boolean): whether samples must be mirrored
--  meanTensor (tensor): mean tensor to use for mean subtraction
--  isTrain (boolean): whether this is a training database (e.g. mirroring not applied to validation database)
--  shuffle (boolean): whether samples should be shuffled
--  classification (boolean): whether this is a classification task
function DBSource:new (backend, db_path, labels_db_path, meanTensor, isTrain, shuffle, classification)
    local self = copy(DBSource)
    local paths = require('paths')

    if backend == 'lmdb' then
        check_require('pb')
        self.datum = pb.require"datum"
        local lightningmdb_lib=check_require("lightningmdb")
        self.lightningmdb = _VERSION=="Lua 5.2" and lightningmdb_lib or lightningmdb
        self.lmdb_data, self.total = LMDBSource:new(self.lightningmdb, db_path)
        self.keys = self:lmdb_getKeys()
        if #self.keys ~= self.total then
            logmessage.display(2, 'thr-id=' .. __threadid .. ' will be disabled - failed to initialize DB - #keys=' .. #self.keys .. ' #records='.. self.total)
            return nil
        end
        -- do we have a separate database for labels/targets?
        if labels_db_path~='' then
            self.lmdb_labels, total_labels = LMDBSource:new(self.lightningmdb, labels_db_path)
            assert(self.total == total_labels, "Number of records="..self.total.." does not match number of labels=" ..total_labels)
            -- read first entry to check label dimensions
            local v = self.lmdb_labels:get(self.keys[1])
            if not v then
                logmessage.display(2, 'thr-id=' .. __threadid .. ' failed to initialize label DB')
                return nil
            end
            local msg = self.datum.Datum():Parse(v)
            -- assume row vector 1xN labels
            assert(msg.height == 1, "label height=" .. msg.height .. " not supported")
            self.label_width = msg.width
        else
            -- assume scalar label (e.g. classification)
            self.label_width = 1
        end
        -- LMDB-specific functions
        self.getSample = self.lmdb_getSample
        self.close = self.lmdb_close
        self.reset = self.lmdb_reset
        -- read first entry to check image dimensions
        local v = self.lmdb_data:get(self.keys[1])
        local msg = self.datum.Datum():Parse(v)
        self.ImageChannels = msg.channels
        self.ImageSizeX = msg.width
        self.ImageSizeY = msg.height
    elseif backend == 'hdf5' then
        assert(classification, "hdf5 only supports classification yet")
        assert(labels_db_path == '', "hdf5 separate label DB not implemented yet")
        check_require('hdf5')
        -- assume scalar label (e.g. classification)
        self.label_width = 1
        -- list.txt contains list of HDF5 databases --
        logmessage.display(0,'opening HDF5 database: ' .. db_path)
        list_path = paths.concat(db_path, 'list.txt')
        local file = io.open(list_path)
        assert(file,"unable to open "..list_path)
        -- go through all DBs in list.txt and store required info
        self.dbs = {}
        self.total = 0
        for line in file:lines() do
            local fileName = paths.concat(db_path, paths.basename(line))
            -- get number of records
            local myFile = hdf5.open(fileName,'r')
            local dim = myFile:read('/data'):dataspaceSize()
            local n_records = dim[1]
            self.ImageChannels = dim[2]
            self.ImageSizeY = dim[3]
            self.ImageSizeX = dim[4]
            myFile:close()
            -- store DB info
            self.dbs[#self.dbs + 1] = {
                path = fileName,
                records = n_records
            }
            self.total = self.total + n_records
        end
        -- which DB is currently being used (initially set to nil to defer
        -- read to first call to self:nextBatch)
        self.db_id = nil
        -- cursor points to current record in current DB
        self.cursor = nil
        -- set pointers to HDF5-specific functions
        self.getSample = self.hdf5_getSample
        self.close = self.hdf5_close
        self.reset = self.hdf5_reset
    end

    if meanTensor ~= nil then
        self.mean = meanTensor
        assert(self.ImageChannels == self.mean:size()[1])
        if self.mean:nDimension() > 1 then
            -- mean matrix subtraction
            assert(self.ImageSizeY == self.mean:size()[2])
            assert(self.ImageSizeX == self.mean:size()[3])
        end
    end

    logmessage.display(0,'Image channels are ' .. self.ImageChannels .. ', Image width is ' .. self.ImageSizeX .. ' and Image height is ' .. self.ImageSizeY)

    --self.mirror = mirror
    self.train = isTrain
    self.shuffle = shuffle
    self.classification = classification

    self.augOpt.augFlip = 'none'
    self.augOpt.augQuadRot = 'none'
    self.augOpt.augRot = 0
    self.augOpt.augScale = 0
    self.augOpt.augHSV = {H=0, S=0, V=0}
    self.augOpt.ConvertColor = 'none'
    self.augOpt.crop = {use=false, Y=-1, X=-1, len=-1}

    if self.classification then
        assert(self.label_width == 1, 'expect scalar labels for classification tasks')
    end

    return self
end

-- Derived class method setDataAugmentation
function DBSource:setDataAugmentation(augOpt)
    if self.train == true then
        self.augOpt = augOpt -- Copy all
        if self.augOpt.crop.use then
            self.augOpt.crop.Y = self.ImageSizeY - self.augOpt.crop.len + 1
            self.augOpt.crop.X = self.ImageSizeX - self.augOpt.crop.len + 1
        end
    else -- Validation:
        -- For validation, only copy certain options (constructor default is no augmentation)
        -- So we are doing very selective copying of the augmentation options
        if augOpt.crop.use then
            self.augOpt.crop = augOpt.crop
            self.augOpt.crop.Y = math.floor((self.ImageSizeY - self.augOpt.crop.len)/2) + 1
            self.augOpt.crop.X = math.floor((self.ImageSizeX - self.augOpt.crop.len)/2) + 1
        end
    end
end





-- Derived class method inputTensorShape
-- This returns the shape of the input samples in the database
-- There is an assumption that all input samples have the same shape
function DBSource:inputTensorShape()
    local shape = torch.Tensor(3)
    shape[1] = self.ImageChannels
    shape[2] = self.ImageSizeY
    shape[3] = self.ImageSizeX
    return shape
end

-- Derived class method lmdb_getKeys
function DBSource:lmdb_getKeys ()
    local Keys
    if tdsIsInstalled then
        -- use tds.Vec() to allocate memory outside of Lua heap
        Keys = tds.Vec()
    else
        -- if tds is not installed, use regular table (and Lua memory allocator)
        Keys = {}
    end
    local i=0
    local key=nil
    for k,v in all_keys(self.lmdb_data.c,nil,self.lightningmdb.MDB_NEXT) do
        i=i+1
        Keys[i] = k
        key = k
    end
    return Keys
end

-- Derived class method getSample (LMDB flavour)
function DBSource:lmdb_getSample(shuffle, idx)
    local label

    if shuffle then
        idx = math.max(1,torch.ceil(torch.rand(1)[1] * self.total))
    end

    key = self.keys[idx]
    v = self.lmdb_data:get(key)
    assert(key~=nil, "lmdb read nil key at idx="..idx)
    assert(v~=nil, "lmdb read nil value at idx="..idx.." key="..key)

    local total = self.ImageChannels*self.ImageSizeY*self.ImageSizeX
    -- Tensor allocations inside loop consumes little more execution time. So allocated "x" outside with double size of an image and inside loop if any encoded image is encountered with bytes size more than Tensor size, then the Tensor is resized appropriately.
    local x = torch.ByteTensor(total*2):contiguous() -- sometimes length of JPEG files are more than total size. So, "x" is allocated with more size to ensure that data is not truncated while copying.
    local x_size = total * 2 -- This variable is just to avoid the calls to tensor's size() i.e., x:size(1)
    local temp_ptr = torch.data(x) -- raw C pointer using torchffi


    --local a = torch.Timer()
    --local m = a:time().real
    local msg = self.datum.Datum():Parse(v)

    if not self.lmdb_labels then
        -- read label from sample database
        label = msg.label
    else
        -- read label from label database
        v = self.lmdb_labels:get(key)
        local label_msg = self.datum.Datum():Parse(v)
        label = torch.FloatTensor(label_msg.width):contiguous()
        for x=1,label_msg.width do
            label[x] = label_msg.float_data[x]
        end
    end

    if #msg.data > x_size then
        x:resize(#msg.data+1) -- 1 extra byte is required to copy zero terminator i.e., '\0', by ffi.copy()
        x_size = #msg.data
    end

    ffi.copy(temp_ptr, msg.data)
    --print(string.format("elapsed time1: %.6f\n", a:time().real - m))
    --m = a:time().real

    local y=nil
    if msg.encoded==true then
        y = image.decompress(x,msg.channels,'byte'):float()
    else
        x = x:narrow(1,1,total):view(msg.channels,msg.height,msg.width):float() -- using narrow() returning the reference to x tensor with the size exactly equal to total image byte size, so that view() works fine without issues
        if self.ImageChannels == 3 then
            -- unencoded color images are stored in BGR order => we need to swap blue and red channels (BGR->RGB)
            y = torch.FloatTensor(msg.channels,msg.height,msg.width)
            y[1] = x[3]
            y[2] = x[2]
            y[3] = x[1]
        else
            y = x
        end
    end

    return y, label

end

-- Derived class method getSample (HDF5 flavour)
function DBSource:hdf5_getSample(shuffle)
    if not self.db_id or self.cursor>self.dbs[self.db_id].records then
        --local a = torch.Timer()
        --local m = a:time().real
        self.db_id = self.db_id or 0
        assert(self.db_id < #self.dbs, "Trying to read more records than available")
        self.db_id = self.db_id + 1
        local myFile = hdf5.open(self.dbs[self.db_id].path, 'r')
        self.hdf5_data = myFile:read('/data'):all()
        self.hdf5_labels = myFile:read('/label'):all()
        myFile:close()
        -- make sure number of entries match number of labels
        assert(self.hdf5_data:size()[1] == self.hdf5_labels:size()[1], "data/label mismatch")
        self.cursor = 1
        --print(string.format("hdf5 data load - elapsed time1: %.6f\n", a:time().real - m))
    end

    local idx
    if shuffle then
        idx = math.max(1,torch.ceil(torch.rand(1)[1] * self.dbs[self.db_id].records))
    else
        idx = self.cursor
    end
    y = self.hdf5_data[idx]
    channels = y:size()[1]
    label = self.hdf5_labels[idx]
    self.cursor = self.cursor + 1
    return y, label
end

-- Derived class method nextBatch
-- @parameter batchSize Number of samples to load
-- @parameter idx Current index within database
function DBSource:nextBatch (batchsize, idx)
    local Images
    if self.augOpt.crop.use then
        Images = torch.Tensor(batchsize, self.ImageChannels, self.augOpt.crop.len, self.augOpt.crop.len)
    else
        Images = torch.Tensor(batchsize, self.ImageChannels, self.ImageSizeY, self.ImageSizeX)
    end

    local Labels
    if self.label_width == 1 then
        Labels = torch.Tensor(batchsize)
    else
        Labels = torch.FloatTensor(batchsize, self.label_width)
    end

    local i=0

    --local mean_ptr = torch.data(self.mean)
    local image_ind = 0

    for i=1,batchsize do
        -- get next sample
        y, label = self:getSample(self.shuffle, idx + i - 1)

        if self.classification then
            -- label is index from array and Lua array indices are 1-based
            label = label + 1
        end

        Images[i] = PreProcess(y, self.train, self.mean, self.augOpt)

        Labels[i] = label
    end
    return Images, Labels
end

-- Derived class method to reset cursor (HDF5 flavour)
function DBSource:hdf5_reset ()
    self.db_id = nil
    self.cursor = nil
end

-- Derived class method to get total number of Records
function DBSource:totalRecords ()
    return self.total;
end

-- Derived class method close (LMDB flavour)
function DBSource:lmdb_close (batchsize)
    self.lmdb_data:close()
    if self.lmdb_labels then
        self.lmdb_labels:close()
    end
end

-- Derived class method close (HDF5 flavour)
function DBSource:hdf5_close (batchsize)
end

-- Meta class
DataLoader = {}

-- Derived class method new
-- Creates a new instance of a database
-- @param numThreads (int) number of reader threads to create
-- @package_path (string) caller package path
-- @param backend (string) 'lmdb' or 'hdf5'
-- @param db_path (string) path to database
-- @param labels_db_path (string) path to labels database, or nil
-- @param mirror (boolean) whether samples must be mirrored
-- @param mean (Tensor) mean tensor for mean image
-- @param subtractMean (boolean) whether mean image should be subtracted from samples
-- @param isTrain (boolean) whether this is a training database (e.g. mirroring not applied to validation database)
-- @param shuffle (boolean) whether samples should be shuffled
-- @param classification (boolean) whether this is a classification task
function DataLoader:new (numThreads, package_path, backend, db_path, labels_db_path, mean, isTrain, shuffle, classification)
    local self = copy(DataLoader)
    self.backend = backend
    if self.backend == 'hdf5' then
        -- hdf5 wrapper does not support multi-threaded reader yet
        numThreads = 1
    end
    -- create pool of threads
    self.numThreads = numThreads
    self.threadPool = threads.Threads(
        self.numThreads,
        function(threadid)
            -- inherit package path from main thread
            package.path = package_path
            require('data')
            -- executes in reader thread, variables are local to this thread
            db = DBSource:new(backend, db_path, labels_db_path,
                              --mirror, 
                              mean,
                              isTrain,
                              shuffle,
                              classification
                              )
        end
    )
    -- use non-specific mode
    self.threadPool:specific(false)
    return self
end

function DataLoader:getInfo()
    local datasetSize
    local inputTensorShape

    -- switch to specific mode so we can specify which thread to add job to
    self.threadPool:specific(true)
    -- we need to iterate here as some threads may not have a valid DB
    -- handle (as happens when opening too many concurrent instances)
    for i=1,self.numThreads do
        self.threadPool:addjob(
                   i, -- thread to add job to
                   function()
                       -- executes in reader thread, return values passed to
                       -- main thread through following function
                       if db then
                           return db:totalRecords(), db:inputTensorShape()
                       else
                           return nil, nil
                       end
                   end,
                   function(totalRecords, shape)
                       -- executes in main thread
                       datasetSize = totalRecords
                       inputTensorShape = shape
                   end
                   )
        self.threadPool:synchronize()
        if datasetSize then
            break
        end
    end
    -- return to non-specific mode
    self.threadPool:specific(false)
    return datasetSize, inputTensorShape
end

-- schedule next data loader batch
-- @param batchSize (int) Number of samples to load
-- @param dataIdx (int) Current index in database
-- @param dataTable (table) Table to store data into
function DataLoader:scheduleNextBatch(batchSize, dataIdx, dataTable)
    -- send reader thread a request to load a batch from the training DB
    local backend = self.backend
    self.threadPool:addjob(
                function()
                    if backend=='hdf5' and dataIdx==1 then
                        db:reset()
                    end
                    -- executes in reader thread
                    if db then
                        inputs, targets =  db:nextBatch(batchSize, dataIdx)
                        return batchSize, inputs, targets
                    else
                        return batchSize, nil, nil
                    end
                end,
                function(batchSize, inputs, targets)
                    -- executes in main thread
                    dataTable.batchSize = batchSize
                    dataTable.inputs = inputs
                    dataTable.outputs = targets
                end
            )
end

-- returns whether data loader is able to accept more jobs
function DataLoader:acceptsjob()
    return self.threadPool:acceptsjob()
end

-- wait until next data loader job completes
function DataLoader:waitNext()
    -- wait for next data loader job to complete
    self.threadPool:dojob()
    -- check for errors in loader threads
    if self.threadPool:haserror() then -- check for errors
        self.threadPool:synchronize() -- finish everything and throw error
    end
end

-- free data loader resources
function DataLoader:close()
    -- switch to specific mode so we can specify which thread to add job to
    self.threadPool:specific(true)
    for i=1,self.numThreads do
        self.threadPool:addjob(
                    i,
                    function()
                        if db then
                            db:close()
                        end
                    end
                )
    end
    -- return to non-specific mode
    self.threadPool:specific(false)
end


-- set dataset augmentation parameters (calls setDataAugmentation() method of all DB intances)
function DataLoader:setDataAugmentation(augOpt)
    -- switch to specific mode so we can specify which thread to add job to
    self.threadPool:specific(true)
    for i=1,self.numThreads do
        self.threadPool:addjob(
                    i,
                    function()
                        if db then
                            db:setDataAugmentation(augOpt)
                        end
                    end
                )
    end
    -- return to non-specific mode
    self.threadPool:specific(false)
end


return{
    loadLabels = loadLabels,
    loadMean= loadMean,
    PreProcess = PreProcess
}

