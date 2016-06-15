
def build_model(params):

    # Create some wrappers for simplicity
    def conv2d(x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')

    # Create model
    def conv_net(x, weights, biases, dropout):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, params['input_shape'][0], params['input_shape'][1], params['input_shape'][2]])

        # Convolution Layer
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        
        # Apply Dropout
        #fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 20 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, params['input_shape'][2], 20])),
        # 5x5 conv, 20 inputs, 50 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 20, 50])),
        # fully connected, 4*4*16=800 inputs, 500 outputs
        'wd1': tf.Variable(tf.random_normal([4*4*50, 500])),
        # 500 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([500, params['nclasses']]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([20])),
        'bc2': tf.Variable(tf.random_normal([50])),
        'bd1': tf.Variable(tf.random_normal([500])),
        'out': tf.Variable(tf.random_normal([params['nclasses']]))
    }

    dropout_placeholder = tf.placeholder(tf.float32)

    model = conv_net(params['x'], weights, biases, dropout_placeholder)

    return {
        'model' : model, # The predictor model architecture
        'cost' : tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, params['y'])),
        'train_batch_size' : 64,
        'validation_batch_size' : 100,
        #'feed_dict_train' : {dropout_placeholder: 0.75},
        #'feed_dict_val' : {dropout_placeholder: 1.}
        }


