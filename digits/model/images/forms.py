# Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import wtforms
from wtforms import validators

from ..forms import ModelForm
from digits import utils

class ImageModelForm(ModelForm):
    """
    Defines the form used to create a new ImageModelJob
    """

    crop_size = utils.forms.IntegerField('Crop Size',
            validators = [
                    validators.NumberRange(min=1),
                    validators.Optional()
                    ],
            tooltip = "If specified, during training a random square crop will be taken from the input image before using as input for the network."
            )

    use_mean = utils.forms.SelectField('Subtract Mean',
            choices = [
                ('none', 'None'),
                ('image', 'Image'),
                ('pixel', 'Pixel'),
                ],
            default='none',
            tooltip = "Subtract the mean file or mean pixel for this dataset from each image."
            )

    aug_flip = utils.forms.SelectField('Flipping',
            choices = [
                ('none', 'None'),
                ('fliplr', 'Horizontal'),
                ('flipud', 'Vertical'),
                ('fliplrud', 'Horizontal and/or Vertical'),
                ],
            default='none',
            tooltip = "Randomly flips each image during batch preprocessing."
            )

    aug_quad_rot = utils.forms.SelectField('Quadrilateral Rotation',
            choices = [
                ('none', 'None'),
                ('rot90', '0, 90 or 270 degrees'),
                ('rot180', '0 or 180 degrees'),
                ('rotall', '0, 90, 180 or 270 degrees.'),
                ],
            default='none',
            tooltip = "Randomly rotates (90 degree steps) each image during batch preprocessing."
            )

    #
    # Arbitrary Rotation
    #
    aug_rot_use = utils.forms.BooleanField('Arbitrary Rotation',
            default = False,
            tooltip = "You can augment your dataset by performing rotation on images.",
            validators=[ 
                ]
            )
    aug_rot = utils.forms.IntegerField(u'Rotation (+- deg)',
            default=45,
            validators=[ 
                validators.NumberRange(min=0, max=360)
                ],
            tooltip = "The random rotation angle that will be performed during batch preprocessing."
            )


    #
    # Arbitrary Scale
    #
    aug_scale_use = utils.forms.BooleanField('Random Scaling',
            default = False,
            tooltip = "You can augment your dataset by performing random scaling on images.",
            validators=[ 
                ]
            )
    aug_scale = utils.forms.FloatField(u'Rescale (stddev)',
            default=0.05,
            validators=[ 
                validators.NumberRange(min=0, max=1)
                ],
            tooltip = "Retaining image size, the image is rescaled with a +-stddev of this parameter, this option makes cropping redundant."
            )


    #
    # HSV Shifting
    #
    aug_hsv_use = utils.forms.BooleanField('HSV Shifting',
            default = False,
            tooltip = "You can augment your dataset by performing rotation on images.",
            validators=[ 
                ]
            )
    aug_hsv_h = utils.forms.FloatField(u'Hue',
            default=0.01,
            validators=[ 
                validators.NumberRange(min=0, max=0.5)
                ],
            tooltip = "Standard deviation of a shift that will be performed during preprocessing."
            )
    aug_hsv_s = utils.forms.FloatField(u'Saturation',
            default=0.01,
            validators=[ 
                validators.NumberRange(min=0, max=0.5)
                ],
            tooltip = "Standard deviation of a shift that will be performed during preprocessing."
            )
    aug_hsv_v = utils.forms.FloatField(u'Value',
            default=0.01,
            validators=[ 
                validators.NumberRange(min=0, max=0.5)
                ],
            tooltip = "Standard deviation of a shift that will be performed during preprocessing."
            )



    aug_conv_color = utils.forms.SelectField('Colorspace Conversion',
            choices = [
                ('none', 'None'),
                ('grayscale', 'Grayscale (Y)'),
                ('LCS', 'Local Contrast Normalization'),
                ],
            default='none',
            tooltip = "Changes the color space during preprocessing. Note that this might change the amount of input channels!"
            )

















