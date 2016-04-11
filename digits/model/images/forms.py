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
            default='image',
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

    aug_quadrot = utils.forms.SelectField('Quadrilateral Rotation',
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



























