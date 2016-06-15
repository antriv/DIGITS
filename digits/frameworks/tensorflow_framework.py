# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
import re
import subprocess
import time
import tempfile

import flask

from .errors import Error, NetworkVisualizationError, BadNetworkError
from .framework import Framework
import digits
from digits import utils
from digits.config import config_value
from digits.model.tasks import TensorflowTrainTask
from digits.utils import subclass, override

@subclass
class TensorflowFramework(Framework):

    """
    Defines required methods to interact with the Tensorflow framework
    """

    # short descriptive name
    NAME = 'Tensorflow (experimental)'

    # identifier of framework class
    CLASS = 'tensorflow'

    # whether this framework can shuffle data during training
    CAN_SHUFFLE_DATA = True

    SUPPORTED_SOLVER_TYPES = ['ADAM']

    def __init__(self):
        super(TensorflowFramework, self).__init__()
        # id must be unique
        self.framework_id = self.CLASS

    @override
    def create_train_task(self, **kwargs):
        """
        create train task
        """
        return TensorflowTrainTask(framework_id = self.framework_id, **kwargs)

    @override
    def get_standard_network_desc(self, network):
        """
        return description of standard network
        """
        networks_dir = os.path.join(os.path.dirname(digits.__file__), 'standard-networks', self.CLASS)

        for filename in os.listdir(networks_dir):
            path = os.path.join(networks_dir, filename)
            if os.path.isfile(path):
                match = None
                match = re.match(r'%s.py' % network, filename)
                if match:
                    with open(path) as infile:
                        return infile.read()
        # return None if not found
        return None

    @override
    def get_network_from_desc(self, network_desc):
        """
        return network object from a string representation
        """
        # return the same string
        return network_desc

    @override
    def get_network_from_previous(self, previous_network, use_same_dataset):
        """
        return new instance of network from previous network
        """
        # note: use_same_dataset is ignored here because for Tensorflow, DIGITS
        # does not change the number of outputs of the last linear layer
        # to match the number of classes in the case of a classification
        # network. In order to write a flexible network description that
        # accounts for the number of classes, the `nClasses` external
        # parameter must be used, see documentation.

        # @TODO: what is this function exactly?

        # return the same network description
        return previous_network

    @override
    def validate_network(self, data):
        """
        validate a network
        """
        return True

    @override
    def get_network_visualization(self, desc):
        """
        return visualization of network
        """
        return NotImplementedError() #@TODO





