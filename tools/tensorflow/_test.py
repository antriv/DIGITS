#!/usr/bin/python

# This file is used to test main.py

import os

#dataset_dir = "/Users/tzaman/Dropbox/code/DIGITS/digits/jobs/20160715-230349-5f23" #CIFAR100
dataset_dir = "/Users/tzaman/Dropbox/code/DIGITS/digits/jobs/20160715-230434-21a4" #CIFAR10
#dataset_dir = "/Users/tzaman/Dropbox/code/DIGITS/digits/jobs/20160615-215643-75fd" #MNIST

# TIM'S OVERRIDES:
args = (
	" --labels=" + dataset_dir + "/labels.txt"
	" --networkDirectory=../../digits/standard-networks/tensorflow"
	#" --network=lenet.py"
	" --network=lenet_slim.py"
	" --train=" + dataset_dir + "/train_db"
	" --validation=" + dataset_dir + "/val_db"
	" --summaries_dir=/tmp/tb/"
	" --save=/Users/tzaman/Desktop/result"
	" --seed=1"
	" --epoch=5"
	" --learningRate=0.01"
	" --learningRateDecay=0.001"
	" --tf_summaries_dir=/Users/tzaman/Desktop/tb/"
	" --shuffle=true"
	" --optimization=sgd"
	#" --optimization=adam"
	)
print(args)

os.system("python main.py " + args)