#!/usr/bin/python

# This file is used to test main.py

import os

# TIM'S OVERRIDES:
args = (
	" --labels=/Users/tzaman/Desktop/20160203-153604-2bf4/labels.txt"
	" --networkDirectory=../../digits/standard-networks/tensorflow"
	" --network=lenet_slim.py"
	" --train=/Users/tzaman/Desktop/20160203-153604-2bf4/train_db"
	" --validation=/Users/tzaman/Desktop/20160203-153604-2bf4/val_db"
	" --summaries_dir=/tmp/tb/"
	" --save=/Users/tzaman/Desktop/result"
	" --seed=1"
	" --epoch=5"
	" --learningRate=0.0001"
	" --learningRateDecay=0.001"
	" --tf_summaries_dir=/Users/tzaman/Desktop/tb/"
	#" --shuffle=true"
	" --optimization=adam"
	)
print(args)

os.system("python main.py " + args)