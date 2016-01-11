#! /usr/bin/env python
# This script takes a file containing a list of images and creates three files
# train.txt, validation.txt and test.txt, each containing a subset of the images
# randomly split according to the specified proportion.
# Note that the seed is fixed.

# e.g. usage:
# ./split_dataset.py -f /tmp/all_images.txt -s 70 20 -o ~/tmp

import argparse
import numpy as np
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', help='Path to file containing all images',
        required=True)
parser.add_argument('-s', '--split', type=int, nargs=2, help='Split up for train and validation \
        (in percentage): e.g. 70 20. Remaining is treated as test data', required=True)
parser.add_argument('-o', '--output_dir', help='Output directory for placing the data splits', 
        required=True)

args = vars(parser.parse_args())

split = args['split']
if split[0] + split[1] >= 100 or split[0] <= 0 or split[1] <= 0:
    print 'Invalid data split up: train and validation split must be '\
    'positive and total to less than 100%'
    sys.exit()

output_dir = args['output_dir']
if not os.path.isdir(output_dir):
    print 'Non-existent output directory'
    sys.exit()

all_images_file = open(args['file'])
lines = all_images_file.read().splitlines()
all_images_file.close()

total_images = len(lines)
num_train_images = int(split[0] * total_images / 100.0)
num_validation_images = int(split[1] * total_images / 100.0)
num_test_images = total_images - (num_train_images + num_validation_images)

print 'Splitting {} images into {} train, {} validation and {} test \
images'.format(total_images, num_train_images, num_validation_images, num_test_images)

# Fix seed for debugging.
np.random.seed(0)
shuffled_images = np.random.permutation(lines)

train_images = shuffled_images[:num_train_images]
validation_images = shuffled_images[num_train_images:num_train_images +
        num_validation_images]
test_images = shuffled_images[num_train_images+num_validation_images:]

# Write the outputs
train_file = open(output_dir + '/train.txt', 'w')
validation_file = open(output_dir + '/validation.txt', 'w')
test_file = open(output_dir + '/test.txt', 'w')

for image_name in train_images:
    train_file.write('%s\n' % image_name)
for image_name in validation_images:
    validation_file.write('%s\n' % image_name)
for image_name in test_images:
    test_file.write('%s\n' % image_name)

train_file.close()
validation_file.close()
test_file.close()
