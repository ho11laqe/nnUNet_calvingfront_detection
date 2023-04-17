from skimage import io
import cv2
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path",
                        help="percentage of the dataset used for training validation and test")
    args = parser.parse_args()
    data_path = args.data_path
    output_path = join(data_path, 'zone_fronts')

    train_front_path = join(data_path, 'fronts_dilated_5', 'train')
    test_front_path = join(data_path, 'fronts_dilated_5', 'test')
    train_zone_path = join(data_path, 'zones', 'train')
    test_zone_path = join(data_path, 'zones', 'test')

    train_output_path = join(output_path, 'train')
    test_output_path = join(output_path, 'test')

    maybe_mkdir_p(output_path)
    maybe_mkdir_p(train_output_path)
    maybe_mkdir_p(test_output_path)

    kernel = np.ones((5, 5), 'uint8')


    # Train
    for train_file in os.listdir(train_front_path):
        print(train_file)
        # load image
        front_path = join(train_front_path, train_file)
        zone_path = join(train_zone_path, train_file[:-len('front.png')]+'zones.png')
        front = io.imread(front_path)
        zone = io.imread(zone_path)

        zone[front==255] = 32

        # store image
        output_file_path = join(train_output_path, train_file[:-len('front.png')]+'.png')
        io.imsave(output_file_path, zone)


    for test_file in os.listdir(test_front_path):
        print(test_file)
        # load image
        front_path = join(test_front_path, test_file)
        zone_path = join(test_zone_path, test_file[:-len('front.png')] + 'zones.png')
        front = io.imread(front_path)
        zone = io.imread(zone_path)

        zone[front == 255] = 32

        # store image
        output_file_path = join(test_output_path, test_file[:-len('front.png')] + '.png')
        io.imsave(output_file_path, zone)