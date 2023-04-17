from skimage import io
import cv2
import matplotlib.pyplot as plt
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path",
                        help="percentage of the dataset used for training validation and test")
    args = parser.parse_args()
    data_path = args.data_path
    output_path =  join(data_path, 'boundaries_dilated_5')


    train_data_path = join(data_path, 'boundaries', 'train')
    test_data_path = join(data_path,'boundaries', 'test')

    train_output_path = join(output_path, 'train')
    test_output_path = join(output_path, 'test')

    maybe_mkdir_p(output_path)
    maybe_mkdir_p(train_output_path)
    maybe_mkdir_p(test_output_path)

    kernel = np.ones((5, 5), 'uint8')

    # Train
    for train_file in os.listdir(train_data_path):
        print(train_file)
        # load image
        file_path = join(train_data_path, train_file)
        front = io.imread(file_path)

        # dilate boundary
        boundary_dil = cv2.dilate(front, kernel)

        # store image
        output_file_path = join(train_output_path, train_file)
        io.imsave(output_file_path, boundary_dil)

    # Test
    for test_file in os.listdir(test_data_path):
        print(test_file)
        # load image
        file_path = join(test_data_path, test_file)
        front = io.imread(file_path)

        # dilate front
        boundary_dil = cv2.dilate(front, kernel)

        # store image
        output_file_path = join(test_output_path, test_file)
        io.imsave(output_file_path, boundary_dil)