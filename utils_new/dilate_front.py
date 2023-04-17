from skimage import io
import cv2
import matplotlib.pyplot as plt
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *


if __name__ == '__main__':
    data_path = '/home/ho11laqe/PycharmProjects/data_raw'
    output_path =  join(data_path, 'fronts_dilated_5')


    train_data_path = join(data_path, 'fronts', 'train')
    test_data_path = join(data_path,'fronts', 'test')

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

        # dilate front
        front_dil = cv2.dilate(front, kernel)

        # store image
        output_file_path = join(train_output_path, train_file)
        io.imsave(output_file_path, front_dil)

    # Test
    for test_file in os.listdir(test_data_path):
        print(test_file)
        # load image
        file_path = join(test_data_path, test_file)
        front = io.imread(file_path)

        # dilate front
        front_dil = cv2.dilate(front, kernel)

        # store image
        output_file_path = join(test_output_path, test_file)
        io.imsave(output_file_path, front_dil)