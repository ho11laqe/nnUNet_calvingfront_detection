from skimage import io
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import argparse

def extract_boundaries(sample_path):
    zone_orig = io.imread(sample_path)

    zone_pad = np.pad(zone_orig, ((1,1),(1,1)), 'reflect')
    zone_b = np.pad(zone_orig, ((2,0),(1,1)), "reflect")
    zone_t = np.pad(zone_orig, ((0,2),(1,1)), "reflect")
    zone_r = np.pad(zone_orig, ((1,1),(0,2)), "reflect")
    zone_l = np.pad(zone_orig, ((1,1),(2,0)), "reflect")

    boundaries = np.zeros_like(zone_pad)
    isboundary = np.logical_or.reduce((zone_pad != zone_b, zone_pad != zone_t, zone_pad != zone_r, zone_pad != zone_l))
    boundaries[np.logical_and(zone_pad == 127, isboundary)] = 255

    return boundaries[1:-1, 1:-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path",
                        help="percentage of the dataset used for training validation and test")
    args = parser.parse_args()
    data_path = args.data_path
    output_path =  join(data_path, 'boundaries')

    train_data_path = join(data_path, 'zones', 'train')
    test_data_path = join(data_path,'zones', 'test')

    train_output_path = join(output_path, 'train')
    test_output_path = join(output_path, 'test')

    maybe_mkdir_p(output_path)
    maybe_mkdir_p(train_output_path)
    maybe_mkdir_p(test_output_path)

    # Train
    for train_file in os.listdir(train_data_path):
        print(train_file)
        # load image
        sample_path = join(train_data_path, train_file)
        boundary = extract_boundaries(sample_path)
        output_file = train_file[:-len('zones.png')] + 'boundary.png'
        output_path = join(train_output_path, output_file)
        io.imsave(output_path, boundary)

    # Test
    for test_file in os.listdir(test_data_path):
        print(test_file)
        # load image
        sample_path = join(test_data_path, test_file)
        boundary = extract_boundaries(sample_path)
        output_file = test_file[:-len('zones.png')]+ 'boundary.png'
        output_path = join(test_output_path, output_file)
        io.imsave(output_path, boundary)