import SimpleITK as sitk
import argparse

import numpy as np
import torch
import os
from batchgenerators.utilities.file_and_folder_operations import join, isdir, maybe_mkdir_p
from data_processing.data_postprocessing import extract_front_from_zones
from PIL import Image
import cv2



def main(input_folder):
    files = os.listdir(input_folder)
    output_folder = join(input_folder, 'pngs')
    maybe_mkdir_p(output_folder)

    kernel = np.ones((7, 7), np.uint8)
    for file in files:
        if file.endswith('.gz'):
            print(file)
            file_path = join(input_folder, file)
            image = sitk.ReadImage(file_path)
            image = sitk.GetArrayFromImage(image)
            image = image[0]

            color_zone = np.zeros_like(image)
            color_zone[image == 0] = 0
            color_zone[image == 1] = 64
            color_zone[image == 2] = 127
            color_zone[image == 3] = 254
            color_zone[image == 4] = 254

            glacier = np.zeros_like(image)
            glacier[image == 2] = 1
            glacier_dil = cv2.dilate(glacier, kernel, iterations=1)
            color_zone[glacier_dil == 1] = 127

            color_front = extract_front_from_zones(color_zone, 10)
            color_front[color_front==255] =1

            output_path_zone = join(output_folder, file[:-len('.nii.gz')] + '_zone.tif')
            output_path_front = join(output_folder, file[:-len('.nii.gz')] + '_front.tif')

            img_zone = Image.fromarray(color_zone)
            img_front = Image.fromarray(color_front)
            img_zone.save(output_path_zone)
            img_front.save(output_path_front)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', help="Folder with NIfTI files")
    args = parser.parse_args()
    input_folder = args.input_folder
    main(input_folder)
