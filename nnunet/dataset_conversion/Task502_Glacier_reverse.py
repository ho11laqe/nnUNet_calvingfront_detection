import SimpleITK as sitk
import argparse

import numpy as np
import torch
import os
from batchgenerators.utilities.file_and_folder_operations import join, isdir, maybe_mkdir_p
import matplotlib.image as pltimage

def main(input_folder):


    files = os.listdir(input_folder)
    output_folder = join(input_folder, 'pngs')
    maybe_mkdir_p(output_folder)
    for file in files:
        if file.endswith('.gz'):
            file_path = join(input_folder, file)
            image = sitk.ReadImage(file_path)
            image = sitk.GetArrayFromImage(image)
            image = image[0]

            color_image = np.zeros_like(image)
            color_image[image == 0] = 0
            color_image[image == 1] = 64
            color_image[image == 2] = 127
            color_image[image == 3] = 254
            output_path = join(output_folder, file[:-len('.nii.gz')] + '_zone.png')
            pltimage.imsave(output_path, color_image, cmap='gray', vmax=255)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', help="Folder with NIfTI files")
    args = parser.parse_args()
    input_folder = args.input_folder
    main(input_folder)