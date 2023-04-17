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
            front = image[0]
            zones = image[1]
            boundary = image[2]

            color_zones = np.zeros_like(zones)
            color_zones[zones == 0] = 0
            color_zones[zones == 1] = 64
            color_zones[zones == 2] = 127
            color_zones[zones == 3] = 254

            color_fronts = np.zeros_like(front)
            color_fronts[front == 0] = 0
            color_fronts[front == 1] = 255

            color_boundary = np.zeros_like(boundary)
            color_boundary[boundary == 0] =0
            color_boundary[boundary == 1] = 255

            output_path_zone = join(output_folder, file[:-len('.nii.gz')] + '_zone.png')
            pltimage.imsave(output_path_zone, color_zones, cmap='gray', vmax=255)

            output_path_front = join(output_folder, file[:-len('.nii.gz')] + '_front.png')
            pltimage.imsave(output_path_front, color_fronts, cmap='gray', vmax=255)

            output_path_recon = join(output_folder, file[:-len('.nii.gz')] + '_boundary.png')
            pltimage.imsave(output_path_recon, color_boundary, cmap='gray', vmax=255)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', help="Folder with NIfTI files")
    args = parser.parse_args()
    input_folder = args.input_folder
    main(input_folder)