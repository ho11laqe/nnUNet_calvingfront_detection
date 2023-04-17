import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
from nnunet.utilities.file_conversions import *
import argparse
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_percentage", default=100,
                        help="percentage of the dataset used for training validation and test")
    parser.add_argument("-base",
                        help="path to directory of data_raw")
    args = parser.parse_args()
    data_percentage = args.data_percentage

    """
    nnU-Net was originally built for 3D images. It is also strongest when applied to 3D segmentation problems because a 
    large proportion of its design choices were built with 3D in mind. Also note that many 2D segmentation problems, 
    especially in the non-biomedical domain, may benefit from pretrained network architectures which nnU-Net does not
    support.
    Still, there is certainly a need for an out of the box segmentation solution for 2D segmentation problems. And 
    also on 2D segmentation tasks nnU-Net cam perform extremely well! We have, for example, won a 2D task in the cell 
    tracking challenge with nnU-Net (see our Nature Methods paper) and we have also successfully applied nnU-Net to 
    histopathological segmentation problems. 
    Working with 2D data in nnU-Net requires a small workaround in the creation of the dataset. Essentially, all images 
    must be converted to pseudo 3D images (so an image with shape (X, Y) needs to be converted to an image with shape 
    (1, X, Y). The resulting image must be saved in nifti format. Hereby it is important to set the spacing of the 
    first axis (the one with shape 1) to a value larger than the others. If you are working with niftis anyways, then 
    doing this should be easy for you. This example here is intended for demonstrating how nnU-Net can be used with 
    'regular' 2D images. We selected the massachusetts road segmentation dataset for this because it can be obtained 
    easily, it comes with a good amount of training cases but is still not too large to be difficult to handle.
    """

    # download dataset from https://www.kaggle.com/insaff/massachusetts-roads-dataset
    # extract the zip file, then set the following path according to your system:
    base = args.base
    # this folder should have the training and testing subfolders

    # now start the conversion to nnU-Net:
    task_name = 'Task504_Glacier_mtl_recon'
    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")
    target_labelsTr = join(target_base, "labelsTr")


    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTs)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    # convert the training examples. Not all training images have labels, so we just take the cases for which there are
    # labels
    label_fronts_dir_tr = join(base, 'fronts_dilated', 'train')
    label_zones_dir_tr = join(base, 'zones', 'train')
    images_dir_tr = join(base, 'sar_images', 'train')

    training_cases = subfiles(label_fronts_dir_tr, suffix='.png', join=False)
    num_samples = int(len(training_cases)/100 * int(data_percentage))
    training_cases_sampled = random.sample(training_cases, num_samples)

    print('Train samples:')
    for label_front_tr in training_cases_sampled:
        unique_name = label_front_tr[:-len('_front.png')]# just the filename with the extension cropped away, so img-2.png becomes img-2 as unique_name
        print(unique_name)
        image_tr = unique_name + '.png'
        label_zone_tr = unique_name + '_zones.png'

        input_front_file = join(label_fronts_dir_tr, label_front_tr)
        input_zone_file = join(label_zones_dir_tr, label_zone_tr)
        input_image_file = join(images_dir_tr, image_tr)

        output_image_file = join(target_imagesTr, unique_name)  # do not specify a file ending! This will be done for you
        output_seg_file = join(target_labelsTr, unique_name)  # do not specify a file ending! This will be done for you

        # this utility will convert 2d images that can be read by skimage.io.imread to nifti. You don't need to do anything.
        # if this throws an error for your images, please just look at the code for this function and adapt it to your needs
        convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)

        convert_mtl_recon_image_to_nifti(input_image_file, input_front_file, input_zone_file, output_seg_file, is_seg=True)

    # now do the same for the test set
    label_fronts_dir_ts = join(base, 'fronts', 'test')
    label_zones_dir_ts = join(base, 'zones', 'test')
    images_dir_ts = join(base, 'sar_images', 'test')

    testing_cases = subfiles(label_fronts_dir_ts, suffix='.png', join=False)
    num_samples = int(len(testing_cases) / 100 * int(data_percentage))
    testing_cases_sampled = random.sample(testing_cases, num_samples)
    print('Test samples:')
    for label_front_ts in testing_cases_sampled:
        unique_name = label_front_ts[:-len('_front.png')]
        print(unique_name)
        image_ts = unique_name + '.png'
        label_zone_ts = unique_name + '_zones.png'
        input_front_file = join(label_fronts_dir_ts, label_front_ts)
        input_zone_file = join(label_zones_dir_ts, label_zone_ts)
        input_image_file = join(images_dir_ts, image_ts)

        output_image_file = join(target_imagesTs, unique_name)
        output_seg_file = join(target_labelsTs, unique_name)

        convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)
        convert_mtl_recon_image_to_nifti(input_image_file, input_front_file, input_zone_file, output_seg_file, is_seg=True)

    # finally we can call the utility for generating a dataset.json
    generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('SAR',),
                          labels={'label0': {0: 'background', 1: 'front'},
                                  'label1': {0: 'background', 1: 'stone', 2: 'glacier', 3: 'ocean'},
                                  'label2': {0: 'sar'}},
                          dataset_name=task_name, license='hands off!')

    """
    once this is completed, you can use the dataset like any other nnU-Net dataset. Note that since this is a 2D
    dataset there is no need to run preprocessing for 3D U-Nets. You should therefore run the 
    `nnUNet_plan_and_preprocess` command like this:
    
    > nnUNet_plan_and_preprocess -t 120 -pl3d None
    
    once that is completed, you can run the trainings as follows:
    > nnUNet_train 2d nnUNetTrainerV2 120 FOLD
    
    (where fold is again 0, 1, 2, 3 and 4 - 5-fold cross validation)
    
    there is no need to run nnUNet_find_best_configuration because there is only one model to choose from.
    Note that without running nnUNet_find_best_configuration, nnU-Net will not have determined a postprocessing
    for the whole cross-validation. Spoiler: it will determine not to run postprocessing anyways. If you are using
    a different 2D dataset, you can make nnU-Net determine the postprocessing by using the
    `nnUNet_determine_postprocessing` command
    """
