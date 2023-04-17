from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.file_conversions import convert_2d_image_to_nifti
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
from nnunet.dataset_conversion.utils import generate_dataset_json

import argparse


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

    # now start the conversion to nnU-Net:
    task_name = 'Task500_Glacier_zonefronts'
    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")
    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_imagesTs)


    # convert the training examples. Not all training images have labels, so we just take the cases for which there are
    # labels
    samples = subfiles(base, suffix='.png', join=False)
    num_samples = int(len(samples)/100 * int(data_percentage))

    print('Samples conversion:')
    print(target_base)
    for sample_file in samples:
        unique_name = sample_file[:-len('.png')]  # just the filename with the extension cropped away, so img-2.png becomes img-2 as unique_name
        print(unique_name)

        input_image_file = join(base, sample_file)
        output_image_file = join(target_imagesTs, unique_name)  # do not specify a file ending! This will be done for you
        print(output_image_file)
        # this utility will convert 2d images that can be read by skimage.io.imread to nifti. You don't need to do anything.
        # if this throws an error for your images, please just look at the code for this function and adapt it to your needs
        convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)


    # finally we can call the utility for generating a dataset.json
    generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('SAR',),
                          labels={'label0':{0: 'background', 1: 'stone', 2: 'glacier', 3: 'ocean', 4: 'front'}}, dataset_name=task_name, license='hands off!')

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
