#!/bin/bash -l

# Point to the folder with the SAR images
export data_raw=$1

# Folders for processing
export nnUNet_raw_data_base=$data_raw'data_nnUNet_preprocessed/NIFTI/nnUNet_raw_data/Task500_Glacier_zonefronts/imagesTs/'
export nnUNet_preprocessed=$data_raw'data_nnUNet_preprocessed/'
export RESULTS_FOLDER=$data_raw'calvingfront/'

# Convert & Preprocess
#python3 nnunet/dataset_conversion/Task500_Glacier_inference.py -data_percentage 100 -base $data_raw

# Inference
python3 nnunet/inference/predict_simple.py -i $nnUNet_raw_data_base -o $RESULTS_FOLDER/fold_0 -t 500 -m 2d -f 0 -p nnUNetPlansv2.1 -tr nnUNetTrainerV2 -model_folder_name $2

# Convert model output to PNG/TIF
python3 nnunet/dataset_conversion/Task500_Glacier_reverse.py -i $RESULTS_FOLDER
