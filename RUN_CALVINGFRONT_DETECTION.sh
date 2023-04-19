#!/bin/bash -l

while getopts ":m:d:" opt; do
  case $opt in
    m) model="$OPTARG";;
    d) data="$OPTARG";;
    *) echo "Unknown error occurred."
       exit 1;;
  esac
done
# Point to the folder with the SAR images
export data_raw=$data

# Folders for processing
export nnUNet_raw_data_base=$data_raw'/data_nnUNet_preprocessed/NIFTI/'
export nnUNet_preprocessed=$data_raw'/data_nnUNet_preprocessed/'
export RESULTS_FOLDER=$data_raw'/calvingfronts/'

# Convert & Preprocess
python3 nnunet/dataset_conversion/Task500_Glacier_inference.py -data_percentage 100 -base $data_raw

# Inference
python3 nnunet/inference/predict_simple.py -i $nnUNet_raw_data_base'nnUNet_raw_data/Task500_Glacier_zonefronts/imagesTs/' -o $RESULTS_FOLDER/fold_0 -t 500 -m 2d -f 0 -p nnUNetPlansv2.1 -tr nnUNetTrainerV2 -model_folder_name $model

# Convert model output to PNG/TIF
python3 nnunet/dataset_conversion/Task500_Glacier_reverse.py -i $RESULTS_FOLDER'fold_0/'
