#!/bin/bash -l
#SBATCH --nodes=1 --gres=gpu:1 --time=24:00:00
#SBATCH --job-name=Task500_glacier_zonefronts_3

export data_raw="/home/woody/iwi5/iwi5039h/data_raw"
export nnUNet_raw_data_base="/home/woody/iwi5/iwi5039h/nnUNet_data/nnUNet_raw_data_base/"
export nnUNet_preprocessed="/home/woody/iwi5/iwi5039h/nnUNet_data/nnUNet_preprocessed/"
export RESULTS_FOLDER="/home/woody/iwi5/iwi5039h/nnUNet_data/RESULTS_FOLDER"

cd nnunet_glacer
pwd
conda activate nnunet

# Convert & Preprocess
#python3 combine_labels.py -data_path $data_raw
#python3 nnunet/dataset_conversion/Task500_Glacier_zonefronts.py -data_percentage 100 -base $data_raw
#python3 nnunet/experiment_planning/nnUNet_plan_and_preprocess.py -t 500 -pl3d None

# Train and Predict 5-fold crossvalidation
#python3 nnunet/run/run_training.py 2d nnUNetTrainerV2 500 3 --disable_postprocessing_on_folds
python3 nnunet/inference/predict_simple.py -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_Glacier_zonefronts/imagesTs -o $RESULTS_FOLDER/test_predictions/Task500_Glacier_zonefronts/fold_3 -t 500 -m 2d -f 3 -p nnUNetPlansv2.1 -tr nnUNetTrainerV2 -z
#python3 nnunet/dataset_conversion/Task500_Glacier_reverse.py -i $RESULTS_FOLDER/test_predictions/Task500_Glacier_zonefronts/fold_3
#python3 ./evaluate_nnUNet.py --predictions $RESULTS_FOLDER/test_predictions/Task500_Glacier_zonefronts/fold_3/pngs --labels_fronts $data_raw/fronts/test --labels_zones $data_raw/zones/test --sar_images $data_raw/sar_images/test
