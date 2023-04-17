#!/bin/bash -l
#SBATCH --nodes=1 --gres=gpu:1 --time=24:00:00
#SBATCH --job-name=Task502_glacier_zone_2

export data_raw="/home/woody/iwi5/iwi5039h/data_raw"
export nnUNet_raw_data_base="/home/woody/iwi5/iwi5039h/nnUNet_data/nnUNet_raw_data_base/"
export nnUNet_preprocessed="/home/woody/iwi5/iwi5039h/nnUNet_data/nnUNet_preprocessed/"
export RESULTS_FOLDER="/home/woody/iwi5/iwi5039h/nnUNet_data/RESULTS_FOLDER"

cd nnunet_glacier
pwd
conda activate nnunet

python3 nnunet/run/run_training.py 2d nnUNetTrainerV2 502 2 --disable_postprocessing_on_folds
python3 nnunet/inference/predict_simple.py -i $nnUNet_raw_data_base/nnUNet_raw_data/Task502_Glacier_zone/imagesTs -o $RESULTS_FOLDER/test_predictions/Task502_Glacier_zone/fold_2 -t 502 -m 2d -f 2
python3 nnunet/dataset_conversion/Task502_Glacier_reverse.py -i $RESULTS_FOLDER/test_predictions/Task502_Glacier_zone/fold_2
python3 ./evaluate_nnUNet.py --predictions $RESULTS_FOLDER/test_predictions/Task502_Glacier_zone/fold_2/pngs --labels_fronts $data_raw/fronts/test --labels_zones $data_raw/zones/test --sar_images $data_raw/sar_images/test
