#!/bin/bash -l
#SBATCH --nodes=1 --gres=gpu:1 --time=24:00:00
#SBATCH --job-name=Task501_glacier_front_1

export data_raw="/home/woody/iwi5/iwi5039h/data_raw"
export nnUNet_raw_data_base="/home/woody/iwi5/iwi5039h/nnUNet_data/nnUNet_raw_data_base/"
export nnUNet_preprocessed="/home/woody/iwi5/iwi5039h/nnUNet_data/nnUNet_preprocessed/"
export RESULTS_FOLDER="/home/woody/iwi5/iwi5039h/nnUNet_data/RESULTS_FOLDER"

cd nnunet_glacer
pwd
conda activate nnunet

python3 nnunet/run/run_training.py 2d nnUNetTrainerV2 501 1 --disable_postprocessing_on_folds --disable_deepsupervision
python3 nnunet/inference/predict_simple.py -i $nnUNet_raw_data_base/nnUNet_raw_data/Task501_Glacier_front/imagesTs -o $RESULTS_FOLDER/test_predictions/Task501_Glacier_front/fold_1 -t 501 -m 2d -f 1 -p nnUNetPlansv2.1 -tr nnUNetTrainerV2
python3 nnunet/dataset_conversion/Task501_Glacier_reverse.py -i $RESULTS_FOLDER/test_predictions/Task501_Glacier_front/fold_1
python3 ./evaluate_nnUNet.py --predictions $RESULTS_FOLDER/test_predictions/Task501_Glacier_front/fold_1/pngs --labels_fronts $data_raw/fronts/test --labels_zones $data_raw/zones/test --sar_images $data_raw/sar_images/test

