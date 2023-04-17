#!/bin/bash -l
#SBATCH --nodes=1 --gres=gpu:1 --time=24:00:00
#SBATCH --job-name=Task505_glacier_mtl_early_boundary_1

export data_raw="/home/woody/iwi5/iwi5039h/data_raw"
export nnUNet_raw_data_base="/home/woody/iwi5/iwi5039h/nnUNet_data/nnUNet_raw_data_base/"
export nnUNet_preprocessed="/home/woody/iwi5/iwi5039h/nnUNet_data/nnUNet_preprocessed/"
export RESULTS_FOLDER="/home/woody/iwi5/iwi5039h/nnUNet_data/RESULTS_FOLDER"

cd nnunet_glacer
pwd
conda activate nnunet

#python3 nnunet/dataset_conversion/Task504_Glacier_mtl_recon.py -data_percentage 100 -base $data_raw
#python3 nnunet/experiment_planning/nnUNet_plan_and_preprocess.py -t 504 -pl3d None -pl2d ExperimentPlanner2D_mtl

python3 nnunet/run/run_training.py 2d nnUNetTrainerMTLearly_boundary 505 1 -p nnUNetPlans_mtl --disable_postprocessing_on_folds
python3 nnunet/inference/predict_simple.py -i $nnUNet_raw_data_base/nnUNet_raw_data/Task505_Glacier_mtl_boundary/imagesTs -o $RESULTS_FOLDER/test_predictions/Task505_Glacier_mtl_boundary/early/fold_1 -t 505 -m 2d -f 1 -p nnUNetPlans_mtl -tr nnUNetTrainerMTLearly_boundary
python3 nnunet/dataset_conversion/Task505_Glacier_mtl_recon_reverse.py -i $RESULTS_FOLDER/test_predictions/Task505_Glacier_mtl_boundary/early/fold_1
python3 ./evaluate_nnUNet.py --predictions $RESULTS_FOLDER/test_predictions/Task505_Glacier_mtl_boundary/early/fold_1/pngs --labels_fronts $data_raw/fronts/test --labels_zones $data_raw/zones/test --sar_images $data_raw/sar_images/test
