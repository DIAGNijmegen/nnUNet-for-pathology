#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus-per-task=1
#SBATCH --nodelist=dlc-articuno,dlc-lugia,dlc-nidoking,dlc-moltres,dlc-zapdos,dlc-tornadus,dlc-meowth,dlc-groudon
#SBATCH --mem=28G
#SBATCH --time=5-00:00:00
#SBATCH --job-name=nnunetv2
#SBATCH --output=$HOME/logs/slurm-%j.out  # Standard output log
#SBATCH --error=$HOME/logs/slurm-%j.err   # Separate error log
#SBATCH --container-mounts=/data/temporary:/data/temporary
#SBATCH --container-image="doduo1.umcn.nl/#nnunet_for_pathology/sol2:latest"

DATASET=$1
FOLD=$2
# TRAINER=$3
TRAINER="nnUNetTrainer_WSD_points_bal_i0_nnunet_aug_json"

SCRIPT_PATH="/data/temporary/joey/github/nnUNet-for-pathology_v2/pathology_code_and_utils/installs_and_run_training.sh"

bash $SCRIPT_PATH $DATASET $FOLD $TRAINER