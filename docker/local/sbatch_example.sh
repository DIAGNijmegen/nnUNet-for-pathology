#!/bin/bash
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=28G
#SBATCH --time=8:00:00
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/temporary:/data/temporary
#SBATCH --container-image="doduo1.umcn.nl/#nnunet_for_pathology/sol2:latest"
#SBATCH --output=/home/%u/logs/slurm-%j.out

PORT='YOUR RANDOM PORT NUMBER'
NODE=$(hostname)
USERNAME=$(whoami)
SSH_ID_RSA_FOLDER=/path/to/your/local/.ssh/id_rsa

echo "vscode://vscode-remote/ssh-remote+${USERNAME}@${NODE}:${PORT}/home/${USERNAME}?ssh=${SSH_ID_RSA_FOLDER}"

echo "Started SSH on port $PORT"
/usr/sbin/sshd -D -p $PORT