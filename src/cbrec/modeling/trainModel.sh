#!/bin/bash -l        
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem=62g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=levon003@umn.edu
#SBATCH --job-name=cb_recsys_train_model
#SBATCH -o /home/lana/shared/caringbridge/data/projects/recsys-peer-match/sbatch/train_model.stdout
#SBATCH -e /home/lana/shared/caringbridge/data/projects/recsys-peer-match/sbatch/train_model.stderr

# to submit: sbatch -p amdsmall trainModel.sh
working_dir="/home/lana/levon003/repos/recsys-peer-match/src"
cd $working_dir
source activate pytorch-cpuonly || exit 1
echo "In '$working_dir', running script."
python cbrec/modeling/trainModel.py
echo "Finished sbatch script."

