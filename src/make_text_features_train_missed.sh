#!/bin/bash -l        
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=62g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=levon003@umn.edu 
#SBATCH --job-name=cb_recsys_text_train_m
#SBATCH -o /home/lana/shared/caringbridge/data/projects/recsys-peer-match/sbatch/text_train_m.stdout
#SBATCH -e /home/lana/shared/caringbridge/data/projects/recsys-peer-match/sbatch/text_train_m.stderr

# to submit: sbatch -p amdsmall make_text_features_train_missed.sh
working_dir="/home/lana/levon003/repos/recsys-peer-match/src"
cd $working_dir
source activate pytorch-cpuonly || exit 1
echo "In '$working_dir', running script."
python cbrec/text/createTextFeatureSqlite.py --text-id-txt /home/lana/shared/caringbridge/data/projects/recsys-peer-match/model_data/missing_train_journal_oids.txt
echo "Finished sbatch script."

