#!/bin/bash -l        
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=62g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=levon003@umn.edu 
#SBATCH --job-name=cb_recsys_text_predict_c
#SBATCH -o /home/lana/shared/caringbridge/data/projects/recsys-peer-match/sbatch/text_predict_c.stdout
#SBATCH -e /home/lana/shared/caringbridge/data/projects/recsys-peer-match/sbatch/text_predict_c.stderr

# to submit: sbatch -p amdsmall make_text_features_predict_candidate.sh
working_dir="/home/lana/levon003/repos/recsys-peer-match/src"
cd $working_dir
source activate pytorch-cpuonly || exit 1
echo "In '$working_dir', running script."
python cbrec/text/createTextFeatureSqlite.py --text-id-txt /home/lana/shared/caringbridge/data/projects/recsys-peer-match/model_data/predict_candidate_journal_oids.txt
echo "Finished sbatch script."

