#!/bin/bash -l        
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=62g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=levon003@umn.edu 
#SBATCH --job-name=cb_recsys_text_predict
#SBATCH -o /home/lana/shared/caringbridge/data/projects/recsys-peer-match/sbatch/text_predict.stdout
#SBATCH -e /home/lana/shared/caringbridge/data/projects/recsys-peer-match/sbatch/text_predict.stderr

# to submit: sbatch -p amdsmall make_text_features_predict.sh
working_dir="/home/lana/levon003/repos/recsys-peer-match/src"
cd $working_dir
source activate pytorch-cpuonly || exit 1
echo "In '$working_dir', running script."
model_data_dir="/home/lana/shared/caringbridge/data/projects/recsys-peer-match/model_data"
cat "${model_data_dir}/predict_candidate_journal_oids.txt" "${model_data_dir}/predict_source_journal_oids.txt" > "${model_data_dir}/predict_journal_oids.txt"
python cbrec/text/createTextFeatureSqlite.py --text-id-txt "${model_data_dir}/predict_journal_oids.txt"
echo "Finished sbatch script."

