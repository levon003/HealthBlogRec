#!/bin/bash -l        
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=62g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=levon003@umn.edu 
#SBATCH --job-name=cb_recsys_text_allreq
#SBATCH -o /home/lana/shared/caringbridge/data/projects/recsys-peer-match/sbatch/text_allreq.stdout
#SBATCH -e /home/lana/shared/caringbridge/data/projects/recsys-peer-match/sbatch/text_allreq.stderr

# input file created via: cat predict_participant_journal_oids.txt predict_candidate_journal_oids.txt missing_train_journal_oids.txt test2train_required_journal_oids.txt > all_required_journal_oids.txt


# to submit: sbatch -p amdsmall make_all_required_text_features.sh
working_dir="/home/lana/levon003/repos/recsys-peer-match/src"
cd $working_dir
source activate pytorch-cpuonly || exit 1
echo "In '$working_dir', running script."
python cbrec/text/createTextFeatureSqlite.py --text-id-txt /home/lana/shared/caringbridge/data/projects/recsys-peer-match/model_data/all_required_journal_oids.txt
echo "Finished sbatch script."

