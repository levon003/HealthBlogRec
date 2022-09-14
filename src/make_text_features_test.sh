#!/bin/bash -l
#SBATCH --time=32:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --mem=110g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=levon003@umn.edu 
#SBATCH --job-name=cb_recsys_text_test
#SBATCH -o /home/lana/shared/caringbridge/data/projects/recsys-peer-match/sbatch/text_test.stdout
#SBATCH -e /home/lana/shared/caringbridge/data/projects/recsys-peer-match/sbatch/text_test.stderr

# to submit: sbatch -p agsmall make_text_features_test.sh
working_dir="/home/lana/levon003/repos/recsys-peer-match/src"
cd $working_dir
source activate pytorch-cpuonly || exit 1
echo "In '$working_dir', running script."
python cbrec/text/createTextFeatureSqlite.py --text-id-txt /home/lana/shared/caringbridge/data/projects/recsys-peer-match/model_data/test_journal_oids.txt --text-feature-db-filename test_text_feature.sqlite --n-processes 48
echo "Finished sbatch script."

