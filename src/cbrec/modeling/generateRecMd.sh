#!/bin/bash -l        
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem=62g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=levon003@umn.edu 
#SBATCH --job-name=generate_rec_contexts
#SBATCH -o /home/lana/shared/caringbridge/data/projects/recsys-peer-match/sbatch/generateRecMd.stdout
#SBATCH -e /home/lana/shared/caringbridge/data/projects/recsys-peer-match/sbatch/generateRecMd.stderr

# to submit: sbatch -p agsmall generateRecMd.sh
working_dir="/home/lana/levon003/repos/recsys-peer-match/src"
cd $working_dir
source activate pytorch-cpuonly || exit 1
echo "From '$working_dir', generate recontext meta data"
python cbrec/modeling/generateRecMd.py
echo "Finished generating."