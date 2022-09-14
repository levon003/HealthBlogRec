#!/bin/bash -l
#SBATCH --time=90:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=156g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=levon003@umn.edu 
#SBATCH --job-name=cb_recsys_gen
#SBATCH -o /home/lana/shared/caringbridge/data/projects/recsys-peer-match/sbatch/gen.stdout
#SBATCH -e /home/lana/shared/caringbridge/data/projects/recsys-peer-match/sbatch/gen.stderr

# to submit: sbatch -p agsmall gen.sh
working_dir="/home/lana/levon003/repos/recsys-peer-match/src"
cd $working_dir
source activate pytorch-cpuonly || exit 1
echo "In '$working_dir', running script."
python gen.py
echo "Finished sbatch script."

