#!/bin/bash -l        
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=196g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=levon003@umn.edu 
#SBATCH --job-name=cb_recsys_genc
#SBATCH -o /home/lana/shared/caringbridge/data/projects/recsys-peer-match/sbatch/gen_continue.stdout
#SBATCH -e /home/lana/shared/caringbridge/data/projects/recsys-peer-match/sbatch/gen_continue.stderr

# to submit: sbatch -p agsmall gen_continue.sh
working_dir="/home/lana/levon003/repos/recsys-peer-match/src"
cd $working_dir
source activate pytorch-cpuonly || exit 1
echo "In '$working_dir', running script."
python gen.py --from-recent-checkpoint
echo "Finished sbatch script."

