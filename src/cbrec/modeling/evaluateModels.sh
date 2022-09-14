#!/bin/bash -l        
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=26g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=levon003@umn.edu
#SBATCH --job-name=cb_recsys_evaluateModels
#SBATCH -o /home/lana/shared/caringbridge/data/projects/recsys-peer-match/sbatch/tmp_evaluateModels.stdout
#SBATCH -e /home/lana/shared/caringbridge/data/projects/recsys-peer-match/sbatch/tmp_evaluateModels.stderr

# to submit: sbatch -p amdsmall evaluateModels.sh
working_dir="/home/lana/levon003/repos/recsys-peer-match/src"
cd $working_dir
source activate pytorch-cpuonly || exit 1
echo "In '$working_dir', running script."
python cbrec/modeling/evaluateModels.py --model-filepath /home/lana/shared/caringbridge/data/projects/recsys-peer-match/torch_experiments/modeling/feature_ablations_20220126230350/outputs/feature_ablations_all.json
echo "Finished sbatch script."

