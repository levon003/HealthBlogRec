#!/bin/bash -l        
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --mem=60g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zentx005@umn.edu
#SBATCH --job-name=cb_recsys_clicksiteanalysis
#SBATCH -o /home/lana/shared/caringbridge/data/projects/recsys-peer-match/sbatch/tmp_clickAnalysis.stdout
#SBATCH -e /home/lana/shared/caringbridge/data/projects/recsys-peer-match/sbatch/tmp_clickAnalysis.stderr

#source activate pytorch-cpuonly || exit 1
echo "Starting sbatch script."
python /home/lana/zentx005/repos/recsys-peer-match/notebook/retention/ClickedSiteAnalysis.py
echo "Finished sbatch script."