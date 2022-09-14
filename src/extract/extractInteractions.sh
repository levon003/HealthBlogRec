#!/bin/bash -l        
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=4g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=levon003@umn.edu 
#SBATCH --job-name=recsys_extractInteractions
#SBATCH -o /home/lana/shared/caringbridge/data/projects/recsys-peer-match/activity/sbatch/extractInteractions.stdout
#SBATCH -e /home/lana/shared/caringbridge/data/projects/recsys-peer-match/activity/sbatch/extractInteractions.stderr

# to submit: sbatch -p amdsmall extractInteractions.sh
working_dir="/home/lana/levon003/repos/recsys-peer-match/src/extract"
cd $working_dir
echo "In '$working_dir', running script."
python extractInteractions.py --collection-name reaction > /home/lana/shared/caringbridge/data/projects/recsys-peer-match/activity/sbatch/extractInteractions_reaction.log 2>&1 &
python extractInteractions.py --collection-name site_profile > /home/lana/shared/caringbridge/data/projects/recsys-peer-match/activity/sbatch/extractInteractions_site_profile.log 2>&1 &
echo "Queued extraction jobs."
wait
echo "Finished sbatch script."

