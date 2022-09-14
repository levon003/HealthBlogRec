#!/bin/bash -l  
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --mem=62g
#SBATCH --mail-type=NONE
#SBATCH --mail-user=levon003@umn.edu
#SBATCH --job-name=cb_recsys_train_model

# SLURM script which runs the python model generation script with personalized parameters. The sbatch commands above are used by default, 
# but any one of these can be overwritten by passing new values into the script (--mail-type=ALL, --mail-user=levon003@umn.edu, etc.).

# Two variables must be passed into the script with the --export flag:
#    USERNAME: the username of the repo you want to run the script from (i.e. mcnam385, levon003)
#    CONFIG: the config json file containing all the parameters for the model we want to run. This file must be stored in src/cbrec/experiment/inputs

# Example script:
# $ sbatch -p agsmall --mail-type=ALL --mail-user=levon003@umn.edu --export=USERNAME='levon003',CONFIG='test.json' model_gen_script.sh

source activate 
source activate pytorch-cpuonly || exit 1

echo "Running python model_gen.py $USERNAME $CONFIG"
python /home/lana/$USERNAME/repos/recsys-peer-match/src/cbrec/experiment/model_gen.py $USERNAME $CONFIG
echo "Finished sbatch script."
