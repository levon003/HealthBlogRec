#!/bin/bash -l        
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --mem=10g

# called from submitEvalFromDirectory.py
# note: the calling script sets a variety of sbatch properties

# Variables must be passed into the script with the --export flag:
#    USERNAME: the username of the repo you want to run the script from (e.g. mcnam385, levon003)
#    MODEL_FILEPATH: the json filepath of the model to evaluate

source activate pytorch-cpuonly || exit 1
echo "Running eval as user '$USERNAME' for model '$MODEL_FILEPATH'."
python /home/lana/$USERNAME/repos/recsys-peer-match/src/cbrec/modeling/evaluateModelFromCacheMp.py --model-filepath $MODEL_FILEPATH $VALIDATION_ONLY $TEST_ONLY
echo "Finished sbatch script."
