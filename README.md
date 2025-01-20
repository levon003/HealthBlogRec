# HealthBlogRec

[![arXiv](https://img.shields.io/badge/arXiv-2209.04973-b31b1b.svg)](https://arxiv.org/abs/2209.04973)
[![License](https://img.shields.io/github/license/levon003/HealthBlogRec)](https://github.com/levon003/HealthBlogRec/blob/main/LICENSE)



HealthBlogRec is a recommendation system for peer health blogs.

![Recommender system overview](/figures/rec_system_overview.png)

## Citation and project history

If any portion of this project is useful to you, please cite the following paper: 

>Zachary Levonian, Matthew Zent, Ngan Nguyen, Matthew McNamara, Loren Terveen, and Svetlana Yarosh. 2022. “Some other poor soul’s problems”: a peer recommendation intervention for health-related social support. arXiv:2209.04973 (September 2022), 58 pages. http://arxiv.org/abs/2209.04973

The project was started in January 2021, a field study with the system was conducted August - December 2021, and a results write-up was submitted to the CSCW conference in July 2022. After a round of major revisions, a rejection, and another round of major revisions, the paper was accepted to CSCW 2025 in September 2024. All development occurred in a private repository hosted [here](https://github.com/umncs-caringbridge/recsys-peer-match) (git logs [here](/gitlog.txt)).

### Code contributors
 - Zachary Levonian <levon003@umn.edu>
 - Matthew Zent
 - Ngan Nguyen
 - Matthew McNamara

## Repository structure

 - `src`: Python source files.
   - Top level: The bash scripts are designed to work with the [Slurm](https://slurm.schedmd.com/documentation.html) scheduling system using on MSI. The core entry points to the cbrec package are convenience scripts: `gen.py` to generate training data and `predict.py` for making predictions from a trained model. `*test*.py` are for limited automated testing.
   - `cbrec`: Python package for recommendation: data processing, model training, and evaluation.
     - Top level: entry point for training and test data generation is `triple_generation.py`. Other files support tracking of recent activity, interaction network structure, and I/O for the generated features and metadata.
     - `text`: For managing RoBERTa embeddings in an sqlite database.
     - `modeling`: Implements the actual model, including routines for data preprocessing and optimization.
     - `experiment`: For generating experimental configurations for offline model training and evaluation. Used for hyperparameter search.
   - `cbsend`: For templating and sending emails to participants.
   - `extract`: For converting MongoDB BSON exports into flattened ndJSON representations.
 - `notebook`: Jupyter Notebooks for analysis and experimentation with recommendation systems models.  Unfortunately, the internal user IDs can be linked to public CaringBridge profiles, so all cell outputs are cleared and all ID/email references stripped. Figures generated for the paper can be found in `figures`, but all other figures have been cleared.
   - `eval`: Modeling, including the weekly train & predict in `PytorchTraining.ipynb`.  Offline evaluation is here too, for both trained models and non-trained baselines.
   - `model_data`: Data preparation, cleaning, and transformation. Includes one-off investigations of outliers and unexpected data distributions.
   - `prerec_evidence`: Data analysis mostly conducted before the study. Includes an analysis of the retention impact of author comments (sec3.1 in [arXiv:2209.04973v1](https://arxiv.org/abs/2209.04973v1)).
   - `retention`: Effect size estimates for the impact of participation on other observed behaviors (sec5.3.3 and Appendix G in [arXiv:2209.04973v1](https://arxiv.org/abs/2209.04973v1)).
   - `sqlite`: Small data transformations for internal testing.
   - `survey`: Survey data analysis and monitoring of participant activity during the field study.
   - `torch_experiments`: Modeling experiments, including some hyperparameter optimization and testing of experiment infrastructure.
 - `data`: small data objects stored with the repository. A few examples are provided. All of the "real" data was stored in compressed BSON, JSON, and SQLite files on MSI.


## Using HealthBlogRec 

### Conda environments

All code was executed in the cluster hosted by the [Minnesota Supercomputing Institute](https://www.msi.umn.edu/) (MSI) at the University of Minnesota.
On MSI, the `pytorch-cpuonly` environment was created with Python 3.9.4.

Conda managed requirements:
 - torch
 - numpy
 
Pip managed requirements:
 - transformers
 - pandas
 - matplotlib
 - sklearn
 - scipy
 - networkx
 - ipykernel
 
### Setup and data cleaning for model training

1. `gen.py` - generate training and testing data, ~50 hours
2. `PytorchTraining.ipynb` - generate train_journal_oids.txt and test_journal_oids.txt
3. `make_text_features_train.sh` and `make_text_features_test.sh` - cache the embeddings for each journal id, ~27 hours for train, ~11 hours for test
4. `PytorchTraining.ipynb` - create and pickle the X_train_raw and y_train_raw numpy matrices, ~1 hour
5. `generateRecMd.py` - cache the prediction and test contexts, ~12 hours

### Offline Training and Evaluation

1. `cbrec.experiment.config_gen.py` - create a training configuration
2. Follow the instructions to train the models
3. Follow the instructions to evaluate the models (`submitEvalFromDirectory.py`)
4. Create a Jupyter notebook to compare the models based on their validation output - sample: ?
5. Given a JSON filepath of a model you want to do test evaluation on, run: `submitEvalFromDirectory.py --model-filepath {.json filepath} --test-only`
6. Load the test metadata filepath (`_test_metadata.ndjson`) and the coverage scores (`_coverage_scores.pkl`).

A separate evaluation process was used for the baselines (see original implementation `notebook/eval/BaselineCompute.ipynb`)

### Can I use this to build my own recommendation system?

Probably not without substantial development effort.  Most of the value is captured in the general approach, which is described in the arXiv paper. You might be interested in the candidate identification and negative sampling (see `src/cbrec/triple_generation.py`), the model (see `src/cbrec/modeling/models/linearnet.py` and similar), the model optimization procedure (see `src/cbrec/modeling/train.py`), or the RoBERTa feature extraction (see `src/cbrec/text/createTextFeatureSqlite.py`).

## Fixed bugs during deployment

We caught and fixed a few bugs during development that affected the system deployed during the user study.

 - During the user study, a bug in the way activity "time elapsed" features was computed produced a bimodal distribution where no recent interaction of that type produced a large positive value.  This bug was corrected for the offline results reported in the paper.
 - During the user study, a timezone bug introduced during a system upgrade produced incorrect timestamps for specifically comments before March 28, 2018.  The data were reprocessed and the bug corrected for the offline evaluations reported in the paper.
 - During the user study, a bug in the feature preprocessing code resulted in source USP activity and network features being omitted (specifically, replaced with duplicates of the candidate USP features) from training triples generated from historical data after July 1, 2020. This bug would likely have had a negative but minor impact on the model performance, roughly equivalent to dropping source USP activity and network features for about 12% of the training data.
 
 
