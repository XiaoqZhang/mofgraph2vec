# MOFgraph2vec

A metal-organic framework (MOF) recommendation system based on Doc2Vec. 

## Installation

We suggest you install the package into a seperate conda environment with Python 3.8 or higher. Follow the steps below:

```
$ conda create -n mof2vec python=3.8 -y
$ conda activate mof2vec
$ git clone https://github.com/XiaoqZhang/mofgraph2vec.git
$ cd mofgraph2vec
$ python -m pip install -e .
```

## Getting started
### Train the model

- To prepare the data, you need a folder that contains a folder with all MOF CIFs and a `.csv` file with geometry and topology information. Examples are shown in `example_data`.
- Change the configurations in `conf/`. See more details in [Configuration parameters](#configuration-parameters). 
- Navigate to the folder `experiments/`.
- Run the model by `$ python train.py`. The first time of featurization may take some time. 

### Load the pre-trained models

- The pre-trained models for ARC-MOF and QMOF databases are attached in Releases. 
- Example of loading the pre-trained models is provided in `dev/example.ipynb`. 

### Similarity analysis

- Examples are given in `dev/example.ipynb`. 

## Configuration parameters
You can easily tune the model parameters in `conf` folder. 

### config.yaml

- `mof2vec_data`: Configuration specific to the MOF embedding data. 
- `mof2vec_model`: Configuration related the MOF embedding model. 
- `doc2label_data`: Configuration for the data for training the downstream regression model. 
- `doc2label_model`: Configuration for the downstream regression model. 
- `sweep`: Wandb hyperparameter sweeping configuration.

#### logger
This section contains setting for wandb logging and tracking experiments. 

#### seed
Fix random seed, ensuring reproducibility in experiments. 

#### sweep
A boolean indicating whether wandb hyperparameter sweeping is enabled or not. 

#### mode
- `mof2vec`: Only run MOF embedding. 
- `doc2lable`: Only run downstream regression model. In this mode, `pretraining` should be set to `False` and MOF embeddings should be provided in `embedding_path` in the configuration file `doc2label_data/default.yaml`. 
- `workflow`: Run the MOF embedding and use the embedding as features for the downstream regression model. 

### mof2vec_data

- `cif_path`: The path to the folder that contains all the `.cif` files to embed. 
- `embed_label`: A boolean indicating whether to parsing geometric properties of MOFs. 
- `label_path`: If `embed_label` is set to True. The `.csv` file that contains the geometry information should be provided. 
- `descriptors_to_embed`: The numeric columns to parse in the `.csv` file. 
- `category_to_embed`: The category columns to parse in the `.csv` file. 
- `id_column`: The column that contains the cif names. 
- `wl_step`: The number of steps in extracting the rooted substructures. 

### mof2vec_model
- `vector_size`: Dimensionality of the embeddings. 
- `window`: The maximum distance between the current and predicted word within a MOF document.
- `min_count`: Ignores all words with total frequency lower than this. 
- `dm`: Defines the training algorithm. If dm=1, ‘distributed memory’ (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is employed.
- `sample`: The threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).
- `workers`: Use these many worker threads to train the model (=faster training with multicore machines).
- `alpha`: The initial learning rate. 
- `epochs`: Number of iterations (epochs) over the corpus. 
More parameters can be added. Find the details in [gensim.models.doc2vec.Doc2Vec](https://radimrehurek.com/gensim/models/doc2vec.html)

### doc2label_data

- `pretraining`: If set `True`, the MOF embedding model is trained from scratch. If set to `False`, load MOF embeddings from pre-trained models. In this case, either `embedding_model_path` or `embedding_path` should be provided. 
- `embedding_model_path`: Pretrained embedding model path. 
- `embedding_path`: MOF embeddings from pre-trained models. 
- `label_path`: The path to the `.csv` file that contains the downstream regression data. 
- `task`: The task column for the downstream regression model. 
- `MOF_id`: The column that contains MOF names. 
- `train_frac`: The training data size. 
- `test_frac`: The test data size. 
- `num_workers`: Specify the number of workers to train the regression model. 

### doc2label_model

#### grid_search
Specify whether to turn on a Grid search or not. If `True`, all the values in `params` should be provided as `list`. The grid search is performed with a 5-fold cross-validation. If `False`, all the values should be `float` or `int`. 

#### params
The hyperparameters for the supervised [XGBoost model](https://xgboost.readthedocs.io/en/stable/parameter.html). 

#### n_jobs
Number of jobs to run in parallel. 

## License

This project is licensed under the MIT License. See the LICENSE file for more information.