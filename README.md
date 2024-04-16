# MOFgraph2vec

A metal-organic framework (MOF) recommendation system based on Doc2Vec. 

## Installation

We suggest you install the package into a seperate conda environment with Python 3.8 or higher. Follow the steps below:

```
$ git clone https://github.com/XiaoqZhang/mofgraph2vec.git
$ cd mofgraph2vec
$ pip install -e .
```

## Getting started
### Train the model

- Change the configurations in `conf/`
- Navigate to `experiments/workflow`
- Run the model by `$ python train.py`

### Load the model

- The pre-trained model for ARC-MOF and QMOF databases are provided

### Similarity analysis

- Examples are given in `dev/similarity.ipynb`

## License

This project is licensed under the MIT License. See the LICENSE file for more information.