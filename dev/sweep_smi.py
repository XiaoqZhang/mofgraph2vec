import os
import wandb
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from mofgraph2vec.utils.saving import save_embedding
from xgboost import XGBRegressor
from mofgraph2vec.utils.loss import get_numpy_regression_metrics

import os
import pandas as pd
from sklearn.model_selection import train_test_split

import wandb

RANDOM_SEED = int(1414)

config = {
    'alpha': {
        'distribution': 'uniform',
        'min': 1e-2,
        'max': 3e-2
    },
    'window': {
        'distribution': 'int_uniform',
        'min': 2,
        'max': 40
    },
    'vector_size': {
        'distribution': 'int_uniform',
        'min': 20,
        'max': 500
    }
}



# return us a sweep id (required for running the sweep)
def get_sweep_id(method):
    sweep_config = {
        'method': method,
        'metric': {
            'name': 'test_mse',
            'goal': 'minimize'
        },
        'early_terminate': {
            'type': 'hyperband',
            's': 2,
            'eta': 3,
            'max_iter': 30
        },
        'parameters': config,
    }
    sweep_id = wandb.sweep(sweep_config, project='mof2vec')

    return sweep_id

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def train():
    # Config is a variable that holds and saves hyperparameters and inputs

    configs = {
        'alpha': 1.3e-2,
        'window': 10,
        'vector_size': 50,
        'min_count': 0,
        'sample': 1.0,
        'dm': 1,
        'epochs': 100
    }

    # Initilize a new wandb run
    wandb.init(project='mof2vec', config=configs)

    config = wandb.config

    documents = []
    mof_names = os.listdir("../data/bbs/")
    for n in mof_names:
        with open("../data/bbs/%s/python_mofid.txt" %n) as file:
            line = file.readline().split()[0]
            documents.append(
                TaggedDocument(
                    words=smi_tokenizer(line),
                    tags=[n.rstrip('.cif')]
                )
            )

    model = Doc2Vec(**config)
    model.build_vocab(documents)
    
    model.train(
        documents, total_examples=model.corpus_count, epochs=100
    )
    
    save_embedding(
        pretraining = True,
        output_path = './test', 
        model = model, 
        documents = documents, 
        doc_dimensions = config.vector_size, 
        topo_vectors = None, 
        topo_dimensions = None
    )

    df_vectors = pd.read_csv('./test/embedding_dv.csv').set_index('type')
    df_label = pd.read_csv('../data/data.csv').set_index('cif.label')
    df = df_vectors.join(df_label.loc[:, "logKH_CO2"])

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    X_train = df_train.values[:, :-1]
    y_train = df_train.values[:, -1].reshape(-1,1)
    X_test = df_test.values[:, :-1]
    y_test = df_test.values[:, -1].reshape(-1,1)

    regressor = XGBRegressor()
    regressor.fit(X_train, y_train)
    pred = regressor.predict(X_test)
    true = y_test.flatten()
    
    metrics = get_numpy_regression_metrics(y_train, regressor.predict(X_train), "train")
    metrics.update(get_numpy_regression_metrics(true, pred, "test"))
    
    wandb.log(metrics)


def main():
    sweep_id = get_sweep_id('bayes')
    wandb.agent(sweep_id, function=train, count=300)


if __name__ == '__main__':
    main()


