from mofgraph2vec.trainer.train import train

if __name__ == "__main__":
    train(
        task="logKH_CO2",
        id_column="cif.label",
        label_path="../../data/data.csv",
        embedding_path="../../data/vec/embedding.csv",
        train_frac=0.8,
        valid_frac=0.1,
        test_frac=0.1,
        batch_size=64,
        seed=2034
    )