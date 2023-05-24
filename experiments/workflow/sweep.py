import hydra
from omegaconf import DictConfig
import os

from mofgraph2vec.sweep.sweep import sweep

@hydra.main(config_path="../../conf", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    for path in [
        "../../data/features/RSM/mof2vec.csv",
        "../../data/features/RSM/geo_mof2vec.csv"
        "../../data/features/RSM/RACs_mof2vec.csv",
        "../../data/features/RSM/stoi120_mof2vec.csv"
    ]:
        config.doc2label_data.embedding_path = path
        sweep(config)
    
    config.doc2label_data.task = ["logKH_CH4"]
    for path in [
        "../../data/features/RSM/geo.csv",
        "../../data/features/RSM/rsm-stoi120.csv",
        "../../data/features/RSM/RACs.csv"
        "../../data/features/RSM/geo_stoi120.csv",
        "../../data/features/RSM/geo_mof2vec.csv",
        "../../data/features/RSM/geo_RACs.csv",
        "../../data/features/RSM/RACs_mof2vec.csv",
        "../../data/features/RSM/stoi120_mof2vec.csv",
    ]:
        config.doc2label_data.embedding_path = path
        sweep(config)

    config.doc2label_data.task = ["pure_uptake_CO2_298.00_15000"]
    for path in [
        "../../data/features/RSM/geo.csv",
        "../../data/features/RSM/rsm-stoi120.csv",
        "../../data/features/RSM/RACs.csv"
        "../../data/features/RSM/geo_stoi120.csv",
        "../../data/features/RSM/geo_mof2vec.csv",
        "../../data/features/RSM/geo_RACs.csv",
        "../../data/features/RSM/RACs_mof2vec.csv",
        "../../data/features/RSM/stoi120_mof2vec.csv",
    ]:
        config.doc2label_data.embedding_path = path
        sweep(config)

    config.doc2label_data.task = ["pure_uptake_CO2_298.00_1600000"]
    for path in [
        "../../data/features/RSM/geo.csv",
        "../../data/features/RSM/rsm-stoi120.csv",
        "../../data/features/RSM/RACs.csv"
        "../../data/features/RSM/geo_stoi120.csv",
        "../../data/features/RSM/geo_mof2vec.csv",
        "../../data/features/RSM/geo_RACs.csv",
        "../../data/features/RSM/RACs_mof2vec.csv",
        "../../data/features/RSM/stoi120_mof2vec.csv",
    ]:
        config.doc2label_data.embedding_path = path
        sweep(config)

    config.doc2label_data.task = ["pure_uptake_methane_298.00_580000"]
    for path in [
        "../../data/features/RSM/geo.csv",
        "../../data/features/RSM/rsm-stoi120.csv",
        "../../data/features/RSM/RACs.csv"
        "../../data/features/RSM/geo_stoi120.csv",
        "../../data/features/RSM/geo_mof2vec.csv",
        "../../data/features/RSM/geo_RACs.csv",
        "../../data/features/RSM/RACs_mof2vec.csv",
        "../../data/features/RSM/stoi120_mof2vec.csv",
    ]:
        config.doc2label_data.embedding_path = path
        sweep(config)

    config.doc2label_data.task = ["pure_uptake_methane_298.00_6500000"]
    for path in [
        "../../data/features/RSM/geo.csv",
        "../../data/features/RSM/rsm-stoi120.csv",
        "../../data/features/RSM/RACs.csv"
        "../../data/features/RSM/geo_stoi120.csv",
        "../../data/features/RSM/geo_mof2vec.csv",
        "../../data/features/RSM/geo_RACs.csv",
        "../../data/features/RSM/RACs_mof2vec.csv",
        "../../data/features/RSM/stoi120_mof2vec.csv",
    ]:
        config.doc2label_data.embedding_path = path
        sweep(config)


if __name__ == "__main__":
    main()
    os.system('say "Your sweep has finished"')