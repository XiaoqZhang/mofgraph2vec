import random
import numpy as np
from loguru import logger
from mofgraph2vec.utils.evaluation import evaluate_model

def cross_validation(data, model, k_foldes, epochs, patience):
    random.shuffle(data)
    num = int(len(data)/k_foldes)

    slices = []
    for k in range(k_foldes):
        if k != (k_foldes -1):
            slices.append(data[k*num: (k+1)*num])
        else:
            slices.append(data[k*num:])

    indexes = np.arange(k_foldes)
    scores = []
    for k in range(k_foldes):
        valid_data = slices[k]
        training_data = [ele for id in indexes if id != k for ele in slices[id]]
        logger.info(f"Cross validation with {len(training_data)} training data and {len(valid_data)} validation data. ")
        model.train(training_data, total_examples=model.corpus_count, epochs=epochs)
        score = evaluate_model(model,valid_data, patience)
        scores.append(score)

        logger.info(f"{k+1}-folder validation accuracy: {score}")
    
    return np.mean(scores), np.std(scores)
