from collections import Counter
import random

def evaluate_model(model, documents):
    selected_idx = random.sample(range(len(documents)), 100)
    ranks = []
    for doc_id in selected_idx:
        inferred_vector = model.infer_vector(documents[doc_id].words)
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
        rank = [docid for docid, _ in sims].index(documents[doc_id].tags[0])
        ranks.append(rank)

    return Counter(ranks)[0]/100