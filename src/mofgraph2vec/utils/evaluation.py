def evaluate_model(model, documents, patience):
    score = 0
    for doc_id in range(len(documents)):
        inferred_vector = model.infer_vector(documents[doc_id].words, epochs=40)
        sims = model.dv.most_similar([inferred_vector], topn=patience)

        if documents[doc_id].tags[0] in list(zip(*sims))[0]:
            score += 1
    
    return score/len(documents)
