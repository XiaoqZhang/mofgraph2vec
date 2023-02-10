from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec
from mofgraph2vec.utils.evaluation import evaluate_model
from loguru import logger

class AccuracyCallback(CallbackAny2Vec):
    def __init__(self, path_prefix, test_corpus, patience):
        super().__init__()
        self.path_prefix = path_prefix
        self.test_corpus = test_corpus
        self.epoch = 0
        self.scores = [0]
        self.patience = patience

    def on_epoch_end(self, model):
        accuracy = evaluate_model(model, self.test_corpus, self.patience)
        output_path = get_tmpfile('{}/epoch{}.model'.format(self.path_prefix, self.epoch))
        if accuracy > self.scores[-1]:
            model.save(output_path)
        self.scores.append(accuracy)
        logger.info("Epoch %i end with model accuracy %.4f. " %(self.epoch, accuracy))
        self.epoch += 1
        return accuracy