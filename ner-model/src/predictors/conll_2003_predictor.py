from overrides import overrides

import numpy as np

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.predictors.predictor import Predictor
from allennlp.models import Model

@Predictor.register("conll_2003_predictor")
class CoNLL2003Predictor(Predictor):
    def __init__(self, model:Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)

        outputs["predicted_labels"] = [self._model.vocab.get_token_from_index(i, 'labels') for i in np.argmax(outputs["tag_logits"], axis=-1)]
        del outputs["tag_logits"]

        return sanitize(outputs)
