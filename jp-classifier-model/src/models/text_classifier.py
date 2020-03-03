from typing import Dict
from overrides import overrides

import torch

from allennlp.models import Model

from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder

from allennlp.nn.util import get_text_field_mask

from allennlp.training.metrics import CategoricalAccuracy

@Model.register('text_classifier')
class TextClassifier(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder

        self.classification_layer = torch.nn.Linear(in_features = encoder.get_output_dim(),
                                                    out_features = vocab.get_vocab_size('labels'))
        self._loss = torch.nn.CrossEntropyLoss()
        self._accuracy = CategoricalAccuracy()

    @overrides
    def forward(self,
                sentence: Dict[str, torch.Tensor],
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)

        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)

        label_logit = self.classification_layer(encoder_out)
        output = {}

        if label is not None:
            self._accuracy(label_logit, label)
            output["loss"] = self._loss(label_logit, label.long())

        return output

    @overrides
    def get_metrics(self,
                    reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self._accuracy.get_metric(reset)}


