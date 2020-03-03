from typing import Dict
from overrides import overrides

import torch

from allennlp.models import Model

from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from allennlp.training.metrics import SpanBasedF1Measure

@Model.register('ner_tagger')
class NERTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder

        self.hidden2tag = torch.nn.Linear(in_features = encoder.get_output_dim(),
                                          out_features = vocab.get_vocab_size('labels'))

        self._f1_metric = SpanBasedF1Measure(vocab, 'labels')

    @overrides
    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)

        embeddings = self.word_embeddings(sentence)

        encoder_out = self.encoder(embeddings, mask)

        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}

        if labels is not None:
            self._f1_metric(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output

    @overrides
    def get_metrics(self,
                    reset: bool = False) -> Dict[str, float]:
        return self._f1_metric.get_metric(reset=reset)

