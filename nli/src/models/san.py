from typing import Any, Dict, List, Optional, Tuple, cast

import torch
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import (
    FeedForward, MatrixAttention, Seq2SeqEncoder, TextFieldEmbedder, TimeDistributed
)
from allennlp.modules.attention import BilinearAttention
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util


@Model.register("san")
class StochasticAnswerNetworks(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 lexical_feedforward: FeedForward,
                 contextual_encoder: Seq2SeqEncoder,
                 attention_feedforward: FeedForward,
                 matrix_attention: MatrixAttention,
                 memory_encoder: Seq2SeqEncoder,
                 output_feedforward: FeedForward,
                 output_logit: FeedForward,
                 answer_steps: int = 5,
                 dropout: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._lexical_feedforward = TimeDistributed(lexical_feedforward)
        self._contextual_encoder = contextual_encoder
        self._attention_feedforward = TimeDistributed(attention_feedforward)
        self._matrix_attention = matrix_attention
        self._memory_encoder = memory_encoder
        self._output_feedforward = output_feedforward
        self._output_logit = output_logit
        self._answer_steps = answer_steps
        self._answer_gru_cell = torch.nn.GRUCell(
            self._memory_encoder.get_output_dim(),
            self._memory_encoder.get_output_dim(),
        )
        self._answer_attention = TimeDistributed(
            torch.nn.Linear(self._memory_encoder.get_output_dim(), 1)
        )
        self._answer_bilinear = BilinearAttention(
            self._memory_encoder.get_output_dim(),
            self._memory_encoder.get_output_dim(),
        )

        check_dimensions_match(text_field_embedder.get_output_dim(), lexical_feedforward.get_input_dim(),
                               "text field embedding dim", "lexical feedforward input dim")
        check_dimensions_match(lexical_feedforward.get_output_dim(), contextual_encoder.get_input_dim(),
                               "lexical feedforwrd input dim", "contextual layer input dim")
        check_dimensions_match(contextual_encoder.get_output_dim(), attention_feedforward.get_input_dim(),
                               "contextual layer output dim", "attention feedforward input dim")
        check_dimensions_match(contextual_encoder.get_output_dim() * 2, memory_encoder.get_input_dim(),
                               "contextual layer output dim", "memory encoder input dim")
        check_dimensions_match(memory_encoder.get_output_dim() * 4, output_feedforward.get_input_dim(),
                               "memory encoder output dim", "output feedforward input")
        check_dimensions_match(output_feedforward.get_output_dim(), output_logit.get_input_dim(),
                               "output feedforward output dim", "output logit input")

        self._dropout = torch.nn.Dropout(dropout) if dropout else None

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.NLLLoss()

        initializer(self)

    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None # pylint: disable=unused-argument
               ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        premise_embeddings = self._text_field_embedder(premise)
        hypothesis_embeddings = self._text_field_embedder(hypothesis)

        premise_mask = util.get_text_field_mask(premise).float()
        hypothesis_mask = util.get_text_field_mask(hypothesis).float()

        # Lexicon Encoding Layer
        premise_lexical_embeddings = self._lexical_feedforward(premise_embeddings)
        hypothesis_lexical_embeddings = self._lexical_feedforward(hypothesis_embeddings)

        # Contextual Encoding Layer
        encoded_premise = self._contextual_encoder(
            premise_lexical_embeddings, premise_mask
        )
        encoded_hypothesis = self._contextual_encoder(
            hypothesis_lexical_embeddings, hypothesis_mask
        )

        # Memory Layer
        premise_memory, hypothesis_memory = self._compute_memory(
            encoded_premise, encoded_hypothesis,
            premise_mask, hypothesis_mask,
        )

        # Answer Module
        label_probs = self._compute_answer(
            premise_memory, hypothesis_memory,
            premise_mask, hypothesis_mask
        )

        output_dict = {"label_probs": label_probs}

        if label is not None:
            label_log_probs = (label_probs + 1e-45).log()
            loss = self._loss(label_log_probs, label.long().view(-1))
            self._accuracy(label_probs, label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}

    def _compute_memory(
            self,
            encoded_premise: torch.Tensor,
            encoded_hypothesis: torch.Tensor,
            premise_mask: torch.Tensor,
            hypothesis_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shape: (batch_size, premise_length, hypothesis_length)
        attention_matrix = self._matrix_attention(
            self._attention_feedforward(encoded_premise),
            self._attention_feedforward(encoded_hypothesis),
        )

        if self._dropout:
            attention_matrix = self._dropout(attention_matrix)

        # Shape: (batch_size, premise_length, hypothesis_length)
        p2h_attention = util.masked_softmax(attention_matrix, hypothesis_mask)
        # Shape: (batch_size, premise_length, embedding_dim)
        attended_hypothesis = util.weighted_sum(encoded_hypothesis, p2h_attention)

        # Shape: (batch_size, hypothesis_length, premise_length)
        h2p_attention = util.masked_softmax(
            attention_matrix.transpose(1, 2).contiguous(), premise_mask)
        # Shape: (batch_size, hypothesis_length, embedding_dim)
        attended_premise = util.weighted_sum(encoded_premise, h2p_attention)

        premise_memory = self._memory_encoder(
            torch.cat([encoded_premise, attended_hypothesis], dim=-1),
            premise_mask,
        )
        hypothesis_memory = self._memory_encoder(
            torch.cat([encoded_hypothesis, attended_premise], dim=-1),
            hypothesis_mask,
        )

        return premise_memory, hypothesis_memory

    def _compute_answer(self,
                        premise_memory: torch.Tensor,
                        hypothesis_memory: torch.Tensor,
                        premise_mask: torch.Tensor,
                        hypothesis_mask: torch.Tensor) -> torch.Tensor:
        batch_size = premise_memory.size(0)
        num_labels = self._output_logit.get_output_dim()

        # Shape: (batch_size, hypothesis_length)
        hypothesis_attention = util.masked_softmax(
            self._answer_attention(hypothesis_memory).squeeze(),
            hypothesis_mask,
        )
        # Shape: (batch_size, embedding_dim)
        answer_state = util.weighted_sum(hypothesis_memory, hypothesis_attention)

        label_prob_steps: torch.Tensor = answer_state.new_zeros(
            (batch_size, num_labels, self._answer_steps)
        )
        for step in range(self._answer_steps):
            # Shape: (batch_size, premise_length)
            premise_attention = self._answer_bilinear(answer_state, premise_memory, premise_mask)
            # Shape: (batch_size, embedding_dim)
            cell_input = util.weighted_sum(premise_memory, premise_attention)

            answer_state = self._answer_gru_cell(cell_input, answer_state)

            output_hidden = torch.cat([
                answer_state,
                cell_input,
                (answer_state - cell_input).abs(),
                answer_state * cell_input,
            ], dim=-1)
            label_logits = self._output_logit(self._output_feedforward(output_hidden))
            label_prob_steps[:, :, step] = label_logits.softmax(-1)

        if self.training and self._dropout:
            # stochastic prediction dropout
            binary_mask = (
                torch.rand((batch_size, self._answer_steps)) > self._dropout.p
            ).to(label_prob_steps.device)
            label_probs = util.masked_mean(
                label_prob_steps, binary_mask.float().unsqueeze(1), dim=2
            )
            label_probs = util.replace_masked_values(
                label_probs, binary_mask.sum(1, keepdim=True).bool().float(), 1.0 / num_labels
            )
        else:
            label_probs = label_prob_steps.mean(2)

        return label_probs
