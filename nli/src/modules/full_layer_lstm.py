from typing import List

import torch
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, Seq2SeqEncoder


@Seq2SeqEncoder.register("full-layer-lstm")
class FullLayerLSTM(Seq2SeqEncoder):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2,
                 bias: bool = True,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 maxout: bool = False) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._maxout = maxout

        self._num_directions = 2 if bidirectional else 1

        self._lstm_layers = [
            PytorchSeq2SeqWrapper(torch.nn.LSTM(
                input_dim, hidden_dim, num_layers=1,
                bias=bias, dropout=dropout, bidirectional=bidirectional,
                batch_first=True,
            ))
        ]
        if self._num_layers > 1:
            for _ in range(1, self._num_layers):
                self._lstm_layers.append(
                    PytorchSeq2SeqWrapper(torch.nn.LSTM(
                        self._num_directions * hidden_dim, hidden_dim, num_layers=1,
                        bias=bias, dropout=dropout, bidirectional=bidirectional,
                        batch_first=True,
                    ))
                )
        for i, lstm_layer in enumerate(self._lstm_layers):
            self.add_module('lstm_layer_%d' % i, lstm_layer)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        batch_size, sequence_length, _embedding_dim = inputs.size()
        lstm_input = inputs

        lstm_outputs: List[torch.Tensor] = []
        for lstm_layer in self._lstm_layers:
            lstm_output = lstm_layer(lstm_input, mask)
            lstm_outputs.append(lstm_output)
            lstm_input = lstm_output

        if self._maxout:
            for i, lstm_output in enumerate(lstm_outputs):
                lstm_outputs[i] = lstm_output.view(
                    batch_size, sequence_length, self._hidden_dim, self._num_directions
                ).max(-1)[0]

        output = torch.cat(lstm_outputs, dim=-1)
        return output

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        if self._maxout:
            return self._hidden_dim * self._num_layers
        return self._hidden_dim * self._num_layers * self._num_directions

    def is_bidirectional(self) -> bool:
        return self._num_directions > 1
