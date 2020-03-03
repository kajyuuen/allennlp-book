local embedding_dim = 10;
local hidden_dim = 16;
local num_layers = 2;
local num_attention_heads = 4;
local projection_dim = hidden_dim;
local feedforward_hidden_dim = hidden_dim * 2;

local model = {
  "type": "composed_seq2seq",
  "source_text_embedder": {
    "token_embedders": {
      "tokens": {
        "type": "embedding",
        "vocab_namespace": "source_tokens",
        "embedding_dim": embedding_dim,
        "trainable": true
      }
    }
  },
  "encoder": {
    "type": "stacked_self_attention",
    "input_dim": embedding_dim,
    "hidden_dim": hidden_dim,
    "projection_dim": projection_dim,
    "feedforward_hidden_dim": feedforward_hidden_dim,
    "num_layers": num_layers,
    "num_attention_heads": num_attention_heads,
  },
  "decoder": {
    "type": "auto_regressive_seq_decoder",
    "target_namespace": "target_tokens",
    "target_embedder": {
      "vocab_namespace": "target_tokens",
      "embedding_dim": hidden_dim,
    },
    "decoder_net": {
       "type": "stacked_self_attention",
       "target_embedding_dim": hidden_dim,
       "decoding_dim": hidden_dim,
       "feedforward_hidden_dim": feedforward_hidden_dim,
       "num_layers": num_layers,
       "num_attention_heads": num_attention_heads,
       "positional_encoding_max_steps": 10,
    },
    "max_decoding_steps": 5,
    "beam_size": 5,
    "tensor_based_metric": {"type": "bleu"},
  }
};

local COMMON = import 'common.jsonnet';
{
  "random_seed":  1,
  "pytorch_seed": 1,
  "dataset_reader": COMMON['dataset_reader'],
  "train_data_path": COMMON['train_data_path'],
  "validation_data_path": COMMON['validation_data_path'],
  "model": model,
  "iterator": COMMON['iterator'],
  "trainer": COMMON['trainer']
}
