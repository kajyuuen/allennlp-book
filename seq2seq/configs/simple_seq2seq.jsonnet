local embedding_dim = 10;
local hidden_dim = 10;
local bidirectional = true;

local model = {
  "type": "simple_seq2seq",
  "source_embedder": {
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
    "type": "lstm",
    "input_size": embedding_dim,
    "hidden_size": hidden_dim,
    "num_layers": 2,
    "dropout": 0.4,
    "bidirectional": bidirectional
  },
  "max_decoding_steps": 5,
  "target_embedding_dim": embedding_dim,
  "target_namespace": "target_tokens",
  "attention": {
    "type": "dot_product"
  },
  "beam_size": 5
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
