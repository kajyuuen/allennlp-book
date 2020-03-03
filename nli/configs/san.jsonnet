local embedding_dim = 300;
local hidden_dim = 300;
local bidirectional = true;
local num_layers = 2;
local maxout = true;

local num_directions = if bidirectional then 2 else 1;


{
    "dataset_reader": {
        "type": "snli",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        }
    },
    "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_train.jsonl",
    "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_dev.jsonl",
    "model": {
        "type": "san",
        "dropout": 0.5,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
                    "embedding_dim": embedding_dim,
                    "trainable": true,
                }
            }
        },
        "lexical_feedforward": {
            "input_dim": embedding_dim,
            "hidden_dims": hidden_dim,
            "num_layers": 1,
            "activations": "relu"
        },
        "contextual_encoder": {
            "type": "full-layer-lstm",
            "input_dim": hidden_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "bidirectional": bidirectional,
            "maxout": maxout,
        },
        "attention_feedforward": {
            "input_dim": hidden_dim * num_layers * (if maxout then 1 else num_directions),
            "hidden_dims": hidden_dim,
            "num_layers": 1,
            "activations": "relu"
        },
        "matrix_attention": {
            "type": "dot_product"
        },
        "memory_encoder": {
            "type": "lstm",
            "input_size": 2 * hidden_dim * num_directions,
            "hidden_size": hidden_dim,
            "num_layers": 1,
            "bidirectional": bidirectional,
        },
        "output_feedforward": {
            "input_dim": 4 * hidden_dim * num_directions,
            "num_layers": 1,
            "hidden_dims": hidden_dim,
            "activations": "relu",
            "dropout": 0.5
        },
        "output_logit": {
            "input_dim": hidden_dim,
            "num_layers": 1,
            "hidden_dims": 3,
            "activations": "linear"
        },
        "initializer": [
            [".*linear_layers.*weight", {"type": "xavier_uniform"}],
            [".*linear_layers.*bias", {"type": "zero"}],
            [".*weight_ih.*", {"type": "xavier_uniform"}],
            [".*weight_hh.*", {"type": "orthogonal"}],
            [".*bias_ih.*", {"type": "zero"}],
            [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
        ]
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["premise", "num_tokens"],
                         ["hypothesis", "num_tokens"]],
        "batch_size": 32
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.0004
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 2,
        "num_epochs": 75,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": 0,
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 0
        }
    }
}
