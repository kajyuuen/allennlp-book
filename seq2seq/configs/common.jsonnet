local train_data_path = "data/train.tsv";
local validation_data_path = "data/valid.tsv";

local dataset_reader = {
  "type": "seq2seq",
  "source_tokenizer": {
    "type": "character"
  },
  "target_tokenizer": {
    "type": "character"
  },
  "source_token_indexers": {
    "tokens": {
      "type": "single_id",
      "namespace": "source_tokens"
      }
  },
  "target_token_indexers": {
    "tokens": {
      "namespace": "target_tokens"
    }
  }
};

{
  "dataset_reader": dataset_reader,
  "train_data_path": train_data_path,
  "validation_data_path": validation_data_path,
  "iterator": {
    "type": "bucket",
    "batch_size" : 100,
    "sorting_keys": [["source_tokens", "num_tokens"]]
  },
  "trainer": {
    "num_epochs": 100,
    "patience": 10,
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.01
    }
  }
}
