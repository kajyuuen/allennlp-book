{
    "dataset_reader": {"type": "sequence_tagging"},
    "train_data_path": "https://raw.githubusercontent.com/allenai/allennlp/master/allennlp/tests/fixtures/data/sequence_tagging.tsv",
    "validation_data_path": "https://raw.githubusercontent.com/allenai/allennlp/master/allennlp/tests/fixtures/data/sequence_tagging.tsv",
    "model": {
        "type": "simple_tagger",
        "text_field_embedder": {
            "token_embedders": {"tokens": {"type": "embedding", "embedding_dim": 5}}
        },
        "encoder": {"type": "lstm", "input_size": 5, "hidden_size": 7, "num_layers": 2}
    },
    "trainer": {
        "type": "callback",
        "optimizer": {"type": "sgd", "lr": 0.01, "momentum": 0.9},
        "num_epochs": 2,
        "callbacks": [
            "checkpoint",
            "track_metrics",
            "validate",
            "mlflow_metrics"
        ]
    },
    "iterator": {"type": "basic", "batch_size": 2}
}
