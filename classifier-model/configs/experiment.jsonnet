local embedding_dim = 10;
local num_filters = 8;
local output_dim = 16;
local num_epochs = 1;
local batch_size = 2;
local learning_rate = 0.1;

{
    dataset_reader: {
        type: 'ag_news_reader',
        token_indexers: {
            tokens: {
                type: 'single_id',
                token_min_padding_length: num_filters
            },
        },
    },
    train_data_path: 'datasets/train.csv',
    test_data_path: 'datasets/test.csv',
    model: {
        type: 'text_classifier',
        word_embeddings: {
            tokens: {
                type: 'embedding',
                embedding_dim: embedding_dim,
                trainable: true
            }
        },
        encoder: {
            type: 'cnn',
            num_filters: num_filters,
            embedding_dim: embedding_dim,
            output_dim: output_dim
        }
    },
    iterator: {
        type: 'bucket',
        batch_size: batch_size,
        sorting_keys: [['sentence', 'num_tokens']]
    },
    trainer: {
        num_epochs: num_epochs,
        optimizer: {
            type: 'sgd',
            lr: learning_rate
        }
    }
}