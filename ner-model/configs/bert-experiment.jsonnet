local embedding_dim = 768;
local hidden_dim = 2;
local num_epochs = 1;
local batch_size = 2;
local learning_rate = 0.1;
local bert_path = "./pretrain_bert";

{
    dataset_reader: {
        type: 'conll_2003_reader',
        token_indexers: {
            bert: {
                "type": "bert-pretrained",
                "pretrained_model": bert_path,
                "do_lowercase": false
            },
        },
    },
    train_data_path: 'datasets/eng.train',
    validation_data_path: 'datasets/eng.testa',
    model: {
        type: 'lstm_tagger',
        word_embeddings: {
            allow_unmatched_keys: true,
            bert: {
                type: 'bert-pretrained',
                pretrained_model: bert_path
            }
        },
        encoder: {
            type: 'lstm',
            input_size: embedding_dim,
            hidden_size: hidden_dim
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