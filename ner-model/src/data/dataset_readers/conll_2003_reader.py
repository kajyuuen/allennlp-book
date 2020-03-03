from typing import Dict, List, Iterator
from overrides import overrides

from allennlp.data import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField

@DatasetReader.register("conll_2003_reader")
class Conll2003Reader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def text_to_instance(self, 
                         tokens: List[Token],
                         tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if tags:
            label_field = SequenceLabelField(labels = tags,
                                             sequence_field = sentence_field)
            fields["labels"] = label_field

        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            sentence, tags = [], []
            for line in f:
                rows = line.strip().split()
                if len(rows) == 0:
                    if len(sentence) > 0:
                        yield self.text_to_instance([Token(word) for word in sentence], tags)
                        sentence, tags = [], []
                    continue
                word, tag = rows[0], rows[3]
                sentence.append(word)
                tags.append(tag)