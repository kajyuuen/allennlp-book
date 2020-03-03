from typing import Dict, List, Iterator
from overrides import overrides

import os

from allennlp.data import Instance
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField

@DatasetReader.register("livedoor_news_reader")
class LivedoorNewsReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def text_to_instance(self, 
                         tokens: List[Token],
                         label: str = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if label:
            label_field = LabelField(label)
            fields["label"] = label_field

        return Instance(fields)

    @overrides
    def _read(self, path: str) -> Iterator[Instance]:
        dirs_path = os.listdir(path)
        category_dirs = [f for f in dirs_path if os.path.isdir(os.path.join(path, f))]
        for category_dir in category_dirs:
            file_dir_path = os.path.join(path, category_dir)
            files = os.listdir(file_dir_path)
            for i, file_name in enumerate(files):
                # 各カテゴリ10文章づつ読み込む
                if i == 10:
                    break
                with open(os.path.join(file_dir_path, file_name)) as f:
                    text = f.read()
                    label = category_dir
                    yield self.text_to_instance([Token(word) for word in text], label)
