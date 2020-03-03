import pytest

from src.data.dataset_readers.conll_2003_reader import Conll2003Reader
from allennlp.common.util import ensure_list

class TestConll2003Reader:
    def test_read_from_file(self):
        conll_reader = Conll2003Reader()
        instances = conll_reader.read("./src/tests/fixtures/data/conll2003.txt")
        instances = ensure_list(instances)

        fields = instances[0].fields
        tokens = [t.text for t in fields["sentence"].tokens]
        assert tokens == ["U.N.", "official", "Ekeus", "heads", "for", "Baghdad", "."]
        assert fields["labels"].labels == ["I-ORG", "O", "I-PER", "O", "O", "I-LOC", "O"]

