import pytest
from src.models.ner_tagger import NERTagger

from allennlp.models import Model
from allennlp.common.testing import ModelTestCase

class TestNERTagger(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            "./src/tests/fixtures/configs/experiment.jsonnet",
            "./src/tests/fixtures/data/conll2003.txt"
        )

    def test_train(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
