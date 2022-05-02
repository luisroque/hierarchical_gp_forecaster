import unittest
import tsaugmentation as tsag
from gpforecaster.model.gpf import GPF
import shutil


class TestModel(unittest.TestCase):

    def setUp(self):
        self.data = tsag.preprocessing.PreprocessDatasets('prison').apply_preprocess()
        self.n = self.data['predict']['n']
        self.s = self.data['train']['s']
        shutil.rmtree("./data/original_datasets")
        self.gpf = GPF('prison', self.data)

    def test_calculate_metrics_dict(self):
        model, like = self.gpf.train(n_iterations=100)
        samples = self.gpf.predict(model, like)
        res = self.gpf.metrics(samples)
        self.assertLess(res['mase']['bottom'], 2.5)
