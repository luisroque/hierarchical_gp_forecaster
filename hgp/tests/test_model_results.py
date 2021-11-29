import unittest
import tsaugmentation as tsag
from hgp.model.hgp import HGP
import shutil


class TestModel(unittest.TestCase):

    def setUp(self):
        self.data = tsag.preprocessing.PreprocessDatasets('prison').apply_preprocess()
        self.n = self.data['predict']['n']
        self.s = self.data['train']['s']
        shutil.rmtree("./original_datasets")
        shutil.rmtree("./transformed_datasets")
        self.hgp = HGP('prison', self.data)

    def test_correct_train(self):
        model, like = self.hgp.train(n_iterations=10)
        self.assertIsNotNone(model)

    def test_predict_shape(self):
        model, like = self.hgp.train(n_iterations=10)
        mean, lower, upper = self.hgp.predict(model, like)
        print(mean.shape)
        self.assertTrue(mean.shape == (1, self.n, self.s))

    def test_results_interval(self):
        model, like = self.hgp.train(n_iterations=100)
        mean, lower, upper = self.hgp.predict(model, like)
        res = self.hgp.metrics(mean)
        self.assertLess(res['mase']['bottom'], 2.5)
