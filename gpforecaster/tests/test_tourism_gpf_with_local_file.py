import unittest
import tsaugmentation as tsag
from gpforecaster.model.gpf import GPF
import shutil


class TestModel(unittest.TestCase):

    def setUp(self):
        self.preproc = tsag.preprocessing.PreprocessDatasets('tourism', test_size=228*10)
        self.data = self.preproc._tourism()
        self.n = self.data['predict']['n']
        self.s = self.data['train']['s']
        self.gpf = GPF(dataset='tourism', groups=self.data,
                       store_prediction_samples=True,
                       store_prediction_points=True)

    def test_results_mean_and_prediction_interval(self):
        model, like = self.gpf.train(n_iterations=10)
        samples = self.gpf.predict(model, like)
        res = self.gpf.metrics(samples)

        # Test shape of results
        self.assertTrue(res['mase']['bottom_ind'].shape == (self.s, ))
        self.assertTrue(res['CRPS']['bottom_ind'].shape == (self.s, ))
        self.assertTrue(res['rmse']['bottom_ind'].shape == (self.s, ))

        # Test shape of predictions
        # Number of prediction samples for the bottom time series (n_points, n_series, n_samples)
        self.assertTrue(res['predictions']['samples']['bottom'].shape == (self.n, self.s, 500))
        self.assertTrue(res['predictions']['points']['bottom'].shape == (self.data['h'], self.s))
        # Number of prediction samples for the total time series
        self.assertTrue(res['predictions']['samples']['total'].shape == (self.n, 500))
        self.assertTrue(res['predictions']['points']['total'].shape == (self.data['h'],))
        # Test number of objects predicted and stored
        self.assertTrue(len(res['predictions']['samples']) == 12)
        self.assertTrue(len(res['predictions']['points']) == 12)

    def test_results_mean_and_prediction_interval_without_storing_results(self):
        self.gpf = GPF(dataset='tourism', groups=self.data,
                       store_prediction_samples=False,
                       store_prediction_points=False)
        model, like = self.gpf.train(n_iterations=10)
        samples = self.gpf.predict(model, like)
        res = self.gpf.metrics(samples)

        # Test shape of results
        self.assertTrue(res['mase']['bottom_ind'].shape == (self.s, ))
        self.assertTrue(res['CRPS']['bottom_ind'].shape == (self.s, ))
        self.assertTrue(res['rmse']['bottom_ind'].shape == (self.s, ))

        # Test shape of predictions
        # Test number of objects predicted and stored
        self.assertTrue(len(res) == 4)

