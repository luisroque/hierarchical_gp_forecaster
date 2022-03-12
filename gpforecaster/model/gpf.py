import torch
import numpy as np
import gpytorch
from .gp import ExactGPModel
from .mean_functions import PiecewiseLinearMean
from gpytorch.mlls import SumMarginalLogLikelihood
from gpforecaster.results.calculate_metrics import CalculateStoreResults
import pickle
import tsaugmentation as tsag
from pathlib import Path
import time


class GPF:

    def __init__(self, dataset, groups, input_dir='./'):
        self.dataset = dataset
        self.groups = groups
        self.input_dir = input_dir
        self.timer_start = time.time()
        self.wall_time_preprocess = None
        self.wall_time_build_model = None
        self.wall_time_train = None
        self.wall_time_predict = None
        self.wall_time_total = None
        self.groups, self.dt = self._preprocess()
        self._create_directories()

        self.train_x = torch.arange(groups['train']['n'])
        self.train_x = self.train_x.type(torch.DoubleTensor)
        self.train_x = self.train_x.unsqueeze(-1)
        self.train_y = torch.from_numpy(groups['train']['data'])

    def _create_directories(self):
        # Create directory to store results if does not exist
        Path(f'{self.input_dir}results').mkdir(parents=True, exist_ok=True)

    def _preprocess(self):
        dt = tsag.preprocessing.utils.DataTransform(self.groups)
        self.wall_time_preprocess = time.time() - self.timer_start
        return dt.std_transf_train(), dt

    def _build_mixtures(self):
        # build the matrix

        #     Group1     |   Group2
        # GP1, GP2, GP3  | GP1, GP2
        # 0  , 1  , 1    | 0  , 1
        # 1  , 0  , 0    | 1  , 0
        # 0  , 1, , 1    | 0  , 1
        # 1  , 0  , 1    | 1  , 0

        idxs = []
        for k, val in self.groups['train']['groups_idx'].items():
            idxs.append(val)

        idxs_t = np.array(idxs).T

        n_groups = np.sum(np.fromiter(self.groups['train']['groups_n'].values(), dtype='int32'))
        known_mixtures = np.zeros((self.groups['train']['s'], n_groups))
        k = 0
        for j in range(self.groups['train']['g_number']):
            for i in range(np.max(idxs_t[:, j]) + 1):
                idx_to_1 = np.where(idxs_t[:, j] == i)
                known_mixtures[:, k][idx_to_1] = 1
                k += 1

        return known_mixtures, n_groups

    def _build_cov_matrices(self):
        known_mixtures, n_groups = self._build_mixtures()
        covs = []
        for i in range(1, n_groups + 1):
            # RBF kernel
            rbf_kernel = gpytorch.kernels.RBFKernel()
            rbf_kernel.lengthscale = torch.tensor([1.])
            scale_rbf_kernel = gpytorch.kernels.ScaleKernel(rbf_kernel)
            scale_rbf_kernel.outputscale = torch.tensor([0.5])

            # Periodic Kernel
            periodic_kernel = gpytorch.kernels.PeriodicKernel()
            periodic_kernel.period_length = torch.tensor([self.groups['seasonality']])
            periodic_kernel.lengthscale = torch.tensor([0.5])
            scale_periodic_kernel = gpytorch.kernels.ScaleKernel(periodic_kernel)
            scale_periodic_kernel.outputscale = torch.tensor([1.5])

            # Cov Matrix
            cov = scale_rbf_kernel + scale_periodic_kernel
            covs.append(cov)

        return covs, known_mixtures, n_groups

    def _apply_mixture_cov_matrices(self):
        covs, known_mixtures, n_groups = self._build_cov_matrices()

        # apply mixtures to covariances
        selected_covs = []
        mixed_covs = []
        for i in range(self.groups['train']['s']):
            mixture_weights = known_mixtures[i]
            for w_ix in range(n_groups):
                w = mixture_weights[w_ix]
                if w == 1.0:
                    selected_covs.append(covs[w_ix])
            mixed_cov = selected_covs[0]
            for cov in range(1, len(selected_covs)):
                mixed_cov += selected_covs[cov]  # because GP(cov1 + cov2) = GP(cov1) + GP(cov2)
            mixed_covs.append(mixed_cov)
            selected_covs = []  # clear out cov list

        return mixed_covs

    def _build_model(self):
        mixed_covs = self._apply_mixture_cov_matrices()
        n_changepoints = 4
        changepoints = np.linspace(0, self.groups['train']['n'], n_changepoints + 2)[1:-1]

        model_list = []
        likelihood_list = []
        for i in range(self.groups['train']['s']):
            likelihood_list.append(gpytorch.likelihoods.GaussianLikelihood())
            model_list.append(ExactGPModel(self.train_x, self.train_y[:, i], likelihood_list[i], mixed_covs[i], changepoints, PiecewiseLinearMean))

        self.wall_time_build_model = time.time() - self.timer_start - self.wall_time_preprocess
        return likelihood_list, model_list

    def train(self, n_iterations=500, lr=1e-3):
        likelihood_list, model_list = self._build_model()

        model = gpytorch.models.IndependentModelList(*model_list)
        likelihood = gpytorch.likelihoods.LikelihoodList(*likelihood_list)

        mll = SumMarginalLogLikelihood(likelihood, model)

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters

        for i in range(n_iterations):
            optimizer.zero_grad()
            output = model(*model.train_inputs)
            loss = -mll(output, model.train_targets)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iterations, loss.item()))
            optimizer.step()

        self.wall_time_train = time.time() - self.timer_start - self.wall_time_build_model
        return model, likelihood

    def predict(self, model, likelihood):
        # Set into eval mode
        model.eval()
        likelihood.eval()

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.arange(self.groups['predict']['n']).type(torch.DoubleTensor)
            predictions = likelihood(*model(*[test_x for i in range(self.groups['predict']['s'])]))

        i = 0
        mean = np.zeros((1, self.groups['predict']['n'], self.groups['predict']['s']))
        lower = np.zeros((1, self.groups['predict']['n'], self.groups['predict']['s']))
        upper = np.zeros((1, self.groups['predict']['n'], self.groups['predict']['s']))
        for pred in predictions:
            mean[:, :, i] = pred.mean
            conf = pred.confidence_region()
            lower[:, :, i] = conf[0].detach().numpy()
            upper[:, :, i] = conf[1].detach().numpy()
            i += 1

        # transform back the data
        mean = ((mean * self.dt.std_data) + self.dt.mu_data)
        lower = ((lower * self.dt.std_data) + self.dt.mu_data)
        upper = ((upper * self.dt.std_data) + self.dt.mu_data)
        self.groups = self.dt.inv_transf_train()

        # Clip predictions to 0 if there are negative numbers
        mean[mean < 0] = 0
        lower[lower < 0] = 0
        upper[upper < 0] = 0

        self.wall_time_predict = time.time() - self.timer_start - self.wall_time_train
        return mean, lower, upper

    def store_metrics(self, res):
        with open(f'{self.input_dir}results/results_gp_cov_{self.dataset}.pickle', 'wb') as handle:
            pickle.dump(res, handle, pickle.HIGHEST_PROTOCOL)

    def metrics(self, mean):
        calc_results = CalculateStoreResults(mean, self.groups)
        res = calc_results.calculate_metrics()
        self.wall_time_total = time.time() - self.timer_start

        res['wall_time'] = {}
        res['wall_time']['wall_time_preprocess'] = self.wall_time_preprocess
        res['wall_time']['wall_time_build_model'] = self.wall_time_build_model
        res['wall_time']['wall_time_train'] = self.wall_time_train
        res['wall_time']['wall_time_predict'] = self.wall_time_predict
        res['wall_time']['wall_time_total'] = self.wall_time_total

        return res

