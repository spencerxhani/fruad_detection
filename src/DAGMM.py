import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import trange
import sys
sys.path.append("../DeepADoTS/src/algorithms/")
# from .algorithm_utils import Algorithm, PyTorchUtils
# from .autoencoder import AutoEncoderModule
#from lstm_enc_dec_axl import LSTMEDModule
import abc
import logging
import random

import numpy as np
import torch
import tensorflow as tf
from tensorflow.python.client import device_lib
from torch.autograd import Variable


class Algorithm(metaclass=abc.ABCMeta):
    def __init__(self, module_name, name, seed, details=False):
        self.logger = logging.getLogger(module_name)
        self.name = name
        self.seed = seed
        self.details = details
        self.prediction_details = {}

        if self.seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __str__(self):
        return self.name

    @abc.abstractmethod
    def fit(self, X):
        """
        Train the algorithm on the given dataset
        """

    @abc.abstractmethod
    def predict(self, X):
        """
        :return anomaly score
        """


class PyTorchUtils(metaclass=abc.ABCMeta):
    def __init__(self, seed, gpu):
        self.gpu = gpu
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
        self.framework = 0

    @property
    def device(self):
        #return 'cuda:0'
        return torch.device(f'cuda:{self.gpu}' if torch.cuda.is_available() and self.gpu is not None else 'cpu')

    def to_var(self, t, **kwargs):
        # ToDo: check whether cuda Variable.
        t = t.to(self.device)
        return Variable(t, **kwargs)

    def to_device(self, model):
        model.to(self.device)


class TensorflowUtils(metaclass=abc.ABCMeta):
    def __init__(self, seed, gpu):
        self.gpu = gpu
        self.seed = seed
        if self.seed is not None:
            tf.set_random_seed(seed)
        self.framework = 1

    @property
    def device(self):
        local_device_protos = device_lib.list_local_devices()
        gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
        return tf.device(gpus[self.gpu] if gpus and self.gpu is not None else '/cpu:0')
    
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange


class AutoEncoder(Algorithm, PyTorchUtils):
    def __init__(self, name: str='AutoEncoder', num_epochs: int=10, batch_size: int=20, lr: float=1e-3,
                 hidden_size: int=5, sequence_length: int=30, train_gaussian_percentage: float=0.25,
                 seed: int=None, gpu: int=None, details=True):
        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.train_gaussian_percentage = train_gaussian_percentage

        self.aed = None
        self.mean, self.cov = None, None

    def fit(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(data.shape[0] - self.sequence_length + 1)]
        indices = np.random.permutation(len(sequences))
        split_point = int(self.train_gaussian_percentage * len(sequences))
        train_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                  sampler=SubsetRandomSampler(indices[:-split_point]), pin_memory=True)
        train_gaussian_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                           sampler=SubsetRandomSampler(indices[-split_point:]), pin_memory=True)

        self.aed = AutoEncoderModule(X.shape[1], self.sequence_length, self.hidden_size, seed=self.seed, gpu=self.gpu)
        self.to_device(self.aed)  # .double()
        optimizer = torch.optim.Adam(self.aed.parameters(), lr=self.lr)

        self.aed.train()
        for epoch in trange(self.num_epochs):
            logging.debug(f'Epoch {epoch+1}/{self.num_epochs}.')
            for ts_batch in train_loader:
                output = self.aed(self.to_var(ts_batch))
                loss = nn.MSELoss(size_average=False)(output, self.to_var(ts_batch.float()))
                self.aed.zero_grad()
                loss.backward()
                optimizer.step()

        self.aed.eval()
        error_vectors = []
        for ts_batch in train_gaussian_loader:
            output = self.aed(self.to_var(ts_batch))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts_batch.float()))
            error_vectors += list(error.view(-1, X.shape[1]).data.cpu().numpy())

        self.mean = np.mean(error_vectors, axis=0)
        self.cov = np.cov(error_vectors, rowvar=False)

    def predict(self, X: pd.DataFrame) -> np.array:
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(data.shape[0] - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.aed.eval()
        mvnormal = multivariate_normal(self.mean, self.cov, allow_singular=True)
        scores = []
        outputs = []
        errors = []
        for idx, ts in enumerate(data_loader):
            output = self.aed(self.to_var(ts))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts.float()))
            score = -mvnormal.logpdf(error.view(-1, X.shape[1]).data.cpu().numpy())
            scores.append(score.reshape(ts.size(0), self.sequence_length))
            if self.details:
                outputs.append(output.data.numpy())
                errors.append(error.data.numpy())

        # stores seq_len-many scores per timestamp and averages them
        scores = np.concatenate(scores)
        lattice = np.full((self.sequence_length, X.shape[0]), np.nan)
        for i, score in enumerate(scores):
            lattice[i % self.sequence_length, i:i + self.sequence_length] = score
        scores = np.nanmean(lattice, axis=0)

        if self.details:
            outputs = np.concatenate(outputs)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, output in enumerate(outputs):
                lattice[i % self.sequence_length, i:i + self.sequence_length, :] = output
            self.prediction_details.update({'reconstructions_mean': np.nanmean(lattice, axis=0).T})

            errors = np.concatenate(errors)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, error in enumerate(errors):
                lattice[i % self.sequence_length, i:i + self.sequence_length, :] = error
            self.prediction_details.update({'errors_mean': np.nanmean(lattice, axis=0).T})

        return scores


class AutoEncoderModule(nn.Module, PyTorchUtils):
    def __init__(self, n_features: int, sequence_length: int, hidden_size: int, seed: int, gpu: int):
        # Each point is a flattened window and thus has as many features as sequence_length * features
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        input_length = n_features * sequence_length

        # creates powers of two between eight and the next smaller power from the input_length
        dec_steps = 2 ** np.arange(max(np.ceil(np.log2(hidden_size)), 2), np.log2(input_length))[1:]
        dec_setup = np.concatenate([[hidden_size], dec_steps.repeat(2), [input_length]])
        enc_setup = dec_setup[::-1]

        layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in enc_setup.reshape(-1, 2)]).flatten()[:-1]
        self._encoder = nn.Sequential(*layers)
        self.to_device(self._encoder)

        layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in dec_setup.reshape(-1, 2)]).flatten()[:-1]
        self._decoder = nn.Sequential(*layers)
        self.to_device(self._decoder)

    def forward(self, ts_batch, return_latent: bool=False):
        flattened_sequence = ts_batch.view(ts_batch.size(0), -1)
        enc = self._encoder(flattened_sequence.float())
        dec = self._decoder(enc)
        reconstructed_sequence = dec.view(ts_batch.size())
        return (reconstructed_sequence, enc) if return_latent else reconstructed_sequence
    
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange

# from .algorithm_utils import Algorithm, PyTorchUtils


class LSTMED(Algorithm, PyTorchUtils):
    def __init__(self, name: str='LSTM-ED', num_epochs: int=10, batch_size: int=20, lr: float=1e-3,
                 hidden_size: int=5, sequence_length: int=30, train_gaussian_percentage: float=0.25,
                 n_layers: tuple=(1, 1), use_bias: tuple=(True, True), dropout: tuple=(0, 0),
                 seed: int=None, gpu: int = None, details=True):
        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.train_gaussian_percentage = train_gaussian_percentage

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.lstmed = None
        self.mean, self.cov = None, None

    def fit(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(data.shape[0] - self.sequence_length + 1)]
        indices = np.random.permutation(len(sequences))
        split_point = int(self.train_gaussian_percentage * len(sequences))
        train_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                  sampler=SubsetRandomSampler(indices[:-split_point]), pin_memory=True)
        train_gaussian_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                           sampler=SubsetRandomSampler(indices[-split_point:]), pin_memory=True)

        self.lstmed = LSTMEDModule(X.shape[1], self.hidden_size,
                                   self.n_layers, self.use_bias, self.dropout,
                                   seed=self.seed, gpu=self.gpu)
        self.to_device(self.lstmed)
        optimizer = torch.optim.Adam(self.lstmed.parameters(), lr=self.lr)

        self.lstmed.train()
        for epoch in trange(self.num_epochs):
            logging.debug(f'Epoch {epoch+1}/{self.num_epochs}.')
            for ts_batch in train_loader:
                output = self.lstmed(self.to_var(ts_batch))
                loss = nn.MSELoss(size_average=False)(output, self.to_var(ts_batch.float()))
                self.lstmed.zero_grad()
                loss.backward()
                optimizer.step()

        self.lstmed.eval()
        error_vectors = []
        for ts_batch in train_gaussian_loader:
            output = self.lstmed(self.to_var(ts_batch))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts_batch.float()))
            error_vectors += list(error.view(-1, X.shape[1]).data.cpu().numpy())

        self.mean = np.mean(error_vectors, axis=0)
        self.cov = np.cov(error_vectors, rowvar=False)

    def predict(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(data.shape[0] - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.lstmed.eval()
        mvnormal = multivariate_normal(self.mean, self.cov, allow_singular=True)
        scores = []
        outputs = []
        errors = []
        for idx, ts in enumerate(data_loader):
            output = self.lstmed(self.to_var(ts))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts.float()))
            score = -mvnormal.logpdf(error.view(-1, X.shape[1]).data.cpu().numpy())
            scores.append(score.reshape(ts.size(0), self.sequence_length))
            if self.details:
                outputs.append(output.data.numpy())
                errors.append(error.data.numpy())

        # stores seq_len-many scores per timestamp and averages them
        scores = np.concatenate(scores)
        lattice = np.full((self.sequence_length, data.shape[0]), np.nan)
        for i, score in enumerate(scores):
            lattice[i % self.sequence_length, i:i + self.sequence_length] = score
        scores = np.nanmean(lattice, axis=0)

        if self.details:
            outputs = np.concatenate(outputs)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, output in enumerate(outputs):
                lattice[i % self.sequence_length, i:i + self.sequence_length, :] = output
            self.prediction_details.update({'reconstructions_mean': np.nanmean(lattice, axis=0).T})

            errors = np.concatenate(errors)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, error in enumerate(errors):
                lattice[i % self.sequence_length, i:i + self.sequence_length, :] = error
            self.prediction_details.update({'errors_mean': np.nanmean(lattice, axis=0).T})

        return scores


class LSTMEDModule(nn.Module, PyTorchUtils):
    def __init__(self, n_features: int, hidden_size: int,
                 n_layers: tuple, use_bias: tuple, dropout: tuple,
                 seed: int, gpu: int):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[0], bias=self.use_bias[0], dropout=self.dropout[0])
        self.to_device(self.encoder)
        self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[1], bias=self.use_bias[1], dropout=self.dropout[1])
        self.to_device(self.decoder)
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features)
        self.to_device(self.hidden2output)

    def _init_hidden(self, batch_size):
        return (self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()),
                self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()))

    def forward(self, ts_batch, return_latent: bool=False):
        batch_size = ts_batch.shape[0]

        # 1. Encode the timeseries to make use of the last hidden state.
        enc_hidden = self._init_hidden(batch_size)  # initialization with zero
        _, enc_hidden = self.encoder(ts_batch.float(), enc_hidden)  # .float() here or .double() for the model

        # 2. Use hidden state as initialization for our Decoder-LSTM
        dec_hidden = enc_hidden

        # 3. Also, use this hidden state to get the first output aka the last point of the reconstructed timeseries
        # 4. Reconstruct timeseries backwards
        #    * Use true data for training decoder
        #    * Use hidden2output for prediction
        output = self.to_var(torch.Tensor(ts_batch.size()).zero_())
        for i in reversed(range(ts_batch.shape[1])):
            output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])

            if self.training:
                _, dec_hidden = self.decoder(ts_batch[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)

        return (output, enc_hidden[1][-1]) if return_latent else output

"""Adapted from Daniel Stanley Tan (https://github.com/danieltan07/dagmm)"""
import logging
import sys
sys.path.append("../DeepADoTS/src/algorithms/")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import trange

# from .algorithm_utils import Algorithm, PyTorchUtils
# from .autoencoder import AutoEncoderModule
#from lstm_enc_dec_axl import LSTMEDModule


class DAGMM(Algorithm, PyTorchUtils):
    class AutoEncoder:
        NN = AutoEncoderModule
        LSTM = LSTMEDModule

    def __init__(self, num_epochs=10, lambda_energy=0.1, lambda_cov_diag=0.005, lr=1e-3, batch_size=50, gmm_k=3,
                 normal_percentile=80, sequence_length=30, autoencoder_type=AutoEncoderModule, autoencoder_args=None,
                 hidden_size: int=5, seed: int=None, gpu: int=None, details=True):
        _name = 'LSTM-DAGMM' if autoencoder_type == LSTMEDModule else 'DAGMM'
        Algorithm.__init__(self, __name__, _name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.lambda_energy = lambda_energy
        self.lambda_cov_diag = lambda_cov_diag
        self.lr = lr
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.gmm_k = gmm_k  # Number of Gaussian mixtures
        self.normal_percentile = normal_percentile  # Up to which percentile data should be considered normal
        self.autoencoder_type = autoencoder_type
        if autoencoder_type == AutoEncoderModule:
            self.autoencoder_args = ({'sequence_length': self.sequence_length})
        elif autoencoder_type == LSTMEDModule:
            self.autoencoder_args = ({'n_layers': (1, 1), 'use_bias': (True, True), 'dropout': (0.0, 0.0)})
        self.autoencoder_args.update({'seed': seed, 'gpu': gpu})
        if autoencoder_args is not None:
            self.autoencoder_args.update(autoencoder_args)
        self.hidden_size = hidden_size

        self.dagmm, self.optimizer, self.train_energy, self._threshold = None, None, None, None

    def reset_grad(self):
        self.dagmm.zero_grad()

    def dagmm_step(self, input_data):
        self.dagmm.train()
        enc, dec, z, gamma = self.dagmm(input_data)
        #print (enc, dec, z, gamma)
        total_loss, sample_energy, recon_error, cov_diag = self.dagmm.loss_function(input_data, dec, z, gamma,
                                                                                    self.lambda_energy,
                                                                                    self.lambda_cov_diag)
        self.reset_grad()
        total_loss = torch.clamp(total_loss, max=1e7)  # Extremely high loss can cause NaN gradients
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dagmm.parameters(), 5)
        # if np.array([np.isnan(p.grad.detach().numpy()).any() for p in self.dagmm.parameters()]).any():
        #     import IPython; IPython.embed()
        self.optimizer.step()
        return total_loss, sample_energy, recon_error, cov_diag

    def fit(self, X: pd.DataFrame):
        """Learn the mixture probability, mean and covariance for each component k.
        Store the computed energy based on the training data and the aforementioned parameters."""
        #X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(X.shape[0] - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.hidden_size = 5 + int(X.shape[1] / 20)
        autoencoder = self.autoencoder_type(X.shape[1], hidden_size=self.hidden_size, **self.autoencoder_args)
        self.dagmm = DAGMMModule(autoencoder, n_gmm=self.gmm_k, latent_dim=self.hidden_size + 2,
                                 seed=self.seed, gpu=self.gpu)
        self.to_device(self.dagmm)
        self.optimizer = torch.optim.Adam(self.dagmm.parameters(), lr=self.lr)

        for _ in trange(self.num_epochs):
            for input_data in data_loader:
                input_data = self.to_var(input_data)
                self.dagmm_step(input_data.float())

        self.dagmm.eval()
        n = 0
        mu_sum = 0
        cov_sum = 0
        gamma_sum = 0
        for input_data in data_loader:
            input_data = self.to_var(input_data)
            _, _, z, gamma = self.dagmm(input_data.float())
            phi, mu, cov = self.dagmm.compute_gmm_params(z, gamma)

            batch_gamma_sum = torch.sum(gamma, dim=0)

            gamma_sum += batch_gamma_sum
            mu_sum += mu * batch_gamma_sum.unsqueeze(-1)  # keep sums of the numerator only
            cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)  # keep sums of the numerator only

            n += input_data.size(0)

    def predict(self, X: pd.DataFrame):
        """Using the learned mixture probability, mean and covariance for each component k, compute the energy on the
        given data."""
        self.dagmm.eval()
        #X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(len(data) - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=sequences, batch_size=1, shuffle=False)
        test_energy = np.full((self.sequence_length, X.shape[0]), np.nan)

        encodings = np.full((self.sequence_length, X.shape[0], self.hidden_size), np.nan)
        decodings = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
        euc_errors = np.full((self.sequence_length, X.shape[0]), np.nan)
        csn_errors = np.full((self.sequence_length, X.shape[0]), np.nan)

        for i, sequence in enumerate(data_loader):
            #print ("shape of sequence",self.to_var(sequence).float().shape)
            enc, dec, z, gamma = self.dagmm(self.to_var(sequence).float())
            sample_energy, _ = self.dagmm.compute_energy(z, size_average=False)
            idx = (i % self.sequence_length, np.arange(i, i + self.sequence_length))
            test_energy[idx] = sample_energy.data.numpy()

            if self.details:
                encodings[idx] = enc.data.cpu().numpy()
                decodings[idx] = dec.data.cpu().numpy()
                euc_errors[idx] = z[:, 1].data.cpu().numpy()
                csn_errors[idx] = z[:, 2].data.cpu().numpy()

        test_energy = np.nanmean(test_energy, axis=0)

        if self.details:
            self.prediction_details.update({'latent_representations': np.nanmean(encodings, axis=0).T})
            self.prediction_details.update({'reconstructions_mean': np.nanmean(decodings, axis=0).T})
            self.prediction_details.update({'euclidean_errors_mean': np.nanmean(euc_errors, axis=0)})
            self.prediction_details.update({'cosine_errors_mean': np.nanmean(csn_errors, axis=0)})

        return test_energy


class DAGMMModule(nn.Module, PyTorchUtils):
    """Residual Block."""

    def __init__(self, autoencoder, n_gmm, latent_dim, seed: int, gpu: int):
        super(DAGMMModule, self).__init__()
        PyTorchUtils.__init__(self, seed, gpu)

        self.add_module('autoencoder', autoencoder)

        layers = [
            nn.Linear(latent_dim, 10),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(10, n_gmm),
            nn.Softmax(dim=1)
        ]
        self.estimation = nn.Sequential(*layers)
        self.to_device(self.estimation)

        self.register_buffer('phi', self.to_var(torch.zeros(n_gmm)))
        self.register_buffer('mu', self.to_var(torch.zeros(n_gmm, latent_dim)))
        self.register_buffer('cov', self.to_var(torch.zeros(n_gmm, latent_dim, latent_dim)))

    def relative_euclidean_distance(self, a, b, dim=1):
        return (a - b).norm(2, dim=dim) / torch.clamp(a.norm(2, dim=dim), min=1e-10)

    def forward(self, x):
        dec, enc = self.autoencoder(x, return_latent=True)

        rec_cosine = F.cosine_similarity(x.view(x.shape[0], -1), dec.view(dec.shape[0], -1), dim=1)
        rec_euclidean = self.relative_euclidean_distance(x.view(x.shape[0], -1), dec.view(dec.shape[0], -1), dim=1)

        # Concatenate latent representation, cosine similarity and relative Euclidean distance between x and dec(enc(x))
        z = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)
        gamma = self.estimation(z)

        return enc, dec, z, gamma

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        # K
        sum_gamma = torch.sum(gamma, dim=0)

        # K
        phi = (sum_gamma / N)

        self.phi = phi.data

        # K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        # z = N x D
        # mu = K x D
        # gamma N x K

        # z_mu = N x K x D
        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data

        return phi, mu, cov

    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = Variable(self.phi)
        if mu is None:
            mu = Variable(self.mu)
        if cov is None:
            cov = Variable(self.cov)

        k, d, _ = cov.size()

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + self.to_var(torch.eye(d) * eps)
            pinv = np.linalg.pinv(cov_k.data.cpu().numpy())
            cov_inverse.append(Variable(torch.from_numpy(pinv)).unsqueeze(0))

            eigvals = np.linalg.eigvals(cov_k.data.cpu().numpy() * (2 * np.pi))
            if np.min(eigvals) < 0:
                pass
                #logging.warning(f'Determinant was negative! Clipping Eigenvalues to 0+epsilon from {np.min(eigvals)}')
            determinant = np.prod(np.clip(eigvals, a_min=sys.float_info.epsilon, a_max=None))
            det_cov.append(determinant)

            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        det_cov = Variable(torch.from_numpy(np.float32(np.array(det_cov))))
#         print ("sum-0",cov_inverse.unsqueeze(0))
#         print ("sum-1",z_mu.ufnsqueeze(-1).cpu())
#         print ("sum", torch.sum(z_mu.unsqueeze(-1).cpu() * cov_inverse.unsqueeze(0), dim=-2))
        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1).cpu() * cov_inverse.unsqueeze(0), dim=-2) * z_mu.cpu(), dim=-1)
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)
#         print ("sample_energy", self.to_var(phi.unsqueeze(0)).cpu())
#         print ("sample_energy-exp_term", exp_term)
        sample_energy = -max_val.squeeze() - torch.log(
            torch.sum(self.to_var(phi.unsqueeze(0)).cpu() * exp_term / (torch.sqrt(self.to_var(det_cov).cpu()) + eps).unsqueeze(0),
                      dim=1) + eps)

        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag

    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):
        recon_error = torch.mean((x.view(*x_hat.shape) - x_hat) ** 2)
        #print (z, gamma)
        phi, mu, cov = self.compute_gmm_params(z, gamma)
        
        #print (z, phi, mu, cov)
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)
#         print ("recon_error",recon_error)
#         print ("lambda_energy",lambda_energy)
#         print ("lambda_cov_diag",lambda_cov_diag)
#         cov_diag = cov_diag.float()
#         print ("cov_diag",cov_diag)
        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag
        
        return loss, sample_energy, recon_error, cov_diag