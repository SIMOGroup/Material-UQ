import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.utils import clamp_probs
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from abc import ABC, abstractmethod
from torch.distributions import Gamma
from torch.distributions.transforms import ExpTransform
from torch.distributions.transformed_distribution import TransformedDistribution
import numpy as np
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from pybnn.sampler.sgld import SGLD
from pybnn.sampler.preconditioned_sgld import PreconditionedSGLD as pSGLD
from pybnn.sampler.sghmc import SGHMC as SGHMC
from pybnn.sampler.adaptive_sghmc import AdaptiveSGHMC as aSGHMC


class BNN(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, X, y):
        """
        X: num_train * dim matrix
        y: num_train vector
        """
        pass

    @abstractmethod
    def sample(self, num_samples = 1):
        """
        Generate `num_sample` samples from the posterior, return a list of neural networks and posterior precisions
        """
        pass

    def validate(self, X, y, num_samples = 20):
        y = y.view(X.shape[0], -1)
        with torch.no_grad():
            nns    = self.sample(num_samples)
            py, pv = self.predict_mv(X, nns)
            rmse   = torch.mean((py - y)**2, dim = 0).sqrt()
            nll    = -1 * torch.distributions.Normal(py, pv.sqrt()).log_prob(y).mean(dim = 0)
        return rmse, nll

    def predict_mv(self, input, nn_samples):
        num_test = input.shape[0]
        preds    = self.sample_predict(nn_samples, input).view(len(nn_samples), num_test, -1)
        return preds.mean(dim = 0), preds.var(dim = 0)


class NN(nn.Module):
    def __init__(self, dim, act=nn.ReLU(), num_hiddens=[50], nout=1):
        super(NN, self).__init__()
        self.dim = dim
        self.nout = nout
        self.act = act
        self.num_hiddens = num_hiddens
        self.num_layers = len(num_hiddens)
        self.nn = self.mlp()
        for l in self.nn:
            if type(l) == nn.Linear:
                nn.init.xavier_uniform_(l.weight)

    def mlp(self):
        layers = []
        pre_dim = self.dim
        for i in range(self.num_layers):
            layers.append(nn.Linear(pre_dim, self.num_hiddens[i], bias=True))
            layers.append(self.act)
            pre_dim = self.num_hiddens[i]
        layers.append(nn.Linear(pre_dim, self.nout, bias=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.nn(x)
        return out


class NoisyNN(NN):
    def __init__(self, dim, act=nn.ReLU(), num_hiddens=[50], logvar=torch.log(torch.tensor(1e-3))):
        super(NoisyNN, self).__init__(dim, act, num_hiddens, nout=1)
        self.logvar = nn.Parameter(logvar)

    def forward(self, input):
        out = self.nn(input)
        logvars = torch.clamp(self.logvar, max=20.) * out.new_ones(out.shape)
        return torch.cat((out, logvars), dim=out.dim() - 1)


class StableRelaxedBernoulli(RelaxedBernoulli):
    """
    Numerical stable relaxed Bernoulli distribution
    """

    def rsample(self, sample_shape=torch.Size()):
        return clamp_probs(super(StableRelaxedBernoulli, self).rsample(sample_shape))


def stable_noise_var(input):
    return F.softplus(torch.clamp(input, min=-35.))


def stable_log_lik(mu, var, y):
    noise_var = stable_noise_var(var)
    return -0.5 * (mu - y) ** 2 / noise_var - 0.5 * torch.log(noise_var) - 0.5 * np.log(2 * np.pi)


def stable_nn_lik(nn_out, y):
    return stable_log_lik(nn_out[:, 0], nn_out[:, 1], y)


def normalize(X, y):
    assert (X.dim() == 2)
    assert (y.dim() == 1)
    x_mean = X.mean(dim=0)
    x_std = X.std(dim=0)
    y_mean = y.mean()
    y_std = y.std()
    x_std[x_std == 0] = torch.tensor(1.)
    if y_std == 0:
        y_std = torch.tensor(1.)
    return x_mean, x_std, y_mean, y_std


class BNN_pSGLD_Method(nn.Module, BNN):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], nout = 1, conf = dict()):
        nn.Module.__init__(self)
        BNN.__init__(self)
        self.dim          = dim
        self.act          = act
        self.num_hiddens  = num_hiddens
        self.nout         = nout
        self.steps_burnin = conf.get('steps_burnin', 2500)
        self.steps        = conf.get('steps',        2500)
        self.keep_every   = conf.get('keep_every',   50)
        self.batch_size   = conf.get('batch_size',   32)
        self.warm_start   = conf.get('warm_start',   False)

        self.lr_weight   = conf.get('lr_weight', 1e-1)
        self.lr_noise    = conf.get('lr_noise',  2e-2)
        self.alpha_n     = torch.as_tensor(1.* conf.get('alpha_n', 0.5))
        self.beta_n      = torch.as_tensor(1.* conf.get('beta_n',  0.4))

        # user can specify a suggested noise value, this will override alpha_n and beta_n
        self.noise_level = conf.get('noise_level', None)
        if self.noise_level is not None:
            prec         = 1 / self.noise_level**2
            prec_var     = (prec * 0.25)**2
            self.beta_n  = torch.as_tensor(prec / prec_var)
            self.alpha_n = torch.as_tensor(prec * self.beta_n)
            print("Reset alpha_n = %g, beta_n = %g" % (self.alpha_n, self.beta_n))

        self.prior_log_precision = TransformedDistribution(Gamma(self.alpha_n, self.beta_n), ExpTransform().inv)

        self.log_precs = nn.Parameter(torch.zeros(self.nout))
        self.nn        = NN(dim, self.act, self.num_hiddens, self.nout)
        self.gain      = 5./3 # Assume tanh activation
        
        self.init_nn()
    
    def init_nn(self):
        self.log_precs.data  = (self.alpha_n / self.beta_n).log() * torch.ones(self.nout)
        for l in self.nn.nn:
            if type(l) == nn.Linear:
                nn.init.xavier_uniform_(l.weight, gain = self.gain)

    def log_prior(self):
        log_p = self.prior_log_precision.log_prob(self.log_precs).sum()
        for n, p in self.nn.nn.named_parameters():
            if "weight" in n:
                std    = self.gain * np.sqrt(2. / (p.shape[0] + p.shape[1]))
                log_p += torch.distributions.Normal(0, std).log_prob(p).sum()
        return log_p

    def log_lik(self, X, y):
        y       = y.view(-1, self.nout)
        nout    = self.nn(X).view(-1, self.nout)
        precs   = self.log_precs.exp()
        log_lik = -0.5 * precs * (y - nout)**2 + 0.5 * self.log_precs - 0.5 * np.log(2 * np.pi)
        return log_lik.sum()

    def sgld_steps(self, num_steps, num_train):
        step_cnt = 0
        loss     = 0.
        while(step_cnt < num_steps):
            for bx, by in self.loader:
                log_prior = self.log_prior()
                log_lik   = self.log_lik(bx, by)
                loss      = -1 * (log_lik * (num_train / bx.shape[0]) + log_prior)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                self.scheduler.step()
                step_cnt += 1
                if step_cnt >= num_steps:
                    break
        return loss

    def train(self, X, y):
        y           = y.view(-1, self.nout)
        num_train   = X.shape[0]
        params      = [
                {'params': self.nn.nn.parameters(), 'lr': self.lr_weight},
                {'params': self.log_precs,          'lr': self.lr_noise}] 

        self.opt       = pSGLD(params)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.opt, lambda iter : np.float32((1 + iter)**-0.33))

        self.loader    = DataLoader(TensorDataset(X, y), batch_size = self.batch_size, shuffle = True)
        step_cnt    = 0
        self.nns    = []
        self.lrs    = []
        if not self.warm_start:
            self.init_nn()
        
        _ = self.sgld_steps(self.steps_burnin, num_train) # burn-in
        
        while(step_cnt < self.steps):
            loss      = self.sgld_steps(self.keep_every, num_train)
            step_cnt += self.keep_every
            prec      = self.log_precs.exp().mean()
            self.nns.append(deepcopy(self.nn))
        print('Number of samples: %d' % len(self.nns))

    def sample(self, num_samples = 1):
        assert(num_samples <= len(self.nns))
        return np.random.permutation(self.nns)[:num_samples]

    def sample_predict(self, nns, input):
        num_samples = len(nns)
        num_x       = input.shape[0]
        pred        = torch.empty(num_samples, num_x, self.nout)
        for i in range(num_samples):
            pred[i] = nns[i](input)
        return pred

    def report(self):
        print(self.nn.nn)
