import torch
from torch import nn
import numpy as np

class zeroInflatedGamma(object):
    # self.output = linear transform
    # self.output_nl = nonlinear transform

    def __init__(self, alpha, beta, q, s_min):
        # must return "as_list" to get ints
        #input_size = alpha.get_shape().as_list()[2]
        self.alpha = alpha
        self.beta = beta
        self.q = q
        #self.s_min = 0.1 # or 0.0307-5e-5 for inequal or (0.1/3)-1e-5 for equal interval sparse sampl
        self.s_min = s_min

    def log_prob_ZIG(self, x):
        # give where spike = 0 a value of 1, to be able to use log_prob gamma.
        # give where spike is not 0 a value of x-s_min
        adjust_x = torch.where(torch.equal(x, 0.0), torch.ones_like(x), x-self.s_min)
        # compute log-likelihood element-wise. Where spike=0 now has a inaccurate value
        loglikelihood_adj_gamma = torch.distributions.gamma.Gamma(self.alpha, self.beta).log_prob(adjust_x)
        # now convert those inaccurate value to be log(1-q)
        # add log(q) to places where x>s_min
        self.loglikelihood = torch.where(torch.equal(x, 0.0), torch.log(1-self.q), loglikelihood_adj_gamma+torch.log(self.q))
        return self.loglikelihood