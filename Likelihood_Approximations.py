#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abcpy.approx_lhd import Approx_likelihood
import numpy as np
import sigkernel
import torch
from einops import rearrange
from itertools import product

'''
This file contains 3 scoring rules: Energy, Kernel, Signature kernel
'''
class EnergyScore(Approx_likelihood):

    def __init__(self, statistics_calc):
        """
        Energy Score Class in ABCpy fasion
        """

        super(EnergyScore, self).__init__(statistics_calc)

    def loglikelihood(self, y_obs, y_sim):
        """Computes the loglikelihood.
        Parameters
        ----------
        y_obs: Python list
            Observed data set.
        y_sim: Python list
            Simulated data set from model at the parameter value.
        Returns
        -------
        float
            Computed approximate loglikelihood.
        """

        stat_obs, stat_sim = self._calculate_summary_stat(y_obs, y_sim)
        
        # preprocess data
        # stat_obs = np.stack([i[1:] for i in stat_obs],axis=0)
        # stat_sim = np.stack([i[1:] for i in stat_sim],axis=0)
        # beta as tuning parameter
        beta = 1.5
        # get number of observations
        n_obs = stat_obs.shape[0]
        n_sim, p = stat_sim.shape
        # compute X-y term
        diff_X_y = stat_obs.reshape(n_obs, 1, -1) - stat_sim.reshape(1, n_sim, p)
        diff_X_y = np.einsum('ijk, ijk -> ij', diff_X_y, diff_X_y)
        # compute X-X' term

        diff_X_tildeX = stat_sim.reshape(1, n_sim, p) - stat_sim.reshape(n_sim, 1, p)
        # exclude diagonal elements which are zero:
        diff_X_tildeX = np.einsum('ijk, ijk -> ij', diff_X_tildeX, diff_X_tildeX)[~np.eye(n_sim, dtype=bool)]
        if beta != 1:
            diff_X_y **= beta
            diff_X_tildeX **= beta
        # return unbiased estimate of energy score
        return  (2 * np.sum(np.mean(diff_X_y, axis=1)) - n_obs * np.sum(diff_X_tildeX) / (n_sim**2))#energy_score/stat_obs.shape[0]
    
class KernelScore(Approx_likelihood):

    def __init__(self, statistics_calc):
        """
        Kernel Score in ABCpy fasion
        """

        super(KernelScore, self).__init__(statistics_calc)

    def loglikelihood(self, y_obs, y_sim):
        """Computes the loglikelihood.
        Parameters
        ----------
        y_obs: Python list
            Observed data set.
        y_sim: Python list
            Simulated data set from model at the parameter value.
        Returns
        -------
        float
            Computed approximate loglikelihood.
        """

        stat_obs, stat_sim = self._calculate_summary_stat(y_obs, y_sim)
        # preprocess the data

        stat_obs = np.stack([i[1:] for i in stat_obs],axis=0)
        stat_sim = np.stack([i[1:] for i in stat_sim],axis=0)
        # rbf kernel
        
        def rbf(x,y):
            gamma = 2

            return np.exp(-np.square(np.linalg.norm(x-y))/(2*np.square(gamma)))
        # number of samples in both simulated and observed data
        
        dim_obs = stat_obs.shape[0]
        dim_sim = stat_sim.shape[0]
        
        k_score_xx = 0
        # 1/m(m-1) * sum(k(xi,xj))
        # tuples have the combinations where i not equal to j
        for i, j in [i for i in list(product(*[range(dim_sim),range(dim_sim)])) if i[0] != i[1]]:

            k_score_xx += rbf(stat_sim[i,:],stat_sim[j,:])
        
        k_score_xx = dim_obs*(k_score_xx/(dim_sim*(dim_sim-1)))
        
        # -2/m * sum (sum(k(xj-yi)))
        k_score_xy = 0
        for i in range(dim_obs):
            for j in range(dim_sim):
                k_score_xy += rbf(stat_sim[j,:],stat_obs[i,:])
            
        k_score = k_score_xx - (2/dim_sim) * k_score_xy
        return -k_score
    
class SigScore(Approx_likelihood):

    def __init__(self, statistics_calc, sigma, allele = True):
        """This class implements the approximate likelihood function which computes the approximate
        likelihood using signature kernel scoring rule.
        
        Salvi, Cristopher, et al. "The Signature Kernel is the solution of a Goursat PDE." 
        SIAM Journal on Mathematics of Data Science 3.3 (2021): 873-899.
        
        Parameters
        ----------
        statistics_calc : abcpy.statistics.Statistics
            Statistics extractor object that conforms to the Statistics class.
        """

        super(SigScore, self).__init__(statistics_calc)
        self.allele = allele
        self.sigma = sigma

    def loglikelihood(self, y_obs, y_sim):
        """Computes the loglikelihood.
        Parameters
        ----------
        y_obs: Python list
            Observed data set.
        y_sim: Python list
            Simulated data set from model at the parameter value.
        Returns
        -------
        float
            Computed approximate loglikelihood.
        """
        
        if self.allele:
            # I only focus on haplo
            y_obs, y_sim = y_obs, y_sim
        else:
            # I also need to deselect alleles
            y_obs, y_sim = y_obs[1], y_sim[1]
     
        dim_t = int(y_obs[0][0])
        dim_total = y_obs[0].shape[0]-1
        stat_obs, stat_sim = self._calculate_summary_stat(y_obs, y_sim)
        
        # Reshape and compute cumulative sums of statistics
        
        stat_obs = np.dstack([np.cumsum(i[1:].reshape(int(dim_total/dim_t),int(dim_t)).T,axis=0) for i in stat_obs])
        stat_sim = np.dstack([np.cumsum(i[1:].reshape(int(dim_total/dim_t),int(dim_t)).T,axis=0) for i in stat_sim])


        
        stat_obs = rearrange(stat_obs, 't b d -> d t b')
        stat_sim = rearrange(stat_sim, 't b d -> d t b')
        
        # # need (obs,time,snp) shape
        
        # # Tuning parameter for RBF kernel
        # 
        static_kernel = sigkernel.RBFKernel(sigma=self.sigma) # hilbert kernel uplift
        dyadic_order = 3
        
        
        signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
        
        # compute the scoring rule

        sr = signature_kernel.compute_scoring_rule(torch.tensor(stat_sim, dtype=torch.float64),torch.tensor(stat_obs, dtype=torch.float64))
        #sr_expected = signature_kernel.compute_expected_scoring_rule(torch.tensor(stat_sim, dtype=torch.float64),torch.tensor(stat_obs, dtype=torch.float64))

        return -sr
    
   
class SigScore_TEMPER(Approx_likelihood):

    def __init__(self, statistics_calc, sigma, allele = True, temper_w = 1):
        """This class implements the approximate likelihood function which computes the approximate
        likelihood using signature kernel scoring rule.
        
        Salvi, Cristopher, et al. "The Signature Kernel is the solution of a Goursat PDE." 
        SIAM Journal on Mathematics of Data Science 3.3 (2021): 873-899.
        
        Parameters
        ----------
        statistics_calc : abcpy.statistics.Statistics
            Statistics extractor object that conforms to the Statistics class.
        """

        super(SigScore_TEMPER, self).__init__(statistics_calc)
        self.allele = allele
        self.sigma = sigma
        self.temper_w = temper_w

    def loglikelihood(self, y_obs, y_sim):
        """Computes the loglikelihood.
        Parameters
        ----------
        y_obs: Python list
            Observed data set.
        y_sim: Python list
            Simulated data set from model at the parameter value.
        Returns
        -------
        float
            Computed approximate loglikelihood.
        """
        
        if self.allele:
            # I only focus on haplo
            y_obs, y_sim = y_obs, y_sim
        else:
            # I also need to deselect alleles
            y_obs, y_sim = y_obs[1], y_sim[1]
     
        dim_t = int(y_obs[0][0])
        dim_total = y_obs[0].shape[0]-1
        stat_obs, stat_sim = self._calculate_summary_stat(y_obs, y_sim)
        
        # Reshape and compute cumulative sums of statistics
        
        stat_obs = np.dstack([np.cumsum(i[1:].reshape(int(dim_total/dim_t),int(dim_t)).T,axis=0) for i in stat_obs])
        stat_sim = np.dstack([np.cumsum(i[1:].reshape(int(dim_total/dim_t),int(dim_t)).T,axis=0) for i in stat_sim])


        
        stat_obs = rearrange(stat_obs, 't b d -> d t b')
        stat_sim = rearrange(stat_sim, 't b d -> d t b')
        
        # # need (obs,time,snp) shape
        
        # # Tuning parameter for RBF kernel
        # 
        static_kernel = sigkernel.RBFKernel(sigma=self.sigma) # hilbert kernel uplift
        dyadic_order = 3
        
        
        signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
        
        # compute the scoring rule

        sr = signature_kernel.compute_scoring_rule(torch.tensor(stat_sim, dtype=torch.float64),torch.tensor(stat_obs, dtype=torch.float64))
        #sr_expected = signature_kernel.compute_expected_scoring_rule(torch.tensor(stat_sim, dtype=torch.float64),torch.tensor(stat_obs, dtype=torch.float64))
        temper_sr = sr* self.temper_w
        return -temper_sr
  