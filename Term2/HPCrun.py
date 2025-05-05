### 
import apache_beam

import os
import weatherbench2
import xarray as xr
import math
from weatherbench2.regions import SliceRegion, ExtraTropicalRegion
from weatherbench2.evaluation import evaluate_in_memory
from weatherbench2 import config
import numpy as np
import sigkernel
import torch
from einops import rearrange
from itertools import product
import cython
import matplotlib.pyplot  as plt
import tqdm
import Functions as fu
import line_profiler


def main():
    # obs_pathlarge = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr'
    # observations240121 = xr.open_zarr(obs_pathlarge)
    obs_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr'
    observations = xr.open_zarr(obs_path)

    obs_pathlarge = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr'
    observations240121 = xr.open_zarr(obs_pathlarge)

    #IFS ENS
    forecast_path = 'gs://weatherbench2/datasets/ifs_ens/2018-2022-64x32_equiangular_conservative.zarr'
    forecast1 = xr.open_zarr(forecast_path)
    forecast1larger= xr.open_zarr('gs://weatherbench2/datasets/ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative.zarr')

    #forecast2 = xr.open_zarr('gs://weatherbench2/datasets/ifs_ens/2018-2022-64x32_equiangular_conservative_mean.zarr')

    score = fu.ensworkflow(observations,forecast1,days=1,lag =10, zero=0)
    print(score)
    np.save('3lagscoreens.npy', score)

    score2 = fu.ensworkflow(observations240121,forecast1larger,days=1,lag =10, zero=0)
    print(score2) #Divide by extra sqrt(ensemble model) #sqrt(50) = 7
    np.save('3lagscore2ens.npy', score2)

    # _,_,dist,_ = fu.workflowfulladjusted(observations,forecast2,days = 1,lag=4,zero=0)
    # print(dist)
    # _,_,dist,_ = fu.workflowfulladjusted(observations,forecast2,days = 1,lag=5,zero=0)
    # print(dist)

    #Getting different values for different lags because of the scaling from observations. Different lags cause different observation subset.


if __name__ == "__main__":
    main()
