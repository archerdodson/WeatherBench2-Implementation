from datetime import datetime, timedelta
import numpy as np
import sigkernel
import torch
import weatherbench2
import xarray as xr
import cython
from multiprocessing import Pool, cpu_count

import time

def workflowfullparallel(observations, forecasts, days, lag,zero):
    ob, fo = timecutting(observations,forecasts,days,lag+zero)
    ob = ob.values
    fo = fo[:,0:lag,:,:].values
    ob, fo = scalebyobsadjusted(ob,fo,fo.shape[2])
    pkarray, pkarraylat, distance, score = pkparallel(ob,fo,zero)

    return (pkarray, pkarraylat, distance, score)


def workflowfulladjusted(observations, forecasts, days, lag,zero):
    ob, fo = timecutting(observations,forecasts,days,lag+zero)
    ob = ob.values
    fo = fo[:,0:lag,:,:].values
    ob, fo = scalebyobsadjusted(ob,fo,fo.shape[2])
    newpk, pkarraylat, distance,score = pkfulladjusted(ob,fo,zero)

    return (newpk, pkarraylat, distance,score)

def timecutting(obsxarray,forsxarray, days, predtimedelta):
    startdate = datetime(2020,1,1)
    newdate = startdate + timedelta(days=days)
    obsnewdate = newdate + timedelta(hours = (predtimedelta-1)*6)
    
    return obsxarray['geopotential'].sel(level = 500, time = slice(startdate, obsnewdate)), forsxarray['geopotential'].sel(level = 500, time = slice(startdate, newdate)), 


def scalebyobsadjusted(observations,forecasts,shapeparam):
    mean = np.mean(observations)
    std = np.std(observations)
    scaledobs = (observations - mean)/(std*np.sqrt(shapeparam))
    scaledfors = (forecasts - mean)/(std*np.sqrt(shapeparam))

    return scaledobs, scaledfors


#Given forecast of time T with predlag, and observations of time T + predlag
#Average each path of length k across all t.
#zero is either 1 or 0 
#latitude weighting
weights = np.array([0.07704437, 0.23039114, 0.38151911, 0.52897285, 0.67133229,
       0.80722643, 0.93534654, 1.05445875, 1.16341595, 1.26116882,
       1.34677594, 1.41941287, 1.47838008, 1.52310968, 1.55317091,
       1.56827425, 1.56827425, 1.55317091, 1.52310968, 1.47838008,
       1.41941287, 1.34677594, 1.26116882, 1.16341595, 1.05445875,
       0.93534654, 0.80722643, 0.67133229, 0.52897285, 0.38151911,
       0.23039114, 0.07704437])

weightslarger = np.array([0.00518318, 0.04146013, 0.08289184, 0.12426674, 0.16555647,
       0.20673274, 0.24776733, 0.28863211, 0.32929907, 0.36974035,
       0.40992823, 0.44983517, 0.48943381, 0.52869701, 0.56759787,
       0.60610973, 0.6442062 , 0.68186115, 0.7190488 , 0.75574364,
       0.79192053, 0.82755468, 0.86262167, 0.89709746, 0.93095842,
       0.96418135, 0.99674348, 1.02862249, 1.05979653, 1.09024424,
       1.11994476, 1.14887772, 1.17702329, 1.20436219, 1.23087569,
       1.2565456 , 1.28135434, 1.3052849 , 1.32832088, 1.35044651,
       1.3716466 , 1.39190663, 1.41121272, 1.42955164, 1.44691081,
       1.46327834, 1.47864302, 1.4929943 , 1.50632236, 1.51861807,
       1.52987299, 1.54007941, 1.54923034, 1.5573195 , 1.56434135,
       1.57029108, 1.57516462, 1.57895861, 1.58167047, 1.58329832,
       1.58384106, 1.58329832, 1.58167047, 1.57895861, 1.57516462,
       1.57029108, 1.56434135, 1.5573195 , 1.54923034, 1.54007941,
       1.52987299, 1.51861807, 1.50632236, 1.4929943 , 1.47864302,
       1.46327834, 1.44691081, 1.42955164, 1.41121272, 1.39190663,
       1.3716466 , 1.35044651, 1.32832088, 1.3052849 , 1.28135434,
       1.2565456 , 1.23087569, 1.20436219, 1.17702329, 1.14887772,
       1.11994476, 1.09024424, 1.05979653, 1.02862249, 0.99674348,
       0.96418135, 0.93095842, 0.89709746, 0.86262167, 0.82755468,
       0.79192053, 0.75574364, 0.7190488 , 0.68186115, 0.6442062 ,
       0.60610973, 0.56759787, 0.52869701, 0.48943381, 0.44983517,
       0.40992823, 0.36974035, 0.32929907, 0.28863211, 0.24776733,
       0.20673274, 0.16555647, 0.12426674, 0.08289184, 0.04146013,
       0.00518318])

def pkparallel_lat_split(lat_chunk, observations_chunk, forecasts_chunk, zero):
    """
    Function to compute results for a chunk of latitudes.
    """
    static_kernel = sigkernel.Linear_ID_Kernel()
    dyadic_order = 2
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)

    time = forecasts_chunk.shape[0]
    lead = forecasts_chunk.shape[1]
    latlength = forecasts_chunk.shape[3]

    pkarray = np.zeros((latlength, time, lead, 3))

    for lat in range(latlength):
        #for t in range(time):
        for lag in range(2, lead + 1):

            batched_fors = torch.cat([
                    torch.tensor(sigkernel.transform(np.expand_dims(forecasts_chunk[t,0:lag,:,lat], axis = 0), scale = 1, at = True, ll = False), dtype=torch.double)
                    for t in range(time)
                ])
            
            batched_obs = torch.cat([
                torch.tensor(sigkernel.transform(np.expand_dims(observations_chunk[2*t+zero:2*t+zero+lag,:,lat], axis = 0), scale = 1, at = True, ll = False), dtype=torch.double)
                for t in range(time)
            ])


            K_Xy = signature_kernel.compute_kernel(batched_fors, batched_obs)
            K_XX = signature_kernel.compute_kernel(batched_fors, batched_fors)
            K_yy = signature_kernel.compute_kernel(batched_obs, batched_obs)


            pkarray[lat,:,lag-1,0] = K_Xy
            pkarray[lat,:,lag-1,1] = K_XX
            pkarray[lat,:,lag-1,2] = K_yy

    return pkarray

def pkparallel(observations, forecasts, zero, batch_size=None):
    """
    Main function to parallelize computation across latitudes.
    """
    latlength = forecasts.shape[3]

    # Determine the number of processes (default to number of cores)
    num_cores = cpu_count()
    print(num_cores)
    batch_size = batch_size or (latlength // num_cores + (latlength % num_cores > 0))

    # Split data into chunks by latitude
    lat_chunks = [
        range(i, min(i + batch_size, latlength))
        for i in range(0, latlength, batch_size)
    ]
    print(lat_chunks)

    observations_chunks = [
        observations[:, :, lat_chunk]
        for lat_chunk in lat_chunks
    ]

    forecasts_chunks = [
        forecasts[:, :, :, lat_chunk]
        for lat_chunk in lat_chunks
    ]

    # Process chunks in parallel
    with Pool(processes=min(num_cores, len(lat_chunks))) as pool:
        results = pool.starmap(
            pkparallel_lat_split,
            [(lat_chunk, obs_chunk, for_chunk, zero)
             for lat_chunk, obs_chunk, for_chunk in zip(
                 lat_chunks, observations_chunks, forecasts_chunks
             )]
        )

    if latlength == 32:
        print("32")
        usedweights = weights
    if latlength == 121:
        print("121")
        usedweights = weightslarger

    # Combine results from all chunks
    #pkarraylat = np.sum(results, axis=0)
    pkarray = np.concatenate(results, axis=0) #Against lat chunks
    pkarraylat = np.sum(pkarray * usedweights[:, None, None, None], axis=0)
    pktime = np.mean(pkarraylat, axis=0)
    distance = pktime[:, 1] + pktime[:, 2] - 2 * pktime[:, 0]
    score = pktime[:, 1] - 2 * pktime[:, 0]

    return pkarray, pkarraylat, distance, score


def pkfulladjusted(observations,forecasts,zero):

        static_kernel = sigkernel.Linear_ID_Kernel()   #Linear_ID_Kernel()  #RBFKernel(sigma=sigma)
        dyadic_order = 2
        signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
        time = forecasts.shape[0]
        lead = forecasts.shape[1]
        latlength = forecasts.shape[3]

        if latlength == 32:
             print("32")
             usedweights = weights
        if latlength == 121:
             print("121")
             usedweights = weightslarger


        #print(time,lead)
        
        pkarray = np.zeros((latlength,time,lead,3))
        #print(pkarray.shape)

        for lat in range(latlength):
            print(lat)

        #Across all t, across all lags
            for t in range(time):
                #print(t)
                for lag in range(2,lead+1):
                    #print(lag)
                    fors = forecasts[t,0:lag,:,lat]
                    #shape lag,64 Length lag, dimension 64
                    obs = observations[2*t+zero:2*t+zero+lag,:,lat]

                    fors = np.expand_dims(fors, axis = 0) #1,lag,64
                    obs = np.expand_dims(obs, axis = 0) #1,64


                    llobs = sigkernel.transform(obs, scale = 1, at = True, ll = False)
                    llfors = sigkernel.transform(fors, scale = 1, at = True, ll = False)

                    X = torch.tensor(llfors, dtype=torch.double)
                    y = torch.tensor(llobs, dtype=torch.double)


                    K_Xy = signature_kernel.compute_Gram(X, y, sym=False, max_batch=100)
                    K_XX = signature_kernel.compute_Gram(X, X, sym=False, max_batch=100)
                    K_yy = signature_kernel.compute_Gram(y, y, sym=False, max_batch=100)

                    pkarray[lat,t,lag-1,0] = K_Xy
                    pkarray[lat,t,lag-1,1] = K_XX
                    pkarray[lat,t,lag-1,2] = K_yy
        
        pkarraylat = np.sum(pkarray*usedweights[:,None,None,None],axis=0)
        pktime = np.mean(pkarraylat, axis=0)
        distance = pktime[:,1]+pktime[:,2]-2*pktime[:,0]
        score = pktime[:,1] - 2*pktime[:,0]

    
        return(pkarray,pkarraylat,distance,score)


def ensworkflow(observations, forecasts, days, lag,zero):
    ob, fo = timecutting(observations,forecasts,days,lag+zero)
    ob = ob.values
    fo = fo[:,:,0:lag,:,:].values #0 to 2
    ob, fo = scalebyobsadjusted(ob,fo,fo.shape[3])
    score = pkens(ob,fo,zero)

    return (score)

def pkens(observations, forecasts, zero, batch_size = None):
    #Forecast (time, number, lag, long, lat)
    #Observations (time, long, lat)

    latlength = forecasts.shape[4]

    # Determine the number of processes (default to number of cores)
    num_cores = cpu_count()
    print(num_cores)
    batch_size = batch_size or (latlength // num_cores + (latlength % num_cores > 0))

    # Split data into chunks by latitude
    lat_chunks = [
        range(i, min(i + batch_size, latlength))
        for i in range(0, latlength, batch_size)
    ]
    print(lat_chunks)

    observations_chunks = [
        observations[:, :, lat_chunk]
        for lat_chunk in lat_chunks
    ]

    forecasts_chunks = [
        forecasts[:,:,:, :, lat_chunk]
        for lat_chunk in lat_chunks
    ]

    # Process chunks in parallel
    print('start process')
    with Pool(processes=min(num_cores, len(lat_chunks))) as pool:
        results = pool.starmap(
            pkparallel_lat_splitens,
            [(lat_chunk, obs_chunk, for_chunk, zero)
             for lat_chunk, obs_chunk, for_chunk in zip(
                 lat_chunks, observations_chunks, forecasts_chunks
             )]
        )

    if latlength == 32:
        print("32")
        usedweights = weights
    if latlength == 121:
        print("121")
        usedweights = weightslarger

    # Combine results from all chunks
    #pkarraylat = np.sum(results, axis=0)

    #Processing... 
    
    #pkarray = np.concatenate(results, axis=0) #Against lat chunks

    pkarraylat = np.sum(results * usedweights[:, None, None], axis=0) #using results
    pktime = np.mean(pkarraylat, axis=0)
    # distance = pktime[:, 1] + pktime[:, 2] - 2 * pktime[:, 0]
    # score = pktime[:, 1] - 2 * pktime[:, 0]

    return pktime


def pkparallel_lat_splitens(lat_chunk, observations_chunk, forecasts_chunk, zero):
    """
    Function to compute results for a chunk of latitudes.
    """
    static_kernel = sigkernel.Linear_ID_Kernel()
    dyadic_order = 2
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)

    time = forecasts_chunk.shape[0]
    number = forecasts_chunk.shape[1]
    lead = forecasts_chunk.shape[2]
    latlength = forecasts_chunk.shape[4]

    pkarray = np.zeros((latlength, time, lead)) ##

    for lat in range(latlength):
        #print(lat)
        for t in range(time): ### Change with batching
            #print(t, "time")
            for lag in range(2, lead + 1):
                #print(lag, "lag")
                #print(forecasts_chunk[t,:,0:lag,:,lat].shape)
                X = torch.tensor(sigkernel.transform(forecasts_chunk[t,:,0:lag,:,lat], scale = 1, at = True, ll = False), dtype=torch.double)
                #print(X.shape)
                y = torch.tensor(sigkernel.transform(np.expand_dims(observations_chunk[2*t+zero:2*t+zero+lag,:,lat], axis = 0), scale = 1, at = True, ll = False), dtype=torch.double)

                score = signature_kernel.compute_scoring_rule(X,y)

                pkarray[lat,t,lag-1] = score

    return pkarray

