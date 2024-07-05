#### IMPORT LIBRARIES #### 

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_kddcup99
from functions import *

#### SELECT THE PICKLE FILE ####
pickle_file = 'data/log1.pkl'
pickle_fileR = 'data/log1U.pkl'

#### SET UP SPARK ####

# import the python libraries to create/connect to a Spark Session
from pyspark.sql import SparkSession

# build a SparkSession 
#   connect to the master node on the port where the master node is listening (7077)
#   declare the app name 
#   configure the executor memory to 512 MB
#   either *connect* or *create* a new Spark Context
spark = SparkSession.builder \
    .master("spark://spark-master:7077")\
    .appName("My first spark application")\
    .config("spark.executor.memory", "512m")\
    .getOrCreate()

# create a spark context
sc = spark.sparkContext

#### IMPORT THE DATA SET ####

data = fetch_kddcup99(return_X_y = True, percent10 = True) # default percent10=True

# collect samples and features (target)
x = data[0]
y = data[1] 

# cut the data fro memory reasons
subLen = 1000
x = x[:subLen,]
y = y[:subLen]

#### PARALLEL

# setting up the output information
totalLogParallelInit = {}
totalLogParallelKmeans = {}
tDurationsParallel = {}
tPreOperationsParallel = {}

# cycle over num_slices to be run
# nSlices = [2, 4, 8, 16, 32, 64]
nSlices = [1, 2]

for nSlice in nSlices:

    tInit = time() # compute the time of the beginning of the iteration over the number of partitions
    print(f"The iteration with {nSlice} number of partition started at time {tInit}")
    
    # parallelize over nSlice partitions
    Rdd = sc.parallelize([(None, {"x": x[i],"y": y[i], "d2":None}) for i in range(len(y))], numSlices = nSlice)

    # cut the categorical attributes
    Rdd = Rdd.map(deleteBytes)\
             .persist()

    # setting the theoretical number of clusters
    kTrue = Rdd.map(lambda datum: datum[1]["y"])\
               .distinct()\
               .count()
    
    # rescale the RDD over the max
    maxS = Rdd.map(lambda datum: datum[1]["x"])\
           .reduce(lambda a, b: np.maximum(a, b))
    minS = Rdd.map(lambda datum: datum[1]["x"])\
           .reduce(lambda a, b: np.minimum(a, b))

    Rdd = Rdd.map(lambda datum: minmaxRescale(datum, minS, maxS))\
             .persist()
    
    # setting up the input and output information for the alghoritm
    logParallelInit = {}
    logParallelKmeans = {}

    k=kTrue
    l=k*2 # rescaling probability to have more centroids than k

    tInitI = time()

    tPreOperation = tInitI - tInit
    print(f"Finished the pre-steps after {tPreOperation} seconds")
          
    # initialization kMeans //
    C_init = parallelInit(Rdd, k, l, logParallelInit)
    
    tInitialization = time() - tInitI
    print(f"Finished the initialization after {tInitialization} seconds")
    
    # run the k-means alghoritm
    C = kMeans(Rdd, C_init, 15, logParallelKmeans)
    
    # time information
    tEnd = time() # compute the time of the end of the iteration over the number of partitions
    tDuration = tEnd - tInit
    
    print(f"The iteration with {nSlice} number of partition ended at time {tEnd} after {tDuration} seconds")

    # output in the correct memory adresses
    totalLogParallelInit[nSlice] = logParallelInit
    totalLogParallelKmeans[nSlice] = logParallelKmeans
    tDurationsParallel[nSlice] = tDuration
    tPreOperationsParallel[nSlice] = tPreOperation

#### TOTAL OUTPUT ON FILE ####

# compute the total log 
logParallel = {"totalLogParallelInit": totalLogParallelInit, "totalLogParallelKmeans": totalLogParallelKmeans, "tDurationsParallel": tDurationsParallel, "tPreOperationsParallel": tPreOperationsParallel}

# save the log file
if not os.path.exists('data'): # create a directory if it doesnt exist
    os.makedirs('data')

with open(pickle_fileP, "wb") as file:
    pickle.dump(logParallel, file)




print("Starting the naive inizialization part")




#### NAIVE RANDOM

# setting up some dictionaries
totalLogNaiveInit = {}
totalLogNaiveKmeans = {}
tDurationsNaive = {}
tPreOperationsNaive = {}

# cycle over num_slices to be run
# nSlices = [2, 4, 8, 16, 32, 64]
nSlices = [1, 2]

for nSlice in nSlices:

    tInit = time() # compute the time of the beginning of the iteration over the number of partitions
    print(f"The iteration with {nSlice} number of partition started at time {tInit}")
    
    # parallelize over nSlice partitions
    Rdd = sc.parallelize([(None, {"x": x[i],"y": y[i], "d2":None}) for i in range(len(y))], numSlices = nSlice)

    # cut the categorical attributes
    Rdd = Rdd.map(deleteBytes)\
             .persist()

    # setting the theoretical number of clusters
    kTrue = Rdd.map(lambda datum: datum[1]["y"])\
               .distinct()\
               .count()
    
    # rescale the RDD over the max
    maxS = Rdd.map(lambda datum: datum[1]["x"])\
           .reduce(lambda a, b: np.maximum(a, b))
    minS = Rdd.map(lambda datum: datum[1]["x"])\
           .reduce(lambda a, b: np.minimum(a, b))

    Rdd = Rdd.map(lambda datum: minmaxRescale(datum, minS, maxS))\
             .persist()
    
    # setting up the input and output information for the alghoritm
    logNaiveInit = {}
    logNaiveKmeans = {}

    k=kTrue
    l=k*2 # rescaling probability to have more centroids than k

    tInitI = time()

    tPreOperation = tInitI - tInit
    print(f"Finished the pre-steps after {tPreOperation} seconds")
          
    # initialization kMeans //
    C_init = naiveInitFromSet(Rdd, k, logNaiveInit)
    
    tInitialization = time() - tInitI
    print(f"Finished the initialization after {tInitialization} seconds")
    
    # run the k-means alghoritm
    C = kMeans(Rdd, C_init, 15, logNaiveKmeans)
    
    # time information
    tEnd = time() # compute the time of the end of the iteration over the number of partitions
    tDuration = tEnd - tInit
    
    print(f"The iteration with {nSlice} number of partition ended at time {tEnd} after {tDuration} seconds")

    # output in the correct memory adresses
    totalLogNaiveInit[nSlice] = logNaiveInit
    totalLogNaiveKmeans[nSlice] = logNaiveKmeans
    tDurationsNaive[nSlice] = tDuration
    tPreOperationsNaive[nSlice] = tPreOperation

#### TOTAL OUTPUT ON FILE ####

# compute the total log 
logNaive = {"totalLogNaiveInit": totalLogNaiveInit, "totalLogNaiveKmeans": totalLogNaiveKmeans, "tDurationsNaive": tDurationsNaive, "tPreOperationsNaive": tPreOperationsNaive}

# save the log file
if not os.path.exists('data'): # create a directory if it doesnt exist
    os.makedirs('data')

with open(pickle_fileR, "wb") as file:
    pickle.dump(logNaive, file)


#### KMEANS ++
#(something like that)? FINISH THIS FUNCTION (COMPUTE THE K-MEANS WITH K ++ INIZIALIZATION)

tInit = time()

kTrue = len(set(y))
k = kTrue

CRandom = naiveInitFromSet(Rdd, 1)
CInit = localPlusPlusInit(CRandom, kTrue)

localLloyds(C_init, k, weights=None, n_iterations=100)




    
    
    

