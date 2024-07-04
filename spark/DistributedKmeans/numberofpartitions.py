#### IMPORT LIBRARIES #### 

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_kddcup99
from functions import *

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

# cut the categorical features
x = x[:, 4:]

# cut the data fro memory reasons
subLen = 1000
x = x[:subLen,]
y = y[:subLen]

# setting the theoretical number of clusters
kTrue = len(set(y))

# setting up the output information
totalLogParallelInit = {}
totalLogParallelKmeans = {}
tDurations = {}

# cycle over num_slices to be run
nSlices = [2, 4, 8, 16, 32, 64]

for nSlice in nSlices:

    tInit = time() # compute the time of the beginning of the iteration over the number of partitions
    print(f"The iteration with {nSlice} number of partition started at time {tInit}")
    
    # parallelize over nSlice partitions
    Rdd = sc.parallelize([(None, {"x": x[i],"y": y[i], "d2":None}) for i in range(len(y)], numSlices = nSlice)

    # check if the parallelization over the partitions is ok
    assert Rdd.getNumPartitions() = nSlice,

    # rescale the RDD over the max
    maxS = Rdd.map(lambda datum: datum[1]["x"])\
           .reduce(lambda a, b: np.maximum(a, b))
    minS = Rdd.map(lambda datum: datum[1]["x"])\
           .reduce(lambda a, b: np.minimum(a, b))

    Rdd = Rdd.map(lambda datum: minmaxRescale(datum, minS, maxS))

    # setting up the input and output information for the alghoritm
    logParallelInit = {}
    logParallelKmeans = {}

    
    k=kTrue
    l=k*2 # rescaling probability to have more centroids than k

    # initialization kMeans //
    C_init = parallelInit(Rdd, k, l, logParallelInit)
    
    tInitialization = time() - tInit
    print(f"Finished the initialization after {tInitalization} seconds")
    
    # run the k-means alghoritm
    C = kMeans(Rdd, C_init, 15, logParallelKmeans)
    
    # time information
    tEnd = time() # compute the time of the end of the iteration over the number of partitions
    tDuration = tEnd - tInit
    
    print(f"The iteration with {nSlice} number of partition ended at time {tEnd} after {tDuration} seconds")

    # output in the correct memory adresses
    totalLogParallelInit[nSlice] = logParallelInit
    totalLogParallelKmeans[nSlice] = logParallelKmeans
    tDurations [nSlice] = tDuration


#### TOTAL OUTPUT ON FILE ####

# compute the total log 
log = {"totalLogParallelInit": totalLogParallelInit, "totalLogParallelKmeans": totalLogParallelKmeans, "tDurations": tDurations}

# save the log file
if not os.path.exists('data'): # create a directory if it doesnt exist
    os.makedirs('data')

pickle_file = 'data/log.pkl'
with open(pickle_file, "wb") as file:
    pickle.dump(log, file)
    
    
    
    
    

