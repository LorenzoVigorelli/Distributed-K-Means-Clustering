import numpy as np
import pandas as pd
from time import time

###### FUNCTIONS #######

def labelToInt(label):
    '''
    Map from set of labels in original dataset (`strings`) into `int` for easier manipulation of rdd
    '''
    uniqueLabels=list(np.unique(y))
    return uniqueLabels.index(label)


def deleteBytes(datum):
    '''
    Clean dataset from categorical attributes, leaving numerical ones
    '''
    x = datum[1]["x"]
    mask = [type(i) != bytes for i in x]
    datum[1]["x"] = np.asarray(x[mask])
    print(x)
    print(mask)
    return datum
    
def localPlusPlusInit(points, k): 
    '''
    KMeans++ initialization.
    Arguments:
    `points`: array (n, dim) of points to be clustered;
    `k`: desired number of centroids. 
    Returns:
    initial array (k, dim) of centroids, k<=n.
    '''
    # Sample one point uniformly from points array
    C=points[np.random.choice(points.shape[0])]
    C=C[np.newaxis, :]
    
    for _ in range(k):
        # Compute array (n,1) of probabilities associated to each point
        probs=np.min(np.sum((points[:,:,np.newaxis]-C.T[np.newaxis,:,:])**2, axis=1), axis=1).flatten()
        # Normalize probability distribution
        probs=probs/np.sum(probs)
        
        # Draw one new centroid according to distrbution
        nextCentroid=points[np.random.choice(points.shape[0], p=probs)][np.newaxis,:]
        # Add centroid to array
        C=np.vstack((C, nextCentroid))
    return C


def weightedAverage(group):
    """
    Compute weighted average of a group from a pd.DataFrame with point coordinates, weights, clusterId.
    Utilized in local (non-distributed) version of Lloyds algorithm, needed also in K-Means//
    """
    weight_column='weights'
    groupby_column='clusterId'
    columns_to_average = group.columns.difference([weight_column, groupby_column])
    weighted_averages = group[columns_to_average].multiply(group[weight_column], axis=0).sum() / group[weight_column].sum()
    return weighted_averages


def localLloyds(points, k, weights=None, n_iterations=100):
    """
    Local (non-distributed) Lloyds algorithm
    Arguments:
    `points`: array (n, dim) of points to cluster;
    `k`: number of desired clusters;
    `weights`: optional, weights for weighted average in centroid re-computing;
    `n_iterations`: optional, number of iteration in lloyds algorithm;
    Return:
    array of best centroids computed.
    """
    df=pd.DataFrame(points)
    
    # If weights not given, assume uniform weights for points
    if weights is None:
        weights=np.ones(shape=len(points))
    df['weights']=weights
    df['clusterId']=np.zeros(shape=len(points))

    # K-Means++ initialization
    C=localPlusPlusInit(points, k)
    clusterId=np.argmin(np.sum((points[:,:,np.newaxis]-C.T[np.newaxis,:,:])**2, axis=1), axis=1)
    for iteration in range(n_iterations):
        # Compute centroid given cluster
        df['clusterId']=clusterId
        C_df=df.groupby('clusterId')\
            .apply(weightedAverage)\
            .reset_index()
        # Compute cluster given centroid
        C_array=C_df[C_df.columns.difference(['weights', 'clusterId'])].reset_index(drop=True).to_numpy()
        clusterId=np.argmin(np.sum((points[:,:,np.newaxis]-C_array.T[np.newaxis,:,:])**2, axis=1), axis=1)
    return C_array   


def minmaxRescale(datum, minS, maxS):
    """
    Rescale datum in [0,1] interval for better clusterization
    Arguments:
    `datum`: see rdd format;
    `minS`: array of min coordinate value among points for each attribute;
    `maxS`: as `minS` with max.
    """
    mask = (minS < maxS).astype(bool)
    feature = datum[1]["x"] 
    feature = (feature[mask] - minS[mask])/(maxS[mask] - minS[mask])
    return (datum[0], {"x": feature, "y": datum[1]["y"], "d2":datum[1]["d2"]}) 


def selectCluster(datum, C, updateDistances=True):
    """
    Associate datum to its centroid and optionally updates squared distance between them.
    Arguments:
    `datum`: see rdd format;
    `C`: array (k, len(datum[1]["x"]));
    `updateDistances`: if True, updates `datum[1]["d2"]` with squared distance between datum point and closest centroid in C;
    """
    distances = np.sum((datum[1]["x"] - C)**2, axis=1)
    print('distances: ',distances)
    clusterId = np.argmin(distances)
    if updateDistances is True:
        return (clusterId, {'x':datum[1]['x'], 'y':datum[1]['y'], 'd2':distances[clusterId]})
    else:
        return (clusterId, datum[1])


def updateCentroids(Rdd):
    """
    Update centroids as spatial average of cluster points
    Argument:
    `Rdd`: see rdd format;
    Return:
    Updated array of centroids.
    """
    C=Rdd.mapValues(lambda xy: (xy['x'], 1))\
              .reduceByKey(lambda a,b : (a[0]+b[0], a[1]+b[1]))\
              .mapValues(lambda a:a[0]/a[1])\
              .values()\
              .collect() 
    C=np.array(C) #check later more carefully if causes some overhead
    return C


def updateDistances(Rdd, C):
    """
    Update Rdd with square distances from centroids, given Rdd with clusters already assigned to each point
    Arguments:
    `Rdd`: see rdd format;
    `C`: array of cluster centroids;
    """
    def datumUpdate(datum, C):
        '''
        Update a datum of the rdd with distance from assigned centroid
        '''
        d2=np.sum((datum[1]['x']-C[datum[0]])**2)
        #return datum
        return (datum[0], {"x": datum[1]["x"], "y": datum[1]["y"], "d2":d2})
    Rdd=Rdd.map(lambda datum:datumUpdate(datum, C))
    return Rdd


def cost(Rdd):
    """
    Calculate global cost of clusterization, from an Rdd with distances from centroids already updated
    """
    my_cost=Rdd.map(lambda datum : datum[1]['d2'])\
               .reduce(lambda a,b: a+b)
    return my_cost 


def kMeans(Rdd, C_init, maxIterations, logParallelKmeans=None):
    """
    Distributed (parallel) Lloyds algorithm
    Arguments:
    `Rdd`: see rdd format;
    `C_init`: array (k, dim) of initial centroids;
    `maxIterations`: max number of iterations;
    `logParallelKmeans`: optional, dictionary {'CostsKmeans', 'tIterations', 'tTotal'} to store cost and time info.
    """
    
    t0 = time()

    # Storing cost and time info
    my_kMeansCosts = []
    tIterations = []
    C=C_init

    for t in range(maxIterations):
        t1 = time()
        RddCached = Rdd.map(lambda datum: selectCluster(datum, C)).persist() ###
        
        # Now we compute the new centroids by calculating the averages of points belonging to the same cluster.
        C=updateCentroids(RddCached)
        my_cost = cost(RddCached)
        
        my_kMeansCosts.append(my_cost)
        t2 = time()
        
        tIteration = t2 - t1
        tIterations.append(tIteration)
        
        #RddCached.unpersist() 

        # Break loop if convergence of cost is reached
        if (len(my_kMeansCosts) > 1) and (my_kMeansCosts[-1] > 0.999*my_kMeansCosts[-2]):
            break

    tEnd = time()
    tTotal = tEnd - t0

    # Store cost and time info in argument dictionary
    if logParallelKmeans is not None:
        logParallelKmeans["CostsKmeans"] = my_kMeansCosts
        logParallelKmeans["tIterations"] = tIterations
        logParallelKmeans["tTotal"] = tTotal

    return C


def naiveInitFromSet(Rdd, k, spark_seed=12345,logNaiveInit=None):
    """
    Uniform sampling of k points from Rdd
    """
    t0 = time()
    # Sampling. Replacement is set to False to avoid coinciding centroids BUT no guarantees that in the original dataset all points are distinct!!!
    kSubset=Rdd.takeSample(False, k, seed=spark_seed)
    C_init=np.array([datum[1]['x'] for datum in kSubset])

    tEnd = time()
    
    if logNaiveInit is not None:
        logNaiveInit["tTotal"] = tEnd - t0
        
    return C_init


def naiveInitFromSpace(k, dim):
    """
    #uniform drawing of k points from euclidean space
    #we assume the Rdd has been mapped into a [0,1]^dim space
    """
    C_init=np.random.uniform(size=(k,dim))
    return C_init


def parallelInit(Rdd, k, l, logParallelInit=None):
    """
    Parallel initialization
    """
    t0 = time()
    
    # initialize C as a point in the dataset
    C=naiveInitFromSet(Rdd, 1) 
    
    # associate each datum to the only centroid (computed before) and computed distances and cost
    Rdd=Rdd.map(lambda datum : (0, datum[1]))
    Rdd=updateDistances(Rdd, C).persist() ###
    
    my_cost=cost(Rdd)

    # number of iterations (log(cost))
    n_iterations=int(np.log(my_cost))
    if(n_iterations<1): n_iterations=1

    
    tSamples = []
    tCentroids = []
    CostInits = [my_cost]
    # iterative sampling of the centroids
    for _ in range(n_iterations):

        t1=time()
        # sample C' according to the probability
        C_prime=Rdd.filter(lambda datum : np.random.uniform()<l*datum[1]['d2']/my_cost)\
                   .map(lambda datum : datum[1]['x'])\
                   .collect()
        C_prime=np.array(C_prime)
        t2=time()

        # stack C and C', update distances, centroids, and cost
        if (C_prime.shape[0]>0):
            C=np.vstack((C, C_prime))
            
            #Rdd.unpersist() ###
            Rdd=Rdd.map(lambda datum: selectCluster(datum, C)).persist() ###
            
            my_cost=cost(Rdd)
        t3=time()

        tSample = t2 -t1
        tCentroid = t3 - t2
        tSamples.append(tSample)
        tCentroids.append(tCentroid)
        CostInits.append(my_cost)
       
    #erase centroids sampled more than once 
    C=C.astype(float)
    C=np.unique(C, axis=0)
    Rdd=Rdd.map(lambda datum: selectCluster(datum, C))
    
    #compute weights of centroids (sizes of each cluster) and put them in a list whose index is same centroid index as C
    wx=Rdd.countByKey()
    weights=np.zeros(len(C))
    weights[[list(wx.keys())]]=[list(wx.values())]
    
    #subselection of k centroids from C, using local Lloyds algorithm with k-means++ initialization
    if C.shape[0]<=k:
        C_init=C
    else:
        C_init=localLloyds(C, k, weights=weights, n_iterations=100) #can be set to lloydsMaxIterations for consistency TODO

    tEnd = time()
    
    if logParallelInit is not None:
        logParallelInit["tSamples"] = tSamples
        logParallelInit["tCentroids"] = tCentroids
        logParallelInit["CostInit"] = CostInits
        logParallelInit["tTotal"] = tEnd - t0

    #Rdd.unpersist() ###
    return C_init

def predictedCentroidsLabeler(C_expected, C_predicted):
    distMatrix=np.sum((C_expected[:,:,np.newaxis]-C_predicted.T[np.newaxis, :,:])**2,axis=1)
    #the labeler i-th entry j, tells that i-th centroid of C_expected is associated to j-th element of C_predicted
    labeler=np.argmin(distMatrix,axis=1)
    #square distance of element of C_expected to nearest point in C_predicted
    distances=distMatrix[np.arange(len(distMatrix)),labeler]
    return labeler, distances

