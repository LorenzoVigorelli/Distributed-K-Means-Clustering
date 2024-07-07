import numpy as np
import pandas as pd
from .helperFunctions import weightedAverage

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

