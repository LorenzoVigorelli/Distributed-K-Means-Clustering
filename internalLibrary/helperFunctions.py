import numpy as np

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

def predictedCentroidsLabeler(C_expected, C_predicted):
    distMatrix=np.sum((C_expected[:,:,np.newaxis]-C_predicted.T[np.newaxis, :,:])**2,axis=1)
    #the labeler i-th entry j, tells that i-th centroid of C_expected is associated to j-th element of C_predicted
    labeler=np.argmin(distMatrix,axis=1)
    #square distance of element of C_expected to nearest point in C_predicted
    distances=distMatrix[np.arange(len(distMatrix)),labeler]
    return labeler, distances