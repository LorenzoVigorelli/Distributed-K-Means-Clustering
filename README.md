# Distributed K-Means Clustering

## The project

### Acknowledgements
This is the final project for the 'Management and Analysis of Physics Dataset (MOD. B)' course in the 'Physics of Data' master program, University of Padua. <br>
Group 1: <a href=https://github.com/paololapo> Paolo Lapo Cerni </a>, <a href=https://github.com/emanuele-quaglio> Emanuele Quaglio </a>, <a href=https://github.com/LorenzoVigorelli> Lorenzo Vigorelli </a>, 
<a href=https://github.com/T3X3K> Arman Singh Bains </a>

### K-Means//
This project aims to adapt the well-known **K-Means** clustering algorithm to **MapReduce-like architectures**, exploiting the parallelization capabilities offered by distributed systems. K-Means consists of two stages: the *initialization* and the *Llyod iterations*. A proper initialization is crucial to obtain good results. At the state of the art, the *K-Means++* can obtain a set of initial centroids close to the optimal one but it's not easily parallelizable. Recently, **K-Means//** has been proposed to overcome this issue. 

Main reference: *Bahmani, Bahman, et al. "Scalable k-means++." arXiv preprint arXiv:1203.6402 (2012)*.

### About this repo
The main results are organized in the `ParallelInitializationRdd` notebook, including a brief exploration of the dataset and the time efficiency analysis. <br>
The `BenchmarkComputation` folder contains the code to run the analysis. The code is divided into three files, depending on whether you want to persist, persist and unpersist, or not persist at all the RDDs during the computation. <br>
The functions are divided into three files in the `internalLibrary`. To access them directly, you can also use the `instaFunctions.py` file. <br>
`Data` contains the logs of each run, defined as a nested structure of dictionaries. 

## The cluster
### CloudVeneto computing
We use **PySpark** as the engine to distribute the analysis, exploiting <a href=https://cloudveneto.it/> CloudVeneto </a> computational resources. 
Our cluster has a master node and two workers. Each machine has $4$ sockets with $1$ single-thread core each and $6.8$ Gb of volatile memory. <br>
Especially if you have little storage available, you can run into unexpected errors. It is crucial to monitor the running processes with
```
ps aux | grep spark
```
and eventually kill them via their `pid`. For instance:
```
sudo kill pid1
```


### Docker local setup
To test the code **locally**, you can use the `docker-compose.yml` file to simulate a cluster. The docker image used in this setup will be pulled from the [remote Docker registry](https://hub.docker.com/repository/docker/jpazzini/mapd-b). <br>
By default, only *one worker is created*, with $4$ cores and $512$ MB of memory. You can use an arbitrary number *N* of workers spawning the services with
```
docker compose up --scale spark-worker=N
```
This docker file will expose the Jupyter-notebook service on the port `8889` while the Spark cluster dashboard will be available on `localhost:8080`. <br>
To shut down the cluster, type:
```
docker compose down
```
