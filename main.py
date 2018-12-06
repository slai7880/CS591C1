import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas
import json
import pickle
import random
import heapq as hq
from multiprocessing import Pool
import multiprocessing
import datetime
import time

countries = ["HR", "HU", "RO"]
edgeFileSuffix = "_edges.csv"
genreFileSuffix = "_genres.json"
defaultDataset = os.path.join("dataset", "deezer_clean_data")
metrics = ["common-friends", "total-friends", "friends-measure", "Jaccard's Coefficient"]
output2File = True

def getSampleGraph(size):
    """Randomly generates a graph with |V| = size.
    """
    matrix = np.random.rand(size, size)
    matrix = matrix.T * matrix
    for i in range(size):
        for j in range(size):
            if matrix[i, j] > 0.4:
                matrix[i, j] = 1
            else:
                matrix[i, j] = 0
    graph = {i : [] for i in range(size)}
    for i in range(size):
        for j in range(i + 1, size):
            if matrix[i, j] == 1:
                graph[i].append(j)
                graph[j].append(i)
    return graph

def readData(directory, country):
    '''
    Reads the raw data and then returns the graph as well as the genre
    lists as dictionaries.
    Parameters
    ----------
    directory : string
        The directory where the csv and json files are located.
    country : string
        The abbreviation of the country name.
    Returns
    -------
    graph : {int : List[int]}
    genreDict : {int : List[string]}
    '''
    edges = pandas.read_csv(os.path.join(directory, country + edgeFileSuffix))
    fJson = open(os.path.join(directory, country + genreFileSuffix))
    genreDict = json.load(fJson)
    fJson.close()
    graph = {i : [] for i in range(len(genreDict))}
    maxFriendGroup = 0
    for i in range(edges.shape[0]):
        graph[edges["node_1"][i]].append(edges["node_2"][i])
        graph[edges["node_2"][i]].append(edges["node_1"][i])
        maxFriendGroup = max(maxFriendGroup, len(graph[edges["node_1"][i]]), len(graph[edges["node_2"][i]]))
    print(maxFriendGroup)
    
    return graph, genreDict

def readDataRS(directory, country, testRatio = 0.05):
    '''
    Reads the raw data, randomly partitions it into train and test sets, and
    then returns them.
    Parameters
    ----------
    directory : string
        The directory where the csv and json files are located.
    country : string
        The abbreviation of the country name.
    testRatio : float
    Returns
    -------
    graph : {int : List[int]}
    genreDict : {int : List[string]}
    '''
    edges = pandas.read_csv(os.path.join(directory, country + edgeFileSuffix))
    fJson = open(os.path.join(directory, country + genreFileSuffix))
    genreDict = json.load(fJson)
    fJson.close()
    edgeList = [None] * edges.shape[0]
    for i in range(edges.shape[0]):
        # ensuring that for each e = (e1, e2), e1 < e2.
        edgeList[i] = (min(edges["node_1"][i], edges["node_2"][i]), max(edges["node_1"][i], edges["node_2"][i]))
    random.shuffle(edgeList)
    testAmount = int(testRatio * edges.shape[0])
    test, train = edgeList[:testAmount], edgeList[testAmount:]
    graph = {i : [] for i in range(len(genreDict))}
    for e in train:
        graph[e[0]].append(e[1])
        graph[e[1]].append(e[0])
    return graph, genreDict, test

def getAdjMat(graph):
    '''Given a dictionary representing a graph, returns the adjacency matrix.
    Parameters
    ----------
    graph : {int : List[int]}
    Returns
    -------
    result : numpy matrix
    '''
    result = np.zeros((len(genreDict), len(genreDict)))
    result = np.zeros((100, 100))
    for i in graph:
        for j in graph[i]:
            result[i, j] = 1
    return result

def intersect(L1, L2):
    '''Finds the intersection between two lists.
    '''
    result = []
    for e in L1:
        if e in L2:
            result.append(e)
    return result
    
def union(L1, L2):
    '''Finds the union between two lists.
    '''
    return list(set(L1 + L2))

def countPaths(graph, N1, N2, L):
    '''Counts the number of paths between N1 and N2 of length L.
    '''
    def explore(graph, previous, current, target, currentL, L):
        if currentL <= L:
            if currentL == L and current == target:
                return 1
            else:
                result = 0
                for next in graph[current]:
                    if not next in previous:
                        previous.append(next)
                        result += explore(graph, previous, next, target, currentL + 1, L)
                        previous.pop()
                return result
        return 0
    return explore(graph, [N1], N1, N2, 0, L)

def getConnectionScore(graph, N1, N2, metric):
    '''Determines the connection score betweet N1 and N2 based on a given metric.
    Parameters
    ----------
    graph : {int : List[int]}
    N1, N2 : int
    metric : string
    '''
    if metric == "common-friends":
        return len(intersect(graph[N1], graph[N2]))
    elif metric == "total-friends":
        return len(union(graph[N1], graph[N2]))
    elif metric == "friends-measure":
        return countPaths(graph, N1, N2, 2) + countPaths(graph, N1, N2, 3)
    elif metric == "Jaccard's Coefficient":
        return len(intersect(graph[N1], graph[N2])) / len(union(graph[N1], graph[N2]))
    else:
        print("Invalid metric name: " + metric)



def findEdgesHelper(args):
    """This function serves as a helper function for findEdges.
    Parameters
    ----------
    args : List[
        graph : {int : List[int]}
        i, j : int
            They represent nodes.
        metric : string
            Must be a member of the global list metrics.
        ]
    Returns
    -------
    {string : (int, int, int)}
        Each key is the name of a metric and each value is a tuple (score, node1, node2).
    """
    graph, i, j, metrics = args
    return {m : (getConnectionScore(graph, i, j, m), i, j) for m in metrics}


def findEdges(graph, metrics = metrics, fillingRate = 0.001, split = 1):
    """Finds a list of top edges for each metric. The size of the list is
    determined by fillingRate.
    Parameters
    ----------
    graph : {int : List[int]}
    metrics : List[string]
        Must be a subset of the global list metrics.
    fillingRate : float
        The desired fillingRate (based on the total number of missing links).
    split : int
        The number of processes to execute on. Multiprocessing is used when
        split > 1 and single-thread processing is used otherwise.
    Returns
    -------
    suggested : {string : List[(int, int)]}
        Each key is a metric name and each element is a list of edges.
    """
    count = 0
    for node in graph:
        count += len(graph[node])
    upperBound = len(graph) * (len(graph) - 1) / 2 - count / 2
    topEdges = {m : [] for m in metrics}
    if split > 1:
        args = []
        for i in range(len(graph)):
            for j in range(i + 1, len(graph)):
                if not j in graph[i]:
                    args.append([graph, i, j, metrics])
        
        pool = Pool(split)
        outcome = pool.map(findEdgesHelper, args)
        pool.close()
        pool.join()
        for i in range(len(outcome)):
            for m in metrics:
                if len(topEdges[m]) < upperBound:
                    hq.heappush(topEdges[m], outcome[j][m])
                elif topEdges[m][0] < outcome[j][m]:
                    hq.heapreplace(topEdges[m], outcome[j][m])
        
    else:
        for i in range(len(graph)):
            for j in range(i + 1, len(graph)):
                if not j in graph[i]:
                    scores = {m : (getConnectionScore(graph, i, j, m), i, j) for m in metrics}
                    
                    for m in metrics:
                        if len(topEdges[m]) < upperBound:
                            hq.heappush(topEdges[m], scores[m])
                        elif topEdges[m][0] < scores[m]:
                            hq.heapreplace(topEdges[m], scores[m])
    results = {}
    for m in metrics:
        results[m] = [(e[1], e[2]) for e in topEdges[m]]
    return results


def evaluate(suggested, test, args):
    """Evaluates the suggested links.
    Parameters
    ----------
    suggested : {string : List[(int, int)]}
        Each key is a metric name and each element is a list of edges.
    test : List[(int, int)]
    Returns
    -------
    None
    """
    result = {}
    for metric in suggested:
        result[metric] = len(intersect(suggested[metric], test)) / len(test)
    print("Arguments: " + str(args))
    print("Recovery rate:")
    print(result)


if __name__ == "__main__":
    if output2File:
        oldStdOut = sys.stdout
        oldStdErr = sys.stderr
        # specify output file here, in general current date and time is used
        now = datetime.datetime.now()
        month = str(now.month)
        if now.month < 10:
            month = "0" + month
        suffix = str(now.hour) + str(now.minute) + str(month) + str(now.day) + str(now.year)
        outName = "stdout/stdout" + suffix + ".txt"
        errName = "stderr/stderr" + suffix + ".txt"
        sys.stdout = open(outName, "w")
        sys.stderr = open(errName, 'w')
    
    country = countries[0]
    fillingRate = 0.001
    
    start = time.time()
    print("###############################################################################")
    print(datetime.datetime.now())
    print("")
    
    '''
    graph = getSampleGraph(10)
    print(graph)
    suggested = findEdges(graph, split = 2)
    print(suggested)
    '''
    
    graph, genreDict, test = readDataRS(defaultDataset, country)
    suggested = findEdges(graph, fillingRate = fillingRate, split = int(sys.argv[1]))
    evaluate(suggested, test, {"Country" : country, "Filling rate" : fillingRate})
    
    
    end = time.time()
    print("Run time = " + str((end - start) // 60) + " minutes")
    print("")
    
    if output2File:
        sys.stderr.flush()
        sys.stderr = oldStdErr
        sys.stdout.flush()
        sys.stdout = oldStdOut
    