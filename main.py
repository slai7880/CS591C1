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
nodeAmount = {"HR" : 54573, "HU" : 47538, "RO" : 41773}
edgeFileSuffix = "_edges.csv"
genreFileSuffix = "_genres.json"
defaultDataset = os.path.join("dataset", "deezer_clean_data")
topDataSubset = os.path.join("dataset", "deezer_clean_data", "top_5000_edges_subset")
randomDataSubset = os.path.join("dataset", "deezer_clean_data", "random_subset")
metrics = ["common-friends", "total-friends", "friends-measure", "Jaccard's Coefficient"]
output2File = False
chunkSize = 10000

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

def readDataRP(directory, country, testRatio = 0.05):
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

def readDataRS(directory, country, nodeLimit, testRatio = 0.05):
    """Randomly sample a graph from the original one with number of
    nodes equal to nodeLimit.
    Parameters
    ----------
    directory : string
    country : string
    nodeLimit : int
    testRatio : float
    save : boolean
    """
    edges = pandas.read_csv(os.path.join(directory, country + edgeFileSuffix))
    fJson = open(os.path.join(directory, country + genreFileSuffix))
    genreDict = json.load(fJson)
    fJson.close()
    edgeList = []
    nodes = random.sample(range(nodeAmount[country]), nodeLimit)
    nodeMap = {nodes[i] : i for i in range(len(nodes))}
    genreDict2 = {nodeMap[n] : genreDict[str(n)] for n in nodeMap}
    for i in range(edges.shape[0]):
        # ensuring that for each e = (e1, e2), e1 < e2.
        n1, n2 = edges["node_1"][i], edges["node_2"][i]
        if n1 in nodeMap and n2 in nodeMap:
            edgeList.append((min(nodeMap[n1], nodeMap[n2]), max(nodeMap[n1], nodeMap[n2])))
    random.shuffle(edgeList)
    testAmount = int(testRatio * len(edgeList))
    test, train = edgeList[:testAmount], edgeList[testAmount:]
    graph = {i : [] for i in range(nodeLimit)}
    for e in train:
        graph[e[0]].append(e[1])
        graph[e[1]].append(e[0])
    f = open(os.path.join(dataset, country + str(nodeLimit)), "w")
    f.write(str(graph))
    f.close()
    f = open(os.path.join(dataset, country + str(nodeLimit) + "Test"), "w")
    f.write(str(test))
    f.close()
    print("Graph and test saved to " + dataset)
    return graph, genreDict2, test

def getAdjMat(graph):
    '''Given a dictionary representing a graph, returns the adjacency matrix.
    Parameters
    ----------
    graph : {int : List[int]}
    Returns
    -------
    result : numpy matrix
    '''
    result = np.zeros((len(graph), len(graph)))
    for i in graph:
        for j in graph[i]:
            result[i, j] = 1
            result[j, i] = 1
    return result

def getGraph(M):
    """Given a matrix representing a graph, returns the dictionary.
    Parameters
    ----------
    graph : numpy matrix
    Returns
    -------
    result : {int : List[int]}
    """
    result = {i : [] for i in range(M.shape[0])}
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] != 0:
                result[i].append(j)
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
        lenUnion = len(union(graph[N1], graph[N2]))
        if lenUnion == 0:
            return 0
        else:
            return len(intersect(graph[N1], graph[N2])) / lenUnion
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
    upperBound = int(fillingRate * (len(graph) * (len(graph) - 1) / 2 - count / 2))
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
                    hq.heappush(topEdges[m], outcome[i][m])
                elif topEdges[m][0][0] < outcome[i][m][0]:
                    hq.heapreplace(topEdges[m], outcome[i][m])
        
    else:
        for i in range(len(graph)):
            for j in range(i + 1, len(graph)):
                if not j in graph[i]:
                    scores = {m : (getConnectionScore(graph, i, j, m), i, j) for m in metrics}
                    
                    for m in metrics:
                        if len(topEdges[m]) < upperBound:
                            hq.heappush(topEdges[m], scores[m])
                        elif topEdges[m][0][0] < scores[m][0]:
                            hq.heapreplace(topEdges[m], scores[m])
    results = {}
    for m in metrics:
        results[m] = []
        while len(topEdges[m]) > 0:
            e = hq.heappop(topEdges[m])
            results[m].append((e[1], e[2]))
            results[m].reverse()
    return results


def runBaselineMethods(dataset, nodeLimit, fillingRate = 0.01, split = 1):
    """Use this function to run baseline methods.
    """
    '''
    graph = getSampleGraph(10)
    print(graph)
    suggested = findEdgesCheckpoint(graph, country = country, fillingRate = 0.5, split = 2)
    print(suggested)
    '''
    suggested = {}
    for country in countries:
        print("country = " + country)
        
        f = open(os.path.join(dataset, country + str(nodeLimit[country])), "r")
        graph = None
        for line in f:
            if len(line) > 1:
                graph = eval(line)
        
        suggested[country] = findEdges(graph, fillingRate = fillingRate, split = split)
    f = open(os.path.join(dataset, "baselinePredicted"), "w")
    f.write(str(suggested))
    f.close()
    
def evaluateHelper(test, predicted):
    count = 0
    for e in test:
        if e in predicted:
            count += 1
    return count / len(test)

# Evaluate a list of predicted edges against the ground truth.
def evaluate(dataset, nodeLimit, predicted):
    results = {}
    for country in countries:
        print("Country = " + country)
        f = open(os.path.join(dataset, country + str(nodeLimit[country]) + "Test"), "r")
        test = None
        for line in f:
            if len(line) > 1:
                test = eval(line)
        f.close()
        results[country] = {}
        for key in predicted[country]:
            L = predicted[country][key]
            results[country][key] = []
            for rate in [0.1 * i for i in range(1, 11)]:
                score = evaluateHelper(test, L[:int(rate * len(L))])
                print("key = " + str(key) + "  rate = " + str(np.round(rate, 2)) + "  score = " + str(score))
                results[country][key].append(score)
    return results
    
def ensemble(dataset, nodeLimit):
    pass

def getTopEdgesHelper(Y, amount):
    topEdges = []
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if i != j:
                if len(topEdges) < amount:
                    hq.heappush(topEdges, (Y[i, j], i, j))
                elif topEdges[0][0] < Y[i, j]:
                    hq.heapreplace(topEdges, (Y[i, j], i, j))
    result = []
    while len(topEdges) > 0:
        e = hq.heappop(topEdges)
        result.append((e[1], e[2]))
    result.reverse()
    return result

# Since TarDPR produces a matrix, we need to get the list of
# top edges with some extra work.
def getTopEdges(dataset, country, nodeLimit, amount):
    degree = [50, 100, 200, 400]
    result = {}
    for d in degree:
        X = np.loadtxt(os.path.join(dataset, "TarDPR", country + str(nodeLimit) + "d" + str(d) + "X.csv"), delimiter = ",")
        Y = np.dot(X.T, X)
        result[d] = getTopEdgesHelper(Y, amount)
    if country == "HU":
        result[800] = [e for e in result[400]]
    else:
        X = np.loadtxt(os.path.join(dataset, "TarDPR", country + str(nodeLimit) + "d800X.csv"), delimiter = ",")
        Y = np.dot(X.T, X)
        result[800] = getTopEdgesHelper(Y, amount)
    return result

def processResults(dataset, nodeLimit):
    fillingRate = 0.01
    tardprEdgeAmount = {"HR" : int(fillingRate * 15182212), "HU" : int(fillingRate * 16527790), "RO" : int(fillingRate * 16739872)}
    tardprPredicted = None
    
    for country in countries:
        tardprResults[country] = getTopEdges(dataset, country, nodeLimit[country], tardprEdgeAmount[country])
    
    f = open(os.path.join(dataset, "tardprPredicted"), "w")
    f.write(str(tardprResults))
    f.close()

    f = open(os.path.join(dataset, "tardprPredicted"), "r")
    for line in f:
        if len(line) > 1:
            tardprPredicted = eval(line)
    f.close()
    evaluated = evaluate(dataset, nodeLimit, tardprPredicted)
    f = open(os.path.join(dataset, "tardprEvaluated"), "w")
    f.write(str(evaluated))
    f.close()
    
    
    
    baselinePredicted = None
    f = open(os.path.join(dataset, "baselinePredicted"), "r")
    for line in f:
        if len(line) > 1:
            baselinePredicted = eval(line)
    f.close()
    evaluated = evaluate(dataset, nodeLimit, baselinePredicted)
    f = open(os.path.join(dataset, "baselineEvaluated"), "w")
    f.write(str(evaluated))
    f.close()
    

def plot(dataset, nodeLimit):
    # plot TarDPR results
    evaluated = None
    f = open(os.path.join(dataset, "tardprEvaluated"), "r")
    for line in f:
        if len(line) > 1:
            evaluated = eval(line)
    
    degrees = [50, 100, 200, 400, 800]
    colormap = {50 : "b", 100 : "r", 200 : "g", 400 : "m", 800 : "k"}
    fig, ax = plt.subplots(3, figsize = (10, 12), dpi = 100)
    for i in range(len(countries)):
        for d in degrees:
            ax[i].plot([0.001 * j for j in range(1, 11)], evaluated[countries[i]][d], color = colormap[d], label = "d = " + str(d))
        ax[i].set_title("TarDPR Results for Country " + countries[i])
        ax[i].set_xlabel("Filling Rate")
        ax[i].set_ylabel("Recover Rate")
        ax[i].legend(fontsize = 10)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "tardprdense.png"))
    plt.show()
    
    
    
    # plot baseline results
    evaluated = None
    f = open(os.path.join(dataset, "baselineEvaluated"), "r")
    for line in f:
        if len(line) > 1:
            evaluated = eval(line)
    
    colormap = {"common-friends" : "b", "total-friends" : "r", "friends-measure" : "g", "Jaccard's Coefficient" : "m"}
    fig, ax = plt.subplots(3, figsize = (10, 12), dpi = 100)
    for i in range(len(countries)):
        for m in metrics:
            ax[i].plot([0.001 * j for j in range(1, 11)], evaluated[countries[i]][m], color = colormap[m], label = m)
        ax[i].set_title("Baseline Results for Country " + countries[i])
        ax[i].set_xlabel("Filling Rate")
        ax[i].set_ylabel("Recover Rate")
        ax[i].legend(fontsize = 10)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "baselinedense.png"))
    # fig.savefig(os.path.join("figures", "baselinerandom.png"))
    plt.show()
    
    
    
    

if __name__ == "__main__":
    if output2File:
        oldStdOut = sys.stdout
        oldStdErr = sys.stderr
        # specify output file here, in general current date and time is used
        now = datetime.datetime.now()
        month = str(now.month)
        if now.month < 10:
            month = "0" + month
        day = str(now.day)
        if now.day < 10:
            day = "0" + day
        suffix = str(now.hour) + str(now.minute) + str(month) + str(day) + str(now.year)
        outName = "stdout/stdout" + suffix + ".txt"
        errName = "stderr/stderr" + suffix + ".txt"
        sys.stdout = open(outName, "w")
        sys.stderr = open(errName, 'w')
    
    
    
    start = time.time()
    print("###############################################################################")
    print(datetime.datetime.now())
    print("")
    
    dataset = randomDataSubset
    nodeLimit = {"HR" : 5000, "HU" : 5000, "RO" : 5000} # used for random subset
    nodeLimit = {"HR" : 3897, "HU" : 4066, "RO" : 4092} # used for top subset
    
    for country in countries:
        graph, genreDict, test = readDataRS(dataset, country, nodeLimit[country])
        A = getAdjMat(graph)
        np.savetxt(os.path.join(dataset, country + str(nodeLimit[country]) + "Mat.csv"), A, delimiter = ",")
    
    
    runBaselineMethods(randomDataSubset, nodeLimit, split = 8)
    
    # TarDPR must be run with the main.m file using Matlab.
    
    # processResults(dataset, nodeLimit)
    # plot(dataset, nodeLimit)

    
    end = time.time()
    print("Run time = " + str((end - start) // 60) + " minutes")
    print("")
    
    
    if output2File:
        sys.stderr.flush()
        sys.stderr = oldStdErr
        sys.stdout.flush()
        sys.stdout = oldStdOut
    