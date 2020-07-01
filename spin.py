from utils import readGraphs,plotGraph,generateData
from GraphCollection import GraphCollection
from algorithm import string2matrix
import numpy as np
import json 

datasets = "mico-10"
graphs = readGraphs('./datasets/{}.outx'.format(datasets))

def extractResultGraph(results):
    for k,v in results.items():
        numNodeGraphs = np.array([string2matrix(k).shape[0] for k,v in results.items()])
        indicesFreq = np.where(numNodeGraphs == numNodeGraphs.max())[0]
        return [string2matrix(list(results.keys())[i]) for i in indicesFreq]



if __name__ == "__main__":
    # generateData()
    # exit(0)
    graphDB = GraphCollection(graphs,1.0)
    print("Frequent edges",len(graphDB.freqEdges.items()))
    nodes = [0,2,3,4,5,7,8,11,12,13,15,16,18]
    freqGraphs = graphDB.frequentGraph(nodes)
    print("freqGraphs",freqGraphs)
    for freqGraph in freqGraphs:
        plotGraph(freqGraph,False)
