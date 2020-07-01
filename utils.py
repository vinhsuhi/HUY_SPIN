import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import random


def read_graph_corpus(path, label_center_path=None):
    graphs = []
    label_centers = []
    with open(path, 'r', encoding='utf-8') as file:
        nodes = {}
        edges = {}
        for line in file:
            if 't' in line:
                if len(nodes) > 0:
                    graphs.append((nodes, edges))
                    # if len(graphs) > 9:
                        # break
                nodes = {}
                edges = {}
            if 'v' in line:
                data_line = line.split()
                node_id = int(data_line[1])
                node_label = int(data_line[2])
                nodes[node_id] = node_label
            if 'e' in line:
                data_line = line.split()
                source_id = int(data_line[1])
                target_id = int(data_line[2])
                label = int(data_line[3])
                edges[(source_id, target_id)] = label
        if len(nodes) > 0:
            graphs.append((nodes,edges))
    return graphs#[10:]

def readGraphs(path):
    rawGraphs = read_graph_corpus(path)
    graphs = []
    for graph in rawGraphs:
        numVertices = len(graph[0])
        g = np.zeros((numVertices,numVertices),dtype=int)
        for v,l in graph[0].items():
            g[v,v] = l
        for e,l in graph[1].items():
            g[e[0],e[1]] = l
            g[e[1],e[0]] = l
        graphs.append(g)#[:15,:15])
    return graphs

def plotGraph(graph : np.ndarray,isShowedID=True):
    edges = []
    edgeLabels = {}
    for i in range(graph.shape[0]):
        indices = np.where(graph[i][i+1:] > 0)[0]
        for id in indices:
            edges.append([i,i+id+1])
            edgeLabels[(i,i+id+1)] = graph[i,i+id+1]
    G = nx.Graph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G)
    plt.figure()    
    nodeLabels = {node:node for node in G.nodes()} if isShowedID else {node:graph[node,node] for node in G.nodes()}
    nx.draw(G,pos,edge_color='black',width=1,linewidths=1,
        node_size=500,node_color='pink',alpha=0.9,
        labels=nodeLabels)
    
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edgeLabels,font_color='red')
    plt.axis('off')
    plt.savefig('./figures/{}.png'.format(np.array2string(graph[0])),format='PNG')
    plt.show()

def generateData():
    random.seed(10)
    # nodes = [1,2,3,5,6,8,9,10,11,12,15,18,19,20]
    nGraphs = 30
    # nNodes = 20
    # nodes = [0,2,3,4,5,7,8,11,12,13,15,16,18]
    nNodes = 10
    # nodes = [0,1,2,4]
    nodes = [0,1,2,4,5,6,8,9]
    maxLabel = 100
    nodesLabel = [random.randint(0,maxLabel) if x in nodes else 0 for x in range(nNodes)]
    print(nodesLabel)
    edges = []
    for i,node in enumerate(nodes):
        for j in range(i+1,len(nodes)):
            if random.randint(0,3) != 0:
                edges.append((nodes[i],nodes[j],random.randint(0,maxLabel)))
    print(edges)
    f = open('./datasets/mico-{}.outx'.format(nNodes),'w')

    for graph in range(nGraphs):
        nodesGraph = nodesLabel.copy()
        edgesGraph = edges.copy()
        for i in range(nNodes):
            if nodesGraph[i] == 0:
                iConnected = i
                # for j in range(i+1,nNodes):
                while iConnected < nNodes:
                    iConnected =  random.randint(iConnected + 1,iConnected + nNodes//2)
                    if iConnected < nNodes:
                        edgesGraph.append((i,iConnected,random.randint(0,maxLabel)))
                    else: 
                        edgesGraph.append((i,nNodes-1,random.randint(0,maxLabel)))

                nodesGraph[i] = random.randint(0,maxLabel)
        f.write("t # {}\n".format(graph))
        for i,label in enumerate(nodesGraph):
            f.write("v {} {}\n".format(i,label))
        for edge in edgesGraph:
            f.write("e {} {} {}\n".format(edge[0],edge[1],edge[2]))
    f.close()
        # print("idGraph",graph)
        # print("nodeGraphs",nodesGraph)
        # print("edgesGraphs",edgesGraph)

        

        



