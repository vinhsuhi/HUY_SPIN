import numpy as np
from typing import List
import operator

class GraphCollections():
    tempTrees = {}
    def __init__(self,graphs_,theta_):
        self.graphs = graphs_
        self.theta = theta_
        self.freqEdges = self.getFrequentEdges(self.graphs,self.theta)
        self.initTempTree()

    def getFrequentEdges(self,graphs : List[np.ndarray],theta):
        frequentEdges = {}
        for idGraph,graph in enumerate(graphs):
            visited = [False]*len(graph)
            edgesSet = set()
            queue = []
            start = 0
            queue.append(start)
            visited[start] = True
            while queue:
                s = queue.pop(0)
                for i,v in enumerate(graph[s]):
                    if i != s and v > 0:
                        if visited[i] == False:
                            # evaluating edge
                            labelNodes = [graph[s,s],graph[i,i]]
                            labelNodes = sorted(labelNodes)
                            # encodeEdges = '{}-{}-{}'.format(labelNodes["tree"],labelNodes["index"],v)
                            encodeEdges = (labelNodes[0],labelNodes[1],v)
                            if encodeEdges not in edgesSet:
                                if encodeEdges not in frequentEdges:
                                    frequentEdges[encodeEdges] = {}
                                    frequentEdges[encodeEdges]['freq'] = 1
                                    frequentEdges[encodeEdges]['edges'] = {}
                                else:
                                    frequentEdges[encodeEdges]['freq'] += 1
                                # frequentEdges[encodeEdges]['freq'] = 0 if encodeEdges not in frequentEdges else frequentEdges[encodeEdges]['freq'] + 1                      
                                edgesSet.add(encodeEdges)
                                frequentEdges[encodeEdges]['edges'][idGraph] = [(s,i)]
                            else:
                                frequentEdges[encodeEdges]['edges'][idGraph].append((s,i)) 
                            # end evaluating
                            queue.append(i)
                            visited[i] = True

        # tempFrequents = [{k: v['edges']} for k, v in frequentEdges.items() if v['freq'] >theta*len(graphs)]
        frequents = {}
        for k,v in frequentEdges.items():
            if v['freq'] > theta*len(graphs):
                frequents[k] = v['edges']
        return frequents

    def initTempTree(self):
        for edge,matches in self.freqEdges.items():
            # edge,matches = list(freqEdge.items())[0]
            matrix = np.array([[edge[0],edge[2]],[edge[2],edge[1]]])
            encodeEdge = np.array2string(matrix)
            self.tempTrees[encodeEdge] = {}
            for i in matches.keys():# range(len(self.graphs)):
                topo = []
                for e in matches[i]:
                    m = np.array([[e[0],edge[2]],[edge[2],e[1]]])
                    topo.append(m)
                self.tempTrees[encodeEdge][i] = topo  

    def encodeGraph(self,graph):
        visited = [False]*len(graph)
        queue = []
        queue.append(0)
        visited[0] = True
        code = str(graph[0,0]) + '$'
        while queue:
            s = queue.pop(0)
            levelStr = ''
            for i in np.where(graph[s]>0)[0][s+1:]:
                queue.append(i)
                levelStr += str(graph[s,i]) + "_" + str(graph[i,i]) + "_"
                visited[i] = True 
            if levelStr != '':
                code += levelStr[:-1] +  '$'
        code += '#'

        return code


    def canonicalForm(self,graph: np.ndarray):
        labelNodes = graph.diagonal()
        start = np.zeros((1,1),dtype=int)
        maxNodes = np.where(labelNodes == np.max(labelNodes))[0]
        start[0,0] = np.max(labelNodes)
        canonical = {
            "code" : ''
        }
        for idStart in maxNodes:
            S = {
                "tree" : start,
                "index" : np.array([idStart]),
                "code" : encodeGraph(start)
            }

            while (len(S["index"]) < len(labelNodes)):
                # trees = []
                newCandidates = {}
                for i in range(graph.shape[0]):
                    if i in S["index"]:
                        continue
                    Q = []
                    t = S["tree"]
                    for id,j in enumerate(S["index"]):
                        if graph[i,j] == 0:
                            continue
                        rowExpand = np.zeros((1,t.shape[0]),dtype=int)
                        rowExpand[0,id] = graph[i,j]
                        tree = np.r_[t,rowExpand]
                        colExpand = np.zeros((tree.shape[0],1),dtype=int)
                        colExpand[id,0] = graph[i,j]
                        colExpand[tree.shape[0]-1,0] = graph[i,i]
                        tree = np.c_[tree,colExpand]
                        indexTree = np.concatenate([S["index"],np.array([i])])
                        codeTree = self.encodeGraph(tree)
                        newCandidates[codeTree] = {
                            "tree" : tree,
                            "index" : indexTree,
                            "code" : codeTree
                        }

                S = newCandidates[max(newCandidates.keys())]
            canonical = S if canonical["code"] < S["code"] else canonical 
        print(canonical)            
        return canonical

    def joinCase3bFFSM(self,graphX: np.ndarray,graphY: np.ndarray):
        n = graphX.shape[0]
        # if not np.array_equal(graphX[:-1,:-1],graphY[:-1,:-1]):
            # return None
        rowExpand = np.zeros((1,n),dtype=int)
        rowExpand[0] = graphY[n-1]
        rowExpand[0,n-1] = 0
        graphX = np.r_[graphX,rowExpand]
        colExpand = np.concatenate([rowExpand[0],np.array([graphY[n-1,n-1]])])
        colExpand = np.reshape(colExpand,(n+1,1))
        graphX = np.c_[graphX,colExpand]
        # print(graphX)
        return graphX

    def extend(self,X: np.ndarray,pad: np.ndarray):
        n = X.shape[0]
        X = np.r_[X,pad[:,:-1]]
        pad = np.reshape(pad,(pad[0].shape[0],1))
        X = np.c_[X,pad]
        return X


    def exploreFFSM(self,C):
        # newTempTrees = {}
        Q = []
        for X in C:
            S = []
            newTempTrees = {}
            for Y in C:
                if np.array_equal(X[:-1,:-1],Y[:-1,:-1]) and not np.array_equal(X[-1],Y[-1]):
                    joinedTree = self.joinCase3bFFSM(X,Y)
                    indexAddNode = np.where(joinedTree[-1] > 0)[0][0]
                    embedJoinedTree = np.array2string(joinedTree)
                    # S.append(joinedTree)
                    for i in self.tempTrees[np.array2string(X)].keys():
                        topo= []
                        for subGraph in self.tempTrees[np.array2string(X)][i]:
                            linkedNode = subGraph[indexAddNode,indexAddNode] # node is extended
                            for j in np.where(self.graphs[i][linkedNode] > 0)[0]: # get neighbor of linked node
                                if self.graphs[i][linkedNode,j] == joinedTree[-1,-1] and j not in subGraph.diagonal():
                                    pad = np.zeros((1,subGraph.shape[0]+1),dtype=int)
                                    pad[indexAddNode] = joinedTree[-1,-1]
                                    pad[-1] = j
                                    topo.append(self.extend(subGraph,pad))

                        if len(topo) > 0:
                            if embedJoinedTree not in newTempTrees:
                                newTempTrees[embedJoinedTree] = {}
                            newTempTrees[embedJoinedTree][i] = topo 
                    # if embedJoinedTree in newTempTrees and len(newTempTrees[embedJoinedTree].items()) > self.theta*len(self.graphs):
                        # S.append(joinedTree)
                            # print(newTempTrees)

            # frequent tree
            # newTempTrees = [{k: v} for k, v in newTempTrees.items() if len(v.items()) > self.theta*len(self.graphs)]
            temp = {}
            # S = []
            for k,v in newTempTrees.items():
                if len(v.items()) > self.theta*len(self.graphs):
                    temp[k] = v
            # print(self.tempTrees)
            self.tempTrees = temp
            # print(S)
            print(self.tempTrees)
            # Q = Q.extend(self.exploreFFSM(S))

    def freqEdges2matrix(self):
        matrices = []
        for edge in self.freqEdges.keys():
            matrices.append(
                np.array([[edge[0],edge[2]],[edge[2],edge[1]]])
            )
        return matrices
    

    def frequentGraph(self):
        graphDemo = np.array([
            [2,11,10,11],
            [11,1,0,11],
            [10,0,1,10],
            [11,11,10,1]
        ])

        graphDemo2 = np.array([
            [2,11,10,15],
            [11,1,0,15],
            [10,0,1,16],
            [15,15,16,3]
        ])
        # print(self.tempTrees)
        self.exploreFFSM(self.freqEdges2matrix())
        # print(self.extend(graphDemo,np.array([[0,0,0,0,1]])))
        # canonicalForm(graphDemo)
        return True
                




        






