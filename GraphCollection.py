import numpy as np
from typing import List
import operator
from ExpansionGraph import ExpansionGraph
from algorithm import string2matrix,extendOneNode,canonicalForm
import datetime

class GraphCollection():
    tempTrees = {}
    def __init__(self,graphs_,theta_):
        self.graphs = graphs_
        self.theta = theta_ * len(graphs_)
        self.freqEdges = self.getFrequentEdges(self.graphs,self.theta)
        self.initTempTree()
        # print("theta",self.theta)
        print("freqEdges")#,self.freqEdges)
        for edge in self.freqEdges:
            print(edge)
        print("end freq edges")
        # exit()
        # print("init temp Tree",self.tempTrees)

    def getFrequentEdges(self,graphs : List[np.ndarray],theta):
        frequentEdges = {}
        for idGraph, graph in enumerate(graphs):
            edgesSet = set()
            for i in range(graph.shape[0]):
                indicesEdge = np.where(graph[i,i+1:] > 0)
                for des in indicesEdge[0]:
                    labelNodes = [graph[i,i],graph[i+des+1,i+des+1]]
                    labelNodes = sorted(labelNodes)#,reverse=True)
                    encodeEdges = (labelNodes[0],labelNodes[1],graph[i,i+des+1])
                    if encodeEdges not in edgesSet:
                        if encodeEdges not in frequentEdges:
                            frequentEdges[encodeEdges] = {}
                            frequentEdges[encodeEdges]['freq'] = 1
                            frequentEdges[encodeEdges]['edges'] = {}
                        else:
                            frequentEdges[encodeEdges]['freq'] += 1

                        edgesSet.add(encodeEdges)
                        frequentEdges[encodeEdges]['edges'][idGraph] = [(i,i+des+1) if graph[i,i] == labelNodes[0] else (des + i + 1,i)]
                    else:
                        frequentEdges[encodeEdges]['edges'][idGraph].append((i,i + des + 1) if graph[i,i] == labelNodes[0] else (des + i + 1,i))

        frequents = {}
        for k,v in frequentEdges.items():
            if v['freq'] >= theta:
                frequents[k] = v['edges']
        return frequents


    def initTempTree(self):
        for edge,matches in self.freqEdges.items():
            # print("edge",edge)
            # matrix = self.canonicalForm(np.array([[edge[0],edge[2]],[edge[2],edge[1]]]))
            matrix = np.array([[edge[0],edge[2]],[edge[2],edge[1]]])
            # print("matrix",matrix)
            # matrix = canonicalForm(matrix,code=False)
            # print("after matrix",matrix)
            encodeEdge = np.array2string(matrix)
            self.tempTrees[encodeEdge] = {}
            for i in matches.keys():# range(len(self.graphs)):
                topo = []
                # print("match i ",matches[i])
                for e in matches[i]:
                    m = np.array([[e[0],edge[2]],[edge[2],e[1]]])
                    # print("i",i,"m",m)
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
                "code" : self.encodeGraph(start)
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
        return canonical["tree"]

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

    def extendCase3bFFSM(self,graphX: np.ndarray):
        n = graphX.shape[0]
        pad = np.zeros((1,n),dtype=int)
        pad[0] = graphX[-1]
        pad[0,n-1] = 0
        graphX = np.r_[graphX,pad]
        pad = np.concatenate([pad[0],np.array([graphX[n-1,n-1]])])
        pad = np.reshape(pad,(n+1,1))
        graphX = np.c_[graphX,pad]
        return graphX

    def extend(self,X: np.ndarray,pad: np.ndarray):
        n = X.shape[0]
        X = np.r_[X,pad[:,:-1]]
        pad = np.reshape(pad,(pad[0].shape[0],1))
        X = np.c_[X,pad]
        return X

    def extendFFSM(self,X: np.ndarray,Y: np.ndarray):
        pos =  np.where(Y.diagonal() == X[-1,-1])[0]
        n = X.shape[0]
        extensions = []
        for p in pos:
            if p<n-1:
                indices = np.where(Y[p,p+1:] > 0)[0]
                # indices = np.where(Y[p] > 0)[0]
                for i in indices:
                    # if i!=p:
                    pad = np.zeros((1,n + 1),dtype=int)
                        # pad[0,-1] = Y[i,i]
                        # pad[0,-2] = Y[p,i]
                    pad[0,-1] = Y[p+i+1,p+i+1]
                    pad[0,-2] = Y[p,p+i+1]
                    extensions.append(self.extend(X,pad))
        return extensions

    def hasNoExternalAssEdge(self,tree : np.ndarray,embeddings : dict):
        # return True
        numEmb = 0
        for k in embeddings.keys():
            numEmb += len(embeddings[k])
        print("numEmbedding",numEmb)
        # print("embeddings",embeddings)
        for i in range(tree.shape[0]):
            externalEdges = {}
            # print("iExternal",i)
            for idGraph in embeddings.keys():
                for subGraph in embeddings[idGraph]:
                    curNode = subGraph[i,i]
                    indices = np.where(self.graphs[idGraph][curNode] > 0)[0] # neighbor of curNode
                    indexNodes = subGraph.diagonal() #[subGraph[j,j] for j in np.where(subGraph[i] > 0)[0]]    # in embedding
                    nodes = list(set(indices) - set(indexNodes)) # node not in subgraph of node curNode
                    # print("External-idGraph-subGraph",idGraph,subGraph)
                    # print("incides External",indices)
                    # print("indexNodes Ex",indexNodes)
                    # print("nodes External",nodes)
                    # print("curNode",curNode)
                    edges = set()
                    for node in nodes:
                        if node != curNode:
                            # print("edgeToadd",(self.graphs[idGraph][curNode,curNode],self.graphs[idGraph][node,node],self.graphs[idGraph][curNode,node]))
                            edges.add((self.graphs[idGraph][curNode,curNode],self.graphs[idGraph][node,node],self.graphs[idGraph][curNode,node]))
                    # print("edgesSet",edges)
                    # exit(0)
                    for edge in edges:
                        try:
                            externalEdges[edge]  += 1
                        except:
                            externalEdges[edge]  = 1
            for k,v in externalEdges.items():
                # print("externalEdges key",k)
                # print("externalEdges value",v)
                if v == numEmb:
                    # exit(0)
                    print("hasNoExtenalAssEdge")
                    return False
        # exit(0)
        print("hasAss")
        # print("treeAss",tree)
        # print("AssEmbeding",embeddings)
        return True





    def exploreFFSM(self,C):
        # print("C\n",C)
        if len(C) == 0:
            return
        # newTempTrees = {}
        Q = []
        for X in C:
            S = []
            newTempTrees = {}
            for Y in C:
                candidates = []
                print("X\n",X)
                print("Y\n",Y)
                if np.array_equal(X[:-1,:-1],Y[:-1,:-1]) and not np.array_equal(X[-1],Y[-1]):
                    candidates.append(self.joinCase3bFFSM(X,Y))
                    # print("join",candidates[0],"x",X,"y",Y)
                extensions = self.extendFFSM(X,Y)
                if len(extensions) > 0:
                    print("extension X\n",X,"\nY",Y )
                    candidates.extend(extensions)
                # print("X",X)
                # print("Y",Y)
                # print("candidates",candidates)
                for joinedTree in candidates:
                    # joinedTree = self.joinCase3bFFSM(X,Y)
                    # print("joinedTree",joinedTree)
                    indexAddNode = np.where(joinedTree[-1] > 0)[0][0]
                    # print("index add node",indexAddNode)
                    embedJoinedTree = np.array2string(joinedTree)
                    for i in self.tempTrees[np.array2string(X)].keys():
                        topo= []
                        for subGraph in self.tempTrees[np.array2string(X)][i]:
                            # print("subgraph",subGraph)
                            linkedNode = subGraph[indexAddNode,indexAddNode] # node is extended
                            # print("indexNode",indexAddNode,"linkNoded",linkedNode)
                            # print("joinedTree",joinedTree)
                            for j in np.where(self.graphs[i][linkedNode] > 0)[0]: # get neighbor of linked node
                                if self.graphs[i][linkedNode,j] == joinedTree[-1,indexAddNode] and j not in subGraph.diagonal():
                                    pad = np.zeros((1,subGraph.shape[0]+1),dtype=int)
                                    pad[0,indexAddNode] = joinedTree[-1,indexAddNode]
                                    pad[0,-1] = j
                                    topo.append(self.extend(subGraph,pad))
                        # print("i",i,"topo",topo)
                        if len(topo) > 0:
                            # print("embed",embedJoinedTree)
                            if embedJoinedTree not in newTempTrees:
                                newTempTrees[embedJoinedTree] = {}
                                S.append(joinedTree)
                            newTempTrees[embedJoinedTree][i] = topo
 
            
            temp = {}
            nextCans = []
            i = 0
            print("new tempTrees",newTempTrees)
            # print("theta",self.theta," len",len(self.graphs))
            print("threshold",self.theta)
            for k,v in newTempTrees.items():
                # print("len items",len(v.items()))
                if len(v.items()) > self.theta:
                    temp[k] = v
                    nextCans.append(S[i])
                    # print("S i",S[i])
                i += 1
            if len(temp.items()) == 0:
                return
            self.tempTrees = temp
            # print("new tempTrees",self.tempTrees)
            # print("next candidate",nextCans)
            # Q = Q.extend(self.exploreFFSM(nextCans))
            self.exploreFFSM(nextCans)
            return 0

    def freqEdges2matrix(self):
        matrices = []
        for edge in self.freqEdges.keys():
            matrices.append(
                # self.canonicalForm(np.array([[edge[0],edge[2]],[edge[2],edge[1]]]))
                np.array([[edge[0],edge[2]],[edge[2],edge[1]]])
            )
        return matrices

    def frequentTrees(self,X,restCan,C):
        S = []
        newTempTrees = {}
        print("timestamp frequent trees",datetime.datetime.now())
        # print("X in",X)
        # print("C X in",C)
        for Y in [string2matrix(k) for k in restCan]:
            candidates = []
            # print("X",X)
            # print("Y",Y)
            if np.array_equal(X[:-1,:-1],Y[:-1,:-1]) and not np.array_equal(X[-1],Y[-1]):
                candidates.append(self.joinCase3bFFSM(X.copy(),Y.copy()))
                candidates.append(self.extendCase3bFFSM(X.copy()))
            # print("XX",X)
            # print("YY",Y)
            # print("can join",candidates)
            

            # if X.shape[0] < 3:
                # extensions = extendOneNode(X,Y) #self.extendFFSM(X,Y)
                # if len(extensions) > 0:
                    # candidates.extend(extensions)
            # print("candidates",len(candidates))
            extensions = self.extendFFSM(X,Y)
            if len(extensions) > 0:
                candidates.extend(extensions)
            
            for joinedTree in candidates:
                # print("preJoinedTreeX",X)
                # print("preJoinedTReeY",Y)
                # print("joinedTree",joinedTree)
                indexAddNode = np.where(joinedTree[-1] > 0)[0][0]
                # print("indexAddNode",np.where(joinedTree[-1] > 0)[0])
                embedJoinedTree = np.array2string(joinedTree)
                for i in C[np.array2string(X)].keys():
                    topo= []
                    for subGraph in C[np.array2string(X)][i]:
                        linkedNode = subGraph[indexAddNode,indexAddNode] # node is extended
                        for j in np.where(self.graphs[i][linkedNode] > 0)[0]: # get neighbor of linked node
                            if self.graphs[i][linkedNode,j] == joinedTree[-1,indexAddNode] and j not in subGraph.diagonal():
                                pad = np.zeros((1,subGraph.shape[0]+1),dtype=int)
                                pad[0,indexAddNode] = joinedTree[-1,indexAddNode]
                                pad[0,-1] = j
                                topo.append(self.extend(subGraph,pad))
                    if len(topo) > 0:
                        if embedJoinedTree not in newTempTrees:
                            newTempTrees[embedJoinedTree] = {}
                            S.append(joinedTree)
                        newTempTrees[embedJoinedTree][i] = topo
 
            
        temp = {}
        
        for k,v in newTempTrees.items():
            if len(v.items()) >= self.theta:
                # temp[np.array2string(canonicalForm(string2matrix(k),code=False))] = v
                temp[k] = v
        return temp

    

    def exploreGenericTree(self,C : dict,R : dict):
        # print("C in\n",C)
        # print("Temptrees len", len(tempTrees.items()))
        print("timestamp",datetime.datetime.now())
        Q = {}
        # print("reprGroup",C)
        for idCan,can in enumerate(C.items()):
            # print("canCItems",C.keys())
            # reprGroup,group = can
            X = string2matrix(can[0])
            # print("X",X)
            # Y = [string2matrix(k) for k,v in C.items()]
            # print("Y",Y)
            print('begin frequent trees')
            # print("X ",X)
            # print("C copy",C.copy())
            nextCan = {k:C[k] for k in list(C.keys())[idCan+1:]}
            print("setCan",len(C.keys()))
            # print("canCur",can[0])
            print("nextCan",len(nextCan.keys()))
            # if idCan == 3:
                # exit(0)
            # continue

            S = self.frequentTrees(X,nextCan.keys(),C.copy())
            encodeX = np.array2string(X)
            # print("S freq",S)
            # S - R
            

            canonicalS = {canonicalForm(string2matrix(k)):k for k in S.keys()}
            # print("SSS before",S.keys())
            # print("RRBefore",R.keys())
            for kR in R.keys():
                canonicalKR = canonicalForm(string2matrix(kR))
                if canonicalKR in canonicalS.keys():
                    if canonicalS[canonicalKR] in S:
                        del S[canonicalS[canonicalKR]]
                    

            # print("tempTree after",self.tempTrees)
            # if len(S) != 0:
            print("afterFrequentTRee size S: ",len(S.keys()))
            U,V = self.exploreGenericTree(S,R.copy())
            # print("S empty",S)
            # print("R empty",R)
            # print("U empty",U)
            # print("V empty",V)
            # print("X ok ex",X)
            # print("encode X",encodeX)
            for k,v in U.items():
                Q[k] = v

            R[encodeX] = C[encodeX]
            for k,v in V.items():
                R[k] = v
            if self.hasNoExternalAssEdge(X,C[encodeX]):
                print("ok expansion",X)
                eg = ExpansionGraph(
                    X.copy(),
                    C[encodeX],
                    self.graphs,self.freqEdges,
                    self.theta
                )
                expansionX = eg.expand()
                print("expansion X",expansionX)
                for k,v in expansionX.items():
                    Q[k] = v
                # exit(0)

            
            # del eg
        # print("Q",len(Q.items))
        # print("R",R)
        return Q,R

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
        # self.exploreFFSM(self.freqEdges2matrix())
        
        # self.exploreGenericTree(self.freqEdges2matrix(),[],self.tempTrees)

        results,S = self.exploreGenericTree(self.tempTrees,{})
        # print("final results",results[0])
        # try:
        print("S-final",S)
        print("final result",results)
        if len(results.items()) == 0:
            return []
        numNodeGraphs = np.array([string2matrix(k).shape[0] for k,v in results.items()])
        # print("numNodeGraphs",numNodeGraphs)
        indicesFreq = np.where(numNodeGraphs == numNodeGraphs.max())[0]
        # return results[0]
        # print("indicesFreq",indicesFreq)
        return [string2matrix(list(results.keys())[i]) for i in indicesFreq]
        # except:
            # return []
        # freqGraphs = [for k in freqGraphs]
        # for k,v in results:
            
        
        # print("result extend")
        # print(extendOneNode(graphDemo,graphDemo2))

        # print(np.where(graphDemo == 0))
        # eg = ExpansionGraph(self.tempTrees)
        # print("ca",self.canonicalForm(graphDemo))
        # print(self.extendFFSM(graphDemo,graphDemo2))
        # print(self.joinCase3bFFSM(graphDemo,graphDemo2))
        # print(self.extend(graphDemo,np.array([[0,0,0,0,1]])))
        # canonicalForm(graphDemo)
        return True
                




        






