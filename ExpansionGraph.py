import numpy as np
from utils import plotGraph
from algorithm import canonicalForm,string2matrix,checkConnected


class ExpansionGraph():
    def __init__(self,matrixAdj_ : np.ndarray,topoGraphs_,graphs_,freqEdges_,theta_):
        self.graphs = graphs_
        self.theta = theta_
        self.matrixAdj = matrixAdj_
        self.subGraphs = topoGraphs_
        self.spaceGraphs = {np.array2string(matrixAdj_):topoGraphs_}
        self.canEdges = []
        self.associativeEdges = []
        self.setCandidateEdges(freqEdges_)
        self.setAssociativeEdge()
        # print("topoGraphs",topoGraphs_)
        # print("ConFreqEdges",freqEdges_.keys())
        # print("canEdges: ", self.canEdges)


    def setCandidateEdges(self,freqEdges):
        mapEdges = {(e[0],e[1]):e[2] for e in freqEdges.keys()}
        indices = np.where(self.matrixAdj == 0)
        canEdges = []
        for i in range(len(indices[0])):
            iR = indices[0][i]
            iC = indices[1][i]
            k = tuple(sorted((self.matrixAdj[iR,iR],self.matrixAdj[iC,iC])))
            if k in mapEdges and iR <= iC:
                canEdges.append((iR,iC,mapEdges[k]))   
        self.canEdges = canEdges

    def setAssociativeEdge(self):
        for edge in self.canEdges:
            isAssociative = True
            for graph in self.subGraphs.keys():
                for sub in self.subGraphs[graph]:
                    if self.graphs[graph][sub[edge[0]],sub[edge[1]]] != edge[2]:
                        isAssociative = False
                        break
                if not isAssociative:
                    break
            if isAssociative:
                self.associativeEdges.append(edge)


    def joinEdge(self,graph: np.ndarray,edge):
        graph[edge[0],edge[1]] = edge[2]
        graph[edge[1],edge[0]] = edge[2]
        return graph
    
    countSearch = 0
    def searchGraph(self,graph,canEdges):
        print("lenCanEdges",self.countSearch)
        self.countSearch = self.countSearch + 1
        if len(canEdges) == 0:
            return
        newTempGrapsearchGraphhs = {}
        encodeGraph = np.array2string(graph)

        # bottom-up pruning
        codeFullGraph = self.mergeToGraph(graph,canEdges)
        if codeFullGraph in self.spaceGraphs:
            if len(self.spaceGraphs[codeFullGraph].items()) >= self.theta and not np.array_equal(graph,string2matrix(codeFullGraph)):
                print("originGraph\n",graph)
                print("bottom-up aval\n",codeFullGraph)
                # if np.array_equal(graph,string2matrix(codeFullGraph)):
                    # print("equalArray")
                return {
                    codeFullGraph : self.spaceGraphs[codeFullGraph]
                }


        #end bottom-up  
        for i,edge in enumerate(canEdges):
            print("Jedge",len(canEdges))
            canGraph = self.joinEdge(graph.copy(),edge)
            # print("canGraph",canGraph)
            embedCanGraph = np.array2string(canGraph)
            topo = {}
            for j in self.spaceGraphs[encodeGraph].keys():
                # print("keySpace",self.spaceGraphs[encodeGraph][j])
                # for subGraph in self.spaceGraphs[encodeGraph][j]:
                    # print("sub",j,subGraph)
                subGraph = self.spaceGraphs[encodeGraph][j]
                # print("jsub",j,subGraph)
                sNode = subGraph[edge[0],edge[0]] # id source node
                dNode = subGraph[edge[1],edge[1]] # id destination node
                if self.graphs[j][sNode,dNode] == edge[2]:
                    # subGraph[sNode,dNode] = edge[2]
                    # subGraph[dNode,sNode] = edge[2]
                    connectedSub = subGraph.copy()
                    # print("connectedSub",connectedSub)
                    connectedSub[edge[0],edge[1]] = edge[2]
                    connectedSub[edge[1],edge[0]] = edge[2]
                    topo[j] = connectedSub
            # print("PREembedCanGraph",embedCanGraph)
            # print("EncodedGraph",self.spaceGraphs[encodeGraph])
            # print("PRETopo",len(topo),self.theta)
            if len(topo.items()) >= self.theta:
                if embedCanGraph not in self.spaceGraphs:
                    # print("embedCanGraph",embedCanGraph)
                    self.spaceGraphs[embedCanGraph] = {}
                self.spaceGraphs[embedCanGraph] = topo
            if embedCanGraph in self.spaceGraphs:
                self.searchGraph(canGraph,canEdges[i+1:]) 
            # else:
                # self.searchGraph(graph,canEdges[i+1:])
        # print("returnHere")
        return

    def mergeToGraph(self,graph,canEdges):
        encodeGraph = np.array2string(graph)
        fullGraph = graph.copy()
        for i,edge in enumerate(canEdges):
            fullGraph = self.joinEdge(fullGraph,edge)

        codeFullGraph = np.array2string(fullGraph)
        for idGraph in self.spaceGraphs[encodeGraph].keys():
            # topo = []
            # for sub in self.spaceGraphs[encodeGraph][idGraph]:
            subGraph = self.spaceGraphs[encodeGraph][idGraph].copy()#sub.copy()
            # print("subGraph",subGraph)
            flag = True
            for i,edge in enumerate(canEdges):
                if  self.graphs[idGraph][subGraph[edge[0],edge[0]],subGraph[edge[1],edge[1]]] != edge[2]:
                    flag = False
                    break
                else:
                    subGraph[edge[0],edge[1]] = edge[2]
                    subGraph[edge[1],edge[0]] = edge[2]

            if flag:
                # topo.append(subGraph)
            # if len(topo) > 0:
                if codeFullGraph not in self.spaceGraphs:
                    self.spaceGraphs[codeFullGraph] = {}
                self.spaceGraphs[codeFullGraph][idGraph] = subGraph #np.array(topo)
        return codeFullGraph

    def checkLethal(self):
        initialTree = self.matrixAdj.copy()
        for asEdge in self.associativeEdges:
            self.matrixAdj = self.joinEdge(self.matrixAdj,asEdge)
        

        if canonicalForm(initialTree)['code'] != canonicalForm(self.matrixAdj)['code']:
            return True
        
        self.mergeToGraph(initialTree,self.associativeEdges)
        return False 

    def eliminateAssEdges(self):
        newCans = []
        for edge in self.canEdges:
            if edge not in self.associativeEdges:
                newCans.append(edge)
        self.canEdges = newCans

    def expand(self):
        # if self.checkLethal():
            # return {}
        # for asEdge in self.associativeEdges:
            # self.matrixAdj = self.joinEdge(self.matrixAdj,asEdge)
        
        # self.eliminateAssEdges()
        self.searchGraph(self.matrixAdj,self.canEdges)
        print("end search Graph")
        frequents = {}
        for k,v in self.spaceGraphs.items():
            if len(v.items()) >= self.theta:
                frequents[k] = v
        
        eqGraphClasses = {}
        # canTree = canonicalForm(self.matrixAdj)['code']    
        if len(frequents.items()) > 0:
            maxCan = ''
            maxKey = ''
            for k,v in frequents.items():
                # print("expandKey: ",k)
                subGraph = string2matrix(k)
                if checkConnected(subGraph) and np.where(subGraph > 0)[0].shape[0] > subGraph.shape[0]:
                    # print("subCheck",subGraph)
                    can = canonicalForm(subGraph)
                    # print("canCode",can['code'])
                    if can['code'] > maxCan:
                        maxKey = k
                        maxCan = can['code'] 
                # cam = canonicalForm(subGraph)
                # if cam['code'] == canTree:
                # eqGraphClasses[k] = v
            if maxKey != '':
                eqGraphClasses[maxKey] = frequents[k]
        print("eqGraphClass",eqGraphClasses)

        return eqGraphClasses
        



