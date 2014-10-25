import numpy
import scipy
import scipy.spatial
import os
import sys
import matplotlib.pyplot as plt 
import matplotlib as mpl


nNodes = 50 
maxDistance = 0.35
P = {}


#------------------------------------------------------------------------
def createGraph():


	#Empty Graph G
	G = {}


	#Create Vertices
	V = numpy.random.random((nNodes,2))


	#Compute distances
	kdtree = scipy.spatial.KDTree(V,leafsize=10)


	#Loop Through Nodes and add their edges
	for i in range(nNodes):
		distance,neighbors = kdtree.query(V[i,:],nNodes)
		distance = distance[1:]
		neighbors = neighbors[1:]
		indices = numpy.where(distance < maxDistance)[0]

		for index in indices:
			v = i
			w = neighbors[index]
			d = distance[index]
			edge1 = {w:d}
			edge2 = {v:d}

			if G.has_key(v):
				G[v].update(edge1)
			else:
				G[v] = edge1

			if G.has_key(w):
				G[w].update(edge2)
			else:
				G[w] = edge2

	#Return
	return G,V


#------------------------------------------------------------------------
def DFS(G,curnode,marked,distTo,P):

	#Recursive DFS
	if (curnode in G):
		for v in G[curnode]:
			if (marked[v] == 0):
				marked[v] = 1
				P[v] = curnode
				curnode = v
				DFS(G,curnode,marked,distTo,P)
		 

#------------------------------------------------------------------------
def runDFS(G,startnode,endnode):


	P = {}
	distTo = numpy.zeros((nNodes,1))
	marked = numpy.zeros((nNodes,1))
	marked[startnode] = 1


	#Run DFS
	DFS(G,startnode,marked,distTo,P)

	
	#Find Path
	path = [];
	curnode = endnode
	path.append(endnode)
	while (curnode != startnode):
		if curnode in P:
			curnode = P[curnode]
			path.append(curnode)
		else:
			path = []
			break

	#Return path
	return path


#------------------------------------------------------------------------
def plotGraph(G,V,DFSPath,MST,DPath):


	#Remove sidebar
	mpl.rcParams['toolbar'] = 'None'



	#Loop through nodes
	for i in range(nNodes):
		plt.scatter(V[i,0],V[i,1],50,c=[0,0,0],alpha=1,zorder=2)
		if i in G:
			for g in G[i]:
				plt.plot([V[i,0],V[g,0]],[V[i,1],V[g,1]],'k',alpha=0.1,zorder=1)



	#Plot DFS path
	for i in range(len(DFSPath)-1):
		v = DFSPath[i]
		w = DFSPath[i+1]
		plt.scatter(V[v,0],V[v,1],50,c=[1,0,0],alpha=1,zorder=2)
		plt.scatter(V[w,0],V[w,1],50,c=[1,0,0],alpha=1,zorder=2)
		plt.plot([V[v,0],V[w,0]],[V[v,1],V[w,1]],'r',alpha=1,zorder=1,linewidth=3)



	#Loop through nodes
	for i in range(nNodes):
		if i in MST:
			for g in MST[i]:
				plt.plot([V[i,0],V[g,0]],[V[i,1],V[g,1]],'c',alpha=1,zorder=1,linewidth=3)



	#Plot DFS path
	for i in range(len(DPath)-1):
		v = DPath[i]
		w = DPath[i+1]
		plt.scatter(V[v,0],V[v,1],50,c=[1,0,0],alpha=1,zorder=2)
		plt.scatter(V[w,0],V[w,1],50,c=[1,0,0],alpha=1,zorder=2)
		plt.plot([V[v,0],V[w,0]],[V[v,1],V[w,1]],'g',alpha=1,zorder=1,linewidth=3)



	#Plot
	plt.show()


#------------------------------------------------------------------------
def dijkstra(G,startnode,endnode):

	#Init
	path = []
	P = {}
	D = numpy.zeros((nNodes,1)) + numpy.inf
	Q = []
	D[startnode] = 0

	#Add startnode
	Q.append((startnode,0))


	#Run Through
	while (len(Q) > 0):
		
		#Pop
		v,d = Q[0]
		Q = Q[1:]

		#Loop
		for w in G[v]:
			vw = G[v][w]
			if (D[w] > D[v] + vw):
				D[w] = D[v] + vw
				P[w] = v

				#Check whether w is in Q
				index = [i for i,(key,value) in enumerate(Q) if key == w]
				if (len(index) == 0):
					Q.append((w,vw))
				else:
					Q[index[0]] = (w,vw)



	curnode = endnode
	path.append(curnode)
	while (curnode != startnode):
		curnode = P[curnode]
		path.append(curnode)


	#Return path
	return path


#------------------------------------------------------------------------
def computeMST(G):


	#Create Empty Graph
	MST = {}
	Q = []


	#Create Priority Queue and sort
	for g in G:
		for v in G[g]:
			Q.append((G[g][v],g,v))
	Q.sort(key=lambda tup: tup[0])
	Q = Q[::2]


	#Pop Edges
	while (len(Q) > 0):
		d,v,w = Q[0]
		Q = Q[1:]

		#Can I already go from node1 to node2?
		DFSpath = runDFS(MST,v,w)
		
		#If Not, add to MST
		if (len(DFSpath) == 0):
			edge1 = {w:d}
			edge2 = {v:d}
			if MST.has_key(v):
				MST[v].update(edge1)
			else:
				MST[v] = edge1
			if MST.has_key(w):
				MST[w].update(edge2)
			else:
				MST[w] = edge2


	#Return MST
	return MST


#------------------------------------------------------------------------
if __name__ == '__main__':

	#Create Graph
	G,V = createGraph()

	#DFS
	DFSpath = runDFS(G,0,5)


	#Dijkstra
	Dpath = dijkstra(G,0,5)


	#MST
	MST = computeMST(G)


	#Plot Graph
	plotGraph(G,V,DFSpath,MST,Dpath)