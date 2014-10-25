import numpy
import scipy
import scipy.spatial
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl


#Globals
nNodes = 100
maxDistance = 0.15

#--------------------------------------------------------------------------------
def startMergeSort(X):


	#Start sort
	aux = numpy.copy(X)
	S = sort(X,aux,0,X.shape[0]-1)


	#Return
	return S


#--------------------------------------------------------------------------------
def sort(a,aux,lo,hi):


	#Recursive sort
	if (hi <= lo):
		return a
	mid = lo + (hi - lo) / 2;
	a = sort(a, aux, lo, mid);
	a = sort(a, aux, mid + 1, hi);
	a = merge(a, aux, lo, mid, hi);	
	
	#Return when done
	return a


#--------------------------------------------------------------------------------
def merge(a,aux,lo,mid,hi):


	#Copy subrange into aux
	for k in range(lo, hi+1):
		aux[k] = a[k]

	#Merge back to a
	i = lo
	j = mid + 1
	for k in range(lo,hi+1):
		if (i > mid):
			a[k] = aux[j]
			j = j + 1
		elif (j > hi):
			a[k] = aux[i]
			i = i + 1
		elif (aux[j] < aux[i]):
			a[k] = aux[j]
			j = j + 1
		else:
			a[k] = aux[i]
			i = i + 1

	#return
	return a


#--------------------------------------------------------------------------------
def createGraph():

	
	#Random Vertices
	G = {}
	V = numpy.random.random((nNodes,2))

	
	#Build KD Tree
	kdtree = scipy.spatial.KDTree(V,leafsize=10)


	#Loop through Vertices to add edges
	for i in range(V.shape[0]):
		
		#Find Niefhbors within maxDistance
		distances, neighbors = kdtree.query(V[i,:],nNodes)
		distances = distances[1:]
		neighbors = neighbors[1:]
		indices = numpy.where(distances < maxDistance)
		neighbors = neighbors[indices]
		distances = distances[indices]
		
		for j,neighbor in enumerate(neighbors):
			edge1 = {neighbor:distances[j]}
			edge2 = {i:distances[j]}

			if G.has_key(i):
				G[i].update(edge1)
			else:
				G[i] = edge1

			if G.has_key(neighbor):
				G[neighbor].update(edge2)
			else:
				G[neighbor] = edge2


	#Return V,G
	return V,G


#--------------------------------------------------------------------------------
def DFS(G,node,marked,edgeTo):


	#Recursive DFS
	if node in G:
		for v in G[node]:
			if (v not in marked):
				marked[v] = 1
				edgeTo[v] = node
				edgeTo,marked = DFS(G,v,marked,edgeTo)

	#Return
	return edgeTo,marked


#--------------------------------------------------------------------------------
def runDepthFirstSearch(G,startnode,endnode):


	#Init
	path = []
	edgeTo = {}
	marked = {}


	#DFS
	edgeTo,marked = DFS(G,startnode,marked,edgeTo)


	#Find Path
	curnode = endnode
	path.append(curnode)
	while (curnode != startnode):
		if (curnode in edgeTo):
			curnode = edgeTo[curnode]
			path.append(curnode)
		else:
			path = []
			break;


	#Return path
	return path


#--------------------------------------------------------------------------------
def computeMST(G):

	#Init
	MST = {}
	Q = []

	#Put all edges in a pq
	for v in G:
		for w in G[v]:
			Q.append((G[w][v],v,w))


	#Sort the list
	Q.sort(key=lambda tup: tup[0])


	#Remove Duplicates
	Q = Q[::2]


	#Loop through edges
	while (len(Q) > 0):
		
		#Run DFS on existing MST
		vw,v,w = Q[0]
		Q = Q[1:]
		edgeTo = {}
		marked = {}
		edgeTo,marked = DFS(MST,v,marked,edgeTo)
		
		#Add if not triangle is created
		if w not in edgeTo:
			edge1 = {w:vw}
			edge2 = {v:vw}

			if MST.has_key(v):
				MST[v].update(edge1)
			else:
				MST[v] = edge1

			if MST.has_key(w):
				MST[w].update(edge2)
			else:
				MST[w] = edge2


	#Return
	return MST



#--------------------------------------------------------------------------------
def dijkstra(G,startnode):

	#Init
	Q = []
	D = numpy.zeros((nNodes,1)) + numpy.inf
	P = {}

	#Start
	Q.append((0,0))
	D[startnode] = 0

	while (len(Q) > 0):
		v,vw = Q[0]
		Q = Q[1:]

		for w in G[v]:
			vw = G[v][w]
			if (D[w] > D[v] + vw):
				D[w] = D[v] + vw
				P[w] = v

				#Update or change
				index = [i for i,(key,value) in enumerate(Q) if key == w]
				if (len(index) == 0):
					Q.append((w,vw))
				else:
					Q[index[0]] = (w,vw)


	#Return
	return P

#--------------------------------------------------------------------------------
def runDijkstra(G,startnode,endnode):


	#Run Dijkstra
	path = []
	P = dijkstra(G,startnode)


	#Get path
	curnode = endnode
	path.append(curnode)
	while (curnode != startnode):
		if (curnode in P):
			curnode = P[curnode]
			path.append(curnode)
		else:
			path = []
			break

	print path
	#Return path
	return path

#--------------------------------------------------------------------------------
def plotGraph(G,V,DFSpath,DijkstraPath,MST):

	mpl.rcParams['toolbar'] = 'None'


	#Plot nodes
	for v in G:
		for w in G[v]:
			plt.plot([V[v,0],V[w,0]],[V[v,1],V[w,1]],c=(0,0,0),alpha=0.2,zorder=2)
			plt.scatter(V[v,0],V[v,1],c=(0,0,0),s=50,zorder=1)


	#Plot MST
	for v in MST:
		for w in MST[v]:
			plt.plot([V[v,0],V[w,0]],[V[v,1],V[w,1]],c=(1,0,0),alpha=1.0,zorder=2,linewidth=2)
			plt.scatter(V[v,0],V[v,1],c=(1,0,0),s=50,zorder=1)


	#Plot Dijkstra
	for i in range(len(DijkstraPath)-1):
		v = DijkstraPath[i]
		w = DijkstraPath[i+1]
		plt.plot([V[v,0],V[w,0]],[V[v,1],V[w,1]],c=(0,0,1),alpha=1.0,zorder=2,linewidth=2)
		plt.scatter(V[v,0],V[v,1],c=(0,0,1),s=50,zorder=3)
		plt.scatter(V[w,0],V[w,1],c=(0,0,1),s=50,zorder=3)


	#Show
	plt.show()


#--------------------------------------------------------------------------------
if __name__ == '__main__':


	#Quicksort
	X = numpy.random.random((10,1))
	S = startMergeSort(X)


	#Create Graph
	V,G = createGraph()


	#DFS
	DFSpath = runDepthFirstSearch(G,0,24)


	#Dijkstra
	DijkstraPath = runDijkstra(G,0,24)


	#MST
	MST = computeMST(G)


	#Plot Graph
	plotGraph(G,V,DFSpath,DijkstraPath,MST)