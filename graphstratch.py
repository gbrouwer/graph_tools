import numpy
import scipy
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import spatial


#Globals
nNodes = 50
maxDistance = 0.35


#-----------------------------------------------------------------------------
def createGraph(nNodes,maxDistance):


	#Create Graph Vertices
	V = numpy.random.random((nNodes,2))


	#Create KD neighbor tree
	kdtree = spatial.KDTree(V,leafsize=10)


	#Create Edges
	G = {}
	for i in range(nNodes):

		#Find Neighbors within Range
		distance,neighbors = kdtree.query(V[i,:],nNodes)
		distance = distance[1:]
		neighbors = neighbors[1:]
		indices = numpy.where(distance < maxDistance)[0]
		
		#Add to Graph
		for index in indices:
			v = i
			w = neighbors[index]
			d = distance[index]
			
			edge1 = {w:d}
			edge2 = {v:d}

			#Direction 1
			if G.has_key(v):
				G[v].update(edge1)
			else:
				G[v] = edge1

			#Direction 2
			if G.has_key(w):
				G[w].update(edge2)
			else:
				G[w] = edge2

	return G,V


#-----------------------------------------------------------------------------
def dijkstra(G,startnode):

	#Init Dijkstra
	Q = []
	P = {}
	D = numpy.zeros((nNodes)) + numpy.inf
	D[startnode] = 0

	
	#Priority Queue
	Q.append((startnode,0))



	while (len(Q) > 0):

		#Pop (grab first element)
		v,d = Q[0]
		Q = Q[1:]

		for w in G[v]:
			vw = G[v][w]
			if (D[w] > D[v] + vw):
				D[w] = D[v] + vw
				P[w] = v

				#If w doesn't exist in Q add, otherwise update
				ew = (w,D[w])

				hit = [i for i,(key,value) in enumerate(Q) if key == w]

				if (len(hit) > 0):
					Q[hit[0]] = ew
				else:
					Q.append(ew)

	
	return P,D


#-----------------------------------------------------------------------------
def rundijkstra(G,startnode,endnode):

	#Run Algorithm
	P,D = dijkstra(G,startnode)
	path = []


	#Find Path
	curnode = endnode
	path.append(curnode)
	while (curnode != startnode):
		curnode = P[curnode]
		path.append(curnode)

	#Return
	return path


#-----------------------------------------------------------------------------
def plotGraph(G,X,P):


	#Plot Nodes
	mpl.rcParams['toolbar'] = 'None'
	plt.scatter(X[:,0], X[:,1], 100,c='k',alpha=1.00, edgecolors='none',zorder = 1)
	for g in G:
		for v in G[g]:
			g = int(g)
			v = int(v)
			line = plt.plot([X[g][0],X[v][0]],[X[g][1],X[v][1]],'k',alpha=0.1, zorder =0)
			plt.setp(line,linewidth=1)

	for i in range(len(P)-1):
		g = P[i]
		v = P[i+1]
		print g,v
		line = plt.plot([X[g][0],X[v][0]],[X[g][1],X[v][1]],'r',alpha=1, zorder =0)
		plt.setp(line,linewidth=4)
		plt.scatter(X[g,0], X[g,1], 100,c='r',alpha=1.00, edgecolors='none',zorder = 1)
		plt.scatter(X[v,0], X[v,1], 100,c='r',alpha=1.00, edgecolors='none',zorder = 1)		

	plt.show()



#-----------------------------------------------------------------------------
if __name__ == '__main__':



	#Create Graph
	G,V = createGraph(nNodes,maxDistance)


	#Dijkstra
	P = rundijkstra(G,0,24)


	#Plot Graph
	plotGraph(G,V,P)