import numpy
import scipy
import scipy.spatial
import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import priorityqueue



#------------------------------------------------------------------
class Graph:


	#--------------------------------------------------------------
	def __init__(self):

		#Init
		self.G = {}
		self.MST = {}
		self.hasMST = 0
		self.P = {}
		self.vertices = []
		self.pagerank = []
		self.colorvalues = []
		self.nNodes = 0
		self.path = []
		self.marked = []
		self.edgeTo = {}
		self.dist = []
		self.C = []
		self.identifiers = []



	#--------------------------------------------------------------
	def load(self,filename):

		#Init
		self.G = {}
		self.MST = {}
		self.vertices = []
		self.path = []
		self.marked = []
		self.edgeTo = {}
		self.pagerank = []
		self.colorvalues = []
		self.identifiers = []


		#Read from file
		with open(filename,'r') as f:
			self.nNodes = int(f.readline().rstrip())
			self.vertices = numpy.zeros((self.nNodes,2))
			for i in range(self.nNodes):
				line = f.readline().rstrip()
				elements = line.split(',')
				self.identifiers.append(elements[0])
				self.vertices[i,0] = float(elements[1])
				self.vertices[i,1] = float(elements[2])
				print self.vertices[i,:]
			for line in f:
				elements = line.rstrip().split(',')
				left = self.identifiers.index(elements[0])
				right = self.identifiers.index(elements[1])
				edge1 = {right:float(elements[2])}
				edge2 = {left:float(elements[2])}

				if self.G.has_key(left):
					self.G[left].update(edge1)
				else:
					self.G[left] = edge1

				if self.G.has_key(right):
					self.G[right].update(edge2)
				else:
					self.G[right] = edge2


		#Create Uniform Color Matrix
		self.colorvalues = numpy.zeros((self.nNodes,3)) + 0.5


	#--------------------------------------------------------------
	def create(self,nNodes,maxDistance):


		#Create Vertices
		self.nNodes = nNodes
		self.vertices = numpy.random.random((self.nNodes,2))


		#KD Distance Tree
		kdtree = scipy.spatial.KDTree(self.vertices,leafsize=10)


		#Create Uniform Color Matrix
		self.colorvalues = numpy.zeros((self.nNodes,3)) + 0.5


		#Create Graph
		self.G = {}
		self.MST = {}
		for i in range(nNodes):
			node = self.vertices[i,:]

			#Get neighbors and loop through them
			distances,neighbors = kdtree.query(node,nNodes)
			for j,distance in enumerate(distances):
				if (distance < maxDistance and distance > 0):

					#Create Edge	
					edge = {neighbors[j]:distance}

					#Add Edge, or update if it already exists
					if self.G.has_key(i):
						self.G[i].update(edge)
					else:
						self.G[i] = edge



	#--------------------------------------------------------------
	def show(self,graphtype):


		#Draw nodes and edges
		X = self.vertices
		if (len(graph.pagerank) > 0 and graphtype != 'centrality'):
			self.colorvalues[:,0] = graph.pagerank[:,0]
			self.colorvalues[:,1] = 0.0
			self.colorvalues[:,2] = graph.pagerank[:,0] * 0.5
		else:
			self.colorvalues[:,0] = 0
			self.colorvalues[:,1] = 0
			self.colorvalues[:,2] = 0


		#Draw Centrality
		if (len(self.C) > 0 and graphtype == 'centrality'):
			self.C = self.C - numpy.min(self.C)
			self.C = self.C  / numpy.max(self.C)
			self.colorvalues[:,0] = self.C[:,0]
			self.colorvalues[:,1] = 0.0
			self.colorvalues[:,2] = self.C[:,0] * 0.5


		#Plot Nodes
		mpl.rcParams['toolbar'] = 'None'
		for g in self.G:
			for v in self.G[g]:
				g = int(g)
				v = int(v)
				line = plt.plot([X[g][0],X[v][0]],[X[g][1],X[v][1]],'k',alpha=0.1, zorder =0)
				plt.setp(line,linewidth=1)
		plt.scatter(X[:,0], X[:,1], 100, c=self.colorvalues,alpha=1.00, edgecolors='none',zorder = 1)


		#Draw MST
		if (self.hasMST > 0 and graphtype == 'MST'):
			for g in self.MST.G:
				for v in self.MST.G[g]:
					line = plt.plot([X[g][0],X[v][0]],[X[g][1],X[v][1]],'k',alpha=1, linewidth=2,zorder =0)


		#Draw Dijkstra Path
		if (len(self.path) > 0 and graphtype == 'dijkstra'):
			for i in range(len(self.path)-1):
				g = self.path[i]
				v = self.path[i+1]
				line = plt.plot([X[g][0],X[v][0]],[X[g][1],X[v][1]],'k',alpha=1, linewidth=2,zorder =0)


		#Show Plot
		plt.show()



	#--------------------------------------------------------------
	def distancenMatrix(self):

		#Compute Distance Matrix
		self.D = numpy.zeros((self.nNodes,self.nNodes))
		for i in range(self.nNodes):
			if i in self.G:
				w = self.G[i]
				for v in w:
					self.D[i,v] = w[v]



	#--------------------------------------------------------------
	def transitionMatrix(self):

		#Compute Transition Matrix
		self.T = numpy.zeros((self.nNodes,self.nNodes))
		for i in range(self.nNodes):
			if i in self.G:
				w = self.G[i]
				for v in w:
					self.T[i,v] = w[v]

		#Normalize
		self.T = self.T > 0
		self.T = self.T.astype(float)
		summed = numpy.sum(self.T,axis=0)
		for i in range(self.nNodes):
			if (summed[i] > 0):
				self.T[:,i] = self.T[:,i] / summed[i]
			else:
				self.T[:,i] = 0


	#--------------------------------------------------------------
	def pageRank(self,beta):

		#Init
		N = numpy.zeros((self.nNodes,1)) + 1/float(self.nNodes)
		c = numpy.ones((self.nNodes,1)) * (1-beta)/float(self.nNodes)

	
		#Iterate
		for i in range(100):
			N = numpy.dot(beta*self.T,N) + c


		#Normalize N for visualization
		N = N - numpy.min(N)
		N = N / numpy.max(N)


		#Store
		self.pagerank = N



	#--------------------------------------------------------------
	def findPath(self,startnode,endnode):

		#Find Path
		v = endnode;
		self.path = []
		self.path.append(v)
		while (v != startnode):
			if (v not in self.edgeTo):
				break
			else:
				v = self.edgeTo[v]
				self.path.append(v)



	#--------------------------------------------------------------
	def depthFirstSearch(self,node):

		#Depth First Search (Recursive)
		self.marked.append(node)
		if self.G.has_key(node):
			for w in self.G[node]:
				if w not in self.marked:
					self.edgeTo[w] = node
					self.depthFirstSearch(w)



	#--------------------------------------------------------------
	def DFS(self,startnode,endnode):

		#Start DFS
		marked = []
		edgeTo = []
		self.depthFirstSearch(startnode)
		self.findPath(startnode,endnode)



	#--------------------------------------------------------------
	def dijkstra(self,startnode):

		#Init Distance and Path 
		self.dist = numpy.zeros((self.nNodes)) + numpy.inf
		self.P = {}


		#Setup/Init Priority Queue
		Q = priorityqueue.PriorityQueue('min')
		Q.push((startnode,0))
		self.dist[startnode] = 0


		#Run recursively
		while not Q.isempty():
			
			#Pop Elements
			v,tmp = Q.pop()

			#With all edges of node v
			for w in self.G[v]:
				vw = self.G[v][w]

				#Update the distance to w if a smaller distance
				if (self.dist[w] > self.dist[v] + vw):
					self.dist[w] = self.dist[v] + vw
					self.P[w] = v

					#If priorty contains w, update, otherwise add
					if w in Q.keys:
						index = Q.keys.index(w)
						Q.values[index] = self.dist[w]
					else:
						Q.push((w,self.dist[w]))



	#--------------------------------------------------------------
	def rundijkstra(self,startnode,endnode):


		#Run dijkstra
		self.dijkstra(startnode)


		#Find Shortest Path
		curnode = endnode
		self.path.append(curnode)
		while (curnode != startnode):
			curnode = self.P[curnode]
			self.path.append(curnode)


	#--------------------------------------------------------------
	def centrality(self):

		#Emmpty counter
		C = numpy.zeros((self.nNodes,1))

		#Loop through nodes
		for i in range(self.nNodes):

			#Run Dijkstra
			self.dijkstra(i)

			#Find Path from i to every other node (but itself)
			for j in range(self.nNodes):
				if (i != j):
					path = []
					curnode = j
					path.append(curnode)
					C[curnode] = C[curnode] + 1
					while (curnode != i):
						curnode = self.P[curnode]
						C[curnode] = C[curnode] + 1
						path.append(curnode)
		
		#Store
		self.C = C / numpy.sum(C)


	#--------------------------------------------------------------
	def computeMST(self):

		#Init
		self.MST = Graph()
		self.hasMST = 1
		pq = priorityqueue.PriorityQueue('min')

		
		#Put Edge Values in Priority Queue
		for g in self.G:
			for v in self.G[g]:
				mytuple = self.G[g][v],(g,v)
				pq.push(mytuple)


		#Loop through edges until queue is empty
		while (not pq.isempty()):
			d,edge = pq.pop()
			v,w = edge
			self.MST.marked = []
			self.MST.depthFirstSearch(v)
			if w not in self.MST.marked:
				edge1 = {w:d}
				edge2 = {v:d}
				if (self.MST.G.has_key(v)):
					self.MST.G[v].update(edge1)
				else:
					self.MST.G[v] = edge1
				if (self.MST.G.has_key(w)):
					self.MST.G[v].update(edge2)
				else:
					self.MST.G[w] = edge2
	


#------------------------------------------------------------------
if __name__ == '__main__':


	#Init Graph
	graph = Graph()



	#Create Graph
	graph.create(25,0.5)


	
	#Graph Dijkstra
	graph.rundijkstra(0,23)



	#Calculate Transition Matrix
	graph.transitionMatrix()



	#Calculate Page Rank
	graph.pageRank(0.85)



	#Calculate Betweenness Centrality
	graph.centrality()


	#Depth First Search
	#graph.DFS(0,23)


	
	#Minimum Spanning Tree
	#graph.computeMST()



	#Visualize
	graph.show('centrality')
