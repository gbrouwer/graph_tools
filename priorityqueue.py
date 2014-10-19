import numpy
import scipy
import sys
import os


#------------------------------------------------------------------
class PriorityQueue:


	#Init
	def __init__(self,type):
		self.type = type
		self.keys = []
		self.values = []



	#Show
	def show(self):

		for i,key in enumerate(self.keys):
			print key, self.values[i]



	#Exchange
	def exchange(self,i,j):
		
		tmp = self.keys[i-1]
		self.keys[i-1] = self.keys[j-1]
		self.keys[j-1] = tmp
		tmp = self.values[i-1]
		self.values[i-1] = self.values[j-1]
		self.values[j-1] = tmp



	#Compare
	def compare(self,i,j):
		
		if (self.type == 'max'):
			return self.keys[i-1] < self.keys[j-1]
		else:
			return self.keys[i-1] > self.keys[j-1]


	#Swim
	def swim(self,k):
		
		while (k > 1) and (self.compare(k/2,k)):
			self.exchange(k/2,k)
			k = k / 2
	

	#Sink
	def sink(self,k):

		N = len(self.keys)
		while (2*k <= N):
			j = k * 2
			if ((j < N) and self.compare(j,j+1)):
				j = j + 1
			if (not self.compare(k,j)):
				break
			self.exchange(k,j)
			k = j


	#Pop
	def isempty(self):

		if (len(self.keys) == 0):
			return True
		else:
			return False


	#Pop
	def pop(self):
		
		key = self.keys[0]
		value = self.values[0]
		self.exchange(1,len(self.keys))
		self.keys.pop()
		self.values.pop()
		self.sink(1)
		return key,value



	#Push
	def push(self,mytuple):
		key,value = mytuple
		self.keys.append(key)
		self.values.append(value)
		self.swim(len(self.keys))




#------------------------------------------------------------------
if __name__ == '__main__':


	#Test client
	pq = PriorityQueue('min')


	#Push
	for i in range(100):
		d = numpy.random.random()
		v = numpy.random.randint(0,100)
		w = numpy.random.randint(0,100)
		mytuple = (d,(v,w))
		pq.push(mytuple)


	#Pop
	for i in range(100):
		key,value = pq.pop()
		print key,value
	

