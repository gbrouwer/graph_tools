import numpy
import scipy
import sys
import os


#------------------------------------------------------------------
class Queue:

	def __init__(self):
		self.keys = []
		self.values = []

	def show(self):
		for i,key in enumerate(self.keys):
			print key,self.values[i]

	def push(self,mytuple):
		key,value = mytuple
		self.keys.append(key)
		self.values.append(value)

	def pop(self):
		key = self.keys[0]
		value = self.values[0]
		self.keys = self.keys[1:]
		self.values = self.values[1:]
		return key,value


#------------------------------------------------------------------
if __name__ == '__main__':


	#Test Client
	queue = Queue()

	#Push
	for i in range(10):
		d = numpy.random.random()
		v = numpy.random.randint(0,100)
		w = numpy.random.randint(0,100)
		mytuple = (d,(v,w))
		queue.push(mytuple)


	queue.show()
	print ' ---- '
	

	#Pop
	for i in range(10):
		key,value = queue.pop()
		print key,value
	