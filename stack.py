import numpy
import scipy
import sys
import os


#------------------------------------------------------------------
class Stack:

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
		key = self.keys.pop()
		value = self.values.pop()
		return key,value


#------------------------------------------------------------------
if __name__ == '__main__':


	#Test Client
	stack = Stack()

	#Push
	for i in range(10):
		d = numpy.random.random()
		v = numpy.random.randint(0,100)
		w = numpy.random.randint(0,100)
		mytuple = (d,(v,w))
		stack.push(mytuple)


	stack.show()
	print ' ---- '
	

	#Pop
	for i in range(10):
		key,value = stack.pop()
		print key,value
	