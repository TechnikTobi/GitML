import numpy
import math

def NN(m1, m2, w1, w2, b):
	z = m1 * w1 + m2 * w2 + b
	return sigmoid(z)

def sigmoid(x):
	return 1 / (1 + math.e ** (-x))

def cost(b):
	return (b-4) ** 2

def slope(b):
	return 2 * (b-4)

print(slope(0))

b = 8

#training loop
for i in range(1000):
	b = b - 0.1 * slope(b)
	#print(b)
