
import sys
import time
from copy import copy
import random

from node import Node



# Statistics 
start_time = time.time()
# possibleFeatures = set()
classifiedValues = []


# Split dataset into two datasets
def split(attribute, dataset):
	output = {}
	for row in dataset:
		dic = row[1]
		key = dic.get(attribute, '0')
		output[key] = output.get(key, [])
		output[key].append(row)
	return output



# Change in impurity 
def deltaG(g, gA):
	return g - gA



# Compute Gini index 
def gini(dataset):
	dic = {}
	total = len(dataset)
	g = 0
	for row in dataset:
		label = row[0]
		dic[label] = dic.get(label, 0.0) + 1.0
	for label in dic:
		temp = dic[label] / total
		g += (temp**2)
	return 1.0 - g


def isLeaf(node):
	if gini(node.dataset) == 0:
		return True
	return False


# Select best feature to split dataset
def select(dataset, parent, possibleFeatures):
	selectedFeature = ''
	length = len(dataset)
	g = gini(dataset)
	level = None
	minImpurity = -1
	for feature in possibleFeatures:
		dic = split(feature, dataset)
		giniA = 0
		for key in dic:
			giniA += ((float(len(dic[key])) / float(length)) * gini(dic[key]))
		if deltaG(g, giniA) > minImpurity:
			minImpurity = deltaG(g, giniA)
			selectedFeature = feature
			level = dic
	possibleFeatures.remove(selectedFeature)
	parent.setEdge(selectedFeature)
	for key in level:
		node = Node(level[key])
		parent.addNeighbor(key, node)
		if(isLeaf(node)):
			node.setLabel(level[key][0][0])
		else:
			if possibleFeatures == set():   # No more features, set label as the majority vote
				node.setLabel(majorty(node.dataset))
			else:
				select(level[key], node, copy(possibleFeatures))	# Otherwise select a different feature


def majorty(dataset):
	dic = {}
	vote = 0
	k = None
	for row in dataset:
		key = row[0]
		dic[key] = dic.get(key, 0) + 1
	for key in dic:
		if dic[key] >= vote:
			vote = dic[key]
			k = key
	return k


# Open & parse data
def lIBSVM(fileName, possibleFeatures, labels):
	file = open(fileName, 'r')
	output = []
	for line in file:
		dataLine = line.rstrip('\n').split(' ')
		label = dataLine[0]
		labels.add(label)
		dic = {}
		i = 1
		length = len(dataLine)
		while i < length:
			index, value = dataLine[i].split(':')
			dic[index] = value
			possibleFeatures.add(index)
			i += 1
		output.append((label, dic))
	file.close()
	return output


def classifiyTest(root, test):
	for row in test:
		data = row[1]
		node = root
		while(node.label == 'notLeaf'):
			edge = node.edge
			if(data[edge] in node.neighbors):
				node = node.neighbors[data[edge]]
			else:
				node = node.neighbors[pickPath(data[edge], node)]
		classifiedValues.append(node.label)
	return classifiedValues


def pickPath(key, node):
	k1 = int(key)
	k2 = int(key)
	c = 1
	while(True):
		for k in node.neighbors:
			if k1 - c == int(k):
				return k
			elif k2 + c == int(k):
				return k
		c+=1


def prepareOutput(testValues, classifiedValues, labels):
	l = list(labels)
	l.sort()
	leng = len(l)
	arr = [[0 for x in range(leng+1)] for y in range(leng+1)]
	index = 0
	length = len(classifiedValues)
	i = 0
	while i < length:
		arr[int(testValues[i][0])][int(classifiedValues[i])] += 1
		i+=1

	printMatrix(arr)


def printMatrix(arr):
	length = len(arr)
	for i in range(1, length):
		for j in range(1, length):
			print(str(arr[i][j]) + " "),
		print("\n"),
	# f1Score(arr)


def f1Score(arr):
	length = len(arr)
	for i in range(1, length):
		p = arr[i][i]
		r = arr[i][i]
		dp = 0.0
		dr = 0.0
		for j in range(1, length):
			dp+=arr[i][j]
		for j in range(1, length):
			dr += arr[j][i]

		if(p == 0 or r == 0):
			print("F1 score: %d", 0)
		else:
			p /= dp
			r /= dr
			print("F1 score: %d", (p*r*2/(p+r)))



def accuracy(testValues, classifiedValues):
	count = 0
	i = 0
	for row in testValues:
		if(row[0] == classifiedValues[i]):
			count += 1
		i += 1
	print(float(count)/float(len(testValues)))

def main():
	arg1 = sys.argv[1]
	arg2 = sys.argv[2]
	possibleFeatures = set()
	labels = set()
	train = lIBSVM(arg1, possibleFeatures, labels)
	root = Node(train)
	select(train, root, possibleFeatures)
	test = lIBSVM(arg2, possibleFeatures, labels)
	classifiyTest(root, test)
	# accuracy(test, classifiedValues)
	prepareOutput(test, classifiedValues, labels)





if __name__ == "__main__":
	main()
	# print("--- %s seconds ---" % (time.time() - start_time))