class Node():
	def __init__(self, dataset):
		self.dataset = dataset
		self.neighbors = {}
		self.label = "notLeaf"
		self.edge = ""

	def addNeighbor(self, key, neighbor):
		self.neighbors[key] = neighbor

	def setLabel(self, label):
		self.label = label

	def setEdge(self, edge):
		self.edge = edge