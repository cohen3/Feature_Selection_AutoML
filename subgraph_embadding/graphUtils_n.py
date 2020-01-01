import networkx as nx
import random
import matplotlib.pyplot as plt


def get_graph(filename):
    graph = nx.read_gpickle(filename)
    return graph


def randomWalk(G, walkSize):
    walkList = []
    curNode = random.choice(list(G.nodes()))

    while len(walkList) < walkSize:
        walkList.append(curNode)
        curNode = random.choice(list(G.neighbors(curNode)))
    return walkList


def getStats(G):
    stats = {'num_nodes': nx.number_of_nodes(G), 'num_edges': nx.number_of_edges(G), 'is_Connected': nx.is_connected(G)}


def drawGraph(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos)
    plt.savefig("graph.pdf")
    plt.show()
