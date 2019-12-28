import networkx as nx
import random
import matplotlib.pyplot as plt


def write_graph(filename, G):
    file = open(filename, 'w')
    for edge in G.edges():
        node1 = str(G.node[edge[0]]['label'])
        node2 = str(G.node[edge[1]]['label'])
        file.write(node1 + '\t' + node2 + '\n')
    file.close()


def get_graph(filename):
    graph = nx.read_gpickle(filename)
    return graph


def random_walk(G, walkSize):
    walk_list = []
    cur_node = random.choice(list(G.nodes()))

    while len(walk_list) < walkSize:
        walk_list.append(G.node[cur_node]['label'])
        cur_node = random.choice(list(G.neighbors(cur_node)))
    return walk_list


def getStats(G):
    stats = {'num_nodes': nx.number_of_nodes(G), 'num_edges': nx.number_of_edges(G), 'is_Connected': nx.is_connected(G)}


def drawGraph(G):
    plt.figure()
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos)
    plt.savefig("graph.pdf")
    plt.show()
