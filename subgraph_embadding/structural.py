import gensim.models.doc2vec as doc
import os
from subgraph_embadding import graphUtils_s
import random
import networkx as nx


def arr2str(arr):
    result = ""
    for i in arr:
        result += " " + str(i)
    return result


def generate_degree_walk(Graph, walkSize):
    walk = random_walk_degree_labels(Graph, walkSize)
    # walk = serializeEdge(g,NodeToLables)
    return walk


def random_walk_degree_labels(G, walkSize):
    cur_node = random.choice(list(G.nodes()))
    walk_list = []

    while len(walk_list) < walkSize:
        walk_list.append(G.nodes[cur_node]['label'])
        cur_node = random.choice(list(G.neighbors(cur_node)))
    return walk_list


def get_degree_labelled_graph(G, range_to_labels):
    degree_dict = dict(G.degree(G.nodes()))
    label_dict = {}
    for node in degree_dict.keys():
        val = degree_dict[node] / float(nx.number_of_nodes(G))
        label_dict[node] = in_range(range_to_labels, val)

        nx.set_node_attributes(G, name='label', values=label_dict)

    return G


def in_range(rangeDict, val):
    for key in rangeDict:
        if key[0] < val <= key[1]:
            return rangeDict[key]


def generate_walk_file(dir_name, walk_length, alpha):
    walk_dir_path = dir_name.replace('sub_graphs', 'walks')
    if not os.path.exists(os.path.dirname(walk_dir_path)):
        os.mkdir(os.path.dirname(walk_dir_path))
    walk_file = open(walk_dir_path + '.walk', 'w')
    index_to_name = {}
    range_to_labels = {(0, 0.05): 'z', (0.05, 0.1): 'a', (0.1, 0.15): 'b', (0.15, 0.2): 'c', (0.2, 0.25): 'd',
                       (0.25, 0.5): 'e', (0.5, 0.75): 'f', (0.75, 1.0): 'g'}
    index = 0
    for file in os.listdir(os.path.dirname(dir_name)):
        print(file)
        subgraph = graphUtils_s.get_graph(os.path.join(os.path.dirname(dir_name), file))
        degree_graph = get_degree_labelled_graph(subgraph, range_to_labels)
        degree_walk = generate_degree_walk(degree_graph, int(walk_length * (1 - alpha)))
        walk = graphUtils_s.random_walk(subgraph, int(alpha * walk_length))
        walk_file.write(arr2str(walk) + arr2str(degree_walk) + "\n")
        index_to_name[index] = file.split('.')[0]
        index += 1
    walk_file.close()
    return index_to_name


def structural_embedding(input_dir, iterations=20, dimensions=128, windowSize=2, dm=1, walkLength=64):
    index_to_name = generate_walk_file(input_dir, walkLength, 0.5)
    walk_dir_path = input_dir.replace('sub_graphs', 'walks')
    sentences = doc.TaggedLineDocument(walk_dir_path + '.walk')
    model = doc.Doc2Vec(sentences, size=dimensions, iter=iterations, dm=dm, window=windowSize)
    return list(model.docvecs.vectors_docs), index_to_name
