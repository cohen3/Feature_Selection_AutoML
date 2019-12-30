import gensim.models.doc2vec as doc
import os
from subgraph_embadding import graphUtils_n


def arr2str(arr):
    result = ""
    for i in arr:
        result += " " + str(i)
    return result


def generate_walk_file(dir_name, walkLength):
    walk_dir_path = dir_name.replace("\\sub_graphs\\", "\\walks\\")
    if not os.path.exists(os.path.dirname(walk_dir_path)):
        os.mkdir(os.path.dirname(walk_dir_path))
    walk_file = open(walk_dir_path + '.walk', 'w')
    index_to_name = {}
    index = 0
    for file in os.listdir(os.path.dirname(dir_name)):
        print(file)
        subgraph = graphUtils_n.get_graph(os.path.join(os.path.dirname(dir_name), file))
        walk = graphUtils_n.randomWalk(subgraph, walkLength)
        walk_file.write(arr2str(walk) + "\n")
        index_to_name[index] = file.split('.')[0]
        index += 1
    walk_file.close()

    return index_to_name


def neighborhood_embedding(input_dir, iterations=20, dimensions=128, windowSize=2, dm=1, walkLength=64):
    index_to_name = generate_walk_file(input_dir, walkLength)
    walk_dir_path = input_dir.replace("\\sub_graphs\\", "\\walks\\")
    sentences = doc.TaggedLineDocument(walk_dir_path + '.walk')
    model = doc.Doc2Vec(sentences, size=dimensions, iter=iterations, dm=dm, window=windowSize)
    return list(model.docvecs.vectors_docs), index_to_name
