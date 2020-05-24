from networkx import graph_atlas , write_gml

graph = graph_atlas(15)

write_gml(graph,"/Users/erans/PycharmProjects/metabolic_pathways_/g.gml")

