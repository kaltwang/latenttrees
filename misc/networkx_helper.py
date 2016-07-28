import networkx as nx
import matplotlib.pyplot as plt

def draw_graph_from_adjlist(adjlist):
    lines = [' '.join(str(x) for x in line) for line in adjlist]
    g = nx.parse_adjlist(lines, create_using=nx.DiGraph(), nodetype=int)
    pos = nx.drawing.nx_pydot.graphviz_layout(g, prog='dot')
    nx.draw(g, pos, with_labels=True, arrows=True)
    plt.show()