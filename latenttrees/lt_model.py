import collections, copy, random, warnings, sys, builtins, time, datetime
from itertools import chain

import numpy as np
import networkx as nx
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

#from latenttrees.lt_helper import *
from latenttrees.lt_helper import is_message_1to2, calc_lklhd_parent_messages, imshow_values, \
    NanSelect, select_max_undecorated, select_weighted_random_undecorated, select_random_undecorated, \
    select_random_metropolis_undecorated
from misc.numpy_helper import normalize_convex, is_obj_array, ProgressLine, has_equiv_shape, cut_max, norm_logpdf, \
    normalize_convex_log, expand_array, obj_array_get_N
from misc.python_helper import has_elements, isequal_or_none, get_and_set_attr

# profile magic: define @profile decorator on the fly, if not defined by the kernprof script
# see http://stackoverflow.com/questions/18229628/python-profiling-using-line-profiler-clever-way-to-remove-profile-statements
try:
    builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    builtins.profile = profile


class ObjectRoot(object):
    def __init__(self):
        # type(self).print_enabled = True  # static (class) variable
        self.print_enabled = True  # static (class) variable

    def _print(self, str_):
        if self.print_enabled:
            print(self._print_prefix() + str_)

    def _print_prefix(self):
        classname = type(self).__name__
        timestr = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        prefix = timestr + " [" + classname + "] "
        return prefix


class GraphObject(ObjectRoot):
    COPY_PREFIX = '_copy_'

    def __init__(self):
        super(GraphObject, self).__init__()

    def clear(self, prop_set):
        for p in prop_set:
            setattr(self, p, None)

    def copy(self, prop_set):
        for p in prop_set:
            value_copy = copy.deepcopy(getattr(self, p))
            setattr(self, GraphObject.COPY_PREFIX + p, value_copy)

    def recover_copy(self, prop_set):
        for p in prop_set:
            value_copy = getattr(self, GraphObject.COPY_PREFIX + p)
            setattr(self, p, value_copy)
            delattr(self, GraphObject.COPY_PREFIX + p)



class Node(GraphObject):
    def __init__(self, k_):
        self.k = k_
        self.prior = None
        self.layer = 0
        self.create_new_prior()

    def create_new_prior(self):
        self.prior = DistribFactory.random(1, self.k)

    def set_k(self, k_):
        self.k = k_
        if self.prior is not None:
            self.create_new_prior()


class Edge(GraphObject):
    def __init__(self, k1_, k2_):
        if k1_ > 1:
            self.k1 = k1_
            self.k2 = k2_
            self.distrib = DistribFactory.random(k1_, k2_)
        else:
            raise ValueError('k1_={} must be greater than 1 (parents always need to be categorical)!'.format(k1_))

    def set_k1(self, k1_):
        self.__init__(k1_, self.k2)

    def set_k2(self, k2_):
        self.__init__(self.k1, k2_)


class NodeEdgeFactory(object):
    def __init__(self, properties_=None):
        self._instances_created = False
        if properties_ is not None:
            self._properties = set(properties_)  # string list of property names
        else:
            self._properties = set()

    def _init_properties(self, obj):
        obj.clear(self._properties)
        self._instances_created = True

    def register_properties(self, prop_set):
        # check first if any element from prop_set is actually new
        if not self._properties >= prop_set:
            if self._instances_created:
                raise RuntimeError('Properties must be added before the first class instantiation!')
            self._properties.update(prop_set)


class NodeFactory(NodeEdgeFactory):
    def __init__(self, properties_=None):
        super(NodeFactory, self).__init__(properties_)

    def create_node(self, k):
        result = Node(k)
        super(NodeFactory, self)._init_properties(result)
        return result


class EdgeFactory(NodeEdgeFactory):
    def __init__(self, properties_=None):
        super(EdgeFactory, self).__init__(properties_)

    def create_edge(self, k1, k2):
        result = Edge(k1, k2)
        super(EdgeFactory, self)._init_properties(result)
        return result


class Graph(GraphObject):
    OBJ_STR = 'obj'  # dictionary key string that is used to access node and edge objects within networkx graph

    def __init__(self):
        super(Graph, self).__init__()
        self.__id_node_next = 0
        self.__nxgraph = nx.DiGraph()
        self.__node_factory = NodeFactory()
        self.__edge_factory = EdgeFactory()
        self.__id_roots = set()

        self.required_axes_total = 0
        self.axes = None
        self.figure = None
        self.required_axes = 1
        self.id_axes = self.register_axes(self.required_axes)
        self.print_enabled = False

    def get_dof(self, id_node):
        dof = 0
        prior = self.node(id_node).prior
        if prior is not None:
            dof += prior.get_dof()
        for id_node1, id_node2, edge in self.edges_iter(nbunch=id_node, data=True):
            dof_act = edge.distrib.get_dof()
            dof += dof_act
        return dof

    def set_k(self, id_node, k):
        node = self.node(id_node)
        node.set_k(k)
        for id_node1, id_node2, edge in self.edges_iter(nbunch=id_node, data=True):
            if id_node1 == id_node:
                edge.set_k1(k)
            else:
                assert id_node2 == id_node
                edge.set_k2(k)

    def get_axes(self, id_axes):
        if self.figure is None or (not plt.fignum_exists(self.figure.number)):
            # create figure and axes
            self.figure, self.axes = plt.subplots(nrows=1, ncols=self.required_axes_total, squeeze=True, figsize=(20, 6.66))
        axes_curr = [self.axes[id] for id in id_axes]
        for ax in axes_curr:
            ax.cla()
        return axes_curr


    def register_axes(self, required_axes):
        id_axes_new = [x for x in range(self.required_axes_total, self.required_axes_total+required_axes)]
        self.required_axes_total += required_axes
        return id_axes_new

    def get_id_node_next(self):
        return self.__id_node_next

    def has_node(self, id_node):
        return self.__nxgraph.has_node(id_node)

    def number_of_nodes(self):
        return self.__nxgraph.number_of_nodes()

    def has_edge(self, id_node1, id_node2):
        return self.__nxgraph.has_edge(id_node1, id_node2)

    def has_parent(self, id_node):
        edges = self.in_edges_iter(id_node, data=False)
        return has_elements(edges)

    def get_parent(self, id_node):
        id_parents = [x for x in self.in_edges_iter(id_node, data=False)]
        num_parents = len(id_parents)
        assert num_parents <= 1  # tree structure allows one parent at most
        if num_parents == 1:
            return id_parents[0]
        else:
            return None

    def get_root(self, id_node):
        id_parent = self.get_parent(id_node)
        if id_parent is None:
            return id_node
        else:
            return self.get_root(id_parent)

    def get_id_roots(self):
        return self.__id_roots

    def get_num_roots(self):
        return len(self.__id_roots)

    def degree(self, id_node):
        return self.__nxgraph.degree(id_node)

    def out_degree(self, id_node):
        return self.__nxgraph.out_degree(id_node)

    def node(self, id_node):
        """Access node object.
        :param id_node: the node id
        """
        return self.__nxgraph.node[id_node][Graph.OBJ_STR]

    def edge(self, id_node1, id_node2):
        """Access edge object for edge from id_node1 to id_node2.
        :param id_node1: edge origin
        :param id_node2: edge destination
        """
        return self.__nxgraph[id_node1][id_node2][Graph.OBJ_STR]

    def add_node(self, k):
        """Adds a new node to the graph.
        :param k: cardinality of the node (k==1 for Gaussian and k>1 for categorical random variables)
        """
        id_node = self.__id_node_next
        self.__id_node_next += 1

        node_ = self.__node_factory.create_node(k)
        self.__nxgraph.add_node(id_node, {Graph.OBJ_STR: node_})

        # side effects
        self.__add_root(id_node)  # initially, every node is a root
        self._print("add id_node={}".format(id_node))
        return id_node

    def add_nodes(self, K):
        """Create nodes defined by the types vector k
        :param K: iterable that contains k values for all nodes to create
        """
        self._print("adding {} nodes".format(len(K)))
        id_nodes = [self.add_node(k) for k in K]
        return id_nodes

    def remove_node(self, id_node):
        self._print("remove id_node={}".format(id_node))
        assert self.has_node(id_node)

        # side effects
        self.__remove_root(id_node)

        g = self.__nxgraph
        degree = g.degree(id_node)
        if degree > 0:
            raise ValueError(
                'id_node={} has edges connected to it. Remove all edges first before removing the node!'.format(
                    id_node))
        g.remove_node(id_node)

    def add_edge(self, id_node1, id_node2):
        """Adds a new edge to the graph.
        :param id_node1: edge origin
        :param id_node2: edge destination
        """
        self._print("add edge=({},{})".format(id_node1, id_node2))
        # check the structure
        self.__check_potential_edge(id_node1, id_node2)

        # side effects
        self.__remove_root(id_node2)  # not root anymore

        node1 = self.node(id_node1)
        node2 = self.node(id_node2)
        edge_ = self.__edge_factory.create_edge(node1.k, node2.k)
        self.__nxgraph.add_edge(id_node1, id_node2, {Graph.OBJ_STR: edge_})

        # q needs to be recomputed
        node1.q_dirty = True
        node2.q_dirty = True

        # layer of node1 might change
        # node1.layer = max(node1.layer, node2.layer+1)
        self.__update_layer_recursive(id_node2, node2)

    def __update_layer_recursive(self, id_node, node):
        id_parent = self.get_parent(id_node)
        if id_parent is not None:
            parent = self.node(id_parent)
            parent.layer = max(parent.layer, node.layer+1)
            self.__update_layer_recursive(id_parent, parent)

    def remove_edge(self, id_node1, id_node2):
        """Removes an edge from the graph.
        :param id_node1: edge origin
        :param id_node2: edge destination
        """
        self._print("remove edge=({},{})".format(id_node1, id_node2))
        assert self.has_edge(id_node1, id_node2)
        self.__nxgraph.remove_edge(id_node1, id_node2)

        # side effects
        # by the structure restriction, id_node2 has only one parent and thus becomes a root
        self.__add_root(id_node2)

        # q needs to be recomputed
        self.node(id_node1).q_dirty = True
        self.node(id_node2).q_dirty = True

        # layer of node1 might change
        self.__recalculate_layer(id_node1)

    def __recalculate_layer(self, id_node):
        layer_options = [self.node(id_child).layer + 1 for id_child in self.out_edges_iter(id_node)] + [0]
        layer_new = max(layer_options)
        if self.node(id_node).layer != layer_new:
            self.node(id_node).layer = layer_new
            # we need to recursively check the next parent
            id_parent = self.get_parent(id_node)
            if id_parent is not None:
                self.__recalculate_layer(id_parent)

    def __check_potential_edge(self, id_node1, id_node2):
        """This functions ensures that each node has at most one parent, when adding a potential edge.
        (Note: this function does not prevent the graph from having cycles!)
        :param id_node1: edge origin
        :param id_node2: edge destination
        """
        # first check that both nodes exist
        assert self.has_node(id_node1)
        assert self.has_node(id_node2)
        # then assert the id_node2 does not have a parent yet
        in_degree_node2 = self.__nxgraph.in_degree(id_node2)
        assert in_degree_node2 == 0

    def __add_root(self, id_root):
        self.__id_roots.add(id_root)
        # add prior again
        node = self.node(id_root)
        if node.prior is None:
            node.create_new_prior()

    def __remove_root(self, id_root):
        self.__id_roots.discard(id_root)
        # remove prior
        node = self.node(id_root)
        node.prior = None

    def is_root(self, id_node):
        return id_node in self.__id_roots

    def draw(self, id_highlight=None):
        fig, ax = plt.subplots(1,self.required_axes)
        if not isinstance(ax, collections.Iterable):
            ax = [ax]
        self.draw_axes(ax, id_highlight)
        fig.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.show()

    def draw_axes(self, ax=None, id_highlight=None):
        # plt.ion()  # interactive mode on (doesn't work with pycharm)
        g = self.__nxgraph
        # new call format for networkx 1.11 (uses pygraphviz, which does not support Python 3!)
        # pos = nx.drawing.nx_agraph.graphviz_layout(g, prog='dot')
        # old call format for networkx 1.10
        # pos = nx.graphviz_layout(g, prog='dot')
        # new call format for networkx 1.11 (uses pydot)
        pos = nx.drawing.nx_pydot.graphviz_layout(g, prog='dot')

        if ax is None:
            ax = self.get_axes(self.id_axes)
        assert len(ax) == self.required_axes

        nx.draw(g, pos, ax=ax[0], hold=True, with_labels=True, arrows=True)
        # nx.draw_networkx(self.__graph)

        # plot the roots
        if id_highlight is None:
            id_highlight = self.__id_roots
        for r in id_highlight:
            ax[0].scatter(pos[r][0], pos[r][1], s=500)

        for id_node, node in self.nodes_iter(data=True):
            c = "{}".format(node.layer)
            ax[0].text(pos[id_node][0], pos[id_node][1], "\n" + c, va='top', ha='center', color='blue')

        # plt.show()  # this call blocks further execution until window is closed

    def print_distrib(self):
        pass

    def nodes(self, data=False):
        nx = self.__nxgraph.nodes(data=data)
        if data:
            n = map(lambda x: (x[0], x[1][self.OBJ_STR]), nx)
        else:
            n = nx
        return n

    def nodes_iter(self, data=False):
        nx_iter = self.__nxgraph.nodes_iter(data=data)
        """nx_iter returns tuples where the first element is the id_node and the second element is a dictionary
        of the node data"""
        if data:
            iter_ = map(lambda x: (x[0], x[1][self.OBJ_STR]), nx_iter)
        else:
            iter_ = nx_iter
        return iter_

    def edges_iter(self, nbunch=None, data=False):
        nx_iter_out = self.__nxgraph.out_edges_iter(nbunch=nbunch, data=data)
        nx_iter_in = self.__nxgraph.in_edges_iter(nbunch=nbunch, data=data)
        nx_iter = chain(nx_iter_out, nx_iter_in)
        """nx_iter returns triples where the first two elements are id_node1 and id_node2 and the third element is a
        dictionary of the edge data"""
        if data:
            iter_ = map(lambda x: (x[0], x[1], x[2][self.OBJ_STR]), nx_iter)
        else:
            iter_ = nx_iter
        return iter_

    def edges_iter_except(self, id_node, id_except, data=False):
        iter_ = self.edges_iter(nbunch=id_node, data=data)
        if id_except is not None:
            iter_ = filter(lambda x: (x[0] != id_except) and (x[1] != id_except), iter_)
        return iter_

    def out_edges_iter(self, id_node, data=False):
        nx_iter = self.__nxgraph.out_edges_iter(nbunch=(id_node,), data=data)
        if data:
            iter_ = map(lambda x: (x[1], x[2][self.OBJ_STR]), nx_iter)
        else:
            iter_ = map(lambda x: x[1], nx_iter)
        return iter_

    def in_edges_iter(self, id_node, data=False):
        nx_iter = self.__nxgraph.in_edges_iter(nbunch=(id_node,), data=data)
        if data:
            iter_ = map(lambda x: (x[0], x[2][self.OBJ_STR]), nx_iter)
        else:
            iter_ = map(lambda x: x[0], nx_iter)
        return iter_

    def register_properties(self, prop_graph, prop_node, prop_edge):
        self.clear(prop_graph)
        self.__node_factory.register_properties(prop_node)
        self.__edge_factory.register_properties(prop_edge)

    def clear_properties(self, prop_graph, prop_node, prop_edge):
        """Clear properties from nodes and edges.
        :param prop_graph: set of strings with graph properties
        :param prop_node: set of strings with node properties
        :param prop_edge: set of strings with edge properties
        """
        self.clear(prop_graph)
        for node_id, node in self.nodes_iter(data=True):
            node.clear(prop_node)
        for node_id1, node_id2, edge in self.edges_iter(data=True):
            edge.clear(prop_edge)

    def properties_func(self, id_root, func, recursive=True, prop_node=None, prop_edge=None):
        if prop_node is not None:
            func_node = lambda id, node: func(node, prop_node)
        else:
            func_node = None

        if prop_edge is not None:
            func_edge = lambda id1, id2, edge: func(edge, prop_edge)
        else:
            func_edge = None

        self.func_iter(id_root, recursive=recursive, func_node=func_node, func_edge=func_edge)

    def func_iter(self, id_roots=None, recursive=True, func_node=None, func_edge=None):
        g = self

        if id_roots is None:
            id_roots = g.get_id_roots()
        elif not isinstance(id_roots, collections.Iterable):
            id_roots = (id_roots, )

        for id_root in id_roots:
            self.__func_iter(id_root, recursive=recursive, id_except=None, func_node=func_node, func_edge=func_edge)

    def __func_iter(self, id_node, recursive=True, id_except=None, func_node=None, func_edge=None):
        if func_node is not None:
            node = self.node(id_node)
            func_node(id_node, node)

        edges = self.edges_iter_except(id_node, id_except, data=True)
        for id_node1, id_node2, edge in edges:
            if func_edge is not None:
                func_edge(id_node1, id_node2, edge)
            if recursive:
                _, id_next = is_message_1to2(id_node1, id_node2, id_node)
                self.__func_iter(id_next, recursive=recursive, id_except=id_node, func_node=func_node, func_edge=func_edge)

    def get_adjlist(self):
        # the returned adjacency list is guaranteed to be ordered in a way, that all targets in each line have been
        # defined before.

        #gen = nx.generate_adjlist(self.nxgraph)
        dict_of_lists = nx.to_dict_of_lists(self.__nxgraph)
        # sort according id1 (the second sort below is stable, thus this will be the secondary order)
        # sorting key is the first item, i.e. id1
        list_of_tuples = sorted(dict_of_lists.items(), key=lambda x: x[0])
        adjlist = []
        layer = []
        for id1, ids2 in list_of_tuples:
            lay = self.node(id1).layer
            adjlist.append([id1] + ids2)
            layer.append(lay)

        # sort according layer
        # sorting key is the first item, i.e. the layer
        adjlist_reordered = [i[0] for i in sorted(zip(adjlist, layer), key=lambda x:x[1])]

        # rename the ids to id new, which use consecutive numbers starting from 0
        id_nodes = [line[0] for line in adjlist_reordered]
        id2idn = {} # idn: id new
        for idn, id in enumerate(id_nodes):
            id2idn[id] = idn
        # rename all ids from id to idn and sort them
        adjlist_renamed = []
        for line in adjlist_reordered:
            line_renamed = [id2idn[id] for id in line]
            line_renamed = [line_renamed[0]] + sorted(line_renamed[1:])
            adjlist_renamed.append(line_renamed)

        return adjlist_renamed


class GraphManipulator(ObjectRoot):
    """Interface class for all graph manipulators (data, inference, parameter_update and structure_update)"""
    def __init__(self, graph):
        super(GraphManipulator, self).__init__()
        assert isinstance(graph, Graph)
        self._graph = graph
        self._prop_graph = set()
        self._prop_node = set()
        self._prop_edge = set()
        self.required_axes = 0
        self.id_axes = None

    def _register_properties(self):
        g = self._graph
        g.register_properties(self._prop_graph, self._prop_node, self._prop_edge)
        if self.required_axes > 0:
            self.id_axes = g.register_axes(self.required_axes)

    def clear_properties(self):
        self._graph.clear_properties(self._prop_graph, self._prop_node, self._prop_edge)

    def run(self, id_roots=None, recursive=True):
        g = self._graph

        if id_roots is None:
            id_roots = g.get_id_roots()
        elif not isinstance(id_roots, collections.Iterable):
            id_roots = (id_roots, )

        # self._print('id_roots={}'.format(id_roots))
        lklhd = 0
        for id_root in id_roots:
            lklhd_root = self._run(id_root, recursive=recursive)  # call the run method of the child class
            if lklhd_root is not None:
                lklhd += lklhd_root
        return lklhd

    def draw(self):
        fig, ax = plt.subplots(1,self.required_axes)
        self.draw_axes(ax)
        plt.show()


class Data(GraphManipulator):
    """This class handles the input of observed data and the output of inferred data."""
    def __init__(self, graph):
        super(Data, self).__init__(graph)
        # x contains the observed data for the corresponding node and None otherwise.
        self._prop_node = {'x'}
        self._prop_graph = {'N'}
        self._register_properties()
        self.gauss_init_std = 1
        self.gauss_init_std_rand = 0

    def __insert_samples(self, id_node, samples):
        """Insert data x for specific node.
        :param id_node: 1x1 node identifier
        :param samples: Nx1 numpy array
        """
        g = self._graph
        node = g.node(id_node)
        assert isinstance(samples, np.ndarray)
        # check is samples are a cat. distribution or values
        if is_obj_array(samples):
            # object array with shape (1,)
            samples = samples[0]
            N = samples.shape[0]
            if samples.ndim == 1:
                # samples are values
                distrib = DistribFactory.from_samples(samples, node.k)
            elif node.k >= 2 and node.k == samples.shape[1]:
                #samples describe categorical distributions
                distrib = DistribFactory.empty(N, node.k)
                distrib.init_data(samples)
            else:
                raise ValueError("samples have wrong shape={}, for k={}".format(samples.shape, node.k))
        else:
            N = samples.shape[0]
            distrib = DistribFactory.from_samples(samples, node.k)

        assert isequal_or_none(N, g.N)
        g.N = N
        node.x = distrib

    def insert_samples(self, id_nodes, samples):
        """Insert data x for multiple nodes.
        :param id_nodes: Mx1 node identifier
        :param samples: NxM numpy array
        """
        self._print("inserting N={} samples into M={} nodes".format(obj_array_get_N(samples), len(id_nodes)))
        for i, id_node in enumerate(id_nodes):
            self.__insert_samples(id_node, samples[:, i])

    def __distrib_init(self, id_node):
        """Initialize prior of id_node and outgoing edges by using observed data x
        :param id_node: node ID
        """
        g = self._graph
        node = g.node(id_node)
        # initialize id_node prior, if id_node has prior
        if g.is_root(id_node):
            if node.x is not None:
                node.prior.init_distrib_idx(node.x, idx=None)
            else:
                node.prior.init_random()

        # initialize all outgoing edges (id_node, id_child_n)
        num_edges = g.out_degree(id_node)
        edges = g.out_edges_iter(id_node, data=True)
        idx = np.random.choice(g.N, size=(node.k,), replace=False)
        for id_child, edge in edges:
            child = g.node(id_child)
            has_data = child.x is not None
            # only use random idx of datapoints for Gaussians
            # (Cat datapoints usually are collapsed distributions to a single state)
            is_gaussian = child.k == 1
            if has_data and is_gaussian:
                if num_edges > 1:
                    edge.distrib.init_distrib_idx(child.x, idx)
                else:
                    edge.distrib.init_distrib_equidistant_rand(child.x, self.gauss_init_std, self.gauss_init_std_rand)
            else:
                edge.distrib.init_random()
            child.q_dirty = True
        node.q_dirty = True

    # def _run(self, id_root, recursive, id_except=None):
    #     self.__distrib_init(id_root)
    #
    #     if recursive:
    #         edges = self._graph.edges_iter_except(id_root, id_except)
    #         for id_node1, id_node2 in edges:
    #             _, id_next = is_message_1to2(id_node1, id_node2, id_root)
    #             self._run(id_next, recursive, id_except=id_root)

    def _run(self, id_root, recursive, id_except=None):
        func_node = lambda id, node: self.__distrib_init(id)
        self._graph.func_iter(id_root, recursive, func_node)


class BeliefPropagation(GraphManipulator):
    """This class handles the belief propagation inference algorithm."""
    def __init__(self, graph):
        super(BeliefPropagation, self).__init__(graph)
        # lklhd: N x 1 (conditional) log-likelihoods of the datapoints
        # q: N x k posterior distribution of the node
        # q_dirty: Boolean flag that determines if q needs to be recalculated
        # message_1to2 (alpha): message id_node1 -> id_node2 (from parent to child)
        # message_2to1 (beta):  message id_node1 <- id_node2 (from child to parent)
        self._prop_node = {'lklhd', 'q', 'q_dirty'}
        self._prop_edge = {'message_1to2', 'message_2to1'}
        # requires prop_graph: N
        # requires prop_node: x, prior
        # requires prop_edge: distrib
        self._register_properties()
        self.extract_samples_mode = 'max'

    @profile
    def _run(self, id_root, recursive):
        if recursive:
            rec = sys.maxsize
        else:
            rec = 1

        self.__inward_pass(id_center=id_root, id_in=None, recursive=rec)
        self.__outward_pass(id_center=id_root, id_in=None, recursive=rec)
        lklhd = self.get_lklhd(id_root)
        return lklhd

    @profile
    def get_message(self, id_src, id_dest):
        g = self._graph
        if g.has_edge(id_src, id_dest):
            return g.edge(id_src, id_dest).message_1to2
        else:
            assert g.has_edge(id_dest, id_src)
            return g.edge(id_dest, id_src).message_2to1

    @classmethod
    @profile
    def __get_message_from_edge(cls, id_node1, id_node2, edge, id_dest):
        is_message_1to2_, id_src = is_message_1to2(id_node1, id_node2, id_dest)
        if is_message_1to2_:
            return edge.message_1to2
        else:
            return edge.message_2to1

    @profile
    def __set_message(self, id_dest, id_src, messages_prod):
        # messages_prod : N x k_node
        # message:        N x k_origin
        g = self._graph
        if g.has_edge(id_src, id_dest):
            # update alpha message
            # edge.distrib: k_node x k_origin
            # for alpha message, we need to renormalize first
            # since each distrib is normalized, we only need to set the log_constant to zero
            edge = g.edge(id_src, id_dest)
            messages_prod.set_log_const_zero()
            edge.message_1to2 = messages_prod.dot(edge.distrib)
        else:
            # update beta message
            # edge.distrib: k_origin x k_node
            edge = g.edge(id_dest, id_src)
            edge.message_2to1 = messages_prod.dot_transpose(edge.distrib)
        node = g.node(id_dest)
        node.q_dirty = True  # incoming messages have changed, thus q has changed

    @profile
    def get_messages_prod(self, id_node, id_except, include_prior=True):
        g = self._graph
        node = g.node(id_node)
        messages = []
        if node.x is not None:
            messages.append(node.x)
        if (node.prior is not None) and include_prior:
            messages.append(node.prior)

        edges = g.edges_iter_except(id_node, id_except, data=True)
        for id_node1, id_node2, edge in edges:
            message = self.__get_message_from_edge(id_node1, id_node2, edge, id_node)
            messages.append(message)

        if (len(messages) == 1) and (messages[0].get_k1() == g.N):
            # this is just for speedup, the later 'else' statement would also work
            messages_prod = messages[0].copy()
        else:
            messages_prod = DistribFactory.uniform(g.N, node.k)
            messages_prod.prod(messages)
        return messages_prod

    @profile
    def __inward_pass(self, id_center, id_in=None, recursive=sys.maxsize):  # beta_ci pass
        # sets message that runs towards id_in from id_center
        # to start with a root
        # call __inward_pass(id_root, recursive=1)           for non-recursive and
        # call __inward_pass(id_root, recursive=sys.maxsize) for recursive calls
        g = self._graph

        if recursive > 0:
            edges_oc = g.edges_iter_except(id_center, id_in, data=False)
            for id_node1, id_node2 in edges_oc:
                _, id_out = is_message_1to2(id_node1, id_node2, id_center)
                self.__inward_pass(id_center=id_out, id_in=id_center, recursive=recursive-1)

        if id_in is not None:
            # only if there is a destination for the message
            messages_oc_prod = self.get_messages_prod(id_center, id_in)
            self.__set_message(id_in, id_center, messages_oc_prod)

            # Maybe: save here the product of all incoming messages, in order to be able to calculate later q() and the
            # conditional likelihood C()

    @profile
    def __outward_pass(self, id_center, id_in=None, recursive=sys.maxsize):
        # sets message that run towards id_center from id_in
        g = self._graph

        if id_in is not None:
            # only if there is a source of the message (otherwise it is the root)
            # in the outward pass, id_in and id_center roles are switched (in comparison to the inward pass)
            messages_prod = self.get_messages_prod(id_in, id_center)
            self.__set_message(id_center, id_in, messages_prod)

        if recursive > 0:
            edges_oc = g.edges_iter_except(id_center, id_in, data=False)
            for id_node1, id_node2 in edges_oc:
                _, id_out = is_message_1to2(id_node1, id_node2, id_center)
                self.__outward_pass(id_center=id_out, id_in=id_center, recursive=recursive-1)

    @profile
    def get_lklhd(self, id_node):
        node = self._graph.node(id_node)
        self.__calc_q_single(id_node)
        lklhd = np.mean(node.lklhd)  # normalized log-likelihood; the mean divides implicitly by N
        return lklhd

    @profile
    def get_q(self, id_node):
        node = self._graph.node(id_node)
        self.__calc_q_single(id_node)
        return node.q

    @profile
    def get_lklhd_all(self):
        g = self._graph
        id_roots = g.get_id_roots()
        lklhd = 0
        for id_root in id_roots:
            lklhd += self.get_lklhd(id_root)
        return lklhd

    @profile
    def __calc_q_single(self, id_node):
        g = self._graph
        node = g.node(id_node)
        if (node.q_dirty is not None) and (not node.q_dirty):
            # lazy calculation of q
            return
        messages_prod = self.get_messages_prod(id_node, id_except=None)
        if isinstance(messages_prod, DistribCat):
            lklhd = messages_prod.get_log_const().copy()
            # messages_prod.set_log_const_zero()

            # id_parent = g.get_parent(id_node)
            # if id_parent is not None:
            #     # just to double check
            #     lklhd2 = self.__calc_lklhd_parent(id_node, id_parent)
            #     assert np.allclose(lklhd, lklhd2)
        else:
            id_parent = g.get_parent(id_node)
            assert id_parent is not None # each Gaussian node must have exactly one parent
            # this calculation is only possible if the Gaussian node has exactly one parent.
            lklhd = self.calc_lklhd_parent(id_node, id_parent)

        q = messages_prod

        node.q = q
        node.q_dirty = False  # q has just been updated
        node.lklhd = lklhd

    @profile
    def calc_lklhd_parent(self, id_node, id_parent, message_child2parent_pot=None):
        messages_prod = self.get_messages_prod(id_parent, id_except=id_node)
        if message_child2parent_pot is not None:
            messages_prod.prod([message_child2parent_pot])
        message = self.get_message(id_node, id_parent)
        lklhd = calc_lklhd_parent_messages(messages_prod, message)
        return lklhd

    @profile
    def __extract_samples(self, id_node):
        q = self.get_q(id_node)
        samples = q.extract_samples(mode=self.extract_samples_mode)
        return samples

    @profile
    def extract_samples(self, id_nodes):
        N = self._graph.N
        M = len(id_nodes)
        samples = np.zeros((N, M))
        for i, id_node in enumerate(id_nodes):
            samples[:,i:(i+1)] = self.__extract_samples(id_node)
        return samples

    @profile
    def visual_gauss_edge(self, id_node1, id_node2):
        g = self._graph
        xs, ds = g.node(id_node2).x.visual_get_kde()
        mu = g.node(id_node2).x.get_mu()
        dens = g.edge(id_node1,id_node2).distrib.visual_get_density(xs)
        plt.plot(xs,np.concatenate((ds[:, np.newaxis], dens), axis=1))
        plt.scatter(mu, np.zeros_like(mu))
        plt.show()

class Distrib(object):
    def __init__(self, k1, k2):
        self._k1 = k1
        self._k2 = k2

    def get_k1(self):
        return self._k1

    def get_k2(self):
        return self._k2

# dot(Cat, Cat)
# dot(Cat, Gauss)
# dot(Gauss, Gauss)
# dot(Gauss, Cat)

# transpose(Cat)


class DistribCat(Distrib):
    def __init__(self, k1, k2):
        assert k2 > 1
        super(DistribCat, self).__init__(k1, k2)
        # values = values_norm * exp( log_const )
        self.__values_norm = None  # k1 x k2
        self.__log_const = None  # k1 x 1

    def get_dof(self):
        dof = self._k1 * (self._k2 - 1)
        return dof

    @profile
    def copy(self):
        distrib = DistribFactory.empty(self._k1, self._k2)
        distrib.init_data_raw(self.__values_norm.copy(), self.__log_const.copy())
        return distrib

    def get_values(self):
        return self.__values_norm * np.exp(self.__log_const)

    def get_values_norm(self):
        return self.__values_norm

    def get_log_const(self):
        return self.__log_const

    def set_log_const_zero(self):
        self.__log_const.fill(0)

    @profile
    def prod(self, distribs):
        for distrib in distribs:
            self.__values_norm *= distrib.get_values_norm()
            self.__log_const += distrib.get_log_const()
        self.__normalize_convex()

    @profile
    def dot(self, distrib):
        # self: N x k1 (Cat)
        # distrib: k1 x k2 (Cat / Gauss)
        # out: N x k2 (Cat / Gauss)
        assert (self.get_log_const() == 0).all()  # alpha message should have been normalized
        values_norm = self.__values_norm  # N x k1

        if isinstance(distrib, DistribCat):
            assert (distrib.get_log_const() == 0).all()
            values = values_norm.dot(distrib.get_values_norm())
            result = DistribCat(values.shape[0], values.shape[1])
            result.init_data_unnormalized(values)
        else:
            assert isinstance(distrib, DistribGauss)
            mu = distrib.get_mu()  # k1 x 1
            std = distrib.get_std()  # k1 x 1

            # means are calculated by the same dot product as Cat
            data_m = values_norm.dot(mu)  # N x 1

            # variance within components
            var_within = values_norm.dot(np.square(std))
            # variance between components
            var_between = np.sum(values_norm * np.square(data_m - mu.reshape((1, -1))), axis=1, keepdims=True)
            data_s = np.sqrt(var_within + var_between)
            result = DistribGauss(values_norm.shape[0])
            result.init_data(data_m, data_s)
        return result

    @profile
    def dot_transpose(self, distrib):
        # self: N x k2
        # distrib: k1 x k2
        # out = self * distrib^T: N x k1
        assert isinstance(distrib, DistribCat)  # should never be called for DistribGauss
        assert (distrib.get_log_const() == 0).all()
        values = self.__values_norm.dot(distrib.get_values_norm().transpose())
        result = DistribCat(values.shape[0], values.shape[1])
        result.init_data_unnormalized(values, self.__log_const)
        return result

    @profile
    def init_samples(self, samples, k):
        assert k > 1
        values_norm = self.__samples2distrib(samples, k)
        self.init_data(values_norm)

    @profile
    def extract_samples(self, mode):
        if mode == 'max':
            samples = self.__distrib2samples_max(self.__values_norm)
        elif mode == 'exp':
            samples = self.__distrib2samples_exp(self.__values_norm)
        else:
            assert False
        return samples

    @profile
    def init_uniform(self):
        self.__values_norm = np.empty((self._k1, self._k2))
        self.__values_norm.fill(1/self._k2)
        self.__log_const = np.empty((self._k1, 1))
        # uniform means in this case a distribution of all ones!
        # this ensures that the likelihood corresponds to an unobserved message.
        self.__log_const.fill(np.log(self._k2))

    @profile
    def init_random(self):
        values_norm = np.ones((self._k1, self._k2)) / self._k2
        self.__add_random(values_norm, 0.2)
        self.init_data(values_norm)

    @profile
    def init_data_unnormalized(self, values_norm, log_const=None):
        if log_const is None:
            log_const = np.zeros((values_norm.shape[0], 1))
        log_const += np.log(normalize_convex(values_norm, axis=1))
        self.init_data_raw(values_norm, log_const)

    @profile
    def init_data_raw(self, values_norm, log_const):
        self.__values_norm = values_norm
        self.__log_const = log_const

    @profile
    def init_data(self, values_norm, log_const=None):
        assert has_equiv_shape(values_norm, (self._k1, self._k2))
        assert np.allclose(np.sum(values_norm, axis=1), 1)
        if log_const is not None:
            assert has_equiv_shape(log_const, (self._k1, 1))
        else:
            log_const = np.zeros((self._k1, 1))
        self.__values_norm = values_norm
        self.__log_const = log_const

    @profile
    def init_distrib_idx(self, distrib, idx=None):
        assert isinstance(distrib, DistribCat)
        x = distrib.get_values_norm()
        if idx is None:
            # initialize prior and thus average over all cases
            assert self._k1 == 1
            values_norm = np.mean(x, axis=0, keepdims=True)
        else:
            # select cases idx
            x_idx = x[idx,:]
            assert self._k1 == x_idx.shape[0]
            values_norm = np.copy(x_idx)
        self.__add_random(values_norm, 0.1)
        self.init_data(values_norm)

    @profile
    def update_prior(self, distrib, N_0):
        # self: 1 x k2
        # distrib: N x k2
        M2 = distrib.get_values_norm()
        M2_t = M2[:,:,np.newaxis].transpose((2, 1, 0))  # 1  x k2 x N
        values_norm = self.__normalize_prior(M2_t, N_0)
        self.init_data(values_norm)

    @profile
    def update_cpd(self, distrib1, distrib2, N_0, exclude_self=False):
        # self=x: k1 x k2
        # M1: N x k1
        # M2: N x k2
        M1 = distrib1.get_values_norm()
        M2 = distrib2.get_values_norm()

        M1_t = M1[:,:,np.newaxis].transpose((1, 2, 0))  # k1 x 1  x N
        M2_t = M2[:,:,np.newaxis].transpose((2, 1, 0))  # 1  x k2 x N
        if not exclude_self:
            x = self.get_values_norm()
            x_t  =  x[:,:,np.newaxis]                       # k1 x k2 x 1
            q = M1_t * M2_t * x_t  # k1 x k2 x N
        else:
            q = M1_t * M2_t  # k1 x k2 x N
        q /= np.sum(q, axis=(0, 1), keepdims=True)
        values_norm = self.__normalize_prior(q, N_0)
        assert np.all(np.isfinite(values_norm))
        self.init_data(values_norm)

    def as_str(self, num=None):
        values_norm = cut_max(self.__values_norm, num)
        log_const = cut_max(self.__log_const, num)
        result = 'values_norm=\n{}'.format(values_norm)
        result +='\nlog_const={}\n'.format(log_const.ravel())
        return result

    def plot(self):
        fig, ax = plt.subplots()
        cax = ax.imshow(self.__values_norm, interpolation='nearest', aspect='auto', vmin=0, vmax=1)
        fig.colorbar(cax)

    @profile
    def __normalize_convex(self):
        self.__log_const += np.log(normalize_convex(self.__values_norm, axis=1))

    @staticmethod
    @profile
    def __normalize_prior(q, N_0):
        # see Murphy (2012) Machine Learning - A Probabilistic Perspective, p.80 Eq. (3.47)
        # uniform alpha: alpha_k = alpha -> alpha_0 = K
        # N_0 = K(a-1) (pseudo counts)
        # q: k1 x k2 x N
        K = q.shape[1]
        k1 = q.shape[0]
        values_biased = np.sum(q, axis=2, keepdims=True).squeeze(axis=(2,))  # k1 x k2
        N = np.sum(values_biased, axis=1, keepdims=True)  # k1 x 1
        values_uniform = np.ones((1, K)) / K
        values = values_biased * (1 / (N + N_0))
        values += values_uniform * (N_0 / (N + N_0))

        # # N_0 = K (alpha-1) -> alpha = (N_0 / K) + 1
        # # A pseudo-count of N_0 corresponds to a Dirichlet prior with alpha = (N_0 / K) + 1
        # alpha = (N_0 / K) + 1
        # alpha_all = np.ones((K,))*alpha
        # lklhd_prior = dirichlet.logpdf(values.T, alpha_all).reshape(k1, 1)
        # lklhd_mode = dirichlet.logpdf(values_uniform.T, alpha_all)
        # lklhd_prior -= lklhd_mode

        return values

    @staticmethod
    @profile
    def __samples2distrib(samples, k):
        N = samples.shape[0]

        isnan = np.isnan(samples)
        notnan = np.logical_not(isnan)
        x_real = samples[notnan]

        assert np.all(np.equal(np.mod(x_real, 1), 0))  # all values must be integer
        assert np.all((x_real >= 0) & (x_real < k))  # all values must be within (0, k-1)

        values_norm = np.zeros((N, k))

        ind1 = np.arange(N)[notnan]
        ind2 = x_real.astype(int)
        values_norm[ind1, ind2] = 1.
        values_norm[isnan, :] = 1. / k
        return values_norm

    @staticmethod
    @profile
    def __distrib2samples_max(values_norm):
        idx = np.argmax(values_norm, axis=1)
        samples = idx[:, np.newaxis]
        return samples

    @staticmethod
    @profile
    def __distrib2samples_exp(values_norm):
        k = values_norm.shape[1]
        zero2k = np.arange(k)[:, np.newaxis]  # k x 1
        samples = values_norm.dot(zero2k)  # N x 1
        return samples

    @staticmethod
    @profile
    def __add_random(dist, perc):
        k1 = dist.shape[0]
        k2 = dist.shape[1]
        dist_rand = np.random.rand(k1, k2)
        normalize_convex(dist_rand, axis=1)
        dist *= (1-perc)
        dist += perc * dist_rand

class DistribGauss(Distrib):
    def __init__(self, k1):
        super(DistribGauss, self).__init__(k1, 1)
        self.__mu = None  # k1 x 1 mean
        self.__std = None  # k1 x 1 standard deviation

    def get_dof(self):
        dof = self._k1 * 2
        return dof

    def copy(self):
        distrib = DistribFactory.empty(self._k1, self._k2)
        distrib.init_data(self.__mu.copy(), self.__std.copy())
        return distrib

    def get_mu(self):
        return self.__mu

    def get_std(self):
        return self.__std

    @profile
    def __prod_single(self, distrib):
        mu1 = self.__mu
        std1 = self.__std
        mu2 = distrib.get_mu()
        std2 = distrib.get_std()
        assert has_equiv_shape(mu1, mu2.shape)
        assert mu1.shape[0] == 1 or mu2.shape[0] == 1 or mu1.shape[0] == mu2.shape[0]
        assert mu1.shape[1] == 1 or mu2.shape[1] == 1 or mu1.shape[1] == mu2.shape[1]
        if mu1.shape[0] == 1 and mu2.shape[0] > 1:
            mu1 = np.tile(mu1, (mu2.shape[0],1))
            std1 = np.tile(std1, (mu2.shape[0],1))

        # currently, I did not implement the true product of two Gaussian distributions, only the special cases for
        # either std=Inf or std=0
        # possible cases:
        # std1=Inf -> mu=mu2, std=std2
        # std1=0 -> mu=mu1, std=std1, assert std2!=0
        # 0<std1<Inf -> std2=Inf -> mu=mu1, std=std1
        #               std2=0   -> mu=mu2, std=std2
        #               assert std==Inf or 0
        is_inf1 = np.isinf(std1).ravel()
        is_zero1 = (std1 == 0).ravel()
        is_real1 = ~(is_inf1 | is_zero1)

        is_inf2 = np.isinf(std2).ravel()
        is_zero2 = (std2 == 0).ravel()
        is_real2 = ~(is_inf2 | is_zero2)

        assert np.all(~(is_zero1 & is_zero2))
        assert np.all(~(is_real1 & is_real2))

        # all right hand sides use fancy indexing, thus we do not need to explicitly copy the slices.
        if mu1.shape[0] > 1 and mu2.shape[0] == 1:
            mu1[is_inf1,:] = mu2[0,:]
            std1[is_inf1,:] = std2[0,:]
            if is_zero2:
                mu1[:,:] = mu2[0,:]
                std1[:,:] = std2[0,:]
        else:
            mu1[is_inf1,:] = mu2[is_inf1,:]
            std1[is_inf1,:] = std2[is_inf1,:]
            mu1[is_zero2,:] = mu2[is_zero2,:]
            std1[is_zero2,:] = std2[is_zero2,:]

        # we need the explicit assignment, since the francy indexing above created copies
        self.__mu = mu1
        self.__std = std1

    @profile
    def prod(self, distribs):
        for distrib in distribs:
            self.__prod_single(distrib)

    def dot(self, distrib):
        # self: N x 1 (Gauss)
        # distrib: 1 x k2 (Gauss)
        # out: N x k2 (Cat)
        # should never be called since we never calculate an outgoing alpha from a Gaussian node
        assert False

    @profile
    def dot_transpose(self, distrib):
        # self: N x 1 (Gauss)
        # distrib: k1 x 1 (Gauss)
        # out: N x k1 (Cat)
        assert isinstance(distrib, DistribGauss)

        k1 = distrib.get_k1()
        mu = distrib.get_mu().reshape((1, -1))  # 1 x k1
        std = distrib.get_std().reshape((1, -1))  # 1 x k1

        N = self.get_k1()
        data_m = self.get_mu()  # N x 1
        data_s = self.get_std()  # N x 1

        values_log = np.empty((N, k1))

        # handle unobserved data
        isinf = np.isinf(data_s).ravel()
        if np.any(isinf):
            isnotinf = np.logical_not(isinf)
            # scipy_d = norm(mu, std)  # scipy normal distribution
            # values_log[isnotinf, :] = scipy_d.logpdf(data_m[isnotinf, :])
            # this is ~100 times faster than the scipy implementation
            values_log[isnotinf, :] = norm_logpdf(data_m[isnotinf, :], mu, std)
            values_log[isinf, :] = 0
        else:
            values_log = norm_logpdf(data_m, mu, std)

        log_const = normalize_convex_log(values_log, axis=1)
        distrib_res = DistribCat(N, k1)
        distrib_res.init_data_raw(values_log, log_const)
        return distrib_res

    @profile
    def init_samples(self, samples, k):
        assert k == 1
        mu, std = self.__samples2distrib(samples)
        self.init_data(mu, std)

    @profile
    def extract_samples(self, mode):
        samples = self.__mu
        return samples

    @profile
    def init_uniform(self):
        self.__mu = np.zeros((self._k1, 1))
        self.__std = np.empty((self._k1, 1))
        self.__std.fill(np.inf)

    @profile
    def init_random(self):
        mu = np.random.randn(self._k1, 1) * 0.05
        std = (np.abs(np.random.randn(self._k1, 1)) * 0.05) + 1
        self.init_data(mu, std)

    @profile
    def init_data_raw(self, mu, std):
        self.__mu = mu
        self.__std = std

    @profile
    def init_data(self, mu, std=None):
        assert has_equiv_shape(mu, (self._k1, 1))
        if std is not None:
            assert has_equiv_shape(std, (self._k1, 1))
        self.__mu = mu
        self.__std = std

    @profile
    def init_distrib_idx(self, distrib, idx=None):
        assert isinstance(distrib, DistribGauss)
        x = distrib.get_mu()
        if idx is None:
            # initialize prior and thus average over all cases
            mu = np.nanmean(x, axis=0, keepdims=True)
        else:
            # select cases idx
            mu = x[idx, :]
            idx_nan = np.isnan(mu)
            if np.any(idx_nan):
                # we need to randomly select new values for all NaNs
                idx_good = np.ones_like(idx, dtype=bool)
                idx_good[idx, :] = False
                idx_good[np.isnan(x)] = False
                x_good = x[idx_good, :]
                num_nan = np.count_nonzero(idx_nan)
                mu[idx_nan] = np.random.choice(x_good, num_nan, replace=False)
            mu = np.copy(mu)  # make sure to not overwrite data

        std = np.empty_like(mu)
        std.fill(np.asscalar(np.nanstd(x)))
        self.init_data(mu, std)

    @profile
    def init_distrib_percentiles(self, distrib):
        assert isinstance(distrib, DistribGauss)
        x = distrib.get_mu()
        k = self._k1
        percentiles = 100 * (np.array(range(k)) + 0.5) / k
        mu = np.nanpercentile(x, percentiles)[:,np.newaxis]
        mu = np.copy(mu)  # make sure to not overwrite data

        std = np.empty_like(mu)
        std.fill(np.asscalar(np.nanstd(x)))
        self.init_data(mu, std)

    @profile
    def init_distrib_equidistant(self, distrib, std_rel=1):
        assert isinstance(distrib, DistribGauss)
        x = distrib.get_mu()
        k = self._k1
        percentiles = 1 * (np.array(range(k)) + 0.5) / k
        min_x = np.min(x)
        max_x = np.max(x)
        range_x = max_x - min_x
        mu = min_x + (percentiles * range_x)
        mu = mu[:,np.newaxis]

        std = np.empty_like(mu)
        std.fill(np.asscalar(std_rel * np.nanstd(x) / k))
        self.init_data(mu, std)

    @profile
    def init_distrib_equidistant_rand(self, distrib, std=1, std_rand=0.5):
        self.init_distrib_equidistant(distrib, std)
        x = distrib.get_mu()
        k = self._k1
        self.__mu += np.random.randn(k, 1) * (std_rand * np.nanstd(x) / k)

    # @profile
    # def update_prior(self, distrib, N_0):
    #     # self: 1 x k2=1
    #     # distrib: N x k2=1
    #     mu = distrib.get_mu()
    #     std = distrib.get_std()
    #     assert np.all(std == 0)
    #
    #     self.init_data(np.mean(mu).reshape(1,1), np.std(mu).reshape(1,1))

    @profile
    def update_cpd(self, cat, gauss, N_0, exclude_self=False):
        # self: k1 x 1 (mean + std)
        # cat: N x k1
        # gauss: N x 1 (mean + std)

        if not exclude_self:
            # SPEEDUP possibility: this cat message has been already calculated during inference; could reuse that value
            message_GaussToCat = gauss.dot_transpose(self)
            cat.prod([message_GaussToCat])

        q = cat.get_values_norm().T  # k1 x N
        data_m = gauss.get_mu().T  # 1 x N
        data_s = gauss.get_std().T  # 1 x N

        # ignore unobserved data
        isnotinf = ~np.isinf(data_s).ravel()
        if not np.all(isnotinf):
            q = q[:,isnotinf]
            data_m = data_m[:,isnotinf]
            data_s = data_s[:,isnotinf]

        N_k = np.sum(q, axis=1, keepdims=True)  # k1 x 1
        N_safe = N_k.copy()
        N_safe[N_k == 0] = 1
        mu = q.dot(data_m.T) / N_safe  # k1 x 1
        d_data_sq = (data_m - mu)**2  # k1 x N
        var_biased = np.sum(q * d_data_sq, axis=1, keepdims=True) / (N_k + N_0)
        var_all = np.var(data_m, keepdims=True)
        var = var_biased + (var_all * N_0 / (N_k + N_0))
        std = np.sqrt(var)

        # # A pseudo-count of N_0 and prior variance var_all is equivalent of an inverse-gamma prior with
        # alpha = N_0 / 2
        # scale = var_all * ((N_0 / 2) + 1)  # (=beta)
        # lklhd_prior = invgamma.pdf(var, alpha, 0, scale)
        # lklhd_mode = invgamma.pdf(var_all, alpha, 0, scale)
        # lklhd_prior -= lklhd_mode

        assert np.all(np.isfinite(mu))
        assert np.all(np.isfinite(std))
        assert np.all(std>0)

        self.init_data(mu, std)

    def as_str(self, num=None):
        mu = cut_max(self.__mu, num)
        std = cut_max(self.__std, num)
        result = 'mu={}'.format(mu)
        result += '\nstd={}\n'.format(std)
        return result

    def visual_get_kde(self):
        mu = self.__mu.ravel()
        density = gaussian_kde(mu)
        xs = np.linspace(mu.min(),mu.max(),200)
        density.covariance_factor = lambda : .25
        density._compute_covariance()
        return xs, density(xs)

    def visual_get_density(self, xs):
        xs_distrib = DistribFactory.from_samples(xs, 1)  # DistribGauss
        dens_distrib = xs_distrib.dot_transpose(self)
        dens = dens_distrib.get_values()
        return dens


    @staticmethod
    @profile
    def __samples2distrib(samples):
        # continuous case
        assert np.issubsctype(samples, np.float)
        N = samples.shape[0]
        mu = samples.copy().reshape((N, 1))
        std = np.zeros_like(mu)

        # handle NaNs
        isnan_mu = np.isnan(mu)
        std[isnan_mu] = np.inf
        mu[isnan_mu] = np.nan

        return mu, std


class DistribFactory(object):
    @staticmethod
    def empty(k1, k2):
        # k2 = 1 => DistribGauss
        # k2 > 2 => DistribCat
        # k1 = 1 => Prior or N=1
        # k1 > 1 => CPD or N>1
        if k2 == 1:
            # DistribGauss
            distrib = DistribGauss(k1)
        else:
            # DistribCat
            distrib = DistribCat(k1, k2)
        return distrib

    @classmethod
    def random(cls, k1, k2):
        distrib = cls.empty(k1, k2)
        distrib.init_random()
        return distrib

    @classmethod
    def uniform(cls, k1, k2):
        distrib = cls.empty(k1, k2)
        distrib.init_uniform()
        return distrib

    @classmethod
    def from_samples(cls, samples, k):
        has_equiv_shape(samples, (None, 1))
        N = samples.shape[0]
        distrib = cls.empty(N, k)
        distrib.init_samples(samples, k)
        return distrib

    """
    Init (id_parent, id_child) distribution:
    - needs x (beta) of all children
    - init prior of id_parent
    - init edge of (id_parent, id_child)

    Init Cat prior / CPD:
    - random

    Init Gauss prior:
    - use data from actual node

    Init Gauss CPD:
    - use data from all children with common parent (only observed nodes!)
    """

class ParameterUpdate(GraphManipulator):
    """This class handles the parameter update step."""
    def __init__(self, graph, inference):
        super(ParameterUpdate, self).__init__(graph)
        assert isinstance(inference, BeliefPropagation)
        # requires prop_graph: N
        # requires prop_node: x, prior
        # requires prop_edge: message_1to2, message_2to1
        # updates prop_edge: distrib
        self._register_properties()
        # requires get_messages_prod() from the inference algorithm
        self._inference = inference
        self.N_0 = 1
        self.print_distrib = False
        # pseudo count for categorical/variance prior
        # (sensible range: 0 <= N_0 < Inf)
        # N_0 = 0: no prior, fallback to ML solution (warning: might lead to infinite variances!)

    @profile
    def _run(self, id_root, recursive):
        self.__update_pass(id_root, id_except=None, recursive=recursive)

    @profile
    def __update_pass(self, id_node, id_except, recursive):
        g = self._graph
        func_node = lambda id, node: self.__update_prior(id, node)
        func_edge = lambda id1, id2, edge: self.__update_distrib(id1, id2, edge)
        g.func_iter(id_node, recursive, func_node, func_edge)

    @profile
    def __update_prior(self, id_node, node):
        if node.prior is not None:
            i = self._inference
            messages_prod = i.get_messages_prod(id_node, id_except=None)
            if self.print_distrib:
                self._print('update prior of id_node={}'.format(id_node))
                #self._print(' BEFORE: ' + node.prior.as_str())
            node.prior.update_prior(messages_prod, N_0=self.N_0)
            # if self.print_distrib:
            #     self._print(' AFTER: ' + node.prior.as_str())
            node.q_dirty = True

    @profile
    def __update_distrib(self, id_node1, id_node2, edge):
        i = self._inference
        messages_prod1 = i.get_messages_prod(id_node1, id_except=id_node2)  # M1: N x k1
        messages_prod2 = i.get_messages_prod(id_node2, id_except=id_node1)  # M2: N x k2
        distrib = edge.distrib # distrib: k1 x k2
        if self.print_distrib:
            self._print('update distrib of id_nodes=({},{})'.format(id_node1, id_node2))
            #self._print(' BEFORE: ' + distrib.as_str())
        distrib.update_cpd(messages_prod1, messages_prod2, N_0=self.N_0)
        # if self.print_distrib:
        #     self._print(' AFTER: ' + distrib.as_str())
        self._graph.node(id_node1).q_dirty = True
        self._graph.node(id_node2).q_dirty = True


class ParameterLearning(GraphManipulator):
    """This class handles the parameter update step."""
    def __init__(self, graph, data, inference, parameter_update):
        super(ParameterLearning, self).__init__(graph)
        assert isinstance(data, Data)
        assert isinstance(inference, BeliefPropagation)
        assert isinstance(parameter_update, ParameterUpdate)
        self._prop_graph = set()
        self._prop_node = set()
        self._prop_edge = set()
        self._register_properties()
        # requires run() from the inference algorithm and parameter_updates
        self._data = data
        self._inference = inference
        self._parameter_update = parameter_update
        self.lklhd_mindiff = 1e-3
        self.count_max = 100
        self.restarts = 2
        self.print_restarts = True
        self.print_every = 10
        self.print_em = False
        self.restart_recursive = True
        self.draw_density = False
        self._copy_prop_node = {'prior'}
        self._copy_prop_edge = {'distrib'}

    @profile
    def _run(self, id_root, recursive):
        result = self._find_best_restart(id_root, recursive)
        return result

    @profile
    def find_best_k(self, id_root):
        g = self._graph
        K = 2
        step_log = -1
        increase = True
        K_max = 64

        K_best = None
        bic_best = -np.inf
        cont = True

        print_restarts_bak = self.print_restarts
        self.print_restarts = False

        while K <= K_max:
            g.set_k(id_root, K)
            self._data.run(id_root, recursive=False)
            lklhd = self._find_best_restart(id_root, recursive=False)

            # calculate BIC
            dof = g.get_dof(id_root)
            bic = g.N * lklhd - (dof * np.log(g.N) / 2)
            self._print('binary search: K={}, BIC={}, lklhd={}, dof={}, N={}'.format(K, bic, lklhd, dof, g.N))

            if bic > bic_best:
                K_best = K
                bic_best = bic
                forward = True
            else:
                increase = False
                forward = False

            step_log += 1 if increase else -1
            if step_log < 0:
                break
            step = np.power(2,step_log)
            K += step if forward else -step
            pass

        self._print('binary search: K_best={}, BIC_best={}'.format(K_best, bic_best))
        g.set_k(id_root, K_best)
        self._data.run(id_root, recursive=False)
        self.print_restarts = print_restarts_bak
        lklhd = self._find_best_restart(id_root, recursive=False)
        return lklhd

    @profile
    def _find_best_restart(self, id_root, recursive):
        if not recursive:
            restart_recursive = False
        else:
            restart_recursive = self.restart_recursive

        if self.restarts > 0:
            lklhd = self.__expectation_maximization(id_root, restart_recursive)

            lklhd_best = lklhd
            num_best = 0
            lklhds = [lklhd]
            self.__parameters_copy(id_root, restart_recursive)

            for num in range(1, self.restarts+1):
                self._data.run(id_root, restart_recursive)
                lklhd = self.__expectation_maximization(id_root, restart_recursive)
                lklhds.append(lklhd)
                if lklhd > lklhd_best:
                    lklhd_best = lklhd
                    num_best = num
                    self.__parameters_copy(id_root, restart_recursive)

            lklhd = lklhd_best
            self.__parameters_recover_copy(id_root, restart_recursive)
            lklhd2 = self._inference.run(id_root, restart_recursive)
            assert np.allclose(lklhd, lklhd2)  # just to double-check

            if self.print_restarts:
                self._print('EM restarts: id_root={}, best of {} is {} with lklhd={} (max diff={}); recursive={}; restart_recursive={}'.format(id_root, len(lklhds), num_best+1, lklhd, lklhd - min(lklhds), recursive, restart_recursive))

        # finally do a full recursive run
        if ((not restart_recursive) and recursive) or (self.restarts == 0):
            lklhd = self.__expectation_maximization(id_root, recursive)

        return lklhd

    def __parameters_copy(self, id_root, recursive):
        g = self._graph
        g.properties_func(id_root, GraphObject.copy, recursive, self._copy_prop_node, self._copy_prop_edge)

    def __parameters_recover_copy(self, id_root, recursive):
        g = self._graph
        #g.properties_func(id_root, GraphObject.recover_copy, recursive, self.__copy_prop_node, self.__copy_prop_edge)
        def func_node(id, node):
            GraphObject.recover_copy(node, self._copy_prop_node)
            node.q_dirty = True
        def func_edge(id1, id2, edge):
            GraphObject.recover_copy(edge, self._copy_prop_edge)
        g.func_iter(id_root, recursive=recursive, func_node=func_node, func_edge=func_edge)

    @profile
    def __expectation_maximization(self, id_root, recursive):
        lklhd_last = self._inference.run(id_root, recursive)
        if self.print_em:
            self._print('id_root={}, recursive={}, lklhd={}'.format(id_root, recursive, lklhd_last))
            if self.draw_density:
                self.__draw_density(id_root)
        count = 1
        continue_condition = True
        while continue_condition:
            self._parameter_update.run(id_root, recursive)
            lklhd = self._inference.run(id_root, recursive)
            lklhd_diff = lklhd - lklhd_last

            condition_lklhd = lklhd_diff > self.lklhd_mindiff
            condition_count = count < self.count_max
            continue_condition = condition_count & condition_lklhd

            if self.print_em:
                if (~continue_condition) | (count % self.print_every == 0) | (lklhd_diff < 0):
                    self._print('count={}, lklhd={}, lklhd_diff={}'.format(count, lklhd, lklhd_diff))
                    if self.draw_density:
                        self.__draw_density(id_root)
                if not condition_lklhd:
                    self._print('Terminate because lklhd_diff <= mindiff.')
                if not condition_count:
                    self._print('Terminate because count_max is reached.')

            count += 1
            lklhd_last = lklhd

        if self.draw_density:
            plt.cla()
            #self.__draw_density(id_root)

        return lklhd

    def __draw_density(self, id_root):
        # draw the density of the first child, if it is continuous
        children = [id for id in self._graph.out_edges_iter(id_root)]
        if len(children) > 0 and self._graph.node(children[0]).k == 1:
            self._inference.visual_gauss_edge(id_root, children[0])
            plt.gca().set_prop_cycle(None)


class StructureUpdate(GraphManipulator):
    """Updates graph structure."""
    def __init__(self, graph, data, inference, parameter_learning):
        super(StructureUpdate, self).__init__(graph)
        assert isinstance(data, Data)
        self._data = data
        assert isinstance(inference, BeliefPropagation)
        self._inference = inference
        assert isinstance(parameter_learning, ParameterLearning)
        self._parameter_learning = parameter_learning
        self.required_axes = 2  # for lklhd_pot_diff and lklhd_pot_diff_siblings
        # updates graph structure
        self.k_default = 10  # default k for new hidden nodes
        self.N_0 = 1  # should be the same N_0 as in ParameterUpdate, should find better solution to share the same value
        self.__lklhd_pot_diff = np.zeros((0, 0))
        self.lklhd_pot_diff_root = np.zeros((0,))
        self.lklhd_pot_diff_siblings = np.zeros((0, 0))
        self.lklhd_pot_diff_dirty = np.zeros((0,), dtype=bool)
        self.is_observed = np.zeros((0,))
        self.draw_figure = False
        self.show_value_text = False
        self.keep_old_parents = False
        self.max_of_cp_and_pc = False
        self.balance_k = False  # balance the lklhd_pot_diff value according K of each node; does not seem to work well
        self.find_best_k = False

    @profile
    def add_gaussian_parents(self):
        # add gaussian parents
        nodes = self._graph.nodes()
        for id_node in nodes:
            self.__gaussian_add_parent(id_node)

    @profile
    def remove_gaussian_parents(self):
        g = self._graph
        num_removed = 0
        # can't use iterator, since nodes are deleted while iterating
        nodes_data = g.nodes(data=True)
        for id_node, node in nodes_data:
            if node.k == 1:
                id_parent = g.get_parent(id_node)
                if self._is_single_continuous_parent(id_parent):
                    g.remove_edge(id_parent, id_node)
                    g.remove_node(id_parent)
                    num_removed += 1
        self._print('removed {} gaussian parents'.format(num_removed))

    @profile
    def run(self):
        # TODO:
        # - proper update for removing edges: we need to calculate the lklhd_diff for the siblings of the
        # parent_old as regularization condition
        # - lazy calculation of lklhd_pot_diff and lklhd_pot_diff_siblings (V)
        # - speedup of get_messages_prod()
        # - speedup of update_lklhd_pot_diff!!!
        i = self._inference
        g = self._graph
        lklhd_old = i.get_lklhd_all()
        # self._print('lklhd_pot_diff_dirty={}'.format(np.where(self.lklhd_pot_diff_dirty)[0]))
        self.update_lklhd_pot_diff()

        if self.draw_figure:
            g.draw_axes()
            self.draw_axes()
            plt.show()

        id_parent, id_child, create_node = self.select_update()
        if id_parent is not None:
            same_tree = g.get_root(id_parent) == g.get_root(id_child)
            pot_diff = self.__lklhd_pot_diff[id_parent, id_child]
        else:
            same_tree = True
            pot_diff = None

        self._print("select=({},{}); create_node={}; same_tree={}; pot_diff={}".format(id_parent, id_child, create_node, same_tree, pot_diff))
        id_roots_changed = set()
        if (id_parent is not None) or (id_child is not None):
            assert id_child is not None
            # update dirty
            self._update_lklhd_pot_diff_dirty(id_parent)
            self._update_lklhd_pot_diff_dirty(id_child)

            #pot_diff = self.lklhd_pot_diff[id_parent, id_child]
            # this fails if (id_parent = None) and child.k == 1
            child_is_gauss = g.node(id_child).k == 1
            if id_parent is None and child_is_gauss:
                distrib_new = None
            else:
                _, _, distrib_new = self.__calc_lklhd_pot(id_parent, id_child)

            id_roots_changed |= self.__remove_old_parent(id_child)
            if not create_node:
                # we need to add an edge to the new parent
                self.__add_edge_and_init(id_parent, id_child, distrib_new )
            else:
                assert id_parent is not None  # should never happen that we create a new node and only add a single child
                # first remove the possible old_parent of parent
                id_roots_changed |= self.__remove_old_parent(id_parent)
                # we need to create a new hidden node and two new edges
                id_parent_2nd = g.add_node(self.k_default)
                g.add_edge(id_parent_2nd, id_parent)
                g.add_edge(id_parent_2nd, id_child)
                # TODO: do here non-recursive parameter optimization; then we should be able to guarantee the lklhd_diff in the assert below!
                if self.find_best_k:
                    self._parameter_learning.find_best_k(id_parent_2nd)
                else:
                    self._data.run(id_parent_2nd, recursive=False)

            id_roots_changed |= {g.get_root(id_child)}
            self._print('id_roots_changed={}'.format(id_roots_changed))
            i.run(id_roots_changed)  # run inference on all changed trees
            lklhd = i.get_lklhd_all()
            # double check that out predicted difference is close to the real difference
            lklhd_diff = lklhd - lklhd_old
            if (not create_node) and (not same_tree):
                # if we create a new node, then the increase is only guaranteed after parameter optimization
                # assert np.isclose(lklhd_diff, pot_diff) or (lklhd_diff - pot_diff > 0)
                if not (np.isclose(lklhd_diff, pot_diff) or (lklhd_diff - pot_diff > 0)):
                    warnings.warn("lklhd_diff - pot_diff = {}".format(lklhd_diff - pot_diff))
                    # probably happens when an old_parent is removed and a new edge from parent_parent to siblig is created:
                    # the edge nodes are within the same tree and thus the true distribution needs to be obtained by sending a message between them -> costly!
        else:
            lklhd = None

        return lklhd, id_roots_changed

    @profile
    def __add_edge_and_init(self, id_node1, id_node2, distrib_new):
        # TODO: implemet this function and use it all times a new edge is added.
        # Maybe: use similar function for initializing new node.
        g = self._graph
        i = self._inference
        if id_node1 is not None:
            g.add_edge(id_node1, id_node2)
            g.edge(id_node1, id_node2).distrib = distrib_new
            # init the edge messages, for the rare case that afterwards another __remove_old_parent() is called that
            # uses the same node
            i.run(id_roots=id_node1, recursive=False)
        else:
            # id_node2 might be gaussian, then we need to add a new parent
            id_parent_new = self.__gaussian_add_parent(id_node2)
            if id_parent_new is None:
                # if not, then we can initialize with distrib_new
                g.node(id_node2).prior = distrib_new
            else:
                self._data.run(id_roots=id_parent_new, recursive=False)

    @profile
    def __remove_old_parent(self, id_node):
        g = self._graph
        id_parent = g.get_parent(id_node)
        id_roots_changed = set()
        if id_parent is not None:
            # we need to remove the edge to the old parent first
            g.remove_edge(id_parent, id_node)
            if self.keep_old_parents:
                # nothing else to do, since all parents are kept
                return {g.get_root(id_parent)}

            out_degree_parent = g.out_degree(id_parent)
            id_parent_parent = g.get_parent(id_parent)
            if id_parent_parent is not None:
                parent_has_parent = True
            else:
                parent_has_parent = False
            if out_degree_parent == 1:
                id_sibling = [id for id in g.out_edges_iter(id_parent)][0]
                sibling_continuous = g.node(id_sibling).x is not None
            else:
                id_sibling = None
                sibling_continuous = False
            parent = g.node(id_parent)

            if out_degree_parent <= 1 and not (out_degree_parent==1 and not parent_has_parent and sibling_continuous):
                # case 1: out_degree_parent==0 & parent_has_parent -> (1a) delete edge between parent_parent and parent (1b) delete node parent
                # case 2: out_degree_parent==0 & ~parent_has_parent -> (2a) delete node parent
                # case 3: out_degree_parent==1 & parent_has_parent -> (3a) delete edge between parent_parent and parent, (3b) remove edge between parent and sibling, (3c) add edge between parent_parent and sibling (3d) delete node parent
                # (V) case 4: out_degree_parent==1 & ~parent_has_parent & sibling_continuous -> do nothing
                # case 5: out_degree_parent==1 & ~parent_has_parent & ~sibling_continuous -> (4a) delete edge between parent and sibling and (4b) delete node parent
                if parent_has_parent:
                    # (1a, 3a) delete edge between parent_parent and parent
                    if out_degree_parent==1:
                        g.remove_edge(id_parent_parent, id_parent)
                        # (3b) remove edge between parent and sibling
                        _, _, distrib_new = self.__calc_lklhd_pot(id_parent_parent, id_sibling)
                        g.remove_edge(id_parent, id_sibling)
                        # (3c) add edge between parent_parent and sibling
                        self.__add_edge_and_init(id_parent_parent, id_sibling, distrib_new)
                        id_roots_changed |= {g.get_root(id_parent_parent)}
                    else:
                        # recursively remove the old parent
                        id_roots_changed |= self.__remove_old_parent(id_parent)
                else:
                    if out_degree_parent==1:
                        assert not sibling_continuous  # just to double check, this case should have been excluded in the first 'if'
                        # (4a) delete edge between parent and sibling
                        g.remove_edge(id_parent, id_sibling)
                        id_roots_changed |= {id_sibling}
                # (1b, 2a, 3d, 4b) delete node parent
                g.remove_node(id_parent)
            else:
                id_roots_changed |= {g.get_root(id_parent)}
        return id_roots_changed

    @profile
    def __gaussian_add_parent(self, id_node):
        g = self._graph
        node = g.node(id_node)
        id_parent = None
        if node.k == 1:
            # only for gaussian nodes
            if not g.has_parent(id_node):
                # add new hidden parent if node does not have one
                id_parent = self.__add_node_with_children([id_node])
        return id_parent

    @profile
    def __add_node_with_children(self, id_children):
        g = self._graph
        id_node = g.add_node(self.k_default)
        for id_child in id_children:
            g.add_edge(id_node, id_child)
        return id_node

    @profile
    def add_nodes_from_adjlist(self, adjlist):
        # adjlist must obey the rules:
        # - already existing nodes must have no children
        # - new nodes must have ids in increasing order
        # - ids are only allowed as child if they have been defined before

        self._print('adding {} nodes from adjacency list'.format(len(adjlist)))
        ide2id = {}

        for line in adjlist:
            ide_node = line[0]
            ide_children = line[1:]
            # these checks assume a tree structure
            if self._graph.has_node(ide_node):
                # if node already exists, then it must be one of the data input ones and thus doesn't have children
                assert not ide_children
                ide2id[ide_node] = ide_node
            else:
                # if node does not exist yet, then it should be one of the hidden nodes and thus must have children
                if ide_children:
                    # map ide to id
                    id_children = [ide2id[ide] for ide in ide_children if ide2id[ide] is not None]
                    id_node = self.__add_node_with_children(id_children)
                    ide2id[ide_node] = id_node
                else:
                    # otherwise ignore node
                    ide2id[ide_node] = None
                    self._print('ide_node={} does not have children, leave it out'.format(ide_node))
                # # assert id consistency
                # assert ide_node == id_node
        pass

    @profile
    def update_lklhd_pot_diff(self):
        g = self._graph
        i = self._inference

        id_node_next = g.get_id_node_next()
        self.__lklhd_pot_diff = expand_array(self.__lklhd_pot_diff, (id_node_next, id_node_next))
        self.lklhd_pot_diff_root = expand_array(self.lklhd_pot_diff_root, (id_node_next,))
        self.lklhd_pot_diff_siblings = expand_array(self.lklhd_pot_diff_siblings, (id_node_next, id_node_next))
        self.is_observed = expand_array(self.is_observed, (id_node_next,))
        self.lklhd_pot_diff_dirty = expand_array(self.lklhd_pot_diff_dirty, (id_node_next,), True)

        pl = ProgressLine(prefix=self._print_prefix() + "update_lklhd_pot_diff ")
        for id_child in range(id_node_next):
            # case of new root node (i.e. removing the actual parent)
            if self.lklhd_pot_diff_dirty[id_child]:
                if self.is_allowed_root(id_child):
                    diff_root, _ = self.calc_lklhd_pot_diff(None, id_child)
                else:
                    diff_root = np.nan
                self.lklhd_pot_diff_root[id_child] = diff_root

            # case of adding new edge
            if self.is_allowed_child(id_child):
                for id_parent in range(id_node_next):
                    if self.lklhd_pot_diff_dirty[id_parent] or self.lklhd_pot_diff_dirty[id_child]:
                        # self._print('update lklhd_pot_diff({},{})'.format(id_parent, id_child))
                        if self.is_allowed(id_parent, id_child):
                            diff, message_child2parent_pot = self.calc_lklhd_pot_diff(id_parent, id_child)
                            diff_siblings = self.calc_lklhd_pot_diff_siblings(id_parent, message_child2parent_pot)
                            if diff_siblings.size == 0:
                                diff_siblings_min = np.nan # nan for no siblings
                            else:
                                diff_siblings_min = np.min(diff_siblings)

                            if g.has_parent(id_child):
                                if self.is_allowed_root(id_child):
                                    # we need to add the diff_root, since the old parent will be removed
                                    diff_root = self.lklhd_pot_diff_root[id_child]
                                    assert not np.isnan(diff_root)
                                    diff += diff_root
                        else:
                            diff = np.nan
                            diff_siblings_min = np.nan
                        self.__lklhd_pot_diff[id_parent, id_child] = diff
                        self.lklhd_pot_diff_siblings[id_parent, id_child] = diff_siblings_min
            else:
                self.__lklhd_pot_diff[:, id_child] = np.nan
                self.lklhd_pot_diff_siblings[:, id_child] = np.nan

            perc = int(np.round(100 * (id_child + 1) / id_node_next))
            pl.progress(perc)

        pl.finish()
        # fill in the is_observed variable
        #parents = g.nodes_iter()
        for id_parent in range(id_node_next):
            if g.has_node(id_parent):
                if g.node(id_parent).x is not None:
                    is_observed = True
                else:
                    is_observed = False
            else:
                is_observed = np.nan
            self.is_observed[id_parent] = is_observed

        self.lklhd_pot_diff_dirty = np.zeros_like(self.lklhd_pot_diff_dirty, dtype=bool)

    @profile
    def calc_lklhd_pot_diff(self, id_parent_pot, id_child):
        i = self._inference
        lklhd = i.get_lklhd(id_child)
        lklhd_pot, message_child2parent_pot, _ = self.__calc_lklhd_pot(id_parent_pot, id_child)
        lklhd_pot = np.mean(lklhd_pot)
        lklhd_pot_diff = lklhd_pot - lklhd
        return lklhd_pot_diff, message_child2parent_pot

    @profile
    def calc_lklhd_pot_diff_sibling(self, id_parent_pot, id_sibling, message_child2parent_pot):
        i = self._inference
        lklhd = i.get_lklhd(id_sibling)
        lklhd_pot = i.calc_lklhd_parent(id_sibling, id_parent_pot, message_child2parent_pot)
        lklhd_pot = np.mean(lklhd_pot)
        lklhd_pot_diff = lklhd_pot - lklhd
        return lklhd_pot_diff

    @profile
    def calc_lklhd_pot_diff_siblings(self, id_parent_pot, message_child2parent_pot):
        g = self._graph
        edges = g.out_edges_iter(id_parent_pot)
        lklhd_pot_diff = [self.calc_lklhd_pot_diff_sibling(id_parent_pot, id_sibling, message_child2parent_pot) for id_sibling in edges]
        return np.array(lklhd_pot_diff)

    @profile
    def _get_lklhd_pot_diff(self):
        lklhd_pot_diff = self.__lklhd_pot_diff.copy()
        if self.max_of_cp_and_pc:
            tmp = lklhd_pot_diff.flatten()
            tmpT = lklhd_pot_diff.T.flatten()
            idx = (~np.isnan(tmp)) & (~np.isnan(tmpT))
            tmp[idx] = np.maximum(tmp[idx], tmpT[idx])
            tmp = np.reshape(tmp, lklhd_pot_diff.shape)
            lklhd_pot_diff = tmp

        if self.balance_k:
            k = self._get_node_infos(self._k)
            lklhd_pot_diff *= np.sqrt(k[:,np.newaxis]) * np.sqrt(k[np.newaxis,:])
        return lklhd_pot_diff

    @profile
    def __calc_lklhd_pot_messages(self, messages_prod1, messages_prod2, message_1to2_old):
        """
        :param messages_prod1: q(parent_pot) (WARNING: this variable is changed in-place)
        :param messages_prod2: beta(child)
        :param message_1to2_old: alpha^before(child)
        :return:
        """
        if messages_prod1 is not None:
            k1 = messages_prod1.get_k2()
        else:
            k1 = 1
        k2 = messages_prod2.get_k2()
        distrib = DistribFactory.empty(k1, k2) # distrib: k1 x k2

        messages_prod2_all = messages_prod2.copy()
        messages_prod2_all.prod([message_1to2_old])  # q(child)

        if messages_prod1 is not None:
            # exclude_self needs to be true, since we calculate the CPD without having an edge between the nodes
            distrib.update_cpd(messages_prod1, messages_prod2_all, N_0=self.N_0, exclude_self=True)

            # node is gaussian -> we need to sum over parent_pot
            # (if none of the above applies, then we can sum over either node or parent_pot)

            # we need message node -> parent_pot, i.e. we calculate the lklhd from the parent
            message_child2parent_pot = messages_prod2.dot_transpose(distrib)
            messages_prod1.set_log_const_zero()
            messages_prod1.prod([message_child2parent_pot])
            messages_prod = messages_prod1
        else:
            distrib.update_prior(messages_prod2_all, N_0=self.N_0)

            # parent_pot is None -> we need to sum over node
            # no message exists and we only need to multiply with the prior
            messages_prod2.prod([distrib])
            messages_prod = messages_prod2
            message_child2parent_pot = None

        lklhd = messages_prod.get_log_const()
        return (lklhd, message_child2parent_pot, distrib)

    @profile
    def __calc_lklhd_pot(self, id_parent_pot, id_child):
        g = self._graph
        i = self._inference

        id_parent = g.get_parent(id_child)  # might be None

        # get potential distrib for (parent_pot, node)
        # the child needs to remove its old parent message (either from parent or prior)
        messages_prod2 = i.get_messages_prod(id_child, id_except=id_parent, include_prior=False)  # M2: N x k2

        if id_parent is not None:
            message_1to2_old = i.get_message(id_parent, id_child)
        else:
            message_1to2_old = g.node(id_child).prior

        if id_parent_pot is not None:
            # the potential parent uses all its incoming messages
            messages_prod1 = i.get_messages_prod(id_parent_pot, id_except=None)  # M1: N x k1
        else:
            messages_prod1 = None
        lklhd, message_child2parent_pot, distrib = self.__calc_lklhd_pot_messages(messages_prod1, messages_prod2, message_1to2_old)
        return lklhd, message_child2parent_pot, distrib

    def _update_lklhd_pot_diff_dirty(self, id_node):
        if id_node is None:
            return
        g = self._graph
        id_root = g.get_root(id_node)

        def func_node(id_node_, node_):
            self.lklhd_pot_diff_dirty[id_node_] = True
        g.func_iter(id_roots=id_root, func_node=func_node)

    @profile
    def draw(self):
        fig, ax = plt.subplots(1,self.required_axes)
        if not isinstance(ax, collections.Iterable):
            ax = [ax]
        self.draw_axes(ax)
        plt.show()

    @profile
    def draw_axes(self, ax=None):
        # concatenate lklhd_pot_diff and lklhd_pot_diff_root
        lpd = self.__lklhd_pot_diff
        lpdr = self.lklhd_pot_diff_root[np.newaxis,:]
        pad = np.full_like(lpdr, np.nan)
        data = np.concatenate((lpd, pad, lpdr), axis=0)

        lpds = self.lklhd_pot_diff_siblings

        if ax is None:
            ax = self._graph.get_axes(self.id_axes)
        assert len(ax) == self.required_axes

        # imshow lklhd_pot_diff
        ax[0].set_anchor('N')
        imshow_values(data, ax[0], show_value_text=self.show_value_text)
        # imshow lklhd_pot_diff_siblings
        ax[1].set_anchor('N')
        imshow_values(lpds, ax[1], show_value_text=self.show_value_text)

    @profile
    def _is_single_continuous_parent(self, id_node):
        g = self._graph
        if g.has_node(id_node):
            id_children = [x for x in g.out_edges_iter(id_node)]
            result = (len(id_children) == 1) and (g.node(id_children[0]).k ==1)
        else:
            result = False
        return result

    @profile
    def _is_observed(self, id_node):
        g = self._graph
        if g.has_node(id_node):
            result = g.node(id_node).x is not None
        else:
            result = False
        return result

    @profile
    def _layer(self, id_node):
        g = self._graph
        if g.has_node(id_node):
            result = g.node(id_node).layer
        else:
            result = np.Inf
        return result

    @profile
    def _k(self, id_node):
        g = self._graph
        if g.has_node(id_node):
            result = g.node(id_node).k
        else:
            result = np.NaN
        return result

    @profile
    def _get_node_infos(self, func):
        num_nodes = self.__lklhd_pot_diff.shape[0]
        l = [func(id_node) for id_node in range(num_nodes)]
        ar = np.array(l)
        return ar

    # decorator substitute that is pickle-safe
    select_max = NanSelect(select_max_undecorated)
    select_weighted_random = NanSelect(select_weighted_random_undecorated)
    select_random = NanSelect(select_random_undecorated)
    select_random_metropolis = NanSelect(select_random_metropolis_undecorated)

    @profile
    def is_allowed_child(self, id_child):
        return True

    @profile
    def is_allowed_root(self, id_child):
        return False

class StructureUpdatePredefined(StructureUpdate):
    def __init__(self, graph, data, inference, parameter_learning):
        super(StructureUpdatePredefined, self).__init__(graph, data, inference, parameter_learning)
        self._register_properties()
        self.adjlist = None
        self.__done = False

    def run(self):
        if self.adjlist is None:
            raise ValueError('The variable adjlist is empty, but an adjacency list was expected!')

        if not self.__done:
            self.remove_gaussian_parents()
            self.add_nodes_from_adjlist(self.adjlist)
            self._data.run()
            lklhd = self._inference.run()
            self.__done = True

            # set restart_recursive to True (otherwise we would only learn a part of the tree
            self._parameter_learning.restart_recursive = True
        else:
            # already done adding nodes
            lklhd = None
        id_roots_changed = None
        return lklhd, id_roots_changed
#ToDo:
# Implement within run() method:
# (1) remove gaussian parents (opposite of add_gaussian_parents())
# (2) add_nodes_from_adjlist
# (3) init distributions


class StructureUpdateCVPR2015(StructureUpdate):
    """This class handles the parameter update step."""
    def __init__(self, graph, data, inference, parameter_learning):
        super(StructureUpdateCVPR2015, self).__init__(graph, data, inference, parameter_learning)
        self._register_properties()
        self.lklhd_mindiff = 0.01
        self.lklhd_mindiff_siblings = -0.1
        self.select_strategy = StructureUpdate.select_max
        self.new_node_layer_plusone = True
        self.force_full_tree = False

    @profile
    def is_allowed_child(self, id_child):
        g = self._graph
        if not g.has_node(id_child):
            return False
        child = g.node(id_child)
        if not g.is_root(id_child):
            # check if child's parent is a single parent of a continuous node
            if child.k == 1 and g.out_degree(g.get_parent(id_child)) == 1:
                return True
            else:
                return False
        # check if child is a single parent of a continuous node
        if g.out_degree(id_child) == 1:
            # this check is sufficient, since all other nodes have either degree=0 or degree>1
            return False
        return True

    @profile
    def is_allowed(self, id_parent, id_child):
        g = self._graph
        if not g.has_node(id_parent):
            return False
        if not g.has_node(id_child):
            return False
        parent = g.node(id_parent)
        child = g.node(id_child)
        if parent.k == 1:
            # continuous parents are not allowed
            return False
        if id_parent == id_child:
            # parent and child must be different
            return False
        if g.get_root(id_parent) == g.get_root(id_child):
            # parent and child must be in different subtrees
            return False
        if not g.is_root(id_parent):
            return False
        if not g.is_root(id_child):
            # check if child's parent is a single parent of a continuous node
            if child.k == 1 and g.out_degree(g.get_parent(id_child)) == 1:
                return True
            else:
                return False
        # check if child is a single parent of a continuous node
        if g.out_degree(id_child) == 1:
            # this check is sufficient, since all other nodes have either degree=0 or degree>1
            return False
        return True

    @profile
    @staticmethod
    def _calc_closed(lklhd_pot_diff, cond_is_observed):
        changed = True
        is_closed = np.all(np.isnan(lklhd_pot_diff), axis=1)  # closed are the nodes that cannot add any other node as child
        # observed nodes are always closed
        is_closed = is_closed | cond_is_observed
        count = 0
        while changed:
            lklhd_pot_diff_excl_open_children = lklhd_pot_diff.copy()
            lklhd_pot_diff_excl_open_children[:, ~is_closed] = np.nan  # open nodes cannot be added as child
            is_closed_new = np.all(np.isnan(lklhd_pot_diff_excl_open_children), axis=1)
            is_closed_new = is_closed | is_closed_new
            if np.all(is_closed == is_closed_new):
                changed = False
            is_closed = is_closed_new
            count += 1
            assert count < 999  # just to debug, this should never fail
        return is_closed

    @profile
    def _get_cond_mindiff(self, lklhd_pot_diff):
        with np.errstate(invalid='ignore'):
            cond_mindiff = np.isnan(lklhd_pot_diff) | (lklhd_pot_diff > self.lklhd_mindiff)
            cond_siblings_mindiff = np.isnan(self.lklhd_pot_diff_siblings) | \
                                    (self.lklhd_pot_diff_siblings > self.lklhd_mindiff_siblings)
        return cond_mindiff, cond_siblings_mindiff

    @profile
    def select_update(self):
        lklhd_pot_diff = self._get_lklhd_pot_diff()
        is_observed = self._get_node_infos(self._is_observed)
        layer = self._get_node_infos(self._layer)
        cond_mindiff, cond_siblings_mindiff = self._get_cond_mindiff(lklhd_pot_diff)
        # exclude edges that do not fulfill cond_mindiff
        lklhd_pot_diff[~(cond_mindiff)] = np.nan

        is_closed = self._calc_closed(lklhd_pot_diff, is_observed)
        lklhd_pot_diff[:,~is_closed] = np.nan  # exclude children that are open
        eligible_cond = ~np.isnan(lklhd_pot_diff)

        layer_max = np.maximum(layer[:, np.newaxis], layer[np.newaxis, :])

        # new node is created either if cond_siblings is not fulfilled or if the new parent is observed
        new_node_cond = (~cond_siblings_mindiff) | is_observed[:,np.newaxis]

        if self.new_node_layer_plusone:
            # if we create a new node, then it will be one layer higher
            # this is different from the original cvpr algorithm
            layer_max[new_node_cond] += 1

        if np.any(eligible_cond):
            layer_min_eligible = np.min(layer_max[eligible_cond])

            # priority_cond = (~new_node_cond) & (~is_single_continuous_parent[:,np.newaxis])
            priority_cond = layer_max == layer_min_eligible

            if np.any(eligible_cond & priority_cond):
                # check if we can continue within the priority conditions
                lklhd_pot_diff[~priority_cond] = np.nan

            id_parent, id_child = self.select_strategy(lklhd_pot_diff)
            create_node = new_node_cond[id_parent, id_child]
        else:
            id_parent, id_child, create_node = None, None, None
            # debug printout
            lklhd_pot_diff_orig = self._get_lklhd_pot_diff()
            if np.any(~np.isnan(lklhd_pot_diff_orig)):
                id_parent_max, id_child_max = StructureUpdate.select_max(lklhd_pot_diff_orig)
                self._print('select_max=({},{}); lklhd_pot_diff={}; is_closed[id_child_max]={}'.format(
                    id_parent_max, id_child_max, lklhd_pot_diff_orig[id_parent_max, id_child_max],
                    is_closed[id_child_max]))
                if self.force_full_tree:
                    id_parent, id_child = id_parent_max, id_child_max
                    create_node = new_node_cond[id_parent, id_child]
            else:
                self._print('select_max=(None,None)')
        return id_parent, id_child, create_node


class StructureUpdateCVPR2015Closer(StructureUpdateCVPR2015):
    """This class handles the parameter update step."""
    def __init__(self, graph, data, inference, parameter_learning):
        super(StructureUpdateCVPR2015Closer, self).__init__(graph, data, inference, parameter_learning)
        self._register_properties()
        self.__is_closed_last = np.zeros((0,))

    @profile
    def update_lklhd_pot_diff(self):
        super(StructureUpdateCVPR2015Closer, self).update_lklhd_pot_diff()


        # get is_closed array
        lklhd_pot_diff = self._get_lklhd_pot_diff()
        is_observed = self._get_node_infos(self._is_observed)
        cond_mindiff, _ = self._get_cond_mindiff(lklhd_pot_diff)
        lklhd_pot_diff[~(cond_mindiff)] = np.nan  # exclude edges that do not fulfill cond_mindiff
        is_closed = self._calc_closed(lklhd_pot_diff, is_observed)


        # get newly closed nodes
        id_newly_closed = set()
        for id_root in self._graph.get_id_roots():
            open_last = (id_root >= len(self.__is_closed_last)) or (not self.__is_closed_last[id_root])
            open_now = (id_root >= len(is_closed)) or (not is_closed[id_root])
            if open_last and not open_now:
                id_newly_closed.add(id_root)
        self.__is_closed_last = is_closed

        if id_newly_closed:
            id_newly_closed = sorted(id_newly_closed)
            self._print("recursive EM on id_newly_closed={}".format(id_newly_closed))
            for id_root in id_newly_closed:
                self._parameter_learning.run(id_roots=id_root, recursive=True)
                self._update_lklhd_pot_diff_dirty(id_root)
            super(StructureUpdateCVPR2015Closer, self).update_lklhd_pot_diff()

    @profile
    def select_update(self):
        lklhd_pot_diff = self._get_lklhd_pot_diff()
        is_observed = self._get_node_infos(self._is_observed)
        layer = self._get_node_infos(self._layer)
        cond_mindiff, cond_siblings_mindiff = self._get_cond_mindiff(lklhd_pot_diff)
        lklhd_pot_diff[~(cond_mindiff)] = np.nan  # exclude edges that do not fulfill cond_mindiff

        is_closed = self._calc_closed(lklhd_pot_diff, is_observed)
        lklhd_pot_diff[:,~is_closed] = np.nan  # exclude children that are open
        eligible_cond = ~np.isnan(lklhd_pot_diff)

        layer_max = np.maximum(layer[:, np.newaxis], layer[np.newaxis, :])

        # new node is created either if cond_siblings is not fulfilled or if the new parent is observed
        new_node_cond = (~cond_siblings_mindiff) | is_observed[:,np.newaxis]

        if self.new_node_layer_plusone:
            # if we create a new node, then it will be one layer higher
            # this is different from the original cvpr algorithm
            layer_max[new_node_cond] += 1

        if np.any(eligible_cond):
            layer_min_eligible = np.min(layer_max[eligible_cond])

            # priority_cond = (~new_node_cond) & (~is_single_continuous_parent[:,np.newaxis])
            priority_cond = layer_max == layer_min_eligible

            if np.any(eligible_cond & priority_cond):
                # check if we can continue within the priority conditions
                lklhd_pot_diff[~priority_cond] = np.nan

            id_parent, id_child = self.select_strategy(lklhd_pot_diff)
            create_node = new_node_cond[id_parent, id_child]
        else:
            id_parent, id_child, create_node = None, None, None
        return id_parent, id_child, create_node


class StructureUpdateSampling(StructureUpdate):
    """This class handles the parameter update step."""
    # TODO: implement case to change edges within the same tree
    def __init__(self, graph, data, inference, parameter_learning):
        super(StructureUpdateSampling, self).__init__(graph, data, inference, parameter_learning)
        self._register_properties()
        self.select_strategy = StructureUpdate.select_weighted_random

    @profile
    def is_allowed(self, id_parent, id_child):
        g = self._graph
        if not g.has_node(id_parent):
            return False
        if not g.has_node(id_child):
            return False
        if id_parent == id_child:
            # parent and child must be different
            return False
        parent = g.node(id_parent)
        if parent.k == 1:
            # continuous parents are not allowed
            return False
        child = g.node(id_child)
        if not ((parent.layer == child.layer) or (parent.layer == child.layer+1)):
            return False
        if (parent.layer == child.layer):
            # if a new node is created (i.e. parent.layer == child.layer), then the parent should not be a single parent of a continuous node
            if g.out_degree(id_parent) == 1:
                return False
        # if g.get_root(id_parent) == g.get_root(id_child):
        #     # parent and child must be in different subtrees
        #     return False

        if g.out_degree(id_child) == 1:
            # the child should not be a single parent of a continuous node
            # this check is sufficient, since all other nodes have either degree=0 or degree>1
            return False
        if g.has_edge(id_parent, id_child):
            # the edge should not already be in the tree
            return False
        id_parent_parent = g.get_parent(id_parent)
        if id_parent_parent == g.get_parent(id_child) and id_parent_parent is not None:
            # the nodes should not have already the same parent
            return False
        return True

    @profile
    def is_allowed_root(self, id_child):
        g = self._graph
        if not g.has_node(id_child):
            return False
        if g.is_root(id_child):
            return False
        child = g.node(id_child)
        if child.k == 1:
            # continuous children without parent are not allowed
            return False
        return True

    @profile
    def select_update(self):
        # TODO:
        # - include removing parent option
        # - better lklhd_pot_diff calculation: if old_parents are removed, this needs to be incorporated (for both, parent and child)
        lklhd_pot_diff = self._get_lklhd_pot_diff()
        # lklhd_pot_diff_root = self.lklhd_pot_diff_root.copy()
        layer = self._get_node_infos(self._layer)

        # maybe: include lklhd_pot_diff_root, i.e. the possibility to remove an edge
        # if np.any(~np.isnan(lklhd_pot_diff)) or np.any(~np.isnan(lklhd_pot_diff_root)):
        id_parent, id_child = self.select_strategy(lklhd_pot_diff)
        # id_parent, id_child = self._select_random(lklhd_pot_diff)
        if (id_parent is not None) and (id_child is not None):
            create_node = layer[id_parent] == layer[id_child]
        else:
            create_node = None
        return id_parent, id_child, create_node

class StructureUpdateSamplingFixedLayers(StructureUpdate):
    """This class handles the parameter update step."""
    def __init__(self, graph, data, inference, parameter_learning):
        super(StructureUpdateSamplingFixedLayers, self).__init__(graph, data, inference, parameter_learning)
        self._register_properties()
        self.select_strategy = StructureUpdate.select_weighted_random
        self.__init_fixed_layers()

    @profile
    def __init_fixed_layers(self):
        g = self._graph
        id_node_next = g.get_id_node_next()
        self.__layer = np.zeros((id_node_next,))
        for id_node, node in g.nodes_iter(data=True):
            self.__layer[id_node] = node.layer

    @profile
    def is_allowed(self, id_parent, id_child):
        g = self._graph
        if not g.has_node(id_parent):
            return False
        if not g.has_node(id_child):
            return False
        if id_parent == id_child:
            # parent and child must be different
            return False
        parent = g.node(id_parent)
        if parent.k == 1:
            # continuous parents are not allowed
            return False
        if not (self.__layer[id_parent] > self.__layer[id_child]):
            return False
        if g.has_edge(id_parent, id_child):
            # the edge should not already be in the tree
            return False
        return True

    @profile
    def select_update(self):
        lklhd_pot_diff = self._get_lklhd_pot_diff()
        id_parent, id_child = self.select_strategy(lklhd_pot_diff)
        create_node = False
        return id_parent, id_child, create_node


class StructureUpdateSamplingIterative(StructureUpdate):
    """This class handles the parameter update step."""
    def __init__(self, graph, data, inference, parameter_learning):
        super(StructureUpdateSamplingIterative, self).__init__(graph, data, inference, parameter_learning)
        self._register_properties()
        self.select_strategy = StructureUpdate.select_max
        self.id_node_last = None
        self.id_parent_last = None
        self.id_nodes_next = []
        self.__init_nodes_next()

        self.__keep_old_parents_bak = None

    @profile
    def __init_nodes_next(self):
        if not self.id_nodes_next:
            self.id_nodes_next = []
            for id_node in self._graph.nodes_iter():
                if self.__is_allowed_root(id_node):
                    self.id_nodes_next.append(id_node)
            # randomize order
            random.shuffle(self.id_nodes_next)

    @profile
    def is_allowed(self, id_parent, id_child):
        if self.id_node_last is None:
            return False
        if self.id_node_last != id_child:
            return False
        g = self._graph
        if not g.has_node(id_parent):
            return False
        if not g.has_node(id_child):
            return False
        if id_parent == id_child:
            # parent and child must be different
            return False
        parent = g.node(id_parent)
        if parent.k == 1:
            # continuous parents are not allowed
            return False
        if parent.x is not None:
            # observed parents are not allowed
            return False
        if g.get_root(id_parent) == g.get_root(id_child):
            # parent and child must be in different subtrees
            return False
        return True

    @profile
    def is_allowed_child(self, id_child):
        if self.id_node_last is None:
            return False
        if self.id_node_last != id_child:
            return False
        g = self._graph
        if not g.has_node(id_child):
            return False
        return True

    @profile
    def __is_allowed_root(self, id_child):
        g = self._graph
        if not g.has_node(id_child):
            return False
        if g.is_root(id_child):
            return False
        return True

    @profile
    def select_update(self):
        lklhd_pot_diff = self._get_lklhd_pot_diff()
        if self.id_node_last is not None:
            id_parent, id_child = self.select_strategy(lklhd_pot_diff)
            self.id_node_last = None

            self.__keep_old_parents_bak = self.keep_old_parents
            # in this step, it needs to be set to False, in order to remove single parents of a Gaussian node
            self.keep_old_parents = False

            if id_parent == self.id_parent_last:
                self._print("connected back to previous parent")
            else:
                self._print("found new parent!")
        else:
            id_parent = None
            id_child = self.id_nodes_next.pop(0)
            self.id_node_last = id_child
            if self.__keep_old_parents_bak is not None:
                self.keep_old_parents = self.__keep_old_parents_bak

            self.id_parent_last = self._graph.get_parent(id_child)
        create_node = False

        self.__init_nodes_next()
        return id_parent, id_child, create_node


class StructureUpdateGreedyBinary(StructureUpdate):
    """This class handles the parameter update step."""
    def __init__(self, graph, data, inference, parameter_learning):
        super(StructureUpdateGreedyBinary, self).__init__(graph, data, inference, parameter_learning)
        self._register_properties()
        self.select_strategy = StructureUpdate.select_max
        self.find_best_k = True

    @profile
    def is_allowed(self, id_parent, id_child):
        g = self._graph
        if not g.has_node(id_parent):
            return False
        if not g.has_node(id_child):
            return False
        if id_parent == id_child:
            # parent and child must be different
            return False
        if not g.is_root(id_parent):
            return False
        if not g.is_root(id_child):
            return False
        return True

    @profile
    def is_allowed_root(self, id_child):
        return False

    @profile
    def select_update(self):
        lklhd_pot_diff = self._get_lklhd_pot_diff()
        id_parent, id_child = self.select_strategy(lklhd_pot_diff)
        if (id_parent is not None) and (id_child is not None):
            create_node = True
        else:
            create_node = None
        return id_parent, id_child, create_node


class LogEntry(object):
    ATTR_ORDERED = ['count', 'num_roots', 'lklhd','lklhd_diff','structure_diff','parameter_diff', 'num_nodes']
    def __init__(self, *args):
        assert len(args) == len(self.ATTR_ORDERED)
        for i, attr in enumerate(self.ATTR_ORDERED):
            setattr(self, attr, args[i])

    def as_str(self):
        string = ''
        for attr in self.ATTR_ORDERED:
            value = getattr(self, attr)
            string += attr + "={} ".format(value)
        return string

class Log(object):
    def __init__(self):
        self.log_entries = []

    def append(self, log_entry):
        assert isinstance(log_entry, LogEntry)
        self.log_entries.append(log_entry)

    def as_vector(self, attr):
        result = np.zeros((len(self.log_entries),))
        for i, entry in enumerate(self.log_entries):
            result[i] = getattr(entry, attr)
        return result

class StructureLearning(GraphManipulator):
    def __init__(self, graph, inference,  parameter_learning, structure_update):
        super(StructureLearning, self).__init__(graph)
        assert isinstance(inference, BeliefPropagation)
        assert isinstance(parameter_learning, ParameterLearning)
        assert isinstance(structure_update, StructureUpdate)
        self._register_properties()

        self._inference = inference
        self._parameter_learning = parameter_learning
        self._structure_update = structure_update

        self.lklhd_mindiff = 1.e-6
        self.count_max = 999999
        self.log = Log()

        self.pl_recursive = True

    def __do_logging(self, count, lklhd_parameters, lklhd_structure, lklhd_last):
        lklhd_diff = lklhd_parameters - lklhd_last
        lklhd_diff_structure = lklhd_structure - lklhd_last
        lklhd_diff_parameters = lklhd_parameters - lklhd_structure
        log_entry = LogEntry(count, self._graph.get_num_roots(), lklhd_parameters, lklhd_diff, lklhd_diff_structure,
                             lklhd_diff_parameters, self._graph.number_of_nodes())
        self._print(log_entry.as_str())
        self.log.append(log_entry)


    @profile
    def run(self):
        pl = self._parameter_learning
        su = self._structure_update
        lklhd_last = pl.run()
        self._print('lklhd={}'.format(lklhd_last))
        count = 1
        continue_condition = count-1 < self.count_max

        while continue_condition:
            lklhd_structure, id_roots = su.run()
            if lklhd_structure is not None:
                # only run the parameter learning on the changed subtrees
                pl.run(id_roots=id_roots, recursive=self.pl_recursive)
                lklhd_parameters = self._inference.get_lklhd_all()
                lklhd_diff = lklhd_parameters - lklhd_last

                condition_lklhd = lklhd_diff > self.lklhd_mindiff
                condition_count = count < self.count_max
                continue_condition = condition_count & condition_lklhd

                self.__do_logging(count, lklhd_parameters, lklhd_structure, lklhd_last)
                lklhd_last = lklhd_parameters
                count += 1

                if not condition_lklhd:
                    self._print('Terminate because lklhd_diff <= mindiff.')
                if not condition_count:
                    self._print('Terminate because count_max is reached.')
            else:
                continue_condition = False
                self._print('No more possible structure updates.')
            # flush stdout
            # (for some reason otherwise no stdout is written during this stage, when streams are redirected to a
            #  network file)
            sys.stdout.flush()

        # final parameter optimization
        self._print('Do final recursive parameter optimization.')

        # change the parameter learning variables:
        parameters_new = {'restart_recursive':True,
                     'lklhd_mindiff':1e-5,
                     'count_max':1000,
                     'print_every':100,
                     'print_em':True}
        parameters_original = get_and_set_attr(pl, parameters_new)

        lklhd_structure = lklhd_last
        lklhd_parameters = pl.run()
        self.__do_logging(count, lklhd_parameters, lklhd_structure, lklhd_last)

        # restore old parameters:
        get_and_set_attr(pl, parameters_original)

        return lklhd_last


class LatentTree(ObjectRoot):
    def __init__(self, structure_update=StructureUpdateCVPR2015):
        super(LatentTree, self).__init__()
        # init LT objects
        self.graph = Graph()
        self.data = Data(self.graph)
        self.inference = BeliefPropagation(self.graph)
        self.parameter_update = ParameterUpdate(self.graph, self.inference)
        self.parameter_learning = ParameterLearning(self.graph, self.data, self.inference, self.parameter_update)
        self.structure_update = structure_update(self.graph, self.data, self.inference, self.parameter_learning)
        self.structure_learning = StructureLearning(self.graph, self.inference, self.parameter_learning, self.structure_update)

    @profile
    def set_structure_update(self, structure_update, attr_dict=None):
        self.structure_update = structure_update(self.graph, self.data, self.inference, self.parameter_learning)
        self.structure_learning._structure_update = self.structure_update
        if attr_dict is not None:
            for key, val in attr_dict.items():
                setattr(self.structure_update, key, val)

    @profile
    def training(self, X, K, clear_properties=True):
        # training
        start_time = time.time()
        id_nodes = self.graph.add_nodes(K)  # create the observed nodes
        self.data.insert_samples(id_nodes, X)  # insert the data
        self.structure_update.add_gaussian_parents()  # add gaussian single parents
        self.data.run()  # init distributions random or from data
        self.lklhd = self.structure_learning.run()  # learn the structure
        elapsed_time = time.time() - start_time
        self._print("elapsed training time: {:.1f} min".format(elapsed_time / 60))

        if clear_properties:
            # clean up
            self.clear_properties()

    @profile
    def testing(self, X_o, id_o, id_u, clear_properties=True):
        # testing
        self.data.insert_samples(id_o, X_o)
        lklhd = self.inference.run()
        X_u = self.inference.extract_samples(id_u)

        if clear_properties:
            # clean up
            self.clear_properties()

        return X_u, lklhd

    def clear_properties(self):
        self.data.clear_properties()
        self.inference.clear_properties()
        self.parameter_update.clear_properties()
        self.parameter_learning.clear_properties()
        self.structure_update.clear_properties()
        self.structure_learning.clear_properties()