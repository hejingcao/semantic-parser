# -*- coding: utf-8 -*-

import base64
import functools
import os

from delphin.mrs import Pred
from delphin.mrs.components import links as mrs_links

from framework.common.logger import LOGGER


def _dfs(root, edges_by_node, visited_nodes=None):
    stack = [id(root)]
    if not visited_nodes:
        visited_nodes = set(stack)
    while stack:
        nodeid = stack.pop()
        for edge in edges_by_node.get(nodeid, []):
            for node in edge.nodes:
                if id(node) not in visited_nodes:
                    visited_nodes.add(id(node))
                    stack.append(id(node))
    return visited_nodes


@functools.lru_cache(maxsize=65536)
def _remove_category(cat):
    if cat.endswith('u_unknown'):
        lemma, pos_and_sense = cat.rsplit('/', 1)
        pos_part, sense_part = pos_and_sense.split('_', 1)
        lemma_part = 'X'
    else:
        pred_obj = Pred.stringpred(cat)
        lemma_part = 'X' if cat.startswith('_') else pred_obj.lemma
        pos_part = pred_obj.pos or '#'
        sense_part = pred_obj.sense or '#'
    return lemma_part + '_' + pos_part + '_' + sense_part


class GraphNode:
    __slots__ = ('name', 'is_root')

    def __init__(self, name=None, is_root=False, span=None, label=None):
        self.name = name or base64.b64encode(os.urandom(15)).decode('ascii')
        self.is_root = is_root

    def __str__(self):
        return '<GraphNode {}>'.format(self.name)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, GraphNode) and self.name == other.name


class HyperEdge:
    __slots__ = ('nodes', 'label', 'is_terminal', 'span')

    def __init__(self, nodes, label, is_terminal, span=None):
        self.nodes = tuple(nodes)
        self.label = label
        self.is_terminal = is_terminal
        self.span = span

    def __str__(self):
        return "<{}{}: {}>".format(self.span or '',
                                   self.label,
                                   ' -- '.join(node.name for node in self.nodes))

    def to_tuple(self):
        return (*(x.name for x in self.nodes), self.label)

    @property
    def unique_label(self):
        return self.label, len(self.nodes)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        # return hash(self.nodes) ^ hash(self.label) ^ hash(self.span)
        return hash(self.nodes) ^ hash(self.label)

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
            and self.nodes == other.nodes \
            and self.label == other.label \
            and self.span == other.span

    def new(self, new_nodes, span=None):
        return HyperEdge(new_nodes, self.label, self.is_terminal, span=span)


class PredEdge(HyperEdge):
    __slots__ = ('carg',)

    def __init__(self, pred_node, label, span, carg=None):
        super().__init__([pred_node], label, True, span)
        self.carg = carg

    @classmethod
    def from_eds_node(cls, eds_node, lemma_to_x=False):
        label = str(eds_node.pred)
        if lemma_to_x:
            label = _remove_category(label)
        span = eds_node.lnk.data
        pred_node = GraphNode(eds_node.nodeid)
        pred_edge = cls(pred_node, label, span, eds_node.carg)
        return pred_node, pred_edge

    def new(self, new_nodes, span=None):
        new_nodes = list(new_nodes)
        assert len(new_nodes) == 1
        return PredEdge(new_nodes[0], self.label, span, self.carg)


class HyperGraph:
    __slots__ = ('nodes', 'edges', 'extra')

    def __init__(self, nodes, edges):
        self.nodes = frozenset(nodes)
        self.edges = frozenset(edges)

    def __hash__(self):
        return hash(self.nodes) ^ hash(self.edges)

    def __eq__(self, other):
        return self.nodes == other.nodes and self.edges == other.edges

    def __str__(self):
        return '<HyperGraph nodes={} edges={}>'.format(self.nodes, self.edges)

    def __repr__(self):
        return str(self)

    @property
    def sorted_nodes(self):
        return sorted(self.nodes, key=lambda x: x.name)

    def _get_edges_by_node(self):
        edges_by_node = {}
        for edge in self.edges:
            for node in edge.nodes:
                edges_by_node.setdefault(id(node), []).append(edge)
        return edges_by_node

    def is_connected(self):
        return len(_dfs(next(iter(self.nodes)), self._get_edges_by_node())) == len(self.nodes)

    def to_nodes_and_edges(self, return_properties=True):
        node_mapping = {}
        real_edges = []

        nodes = []
        edges = []
        for edge in self.edges:  # type: HyperEdge
            if len(edge.nodes) == 1:
                main_node = edge.nodes[0]  # type: GraphNode
                mapped_edge = node_mapping.get(main_node)
                if mapped_edge is None:
                    node_mapping[main_node] = edge
                else:
                    LOGGER.warning('Dumplicate node name %s and %s!', mapped_edge, edge.label)
            elif len(edge.nodes) == 2:
                real_edges.append(edge)
            else:
                LOGGER.warning('Invalid hyperedge with node count %d', len(edge.nodes))

        for node, pred_edge in node_mapping.items():
            assert pred_edge.span is not None

            node = node.name, pred_edge.label
            if return_properties:
                node += (pred_edge.span, pred_edge.carg)
            nodes.append(node)

        for edge in real_edges:
            node1, node2 = edge.nodes
            pred_edge1, pred_edge2 = node_mapping.get(node1), node_mapping.get(node2)
            if pred_edge1 is None or pred_edge2 is None:
                LOGGER.warning('No span for edge %s, nodes %s', edge, (pred_edge1, pred_edge2))
                continue
            edges.append((node1.name, node2.name, edge.label))

        return nodes, edges

    def connected_components(self):
        edges_by_node = self._get_edges_by_node()

        visited_nodeids = set()
        for node in self.nodes:
            if id(node) in visited_nodeids:
                continue
            nodeids = _dfs(node, edges_by_node, set(visited_nodeids))
            component = set()
            for nodeid in nodeids - visited_nodeids:
                component.update(edges_by_node.get(nodeid, []))
            yield component
            visited_nodeids = nodeids

    @classmethod
    def from_eds(cls, eds_graph, lemma_to_x=False, modify_label=frozenset()):
        nodes = []
        nodes_by_pred_label = {}
        edges = []
        for node in eds_graph.nodes():
            if 'strip_d' in modify_label and str(node.pred).endswith('_d'):
                nodes_by_pred_label[node.nodeid] = None
                continue

            graph_node, edge = PredEdge.from_eds_node(node, lemma_to_x)
            graph_node.is_root = (node.nodeid == eds_graph.top)

            nodes_by_pred_label[node.nodeid] = graph_node
            nodes.append(graph_node)
            edges.append(edge)

        for node in eds_graph.nodes():
            for label, target_id in eds_graph.edges(node.nodeid).items():
                start_node = nodes_by_pred_label[node.nodeid]
                end_node = nodes_by_pred_label[target_id]
                if start_node is None or end_node is None:
                    continue
                if 'strip-hndl' in modify_label and label.endswith('-HNDL'):
                    continue
                edges.append(HyperEdge([start_node, end_node],
                                       label=label,
                                       is_terminal=True))

        return cls(nodes, edges)

    @classmethod
    def from_mrs(cls, mrs, lemma_to_x=False, modify_label=frozenset()):
        nodes = []
        name_to_number = {}
        nodes_by_pred_label = {}
        edges = []
        for node in mrs.eps():
            if 'strip_d' in modify_label and str(node.pred).endswith('_d'):
                nodes_by_pred_label[node.nodeid] = None
                continue
            graph_node, edge = PredEdge.from_eds_node(node, lemma_to_x)
            graph_node.is_root = (node.label == mrs.top)

            nodes_by_pred_label[node.nodeid] = graph_node
            name_to_number[node.label] = node.nodeid

            nodes.append(graph_node)
            edges.append(edge)

        for start, end, rargname, post in mrs_links(mrs):
            if start == 0:
                continue
            start_node = nodes_by_pred_label[start]
            end_node = nodes_by_pred_label[end]
            if start_node is None or end_node is None:
                continue
            if 'strip-hndl' in modify_label and rargname.endswith('-HNDL'):
                continue

            edges.append(HyperEdge([start_node, end_node],
                                   label=rargname + '/' + post,
                                   is_terminal=True))

        return cls(nodes, edges)

    @classmethod
    def from_nodes_and_edges(cls, nodes, edges):
        hrg_nodes = set()
        hrg_edges = set()
        node_to_pred_node = {}

        for node_name, label, *properties in nodes:
            if properties:
                span, carg = properties
            else:
                span, carg = None, None
            node = GraphNode(node_name)
            edge = PredEdge(node, label, span, carg)

            hrg_edges.add(edge)
            hrg_nodes.add(node)

            node_to_pred_node[node_name] = node

        for source, target, label in edges:
            hrg_edges.add(HyperEdge([node_to_pred_node[source], node_to_pred_node[target]],
                                    label=label,
                                    is_terminal=True))

        return cls(frozenset(hrg_nodes), frozenset(hrg_edges))
