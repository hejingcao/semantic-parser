# -*- coding: utf-8 -*-

from typing import Set

from framework.common.logger import LOGGER
from framework.common.utils import MethodFactory

from .const_tree import ConstTree, Lexicon
from .hyper_graph import HyperEdge, HyperGraph

DETECT_FUNCTIONS = MethodFactory()


def _detect_related(hyper_graph: HyperGraph, cfg_node: ConstTree, direct_edges):
    if not direct_edges:
        return None, None

    related_nodes = set(node for edge in direct_edges for node in edge.nodes)
    related_edges = direct_edges.union(
        set(edge for edge in hyper_graph.edges
            if edge.span is None and related_nodes.issuperset(edge.nodes)))

    return related_nodes, related_edges


def _detect_final(all_nodes: Set[ConstTree],
                  all_edges: Set[HyperEdge],
                  hyper_graph: HyperGraph, comment=None):
    # All edge of an internal node are in related_edges
    internal_nodes = set(node for node in all_nodes
                         if all(edge in all_edges
                                for edge in hyper_graph.edges if node in edge.nodes))
    external_nodes = all_nodes - internal_nodes

    return all_edges, internal_nodes, external_nodes, comment


@DETECT_FUNCTIONS.register('small')
def shrg_detect_small(hyper_graph: HyperGraph, cfg_node: ConstTree, direct_edges):
    related_nodes, related_edges = _detect_related(hyper_graph, cfg_node, direct_edges)
    if related_nodes is None:
        return None

    return _detect_final(related_nodes, related_edges, hyper_graph)


@DETECT_FUNCTIONS.register('lexicalized')
def shrg_detect_lexicalized(hyper_graph: HyperGraph, cfg_node: ConstTree, direct_edges):
    related_nodes, related_edges = _detect_related(hyper_graph, cfg_node, direct_edges)
    if related_nodes is None:
        return None

    def get_outgoing_edges(node):
        # If some external node only have internal edges and outgoing edges, it
        # can be converted into internal node
        return [edge
                for edge in hyper_graph.edges
                if edge.span is None and node == edge.nodes[0]]

    is_lexical = isinstance(cfg_node.children[0], Lexicon)
    if is_lexical:
        outgoing_edges = set(edge
                             for node in related_nodes
                             for edge in get_outgoing_edges(node))
        outgoing_nodes = set(edge.nodes[1] for edge in outgoing_edges)
        related_nodes.update(outgoing_nodes)
        related_edges.update(outgoing_edges)

    return _detect_final(related_nodes, related_edges, hyper_graph)


@DETECT_FUNCTIONS.register('large')
def shrg_detect_large(hyper_graph: HyperGraph, cfg_node: ConstTree, direct_edges):
    related_nodes, related_edges = _detect_related(hyper_graph, cfg_node, direct_edges)
    if related_nodes is None:
        return None

    external_nodes_0 = _detect_final(related_nodes, related_edges, hyper_graph)[-1]

    def get_edges_can_be_internal(node):
        # If some external node only have internal edges and outgoing edges, it
        # can be converted into internal node
        ret = []
        for edge in hyper_graph.edges:
            if node not in edge.nodes:
                continue
            if edge in related_edges:
                continue
            if edge.span is None and node == edge.nodes[0]:
                # edge is terminal, since its span is None
                ret.append(edge)
                continue
            return []
        return ret

    # edges that start with related_nodes
    outgoing_edges = set(edge
                         for node in external_nodes_0
                         for edge in get_edges_can_be_internal(node))
    outgoing_nodes = set(edge.nodes[1] for edge in outgoing_edges)
    related_nodes.update(outgoing_nodes)
    related_edges.update(outgoing_edges)

    return _detect_final(related_nodes, related_edges, hyper_graph)


def _get_edges_by_node(edges):
    node_linked_edges = {}
    for edge in edges:
        for node in edge.nodes:
            node_linked_edges.setdefault(node, []).append(edge)
    return node_linked_edges


def _find_two_step_path(isolated_node, related_nodes, hyper_graph):
    terminal_edges = {}
    for edge in hyper_graph.edges:
        if edge.is_terminal and len(edge.nodes) == 2:  # normal terminal edges
            from_node, to_node = edge.nodes
            terminal_edges.setdefault(from_node, {})[to_node] = edge
            terminal_edges.setdefault(to_node, {})[from_node] = edge
            assert from_node != to_node
    for bridge_node, bridge_edge1 in terminal_edges.get(isolated_node, {}).items():
        assert bridge_node not in related_nodes, 'bridge_node should not be in related_nodes'
        for node in related_nodes:
            if node is isolated_node:
                continue
            bridge_edge2 = terminal_edges.get(node, {}).get(bridge_node)
            if bridge_edge2 is not None:
                return bridge_node, bridge_edge1, bridge_edge2
    return None


@DETECT_FUNCTIONS.register('small-ext')
def shrg_detect_small_ext(hyper_graph: HyperGraph, cfg_node: ConstTree, direct_edges):
    related_nodes, related_edges = _detect_related(hyper_graph, cfg_node, direct_edges)
    if related_nodes is None:
        return None

    edges_by_node = _get_edges_by_node(related_edges)
    isolated_node = None
    for node, edges in edges_by_node.items():
        if len(edges) == 1 and len(edges[0].nodes) == 1:  # isolated edge
            isolated_node = node
            break

    comment = None
    if isolated_node is not None:
        path = _find_two_step_path(isolated_node, related_nodes, hyper_graph)
        if path:
            comment = 'find path {}'.format(path)
            LOGGER.debug('find isolated edge %s', edges_by_node[node])
            LOGGER.debug(comment)
            related_nodes.add(path[0])
            assert path[1] not in related_edges and path[2] not in related_edges, \
                'bridge_edges should not be in related_edges'
            related_edges.add(path[1])
            related_edges.add(path[2])

    return _detect_final(related_nodes, related_edges, hyper_graph, comment)
