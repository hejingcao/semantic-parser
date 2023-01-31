# -*- coding: utf-8 -*-

from collections import defaultdict
from operator import itemgetter

from framework.common.utils import MethodFactory

from .hyper_graph import HyperGraph

TRANSFORMS = MethodFactory()

bilexical_table = {
    ('parenthetical', ('ARG1', 'ARG2')),
    # ('unspec_manner', ('ARG1', 'ARG2')),
    ('poss', ('ARG1', 'ARG2')),
    # ('parg_d', ('ARG1', 'ARG2')),
    ('with_p', ('ARG1', 'ARG2')),
    ('refl_mod', ('ARG1', 'ARG2')),
    ('id', ('ARG1', 'ARG2')),
    ('temp_loc_x', ('ARG1', 'ARG2')),
    # ('measure', ('ARG1', 'ARG2')),
    ('_of_p', ('ARG1', 'ARG2')),
    # ('times', ('ARG2', 'ARG3')),
    # ('compound', ('ARG1', 'ARG2')),
    ('appos', ('ARG1', 'ARG2'))}


@TRANSFORMS.register('bilexical')
def bilexical_transform(hyper_graph: HyperGraph):
    nodes, edges = hyper_graph.to_nodes_and_edges()

    in_edges = defaultdict(set)
    out_edges = defaultdict(set)
    for source, target, edge_label in edges:
        in_edges[target].add((source, edge_label))
        out_edges[source].add((target, edge_label))

    nodes_selected = []
    edges_extra = set()
    for node, label, *properties in nodes:
        if (not label.startswith('_')) and len(out_edges[node]) == 2 and len(in_edges[node]) == 0:
            targets, edge_labels = zip(*sorted(out_edges[node], key=itemgetter(1)))
            if (label, edge_labels) in bilexical_table:
                edges_extra.add((targets[0], targets[1], '##'.join((label, *edge_labels))))
                continue

        nodes_selected.append((node, label, *properties))

    all_edges = list(edges_extra)
    for node, *_ in nodes_selected:
        for target, edge_label in out_edges[node]:
            all_edges.append((node, target, edge_label))

    return HyperGraph.from_nodes_and_edges(nodes_selected, all_edges)


@TRANSFORMS.register('remove-isolated')
def remove_isolated_nodes(hyper_graph: HyperGraph):
    nodes, edges = hyper_graph.to_nodes_and_edges()

    in_edges = defaultdict(set)
    out_edges = defaultdict(set)
    for source, target, edge_label in edges:
        in_edges[target].add((source, edge_label))
        out_edges[source].add((target, edge_label))

    nodes_selected = []
    for node, label, *properties in nodes:
        if not in_edges[node] and not out_edges[node]:
            continue
        nodes_selected.append((node, label, *properties))

    all_edges = []
    for node, *_ in nodes_selected:
        for target, edge_label in out_edges[node]:
            all_edges.append((node, target, edge_label))

    return HyperGraph.from_nodes_and_edges(nodes_selected, all_edges)
