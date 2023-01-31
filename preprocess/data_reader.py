# -*- coding: utf-8 -*-

import copy
import json
from typing import List

from framework.common.dataclass_options import argfield
from framework.common.logger import LOGGER
from framework.common.utils import DotDict
from SHRG.dataset_utils import ReaderBase
from SHRG.graph_transformers import TRANSFORMS
from SHRG.hyper_graph import GraphNode, HyperEdge, HyperGraph, PredEdge

from .carg_utils.wrapper import EdsGraph as MRPEdsWrapper
from .concept_aligner import compute_alignment
from .concept_fixer import EdsWrapper
from .tokenize import tokenize_sentence


def _json_to_hyper_graph(graph):
    nodes = {}
    edges = []
    for node in graph['nodes']:
        properties = dict(zip(node.get('properties', ()), node.get('values', ())))
        anchors = node['anchors'][0]
        carg = properties.get('carg')
        span = anchors['from'], anchors['to']
        label = node['label']
        node['id'] = nodeid = str(node['id'])

        node = GraphNode(nodeid, span=span, label=label)
        pred_edge = PredEdge(node, label, span=span, carg=carg)

        nodes[nodeid] = node
        edges.append(pred_edge)

    for edge in graph['edges']:
        edge['source'] = source = str(edge['source'])
        edge['target'] = target = str(edge['target'])
        edges.append(HyperEdge([nodes[source], nodes[target]],
                               label=edge['label'],
                               is_terminal=True))

    return HyperGraph(frozenset(nodes), frozenset(edges))


class ReaderOptions(ReaderBase.Options):
    modify_label: List[str] = argfield(default_factory=list)
    graph_transformers: List[str] = argfield(default_factory=list)


class DeepBankReader(ReaderBase):
    Options = ReaderOptions

    def on_error(self, filename, error):
        LOGGER.error('%s %s', filename, error)

    @classmethod
    def build_graph(cls, reader_output, filename, options, training, extra_args):
        hyper_graph, eds_graph = reader_output
        sentence = eds_graph.sentence

        original_graph = copy.deepcopy(hyper_graph)

        for transformer in options.graph_transformers:
            hyper_graph = TRANSFORMS.invoke(transformer, hyper_graph)

        original_nodes = []
        original_edges = []
        for node in hyper_graph.nodes:
            original_nodes.append(eds_graph.node(node.name))
            original_edges.append(eds_graph.edges(node.name))

        tokens, spans = tokenize_sentence(eds_graph.sentence)
        wrapped_graph = EdsWrapper(sentence, original_nodes, original_edges)

        hyper_graph.extra = eds_graph.top

        try:
            alignment = compute_alignment(hyper_graph, wrapped_graph,
                                          sentence, tokens, spans, **extra_args)
        except Exception:
            if training:
                raise
            alignment = None

        return DotDict({
            'original': original_graph,
            'graph': hyper_graph,
            'sentence': sentence,
            'alignment': alignment,
            'tokens': tokens,
            'token_spans': spans
        })


class MRPReader:
    Options = ReaderOptions

    def __init__(self, options, data_path, split_patterns, logger=LOGGER, **extra_args):
        self.options = options
        self.logger = logger
        self.extra_args = extra_args

        self._data = {}

        data = {}

        for line in open(data_path):
            graph = json.loads(line)
            graph_id = graph['id']

            for split, pattern in split_patterns:
                if pattern is True or graph_id in pattern:
                    data.setdefault(split, {})[graph_id] = graph
                    break

        for split, _ in split_patterns:
            logger.info('SPLIT %s: %d samples', split, len(data[split]))

        self._data = data

    def get_split(self, split, num_workers=None):
        data = {}
        for graph_id, graph in self._data[split].items():
            try:
                hyper_graph = _json_to_hyper_graph(graph)
                original_graph = copy.deepcopy(hyper_graph)

                for transformer in self.options.graph_transformers:
                    hyper_graph = TRANSFORMS.invoke(transformer, hyper_graph)

                new_nodeids = {node.name for node in hyper_graph.nodes}

                new_nodes = [node for node in graph['nodes'] if node['id'] in new_nodeids]
                new_edges = [edge for edge in graph['edges']
                             if edge['source'] in new_nodeids and edge['target'] in new_nodeids]

                graph['nodes'] = new_nodes
                graph['edges'] = new_edges

                wrapped_graph = MRPEdsWrapper(graph)

                hyper_graph.extra = str(wrapped_graph.top)

                sentence = wrapped_graph.sentence
                tokens, spans = tokenize_sentence(sentence)

                try:
                    alignment = compute_alignment(hyper_graph, wrapped_graph,
                                                  sentence, tokens, spans, **self.extra_args)
                except Exception:
                    if split == 'train':
                        raise
                    alignment = None

                data[graph_id] = DotDict({
                    'original': original_graph,
                    'graph': hyper_graph,
                    'sentence': sentence,
                    'alignment': alignment,
                    'tokens': tokens,
                    'token_spans': spans
                })
            except Exception:
                self.logger.exception('failed %s', graph_id)

        return data
