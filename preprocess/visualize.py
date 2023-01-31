# -*- coding: utf-8 -*-

import html
import itertools
import os
from collections import defaultdict

from graphviz import Digraph

from framework.common.logger import LOGGER, open_file
from framework.common.utils import ProgressReporter
from pipeline.relation_detector import EdsSample
from preprocess.tokenize import convert_char_span_to_word_span

MATCHED_ATTRS = {'color': 'black', 'style': 'solid'}
GOLD_MISMATCH_ATTRS = {'color': 'red', 'style': 'dashed'}
SYSTEM_MISMATCH_ATTRS = {'color': 'grey', 'style': 'dashed'}


def read_node_mapoutput(path):
    all_mappings = []
    for block in open_file(path, 'r').read().strip().split('\n\n'):
        head, summary, *lines = block.strip().split('\n')
        assert head.startswith('Case: ') and summary.startswith('Node: ')

        mappings = defaultdict(list)
        all_mappings.append(mappings)
        for line in lines:
            node_ids, *error_type = line.split()

            if not error_type:
                error_type = 'correct'
            else:
                assert len(error_type) == 1
                error_type = error_type[0]

            mappings[error_type.strip()].append([_ for _ in node_ids.strip().split('<=>') if _])

    return all_mappings


def _draw_node(dot, nodeid, node_label, word_span=(), **kwargs):
    if word_span:
        if word_span[0] == word_span[1]:
            label = '{}<{}>'
        else:
            label = '{}<{}:{}>'
    else:
        label = '{}'

    label = label.format(node_label, *word_span)

    color = kwargs['color']
    if color is not None:
        kwargs['fontcolor'] = color

    dot.node(nodeid, label=label, **kwargs)


def _draw_edge(dot, source, target, edge_label, **kwargs):
    color = kwargs['color']
    if color is not None:
        kwargs['fontcolor'] = color

    dot.edge(source, target, label=edge_label, **kwargs)


def _collect_node_attrs(mappings):
    node_attrs = {}

    for gold_name, system_name in mappings['correct']:
        name = f'm__{gold_name}__{system_name}'
        node_attrs['g' + gold_name] = node_attrs['s' + system_name] = name, MATCHED_ATTRS

    for gold_name in mappings['gold_mismatch']:
        assert len(gold_name) == 1
        gold_name = gold_name[0]
        node_attrs['g' + gold_name] = f'g__{gold_name}', GOLD_MISMATCH_ATTRS

    for system_name in mappings['system_mismatch']:
        assert len(system_name) == 1
        system_name = system_name[0]
        node_attrs['s' + system_name] = f's__{system_name}', SYSTEM_MISMATCH_ATTRS

    for gold_name, system_name in mappings['mismatch']:
        node_attrs['g' + gold_name] = f'g__{gold_name}', GOLD_MISMATCH_ATTRS
        node_attrs['s' + system_name] = f's__{system_name}', SYSTEM_MISMATCH_ATTRS

    return node_attrs


def _collect_nodes(graph, use_span=False):
    if use_span:
        words = graph.words
        token_spans = graph.token_spans
        all_spans = [convert_char_span_to_word_span(span, token_spans, words, strict=False)
                     for spans in graph.spans
                     for span in spans]
    else:
        all_spans = [(word_index, word_index)
                     for word_index, nodes in enumerate(graph.nodes)
                     for _ in range(len(nodes))]

    all_nodes = itertools.chain(*graph.nodes)
    all_lemmas = itertools.chain(*graph.lemmas)
    return [
        (name, (label if lemma is None else lemma), span)
        for ((name, label), lemma, span) in zip(all_nodes, all_lemmas, all_spans)
    ]


def _collect_edges(graph, node_attrs, prefix):
    return {
        (node_attrs[prefix + source][0], node_attrs[prefix + target][0], label)
        for source, target, labels in graph.edges
        for label in labels.split('&&&')
    }


def visualize_graph(system_graph, gold_graph, mappings, output_dir):
    sentence = '&nbsp;'.join([
        '{}<SUP><FONT point-size="8" color="red">{}</FONT></SUP>'.format(html.escape(word), index)
        for index, word in enumerate(gold_graph.words)
    ])

    dot = Digraph(graph_attr={'label': '<' + sentence + '>'})

    node_attrs = _collect_node_attrs(mappings)
    correct_system_to_gold = dict(map(reversed, mappings['correct']))

    gold_nodes = _collect_nodes(gold_graph, use_span=True)
    system_nodes = _collect_nodes(system_graph)

    gold_edges = _collect_edges(gold_graph, node_attrs, 'g')
    system_edges = _collect_edges(system_graph, node_attrs, 's')

    name_to_label = {}
    for name, label, span in gold_nodes:
        name_to_label[name] = label
        nodeid, attrs = node_attrs['g' + name]

        _draw_node(dot, nodeid, label, word_span=span, **attrs)

    for name, label, span in system_nodes:
        nodeid, attrs = node_attrs['s' + name]
        if nodeid[0] == 'm':
            if name_to_label.get(correct_system_to_gold[name]) != label:
                print('???', gold_graph.sample_id)
            continue
        _draw_node(dot, nodeid, label, word_span=span, **attrs)

    for edge in gold_edges:
        source, target, label = edge
        attrs = MATCHED_ATTRS if edge in system_edges else GOLD_MISMATCH_ATTRS
        _draw_edge(dot, source, target, label, **attrs)

    for edge in system_edges:
        source, target, label = edge
        if edge in gold_edges:
            continue
        _draw_edge(dot, source, target, label, **SYSTEM_MISMATCH_ATTRS)

    dot.format = 'svg'
    output_file = os.path.join(output_dir, str(gold_graph.sample_id))
    dot.render(output_file, cleanup=True)


def visualize_graphs(system_graphs, gold_graphs, output_dir, mapping_path=None):
    if mapping_path is None:
        output_files = EdsSample.internal_evaluate(gold_graphs, system_graphs, None,
                                                   cleanup=False)[1]
        try:
            all_mappings = read_node_mapoutput(output_files[-1])
        finally:
            for output_file in output_files:
                try:
                    os.unlink(output_file)
                except OSError:
                    LOGGER.exception('can not remove %s', output_file)
    else:
        all_mappings = read_node_mapoutput(mapping_path)

    os.makedirs(output_dir, exist_ok=True)
    progress = ProgressReporter(len(system_graphs), step=100)
    for system_graph, gold_graph, mappings in \
            progress(zip(system_graphs, gold_graphs, all_mappings)):
        assert system_graph.sample_id == gold_graph.sample_id

        visualize_graph(system_graph, gold_graph, mappings, output_dir)
