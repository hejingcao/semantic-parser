# -*- coding: utf-8 -*-

import json
import os

from preprocess.carg_utils import (CARGTransformer,
                                   EdsGraph,
                                   fix_eds_dash_node_span,
                                   fix_eds_prefix_node_span)


def build_carg_transformer(output_dir, training=False):
    return CARGTransformer(os.path.join(output_dir, 'carg-enabled-labels.txt'),
                           os.path.join(output_dir, 'carg-abbrevs.txt'),
                           os.path.join(output_dir, 'carg-label-mappings.txt'),
                           os.path.join(output_dir, 'carg-label-word-mappings.txt.train' +
                                        ('.train' if training else '')),
                           training=training)


def run_carg_transformer(mrp_path, output_dir, training=True):
    os.makedirs(output_dir, exist_ok=True)

    transformer = build_carg_transformer(output_dir, training)

    error_count_none = 0
    error_count_ignore_dash_dot = 0
    error_count = 0

    total_count = 0
    graphs = []
    for line in open(mrp_path):
        graph = json.loads(line)
        graphs.append(graph)

        original_nodes = graph['nodes']
        sentence = graph['input']
        total_count += len(graph['nodes'])

        eds_graph = EdsGraph(graph)
        try:
            fix_eds_prefix_node_span(eds_graph)
            fix_eds_dash_node_span(eds_graph)
        except Exception as err:
            print('carg preprocess error:', graph['id'], err)

        nodes = [eds_graph.nodes[node['id']] for node in original_nodes]
        labels = [node.label for node in nodes]
        spans = [node.span for node in nodes]
        correct_cargs = [(node.carg if training and not node.extra else None) for node in nodes]
        cargs = transformer(labels, spans, sentence, correct_cargs)

        for carg, node, original_node in zip(cargs, nodes, original_nodes):
            if carg == node.carg:
                continue

            anchor = original_node['anchors'][0]
            word = sentence[anchor['from']:anchor['to']]

            print(node.extra, word, node.label, node.carg, carg, sep='\t')

            error_count += 1
            if node.carg is None and carg is not None:
                error_count_none += 1
            elif node.carg.strip('-') != carg.strip('-'):
                error_count_ignore_dash_dot += 1

    print('wrong pred to none:', error_count_none)
    print('wrong ignore dash:', error_count_ignore_dash_dot)
    print('wrong:', error_count, error_count / total_count)

    if training:
        transformer.save()


if __name__ == '__main__':
    run_carg_transformer('data/mrp2019/training/eds/wsj.mrp', 'data/carg')
