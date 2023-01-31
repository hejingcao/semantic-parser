# -*- coding: utf-8 -*-

from collections import Counter, defaultdict

from framework.common.logger import open_file
from SHRG.utils.lexicon import get_lemma_and_pos, get_wordnet


def anonymize_label(token, label):
    if '+' in label or '-' in label or not label.startswith('_'):  # MWE / special nodes
        return None, label

    lemma_mapping = None
    old_lemma, pos, lemma_start, lemma_end = get_lemma_and_pos(label, True)

    transformed_label = label[:lemma_start] + '{NEWLEMMA}' + label[lemma_end:]
    if pos in ('n', 'v', 'a'):
        pred_lemma = get_wordnet().lemmatize(token, pos)
        lemma_mapping = ((pred_lemma, transformed_label), old_lemma)
    elif token != old_lemma:
        lemma_mapping = ((token, transformed_label), old_lemma)

    return lemma_mapping, transformed_label


class Writer:
    def to_file(self, graphs, output_path, lemma_path=None, write_smatch=False, training=True):
        all_graphs = []
        lemma_mappings = defaultdict(Counter)

        with open_file(output_path, 'w') as fp:
            for graph_id in sorted(graphs):
                graph = graphs[graph_id]
                mappings, aligned_nodes = self.write(fp, graph_id, graph, training=training)
                if mappings is None:
                    continue

                all_graphs.append((graph_id, graph, aligned_nodes))

                for mapping in mappings:
                    lemma_mappings[mapping[0]][mapping[1]] += 1

        if write_smatch:
            with open_file(output_path + '.smatch', 'w') as fp_smatch:
                for item in all_graphs:
                    self.write_smatch(fp_smatch, *item)

        if lemma_path is not None:
            with open_file(lemma_path, 'w') as fp:
                for key, value in lemma_mappings.items():
                    if len(value) != 1:
                        print(key, value)

                    line = list(key)
                    line.append(':::')
                    line.extend([f'{label}@@{count}'
                                 for label, count in value.most_common(1)])
                    fp.write('\t'.join(line) + '\n')

    def write_smatch(self, fp, graph_id, graph, aligned_nodes):
        fp.write(f'# {graph_id}\n')

        nodes, edges = graph.original.to_nodes_and_edges()
        name_to_label = {name: label for name, label, *_ in nodes}

        visited_names = set()
        new_nodes = []
        for current_nodes in aligned_nodes:
            for name, _ in current_nodes:
                visited_names.add(name)
                new_nodes.append((name, name_to_label[name]))

        for name, label in name_to_label.items():  # add rest nodes
            if name not in visited_names:
                new_nodes.append((name, label))

        assert len(nodes) == len(new_nodes)
        nodes = new_nodes

        fp.write(f'{len(nodes)}\n')
        for name, label, *_ in nodes:
            fp.write(f'{name} {label}\n')

        fp.write(f'{len(edges)}\n')
        for source, target, label in edges:
            fp.write(f'{source} {target} {label}\n')
        fp.write('\n')

    def write(self, fp, graph_id, sample, training):
        sentence = sample.sentence
        hyper_graph = sample.graph
        alignment = sample.alignment

        prefix = '###COMMENT###'
        fp.write(f'{prefix} sample_id: {graph_id}\n')
        fp.write(f'{prefix} original_sentence: {repr(sentence)}\n')
        fp.write(f'{prefix} token_spans: {repr(sample.token_spans)}\n')
        fp.write(f'{prefix} top: {repr(hyper_graph.extra)}\n')

        mappings = []
        if alignment is None:
            if training:
                return None, None
            nodes = [[]] * len(sample.tokens)
            lemmas = spans = nodes = [[]] * len(sample.tokens)
            edges = []
            fp.write(f'{prefix} ignore: True\n')
        else:

            assert len(hyper_graph.nodes) == sum(map(len, alignment))

            nodes, edges = hyper_graph.to_nodes_and_edges()

            # merge parallel edges
            new_edges = defaultdict(list)
            for source, target, edge_label in edges:
                new_edges[source, target].append(edge_label)
            edges = [(source, target, '&&&'.join(sorted(edge_labels)))
                     for (source, target), edge_labels in new_edges.items()]

            nodeid_to_span = {}
            for nodeid, _, span, _ in nodes:
                nodeid_to_span[nodeid] = span

            spans = [[] for _ in range(len(alignment))]
            nodes = [[] for _ in range(len(alignment))]
            lemmas = [[] for _ in range(len(alignment))]

            for index, (current_edges, token) in enumerate(zip(alignment, sample.tokens)):
                for edge in current_edges:
                    name = edge.nodes[0].name
                    label = edge.label

                    lemmas[index].append(label)
                    spans[index].append(nodeid_to_span[name])

                    lemma_mapping, label = anonymize_label(token, label)
                    if lemma_mapping is not None:
                        mappings.append(lemma_mapping)

                    nodes[index].append((name, label))

        fp.write(f'{prefix} spans: {repr(spans)}\n')
        fp.write(f'{prefix} nodes: {repr(nodes)}\n')
        fp.write(f'{prefix} edges: {repr(edges)}\n')
        fp.write(f'{prefix} lemmas: {repr(lemmas)}\n')
        fp.write(' '.join(sample.tokens) + '\n')

        return mappings, nodes
