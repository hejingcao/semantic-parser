# -*- coding: utf-8 -*-

import copy

from framework.common.logger import LOGGER
from SHRG.const_tree import ConstTree, Lexicon
from SHRG.hyper_graph import HyperGraph
from SHRG.shrg_compound_split import NodeDistributor

from .concept_fixer import ConceptSpanFixer, _find_span_in_spans_multi, _find_span_in_spans_single

FIXER = ConceptSpanFixer()


class EdsSolver:
    def __init__(self, arg_dict):
        self.arg_dict = arg_dict

    def solve(self, pred_edge, structual_edges=None):
        if pred_edge.label not in self.arg_dict:
            return [(pred_edge.nodes[0], None, None)]

        keys = []
        results = self.arg_dict[pred_edge.label]
        if isinstance(results, tuple):
            results = [results]
        for result in results:
            direction, which_label = result
            if isinstance(which_label, str):
                which_label = [which_label]

            for label in which_label:
                keys.append((pred_edge.nodes[0], direction, label))
        return keys


class ConceptAligner(NodeDistributor):
    def __init__(self, hyper_graph: HyperGraph, sentence, tokens, spans,
                 graph_type='eds',
                 extra_arg_dict={},
                 logger=LOGGER):

        cfg_root = ConstTree('ROOT')
        for index, (token, span) in enumerate(zip(tokens, spans)):
            node = ConstTree('X')
            node.children.append(Lexicon(token, span))
            node.index = index

            cfg_root.children.append(node)

        cfg_root.populate_spans_internal()

        super().__init__(hyper_graph, cfg_root, graph_type, fully_lexicalized=True)

        self.arg_dict.update(extra_arg_dict)

        self.attachment_solver = EdsSolver(self.attachment_solver.arg_dict)

        self.spans = spans
        self.leaves = cfg_root.children
        self.sentence = sentence

    def _log_mapping(self, tree_node, pred_edge, reason=''):
        self.logger.debug('Map "%s" to "%s" @ {%s} (%s)',
                          pred_edge.label,
                          ' '.join(i.string for i in tree_node.generate_lexicons()),
                          self.sentence[pred_edge.span[0]: pred_edge.span[1]],
                          reason)

    def _solve_attachment(self, structural_edges):
        attachments = self.attachment_waiting_list
        for key in list(attachments):
            if key[1] is not None:
                continue
            edge = attachments.pop(key)

            first, last = _find_span_in_spans_multi(edge.span, self.spans)

            if last == first:
                start, end = edge.span
                # _damage_v_1 => undamage.
                while end > start and not self.sentence[end - 1].isalnum():
                    end -= 1
                first = _find_span_in_spans_single((start, end), self.spans)
                if first is not None:
                    reason = 'align_to_inside'
            else:
                reason = 'align_to_first'

            self.edge_to_node[edge.nodes[0]] = tree_node = self.leaves[first]
            self._log_mapping(tree_node, edge, reason)
            self.results[tree_node].add(edge)

        super()._solve_attachment(structural_edges)


def compute_alignment(hyper_graph, wrapped_graph, sentence, tokens, spans, **kwargs):
    hyper_graph = copy.deepcopy(hyper_graph)

    FIXER.fix_stage1(wrapped_graph)
    FIXER.fix_stage2(wrapped_graph)
    FIXER.fix_stage3(wrapped_graph, tokens, spans)

    new_nodes = wrapped_graph.nodes
    for edge in hyper_graph.edges:
        if len(edge.nodes) == 1:
            edge.span = new_nodes[edge.nodes[0].name].span

    aligner = ConceptAligner(hyper_graph, sentence, tokens, spans, **kwargs)

    results = [[] for _ in range(len(tokens))]
    for tree_node, edges in aligner.solve().items():
        results[tree_node.index].extend(edges)

    return results
