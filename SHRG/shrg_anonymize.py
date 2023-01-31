# -*- coding: utf-8 -*-

import pickle
import sys

from framework.common.logger import LOGGER, open_wrapper

from .const_tree import Lexicon
from .hyper_graph import HyperEdge, HyperGraph
from .shrg import CFGRule, HRGRule, SHRGRule
from .shrg_alignment import PREFIX_LABELS
from .shrg_extract import save_rules
from .utils.container import IndexedCounter


def find_prefix_and_neg_edges(hyper_graph):
    neg_edge, prefix_edge = None, None
    for edge in hyper_graph.edges:
        if edge.label in PREFIX_LABELS:
            prefix_edge = edge
        elif edge.label == 'neg':
            neg_edge = edge
        if neg_edge and prefix_edge:
            break
    return neg_edge, prefix_edge


def replace_edge_in_hyper_graph(old_edge, new_edge, hyper_graph: HyperGraph):
    edges = hyper_graph.edges
    new_edges = []
    found = False
    for edge in edges:
        if old_edge.label == edge.label:  # TODO: use 'is' or '=='
            assert not found, f'found multiple matches: {old_edge} {hyper_graph}'
            found = True
            new_edges.append(new_edge)
        else:
            new_edges.append(edge)

    assert found, f'??? {old_edge} is not in {hyper_graph}'
    return HyperGraph(nodes=hyper_graph.nodes, edges=new_edges)


def anonymize_edge(lexicon, edge, hyper_graph):
    lemma_start = edge.label.find('_') + 1
    lemma_end = edge.label.find('_', lemma_start)

    new_edge = HyperEdge(nodes=edge.nodes,
                         label='_X' + edge.label[lemma_end:],
                         is_terminal=edge.is_terminal)
    new_graph = replace_edge_in_hyper_graph(edge, new_edge, hyper_graph)
    return new_edge, new_graph


def anonymize_rule(shrg_rule: SHRGRule):
    hrg = shrg_rule.hrg
    cfg = shrg_rule.cfg
    alignment = []
    for index, (lexicon, edge) in enumerate(cfg.rhs):
        if edge is not None and isinstance(lexicon, Lexicon):
            assert edge.label.startswith('_'), f'Strange condition {edge} {lexicon}'
            alignment.append(index)
    assert len(alignment) <= 1, f'A rule contains two or more lexical words: {cfg.rhs}'
    if not alignment:
        return
    index = alignment[0]
    lexicon, edge = cfg.rhs[index]
    assert edge is not hrg.lhs, 'Terminal edge become rule head'

    neg_edge, prefix_edge = find_prefix_and_neg_edges(hrg.rhs)

    cfg_rhs = list(cfg.rhs)
    if prefix_edge:
        cfg_rhs.insert(index, (f'<{prefix_edge.label}>', prefix_edge))
        index += 1

    new_edge, new_graph = anonymize_edge(lexicon, edge, hrg.rhs)
    extra_edge = edge.span
    if isinstance(extra_edge, HyperEdge):
        new_graph = anonymize_edge(lexicon, extra_edge, new_graph)[1]

    cfg_rhs[index] = ('<$X-neg>' if neg_edge is not None else '<$X>', new_edge)
    return SHRGRule(cfg=CFGRule(cfg.lhs, tuple(cfg_rhs)),
                    hrg=HRGRule(hrg.lhs, new_graph, hrg.comment))


def anonymize_rules(output_prefix):
    _open = open_wrapper(lambda x: output_prefix + x)
    rules, params = pickle.load(_open('.counter.p', 'rb'))
    num_rules = len(rules)
    LOGGER.info('Loaded %d rules (%s)', num_rules, output_prefix)

    anonymize_mapping = {}
    anonymous_rules = IndexedCounter(5)
    num_new_rules = 0
    for i, (rule, counter_item) in enumerate(rules):
        new_rule = anonymize_rule(rule)
        if new_rule is not None:
            num_new_rules += 1
            new_index = rules.add(new_rule, counter_item)
            anonymize_mapping.setdefault(new_index, set()).add(i)
        # accumulate
        anonymous_rules.add(new_rule or rule, counter_item)

    LOGGER.info('Generate %d new rules (merged from %d)', len(rules) - num_rules, num_new_rules)
    save_rules(output_prefix, anonymous_rules, params, extra_suffix='.anonymous')
    save_rules(output_prefix, rules, params, extra_suffix='.merged')
    pickle.dump(anonymize_mapping, _open('.merged.relations.p', 'wb'))


if __name__ == '__main__':
    anonymize_rules(sys.argv[1])
