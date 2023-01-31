# -*- coding: utf-8 -*-

import functools
import operator
from collections import OrderedDict, defaultdict

from framework.common.logger import LOGGER

from .const_tree import ConstTree, Lexicon
from .hyper_graph import HyperGraph
from .utils import span as S
from .utils.lexicon import get_lemma_and_pos, get_wordnet

CATE_ARG_DICTS = {
    'eds': {
        'measure': (0, 'ARG2'), 'times': (0, 'ARG3'), 'plus': (0, 'ARG2'),
        'loc_nonsp': (0, 'ARG2'), 'nominalization': (0, 'ARG1'), 'parg_d': (0, 'ARG1'),
        'superl': (0, 'ARG1'), 'comp': (0, 'ARG1'), 'comp_equal': (0, 'ARG1'),
        'comp_less': (0, 'ARG1'), 'comp_so': (0, 'ARG1'), 'time': (1, 'ARG1'), 'place': (1, 'ARG1'),
        'udef_q': (0, 'BV'), 'proper_q': (0, 'BV'), 'number_q': (0, 'BV'),
        'def_implicit_q': (0, 'BV'), 'def_explicit_q': (0, 'BV'),
        'neg': (0, 'ARG1'), 'subord': (0, 'ARG2'),
        'elliptical_n': (1, 'ARG2'), 'num_seq': (0, 'R-INDEX'), 'ellipsis': (1, 'ARG2'),
        'idiom_q_i': (0, 'BV'), 'place_n': (1, 'ARG1'), 'time_n': (1, 'ARG1'),
        '_pre-_a_ante': (0, 'ARG1'), '_un-_a_rvrs': (0, 'ARG1'), 'pronoun_q': (0, 'BV'),
        'of_p': (0, 'ARG1')
    }
}

LEMMATIZER_EXTRA = {
    'an': 'a', '/': ['and', 'per'], 'auto': 'automobile',
    'hoped': 'hope', 'rated': 'rate'
}

COMPOUND_DICT = {'compound', 'unknown', 'appos', 'part_of', 'implicit_conj', 'generic_entity'}

COMPOUND_ARG_DICTS = {
    'eds': {
        'compound': (0, 'ARG2'), 'unknown': (0, 'ARG'),
        'appos': (0, 'ARG1'), 'part_of': (0, 'ARG1'),
        'implicit_conj': (0, ['L-HNDL', 'L-INDEX']),
        'generic_entity': (1, 'ARG1'),
        'focus_d': (0, 'ARG1'), 'with_p': (0, 'ARG2'),
        'id': (0, 'ARG2'), 'relative_mod': (0, 'ARG2'),
        'parenthetical': (0, 'ARG1'),
        'eventuality': (0, 'ARG1'),
        'pron': (1, 'ARG1')
    }
}


CARG_COMPARE_DICT = {
    'ord': {
        'first': '1', 'second': '2', 'third': '3', 'fourth': '4', 'fifth': '5', 'sixth': '6',
        'seventh': '7', 'eighth': '8', 'ninth': '9', 'tenth': '10', 'eleventh': '11',
        'twelfth': '12', 'thirteenth': '13', 'fourteenth': '14', 'fifteenth': '15',
        'sixteenth': '16', 'seventeenth': '17', 'eighteenth': '18', 'nineteen': '19',
        'twentieth': '20', 'thirtieth': '30', 'fortieth': '40', 'fiftieth': '50',
        'sixtieth': '60', 'seventieth': '70', 'eightieth': '80', 'ninetieth': '90',
        'hundredth': '100', 'thousandth': '1000',
        'millionth': '1000000', 'billionth': '1000000000', 'trillionth': '1000000000000'
    },
    'card': {
        'a': '1', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6',
        'seven': '7', 'eight': '8',
        'nine': '9', 'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17', 'eighteen': '18',
        'nineteenth': '19', 'twenty': '20', 'thirty': '30', 'forty': '40', 'fifty': '50',
        'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90', 'hundred': '100',
        'thousand': '1000', 'million': '1000000', 'billion': '1000000000',
        'trillion': '1000000000000'
    },
    'mofy': {
        'january': 'jan', 'february': 'feb', 'march': 'mar', 'april': 'apr', 'may': 'may',
        'june': 'jun', 'july': 'jul', 'august': 'aug', 'september': 'sep', 'october': 'oct',
        'november': 'nov', 'december': 'dec'
    },
    'named_n': {'_IMF': 'imf', 'u.s.': 'us'}
}

LEXICAL_EDGES_EXTRA = {'much-many_a': ['more']}

IS_NOT_LEXICAL_EDGE = {
    '_pre-_a_ante', '_un-_a_rvrs', '_re-_a_again', '_mis-_a_error',
    '_counter-_a_anti', '_counter-_a_anti', '_co-_a_with'
}
INTERVAL_MARKER = ['to', '–']
SPECIAL_LEXICAL_EDGE = {
    'interval': INTERVAL_MARKER,
    'interval_p_start': INTERVAL_MARKER, 'interval_p_end': INTERVAL_MARKER
}


def lowest_common_ancestor(tree_nodes, nodes):
    for node in tree_nodes:
        all_children = list(node.traverse_postorder())
        if all(i in all_children for i in nodes):
            return node
    raise Exception(f'No common ancestor for {nodes}')


class EDSAttachmentSolver:
    def __init__(self, arg_dict):
        self.arg_dict = arg_dict

    def solve(self, pred_edge, structual_edges=None):
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


class NodeDistributor:
    def __init__(self, hyper_graph: HyperGraph, cfg_root: ConstTree,
                 graph_type='eds',
                 fully_lexicalized=False,
                 logger=LOGGER):
        self.hyper_graph = hyper_graph
        self.cfg_root = cfg_root
        self.logger = logger
        self.results = defaultdict(set)
        self.edge_to_node = {}
        self.attachment_waiting_list = OrderedDict()  # for regularity
        self.internal_waiting_list = []

        self.arg_dict = CATE_ARG_DICTS[graph_type]

        self.fully_lexicalized = fully_lexicalized

        if fully_lexicalized:
            self.arg_dict = self.arg_dict.copy()
            self.arg_dict.update(COMPOUND_ARG_DICTS[graph_type])
            self.compound_dict = {}
        else:
            self.compound_dict = COMPOUND_DICT

        self.attachment_solver = globals()[f'{graph_type.upper()}AttachmentSolver'](self.arg_dict)

    def _log_mapping(self, tree_node, pred_edge, reason=''):
        self.logger.debug('Map "%s" to "%s" @ %s (%s)',
                          pred_edge.label,
                          ' '.join(i.string for i in tree_node.generate_lexicons()),
                          pred_edge.span,
                          reason)

    def _group_tree_nodes_by_span(self):
        # sometimes the span is incorrect
        span_rewrite = {}
        span_to_pred_edges = defaultdict(list)
        span_to_tree_nodes = defaultdict(list)
        structural_edges = set()

        for node in self.cfg_root.generate_preterminals():
            if S.is_empty(node.span):
                continue
            rewrote_span = span_rewrite.get(node.span, node.span)
            for another_span in span_to_tree_nodes:  # find overlapped span
                if S.is_overlapped(node.span, another_span):
                    rewrote_span = another_span
                    span_rewrite[node.span] = another_span
                    break
            span_to_tree_nodes[rewrote_span].append(node)  # group overlapped spans together

        if not self.fully_lexicalized:
            for node in self.cfg_root.traverse_postorder():
                if isinstance(node.children[0], Lexicon):  # skip preterminals
                    continue
                rewrote_span = span_rewrite.get(node.span)
                if rewrote_span is None:
                    children_spans = set(span_rewrite.get(i.span, i.span)
                                         for i in node.children
                                         if not S.is_empty(i.span))
                    if len(children_spans) == 1:  # if it has only one child span, use it
                        rewrote_span = next(iter(children_spans))
                        span_rewrite[node.span] = rewrote_span
                    else:
                        rewrote_span = node.span
                span_to_tree_nodes[rewrote_span].append(node)

        for edge in self.hyper_graph.edges:
            assert edge.is_terminal  # every edge should be terminals at this time
            if edge.span is not None:
                rewrote_span = span_rewrite.get(edge.span, edge.span)
                span_to_pred_edges[rewrote_span].append(edge)
            else:
                structural_edges.add(edge)

        redundant_pred_edges = set(span_to_pred_edges.keys()) - set(span_to_tree_nodes.keys())
        if self.fully_lexicalized:
            for span in redundant_pred_edges:
                for edge in span_to_pred_edges[span]:
                    for key in self.attachment_solver.solve(edge, structural_edges):
                        self.attachment_waiting_list[key] = edge
        else:
            assert not redundant_pred_edges, f'Redundant nodes in graph: {redundant_pred_edges}'

        return span_to_pred_edges, span_to_tree_nodes, structural_edges

    def _solve_first_try(self, span_to_tree_nodes, span_to_pred_edges, structural_edges):
        many_to_many = []

        for span, tree_nodes in span_to_tree_nodes.items():
            pred_edges = span_to_pred_edges.get(span)
            if not pred_edges:
                continue

            terminal_tree_nodes = []
            internal_tree_nodes = []

            for node in tree_nodes:
                if isinstance(node.children[0], Lexicon):
                    terminal_tree_nodes.append(node)
                else:
                    internal_tree_nodes.append(node)

            if len(terminal_tree_nodes) == 1:
                # thre is only one terminal tree node, directly assign the
                # pred_edge to the tree_node
                tree_node = terminal_tree_nodes[0]
                self.results[tree_node].update(pred_edges)
                for edge in pred_edges:  # pred_edge has only one node
                    self.edge_to_node[edge.nodes[0]] = tree_node
            elif len(terminal_tree_nodes) > 1:
                many_to_many.append((span, terminal_tree_nodes))
            elif len(internal_tree_nodes) >= 1 and self.fully_lexicalized:
                for edge in pred_edges:
                    for key in self.attachment_solver.solve(edge, structural_edges):
                        self.attachment_waiting_list[key] = edge
            elif len(internal_tree_nodes) == 1:
                tree_node = internal_tree_nodes[0]
                self.results[tree_node].update(pred_edges)
                for edge in pred_edges:
                    self.edge_to_node[edge.nodes[0]] = tree_node
            elif len(terminal_tree_nodes) == 0 and len(internal_tree_nodes) > 1:
                self.internal_waiting_list.append(pred_edges)
            else:
                fmt = 'Invalid tree nodes: {} {} {} {}'
                raise Exception(fmt.format(span, terminal_tree_nodes,
                                           internal_tree_nodes, pred_edges))
        return many_to_many

    def _solve_leaf(self, leaf_nodes, pred_edges, structural_edges):
        rest_lexical_edges = {}
        top_waiting = set()

        for edge in pred_edges:
            if edge.label in SPECIAL_LEXICAL_EDGE:
                rest_lexical_edges[edge] = None, None, None
            elif edge.carg is not None:
                rest_lexical_edges[edge] = edge.carg.rstrip('-').lower(), None, edge.label
            elif edge.label.startswith('_') and edge.label not in IS_NOT_LEXICAL_EDGE:
                lemma, postag = get_lemma_and_pos(edge.label)
                rest_lexical_edges[edge] = lemma.rstrip('-').lower(), postag, None
            elif edge.label in self.compound_dict:
                top_waiting.add(edge)
            elif edge.label in LEXICAL_EDGES_EXTRA:
                rest_lexical_edges[edge] = LEXICAL_EDGES_EXTRA[edge.label], None, None
            elif edge.label in self.arg_dict:
                for key in self.attachment_solver.solve(edge, structural_edges):
                    self.attachment_waiting_list[key] = edge
            else:
                raise Exception(f'Don\'t know how to solve edge {edge}')

        self.internal_waiting_list.append(top_waiting)

        # to which tree node this pred node is assigned
        rest_leaf_nodes = set(leaf_nodes)

        def assign_edge_to_node(node, pred_edge, reason=''):
            self.results[node].add(pred_edge)
            rest_lexical_edges.pop(pred_edge)
            self.edge_to_node[pred_edge.nodes[0]] = node

            self._log_mapping(node, pred_edge, reason=reason)

        wordnet_lemma = get_wordnet()

        for node in leaf_nodes:
            for pred_edge, (lemma, postag, carg_type) in list(rest_lexical_edges.items()):
                node_string = node.children[0].string
                eq = False
                special_marker = SPECIAL_LEXICAL_EDGE.get(pred_edge.label)
                if special_marker and node_string in special_marker:
                    eq = True

                if not eq:
                    if postag in ('n', 'v', 'a'):
                        for pos in (postag, 'n', 'v', 'a'):
                            node_lemma = wordnet_lemma.lemmatize(node_string, pos)
                            node_lemma = node_lemma.rstrip('-').lower()
                            eq = (node_lemma == lemma)
                            if not eq:
                                graph_lemma_test = wordnet_lemma.lemmatize(lemma, postag)
                                eq = (node_lemma == graph_lemma_test)
                            if eq:
                                break
                    else:
                        node_lemma = node_string.rstrip('-').lower()
                        eq = (node_lemma == lemma)

                # noinspection PyUnboundLocalVariable
                if not eq and node_lemma in LEMMATIZER_EXTRA:
                    extra_mapping = LEMMATIZER_EXTRA[node_lemma]
                    if isinstance(extra_mapping, list):
                        eq = any(i == lemma for i in extra_mapping)
                    else:
                        eq = (extra_mapping == lemma)
                if not eq and carg_type is not None:
                    compare_dict = CARG_COMPARE_DICT.get(carg_type)
                    if compare_dict:
                        eq = (compare_dict.get(node_lemma) == lemma)
                if eq:
                    try:
                        rest_leaf_nodes.remove(node)
                    except KeyError:
                        pass
                    assign_edge_to_node(node, pred_edge)
                    if pred_edge.label not in SPECIAL_LEXICAL_EDGE:
                        # one lexical edge only
                        break

        # last try
        if len(rest_leaf_nodes) == 1 and len(rest_lexical_edges) == 1:
            rest_leaf_node = next(iter(rest_leaf_nodes))
            rest_lexical_edge = next(iter(rest_lexical_edges))
            assign_edge_to_node(rest_leaf_node, rest_lexical_edge, 'rest')

        if rest_lexical_edges:
            raise Exception('Don\'t know how to solve these edges'
                            f' {list(rest_lexical_edges.keys())}')

        self._solve_attachment(structural_edges)

    def _solve_attachment(self, structural_edges):
        can_process = True
        while can_process:
            can_process = False
            # attach to leaf
            if not self.attachment_waiting_list:
                continue
            for edge in structural_edges:
                source, target = edge.nodes
                key_source = (source, 0, edge.label)
                key_target = (target, 1, edge.label)
                as_source = self.attachment_waiting_list.get(key_source)
                as_target = self.attachment_waiting_list.get(key_target)
                if as_source:
                    if target in self.edge_to_node:
                        self.attachment_waiting_list.pop(key_source)

                        if source in self.edge_to_node:
                            self.logger.debug('edge %s is already aligned to %s',
                                              as_source, self.edge_to_node[source])
                        else:
                            tree_node = self.edge_to_node[target]
                            self._log_mapping(tree_node, as_source)
                            self.results[tree_node].add(as_source)
                            self.edge_to_node[source] = tree_node
                            can_process = True
                elif as_target:
                    if source in self.edge_to_node:
                        self.attachment_waiting_list.pop(key_target)

                        if target in self.edge_to_node:
                            self.logger.debug('edge %s is already aligned to %s',
                                              as_target, self.edge_to_node[target])
                        else:
                            tree_node = self.edge_to_node[source]
                            self._log_mapping(tree_node, as_target)
                            self.results[tree_node].add(as_target)
                            self.edge_to_node[target] = tree_node
                            can_process = True

    def _solve_internal(self, pred_edges_list, tree_nodes, structural_edges):
        for pred_edges in pred_edges_list:
            top_waiting = set()
            for edge in pred_edges:
                if edge.label in self.compound_dict:
                    top_waiting.add(edge)
                elif edge.label in self.arg_dict:
                    for key in self.attachment_solver.solve(edge, structural_edges):
                        self.attachment_waiting_list[key] = edge
                else:
                    raise Exception(f'Don\'t know how to solve edge {edge}')

            can_process = True
            while top_waiting and can_process:
                can_process = False
                for top_edge in set(top_waiting):
                    target_tree_nodes = []
                    for edge in structural_edges:
                        if edge.nodes[0] == top_edge.nodes[0]:
                            target = self.edge_to_node.get(edge.nodes[1])
                            target_tree_nodes.append(target)
                    if all(i is not None for i in target_tree_nodes):
                        coverage_nodes = [i for i in tree_nodes
                                          if S.contains(i.span, top_edge.span)]
                        lca = lowest_common_ancestor(coverage_nodes, target_tree_nodes)
                        self.results[lca].add(top_edge)
                        self.edge_to_node[top_edge.nodes[0]] = lca
                        self._log_mapping(lca, top_edge)
                        top_waiting.remove(top_edge)
                        can_process = True

            if top_waiting:
                raise Exception(f'Don\'t know how to solve these edges {top_waiting}')

        self._solve_attachment(structural_edges)

    def solve(self):
        span_to_pred_edges, span_to_tree_nodes, structural_edges = self._group_tree_nodes_by_span()

        many_to_many = self._solve_first_try(span_to_tree_nodes, span_to_pred_edges,
                                             structural_edges)
        for span, terminal_tree_nodes in many_to_many:
            pred_edges = span_to_pred_edges[span]
            self._solve_leaf(terminal_tree_nodes, pred_edges, structural_edges)

        if self.fully_lexicalized:
            self._solve_attachment(structural_edges)
        elif self.internal_waiting_list:
            all_nodes = [i for i in self.cfg_root.traverse_postorder()
                         if not isinstance(i.children[0], Lexicon)]
            self._solve_internal(self.internal_waiting_list, all_nodes, structural_edges)

        flatten_results = functools.reduce(operator.or_, self.results.values(), set())
        for key, value in dict(self.attachment_waiting_list).items():
            if value in flatten_results:
                self.attachment_waiting_list.pop(key)

        if self.attachment_waiting_list:
            values = list(self.attachment_waiting_list.values())
            raise Exception(f'Don\'t know how to solve these edges {values}')

        return self.results
