# -*- coding: utf-8 -*-

import os
import re

from framework.common.logger import LOGGER, open_file
from framework.data_structures.union_find import UnionNode
from SHRG.utils.lexicon import get_lemma_and_pos, get_wordnet


class EdsWrapper:
    class Node:
        __slots__ = [
            'lemma', 'pos', 'carg', 'label', 'span', 'sense',
            'outgoing_edges', 'incoming_edges',
        ]

        def __init__(self, lemma, pos, sense, carg, label, span, outgoing_edges):
            self.lemma = lemma
            self.pos = pos
            self.sense = sense
            self.carg = carg
            self.label = label
            self.outgoing_edges = outgoing_edges
            self.incoming_edges = []
            self.span = span

        def __repr__(self):
            return str(self)

        def __str__(self):
            return '{}<{}:{}>'.format(self.label, self.span[0], self.span[1])

    def __init__(self, sentence, original_nodes, original_edges):
        self.sentence = sentence

        nodes = {}
        for node, edges in zip(original_nodes, original_edges):
            # node span
            span = -1, -1
            if node.lnk is not None:
                span = node.cfrom, node.cto
            # node properties
            nodes[node.nodeid] = self.Node(lemma=node.pred.lemma,
                                           pos=node.pred.pos,
                                           sense=node.pred.sense,
                                           carg=node.carg,
                                           label=node.pred.short_form(),
                                           span=span,
                                           outgoing_edges=edges)

        self.nodes = nodes  # type: Dict[NodeID, 'EdsGraph.Node']

        for source_id in nodes:         # set incoming_edges
            edges = nodes[source_id].outgoing_edges
            for elabel, target_id in edges.items():
                nodes[target_id].incoming_edges.append((elabel, source_id))


def _is_overlapped(span1, span2):
    return span1[0] < span2[1] and span1[1] > span2[0]


def _is_any_overlapped(spans):
    spans = list(spans)
    for index, span1 in enumerate(spans):
        for span2 in spans[:index]:
            if _is_overlapped(span1, span2):
                return True
    return False


def _span_subtract_span(span1, span2):  # span1 - span2
    if span2[0] > span1[0] and span2[1] >= span1[1]:
        return span1[0], span2[0]
    if span2[1] < span1[1] and span2[0] <= span1[0]:
        return span2[1], span1[1]


def _merge_spans(spans):
    missing_spans = []
    last_end = -1
    for start, end in sorted(spans):
        if last_end != -1 and start > last_end:
            missing_spans.append((last_end, start))
        last_end = end
    return missing_spans


def _find_span_in_spans_multi(target_span, spans):
    first, end = 0, -1
    while first < len(spans):
        span = spans[first]
        if target_span[0] <= span[0] and target_span[0] >= end:
            break
        end = span[1]
        first += 1

    last, start = len(spans), spans[-1][1] + 1
    while last > first:
        span = spans[last - 1]
        if target_span[1] >= span[1] and target_span[1] <= start:
            break
        start = span[0]
        last -= 1

    return first, last


def _find_span_in_spans_single(target_span, spans):
    for index, span in enumerate(spans):
        if span[0] <= target_span[0] and span[1] >= target_span[1]:
            return index


def _get_parents_by_label(graph, node, label, return_edge_label=False):
    parents = []
    for elabel, parent_id in node.incoming_edges:
        parent = graph.nodes[parent_id]
        if parent.label == label:
            if return_edge_label:
                parents.append((elabel, parent))
            else:
                parents.append(parent)
    return parents


def _match_label(input_value, pattern):
    if isinstance(pattern, str):
        return input_value == pattern
    if callable(pattern):
        return pattern(input_value)
    if isinstance(pattern, (tuple, list, set)):
        return input_value in pattern

    return re.match(pattern, input_value)


def get_word_bounds(string):
    s, e = 0, len(string)
    while s < e and not string[s].isalnum():
        s += 1
    while s < e and not string[e - 1].isalnum():
        e -= 1
    return s, e


def get_word_bounds2(string, chars):
    s, e = 0, len(string)
    while s < e and string[s] in chars:
        s += 1
    while s < e and string[e - 1] in chars:
        e -= 1
    return s, e


class ConceptSpanFixer:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__),
                                       'concept_fixer_config.py')
        attrs = {}
        code = open_file(config_path, 'r').read()
        exec(compile(code, config_path, 'exec'), {'re': re}, attrs)
        for name, value in attrs.items():
            if name.isupper():
                setattr(self, name, value)

    def get_clusters(self, sentence, nodes):
        nodes = {nodeid: node for nodeid, node in nodes.items()
                 if (node.label.startswith('_')
                     or node.label in self.SPECIAL_LABELS_AS_NORMAL_NODE
                     or node.carg is not None)}

        groups = {nodeid: UnionNode(nodeid) for nodeid in nodes}
        for nodeid, node in nodes.items():
            for other_nodeid, other_node in nodes.items():
                if nodeid != other_nodeid \
                   and _is_overlapped(nodes[other_nodeid].span, node.span):
                    groups[nodeid].union(groups[other_nodeid])

        clusters = {}
        for group in groups.values():
            clusters.setdefault(group.find().value, set()).add(group.value)

        return list(clusters.values())

    def fix_stage1(self, graph):
        """deal with nodes like pre-, mid-, re-"""
        nodes = graph.nodes
        for nodeid, node in nodes.items():
            match = re.match(self.PREFIX_REGEX, node.label)
            arg1_nodeid = node.outgoing_edges.get('ARG1')
            if match and arg1_nodeid:
                LOGGER.debug('prefix: %s', match.group(0))
                prefix = match.group(1).rstrip('-')

                arg1_node = nodes[arg1_nodeid]
                if arg1_node.label == 'nominalization':  # skip nominalization
                    arg1_node = nodes[arg1_node.outgoing_edges.get('ARG1')]

                # change span of arg1_node to the span without prefix
                beg = graph.sentence.lower().find(prefix, *node.span)
                if beg != -1:
                    end = beg + len(prefix)
                    LOGGER.debug('    `%s` <-> `%s`: %s', node.span, (beg, end), node.label)
                    node.span = beg, end
                    if graph.sentence[end] == '-':
                        end += 1
                    assert end < arg1_node.span[1]
                    LOGGER.debug('    `%s` <-> `%s`: %s',
                                 arg1_node.span, (end, arg1_node.span[1]),
                                 arg1_node.label)
                    arg1_node.span = end, arg1_node.span[1]

    def fix_stage2(self, graph):
        nodes = graph.nodes
        sentence = graph.sentence

        clusters = self.get_clusters(sentence, nodes)

        states = []
        for cluster in clusters:
            if len(cluster) > 1:
                cluster_nodes = {nodeid: nodes[nodeid] for nodeid in cluster}
                assert not self._fix_stage2(graph, cluster_nodes), cluster_nodes

        return states

    def _fix_stage2(self, graph, cluster_nodes):
        sentence = graph.sentence
        nodes = graph.nodes

        covered_spans = {}
        for nodeid, node in list(cluster_nodes.items()):
            is_number = _match_label(node.label, self.IS_NUMBER)
            is_name = node.carg is not None or _match_label(node.label, self.IS_NAMED)
            is_much = _match_label(node.label, self.IS_MUCH)
            is_comp = _match_label(node.label, self.IS_COMP)
            is_neg = _match_label(node.label, self.IS_NEG)

            if not (is_number or is_name or is_much or is_comp or is_neg
                    or node.label.startswith('_')):
                continue

            matched_tokens, sep = \
                self._get_matched_tokens(node, sentence, nodes,
                                         is_number, is_name, is_much, is_comp, is_neg)

            if not matched_tokens:
                continue

            matched_tokens = [(span, word)
                              for span, word in matched_tokens
                              if span not in covered_spans]

            if len(matched_tokens) > 1 \
               and all(word == matched_tokens[0][1] for _, word in matched_tokens):
                matched_tokens = matched_tokens[:1]  # select first one

            if len(matched_tokens) == 1:
                span = matched_tokens[0][0]
                for other_span in covered_spans:
                    if _is_overlapped(other_span, span):
                        __import__("pdb").set_trace()
                    assert not _is_overlapped(other_span, span)
                covered_spans[span] = nodeid, node
                node.span = span

        if len(cluster_nodes) != len(covered_spans):
            solved_nodeids = {nodeid for nodeid, _ in covered_spans.values()}
            unsolved_nodes = [node for nodeid, node in cluster_nodes.items()
                              if nodeid not in solved_nodeids]
            unsolved_labels = {node.label for node in unsolved_nodes}
            if unsolved_labels in self.IGNORE_LABEL_SETS:
                return

            if not _is_any_overlapped(node.span for node in cluster_nodes.values()):
                # all nodes in cluster are disconnected
                return

            missing_spans = _merge_spans(covered_spans)
            if not self._fix_stage2_special(sentence, solved_nodeids, unsolved_nodes,
                                            missing_spans, covered_spans):
                return cluster_nodes, covered_spans

    def _fix_stage2_special(self, sentence, solved_nodeids, unsolved_nodes,
                            missing_spans, covered_spans):
        SPECIAL_NEG_REGEX = '(^.+(?:\'t|not)$)|^un|^dis'

        missing_tokens = [sentence[span[0]:span[1]] for span in missing_spans]
        if all(token == '/' for token in missing_tokens) \
           and all(node.label in {'_and_c', '_per_p'} for node in unsolved_nodes) \
           and len(unsolved_nodes) == len(missing_spans):
            for node, span in zip(unsolved_nodes, missing_spans):
                node.span = span
            return True

        if len(unsolved_nodes) == 2:
            node1, node2 = unsolved_nodes
            if node1.span == node2.span:
                span = node1.span
                string = sentence[span[0]:span[1]]
                if re.match(SPECIAL_NEG_REGEX, string.strip(self.PUNCTUATIONS)):
                    if node1.label == 'neg' or node2.label == 'neg':
                        LOGGER.debug('special neg: %s %s %s', node1, node2, string)
                        return True

        if len(unsolved_nodes) == 1:
            node = unsolved_nodes[0]
            rest_span = node.span
            for span in covered_spans:
                rest_span = _span_subtract_span(rest_span, span)
                if rest_span is None:
                    break

            if rest_span is not None:
                string = sentence[rest_span[0]:rest_span[1]]
                s, e = get_word_bounds2(string, '-/')
                node.span = rest_span[0] + s, rest_span[0] + e
                LOGGER.debug('use rest span: %s %s %s', node, string,
                             sentence[node.span[0]:node.span[1]])
                return True

    def _get_tokens(self, node, sentence):
        span = node.span

        string = sentence[span[0]:span[1]].lower()
        # 清理两侧的标点, 但是记录位置
        s, e = get_word_bounds(string)
        stripped_string = string[s:e]

        tokens = None
        try_chars, *special_cases = self.SPLIT_STRINGS
        for matcher, sep, extra in special_cases:
            result = _match_label(stripped_string, matcher)
            if result:
                if extra and isinstance(extra[0], str):
                    tokens = extra
                elif extra and isinstance(extra[0], int):
                    tokens = [result.group(index) for index in extra]
                else:
                    tokens = stripped_string.split(sep)
                break

        if tokens is None:
            for sep in try_chars:
                tokens = stripped_string.split(sep)
                if len(tokens) > 1:
                    break

        token_starts = [s]
        for token in tokens[:-1]:
            token_starts.append(token_starts[-1] + len(sep) + len(token))

        new_token_starts = []
        new_tokens = []
        for token, token_start in zip(tokens, token_starts):
            new_token_starts.append(token_start)
            for subtoken in token.split(' '):
                new_tokens.append(subtoken)
                new_token_starts.append(new_token_starts[-1] + len(subtoken) + 1)
            new_token_starts.pop()

        if len(new_tokens) <= 1:
            return None, None, None

        start = node.span[0]
        for token, token_start in zip(new_tokens, new_token_starts):
            token_start += start
            if token != sentence[token_start:token_start + len(token)].lower():
                __import__("pdb").set_trace()

        return new_tokens, new_token_starts, sep

    def _get_carg(self, node, sep):
        return node.carg.lower() \
            .replace('+', ' ') \
            .replace('_', ' ').strip().rstrip(sep)

    def _get_matcher(self, node, sentence, sep,
                     is_number, is_name, is_much, is_comp, is_neg):
        if is_number:
            carg = self._get_carg(node, sep)
            matcher = self.CARD_TRANSFORM.get(carg, carg)
        elif is_comp:
            matcher = self.COMP_REGEX
        elif is_name:
            matcher = self._get_carg(node, sep)
            # 去掉结尾的 '-'
            if matcher[-1] == sep:
                matcher = matcher[:-1]
            matcher = self.NAME_ENTITY.get(matcher, matcher)
        elif is_neg:
            matcher = self.NEG_REGEX
        elif is_much:
            matcher = self.MUCH_REGEX
        else:
            matcher, _ = get_lemma_and_pos(node.label)

        matcher = self.REGEX_MAP.get(matcher, matcher)

        return matcher

    def _get_lemma(self, token, pos_tag):
        token = token.strip(self.PUNCTUATIONS)
        try:
            return get_wordnet().lemmatize(token, pos_tag)
        except Exception:
            return token

    def _get_matched_tokens(self, node, sentence, nodes,
                            is_number, is_name, is_much, is_comp, is_neg):
        tokens, token_starts, sep = self._get_tokens(node, sentence)
        if tokens is None:
            return None, None

        matcher = self._get_matcher(node, sentence, sep,
                                    is_number, is_name, is_much, is_comp, is_neg)

        matched_tokens = []

        for token, token_start in zip(tokens, token_starts):
            start = token_start + node.span[0]
            token_lemma = self._get_lemma(token, node.pos)
            if isinstance(matcher, str):
                pred = token == matcher \
                    or (token_lemma, matcher) in self.TOKEN_LEMMA_SPECIAL_CASE \
                    or token_lemma.startswith(matcher) \
                    or (len(token_lemma) >= 3 and matcher.startswith(token_lemma)) \
                    or (token_lemma.endswith('ied') and
                        token_lemma[:-3] == matcher[:-1])
            else:
                pred = re.match(matcher, token_lemma)

            if pred:
                matched_tokens.append(((start, start + len(token)), token))

        exact_tokens = [(span, token) for span, token in matched_tokens if token == matcher]
        if len(exact_tokens) == 1:
            if len(matched_tokens) > 1:
                LOGGER.debug('use exact match: %s %s', node, matched_tokens)
            matched_tokens = [exact_tokens[0]]

        return matched_tokens, sep

    def fix_stage3(self, graph, tokens, spans):

        chars = '-/.,"?!;() '
        for node in graph.nodes.values():
            start, end = node.span

            first, last = _find_span_in_spans_multi((start, end), spans)
            if first == last:
                continue

            while first < last \
                    and all((char in chars) for char in tokens[first]):
                first += 1

            while first < last \
                    and all((char in chars) for char in tokens[last - 1]):
                last -= 1

            if first < last:
                span = spans[first][0], spans[last - 1][1]
                if span != node.span:
                    node.span = span
