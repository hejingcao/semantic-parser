# -*- coding: utf-8 -*-

from framework.common.logger import LOGGER

from .hyper_graph import PredEdge

SPECIAL_LEXICAL_LABELS = {'pron', 'poss', 'ellipsis_ref'}

PREFIX_LABELS = {'_pre-_a_ante', '_un-_a_rvrs', '_re-_a_again', '_mis-_a_error', '_co-_a_with'}
SPECIAL_LABEL_PAIRS = {
    ('_like_n_1', '_the_q'), ('_should_v_modal', '_if_x_then'), ('_counter-_a_anti', '_sue_v_1')
}


def _collect_edges(hyper_graph):
    lexical_edges = []
    carg_edges = []
    special_edges = []
    neg_edge = None
    prefix_edge = None
    for edge in hyper_graph.edges:
        if not isinstance(edge, PredEdge):
            continue
        if edge.label.startswith('_'):
            if edge.label in PREFIX_LABELS:
                assert prefix_edge is None, 'Multiple prefix'
                prefix_edge = edge
            else:
                lexical_edges.append(edge)
        elif edge.carg is not None:
            carg_edges.append(edge)
        elif edge.label in SPECIAL_LEXICAL_LABELS:
            special_edges.append(edge)
        elif edge.label == 'neg':
            assert neg_edge is None, 'Multiple <neg>'
            neg_edge = edge
    return lexical_edges, carg_edges, special_edges, neg_edge, prefix_edge


def find_aligned_edge(sentence_id, lexicon, hyper_graph):
    try:
        return _find_aligned_edge(sentence_id, lexicon, hyper_graph)
    except AssertionError:
        LOGGER.warn('Strange condition %s %s', lexicon, hyper_graph.edges)
        return [(lexicon, None)]


def _find_aligned_edge(sentence_id, lexicon, hyper_graph):
    lexical_edges, carg_edges, special_edges, neg_edge, prefix_edge = \
        _collect_edges(hyper_graph)

    if lexical_edges:
        assert not carg_edges or (carg_edges[0].label == 'card')
        num_edges = len(lexical_edges)
        if num_edges == 1:  # only one lexical_edge and no carg edges
            edge = lexical_edges[0]
            return [(lexicon, edge)]

        if num_edges == 2:
            e1, e2 = lexical_edges
            if (e1.label, e2.label) in SPECIAL_LABEL_PAIRS:
                e1.span = e2
                return [(lexicon, e1)]
            elif (e2.label, e1.label) in SPECIAL_LABEL_PAIRS:
                e2.span = e1
                return [(lexicon, e2)]

        LOGGER.warn('Strange condition: %s %s', lexicon, lexical_edges)

    if carg_edges:
        assert not lexical_edges and not special_edges and not neg_edge
        assert len(carg_edges) == 1, 'Multiple carg nodes'
        edge = carg_edges[0]
        rhs = [(f'<{prefix_edge.label}>', prefix_edge)] if prefix_edge is not None else []
        lexicon = f'<{edge.label}>'
        rhs.append((lexicon, edge))
        return rhs

    if special_edges:
        assert not lexical_edges and not prefix_edge

        # !!! do not process special_edges

        # num_edges = len(special_edges)
        # if num_edges == 1:
        #     return [(lexicon, special_edges[0])]
        # elif num_edges == 2:
        #     e1, e2 = special_edges
        #     if e1.label == 'poss' and e2.label == 'pron':
        #         e2.span = e1
        #         return [(lexicon, e2)]
        #     if e1.label == 'pron' and e2.label == 'poss':
        #         e1.span = e2
        #         return [(lexicon, e1)]

    assert not prefix_edge

    return [(lexicon, None)]
