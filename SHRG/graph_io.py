# -*- coding: utf-8 -*-

import re

from framework.common.utils import MethodFactory

from .const_tree import Lexicon
from .hyper_graph import HyperGraph, PredEdge
from .utils import eds_modified as eds

SENTENCE_REGEXP = re.compile(r'\[(.*?)\].*?`(.*)\'')
READERS = MethodFactory()
TEXT_WRITRES = MethodFactory()


def eds_reader_interal(fields):
    if isinstance(fields, str):
        fields = fields.strip().split('\n\n')
    match = re.match(SENTENCE_REGEXP, fields[1])
    assert match is not None, fields[1]
    eds_graph = eds.loads_one(fields[-2])
    return match, eds_graph


@READERS.register('eds')
def eds_reader(fields, params):
    match, eds_graph = eds_reader_interal(fields)
    eds_graph.sentence = match.group(2).strip()
    return HyperGraph.from_eds(eds_graph, modify_label=params.modify_label), eds_graph


@TEXT_WRITRES.register('eds')
def eds_writer(out, sentence_id, eds_graph, modify_label):
    out.write(sentence_id + '\n')
    out.write(eds_graph.sentence + '\n')
    out.write(getattr(eds_graph, 'lemma_sequence', '') + '\n')

    name2index = {}
    hyper_graph = HyperGraph.from_eds(eds_graph, modify_label=modify_label)
    node2pred_edge = {}
    edges = []
    for edge in hyper_graph.edges:
        if isinstance(edge, PredEdge):
            assert len(edge.nodes) == 1, 'strange PredEdge'
            node = edge.nodes[0]
            assert node not in node2pred_edge, 'invalid EDS'
            node2pred_edge[node] = edge
        else:
            edges.append(edge)

    out.write(str(len(hyper_graph.nodes)) + '\n')
    for index, node in enumerate(hyper_graph.sorted_nodes):
        name2index[node.name] = index
        pred_edge = node2pred_edge[node]
        node = eds_graph.node(node.name)
        props = node.properties
        out.write('\t'.join([
            str(index), node.nodeid, pred_edge.label, node.pred.lemma,
            node.pred.pos or '#',
            node.pred.sense or '#',
            node.carg or '#',
            props.get('TENSE', '#'),
            props.get('NUM', '#'),
            props.get('PERS', '#'),
            props.get('PROG', '#'),
            props.get('PERF', '#')
        ]) + '\n')

    out.write(str(name2index[eds_graph.top]) + '\n')
    out.write(str(len(edges)) + '\n')
    for edge in edges:
        assert len(edge.nodes) == 2, 'strange HyperEdge'
        out.write('{:d}\t{:d}\t{}\n'.format(name2index[edge.nodes[0].name],
                                            name2index[edge.nodes[1].name],
                                            edge.label))
    out.write('\n')


def _hyper_edge_to_string(hyper_edge, write_detail):
    fmt = '<{0}: {2} {3}>' if write_detail else '{} {:d} {} {}'
    sep = ' -- ' if write_detail else ' '
    return fmt.format(hyper_edge.label,
                      len(hyper_edge.nodes),
                      sep.join(node.name for node in hyper_edge.nodes),
                      'Y' if hyper_edge.is_terminal else 'N')


def _text_hrg_writer(out, hrg, write_detail):
    edge2index = {None: -1}

    def _get_edge_key(edge):
        return len(edge.nodes), edge.label, tuple(int(node.name) for node in edge.nodes)

    hrg_lhs = hrg.lhs
    hrg_rhs = hrg.rhs

    fmt = 'HRG-RHS: NodeCount={:d} EdgeCount={:d}\n' if write_detail else '{:d} {:d}\n'
    out.write(fmt.format(len(hrg_rhs.nodes), len(hrg_rhs.edges)))

    for index, edge in enumerate(sorted(hrg_rhs.edges, key=_get_edge_key)):
        edge2index[edge] = index
        if write_detail:
            out.write('#{}: '.format(index))
        out.write(_hyper_edge_to_string(edge, write_detail) + '\n')

    fmt = 'EPs: {1}\n' if write_detail else '{} {}\n'
    sep = ' -- ' if write_detail else ' '
    out.write(fmt.format(len(hrg_lhs.nodes), sep.join(node.name for node in hrg_lhs.nodes)))

    return edge2index


def _text_cfg_writer(out, cfg, edge2index, write_detail):
    fmt = '{}@{}' if write_detail else '{} {:d}'
    sep = ' || ' if write_detail else ' '
    cfg_rhs = cfg.rhs or [('???', None)]
    cfg_rhs_string = sep.join(
        fmt.format(item.string if isinstance(item, Lexicon) else item, edge2index.get(edge))
        for item, edge in cfg_rhs
    )

    fmt = 'CFG: {0} => {2}\n\n' if write_detail else '{} {:d} {}\n\n'
    out.write(fmt.format(cfg.lhs, len(cfg_rhs), cfg_rhs_string))


@TEXT_WRITRES.register('mapping')
def text_shrg_mapping_writer(out, hrg2cfg_mapping, head_counter, write_detail=False):
    out.write(str(len(hrg2cfg_mapping)) + '\n')
    for hrg, cfgs in hrg2cfg_mapping.items():
        has_semantics = not isinstance(hrg, str)
        if write_detail:
            out.write('' if has_semantics else 'No semantic part\n')
        else:
            out.write('1\n' if has_semantics else '0\n')

        unique_label = hrg.lhs.unique_label if has_semantics else hrg
        total_count = head_counter[unique_label]
        assert total_count > 0

        edge2index = {None: -1}
        if has_semantics:
            edge2index = _text_hrg_writer(out, hrg, write_detail)

        fmt = 'CFG parts: {}\n' if write_detail else '{}\n'
        out.write(fmt.format(len(cfgs)))
        for index, cfg, count in cfgs:
            fmt = 'Index: {:d} P={:f} {:d}/{:d}\n' if write_detail else '{0:d} {2:d} {3:d}\n'
            out.write(fmt.format(index, count / total_count, count, total_count))

            _text_cfg_writer(out, cfg, edge2index, write_detail)


@TEXT_WRITRES.register('shrg')
def text_shrg_writer(out, rules_counter, write_detail=False):
    out.write(str(len(rules_counter)) + '\n')
    for index, (shrg_rule, counter_item) in enumerate(rules_counter):
        fmt = 'Index: {:d}\n' if write_detail else '{:d}\n'
        out.write(fmt.format(index))

        has_semantics = shrg_rule.hrg is not None
        if write_detail:
            out.write('' if has_semantics else 'No semantic part\n')
        else:
            out.write('1\n' if has_semantics else '0\n')

        edge2index = {None: -1}
        if has_semantics:
            assert shrg_rule.hrg.lhs.label == shrg_rule.cfg.lhs, 'Invalid SHRG Rule'
            edge2index = _text_hrg_writer(out, shrg_rule.hrg, write_detail)

        _text_cfg_writer(out, shrg_rule.cfg, edge2index, write_detail)


@TEXT_WRITRES.register('tree')
def text_tree_writer(out, sentence_id, const_tree, shrg_rules=None):
    """
    @type const_tree: ConstTree
    """
    out.write(sentence_id + '\n')

    inner_nodes = list(const_tree.traverse_postorder())
    assert shrg_rules is None or len(inner_nodes) == len(shrg_rules), 'mismatch !!'

    inner_count = len(inner_nodes)

    lexicons = list(const_tree.generate_lexicons())
    # use id as key, because the __hash__ of Lexicon
    node2index = {id(node): index for index, node in enumerate(inner_nodes)}
    node2index.update({id(lexicon): index
                       for index, lexicon in enumerate(lexicons, inner_count)})
    parent_nodes = [-1] * (inner_count + len(lexicons))

    for index, node in enumerate(inner_nodes):
        for child in node.children:
            parent_nodes[node2index[id(child)]] = index

    inner_nodes.extend(lexicons)
    out.write(str(len(inner_nodes)) + '\n')
    for index, node in enumerate(inner_nodes):
        shrg_rule_index = -1
        if index < inner_count and shrg_rules is not None:
            shrg_rule_index = shrg_rules[index]
        out.write('\t'.join([
            str(index),
            node.tag,
            str(parent_nodes[index]),
            'L' if isinstance(node, Lexicon) else 'T',
            str(shrg_rule_index)
        ]) + '\n')
    out.write('\n')
