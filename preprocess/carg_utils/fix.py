# -*- coding: utf-8 -*-

import re

from framework.data_structures.union_find import UnionNode
from SHRG.utils.lexicon import get_wordnet

REGEX_BAD = re.compile('bad|worse|worst')
REGEX_GOOD = re.compile('good|better|best')
REGEX_NOT_WORD = re.compile(r'[.,\-"?!();#{}+]+')
NUMBERS = [['1', 'one', 'a', 'first'],
           ['2', 'two', 'second'],
           ['3', 'three', 'third'],
           ['4', 'four', 'fourth'],
           ['5', 'five', 'fifth'],
           ['6', 'six', 'sixth'],
           ['7', 'seven', 'seventh'],
           ['8', 'eight', 'eighth'],
           ['9', 'nine', 'ninth'],
           ['10', 'ten', 'tenth'],
           ['11', 'eleven', 'eleventh'],
           ['12', 'twelve', 'twelfth'],
           ['13', 'thirteen', 'thirteenth'],
           ['14', 'fourteen', 'fourteenth'],
           ['15', 'fifteen', 'fifteenth'],
           ['16', 'sixteen', 'sixteenth'],
           ['17', 'seventeen', 'seventeenth'],
           ['18', 'eighteen', 'eighteenth'],
           ['19', 'nineteenth', 'nineteen'],
           ['20', 'twenty', 'twentieth'],
           ['30', 'thirty', 'thirtieth'],
           ['40', 'forty', 'fortieth'],
           ['50', 'fifty', 'fiftieth'],
           ['60', 'sixty', 'sixtieth'],
           ['70', 'seventy', 'seventieth'],
           ['80', 'eighty', 'eightieth'],
           ['90', 'ninety', 'ninetieth'],
           ['100', 'hundred', 'hundredth'],
           ['1000', 'thousand', 'thousandth'],
           ['1000000', 'million', 'millionth'],
           ['1000000000', 'billion', 'billionth'],
           ['1000000000000', 'trillion', 'trillionth']]
CARD_TRANSFORM = {x[0]: re.compile('|'.join(x)) for x in NUMBERS}
CARD_REGEX = re.compile(r'[1-9][0-9]*|' +
                        '|'.join(_ for x in NUMBERS for _ in x[1:]))
MUCH_REGEX = re.compile('more|much')
COMP_REGEX = re.compile('more|less|most|least')
# 前缀
REGEX_PREFIX = re.compile('_([a-z]+-|mid)_')
# 可以作为正常的结点
SPECIAL_LABELS_AS_NORMAL_NODE = ['named', 'card', 'ord',
                                 'dofw',  # day of week
                                 'yofc',  # year of century
                                 'mofy',  # month of year
                                 'numbered_hour',
                                 'holiday', 'fraction', 'season', 'named_n',
                                 'year_range', 'much-many_a']
COMP_NODES = ['comp', 'comp_less', 'superl']
# 对齐时忽略结点
SPECIAL_LABELS_SHOULD_NOT_BE_ALIGNED = [
    # replace with => label.endswith('_q')
    # 'udef_q', 'proper_q', 'pronoun_q',
    # 'def_explicit_q', 'def_implicit_q', 'number_q',
    'part_of', 'generic_entity'
]
# 匹配 'b.a.t' 'U.S.' 这类
REGEX_SPECIAL_NAMED = re.compile('^[a-z](\.[a-z])+\.?$')
# 匹配 SKR200 这类
REGEX_SKR = re.compile('^([^\w]*skr)([0-9\.]+[^\w]*)$')
PUNCTUATIONS = '.,"?!;() '
# 特殊的情况
TOKEN_LEMMA_SPECIAL_CASE = set([('ft', 'foot'),
                                ('gray', 'grey'),
                                ('offshore', 'off-shore'),
                                ('n.m', 'new mexico'),
                                ('hi', 'hawaii'),
                                ('vice', 'co')])
NAME_ENTITY = {
    'un': re.compile('un|u\.n\.?'),
    'us': re.compile('us|u\.s\.?'),
    'att': re.compile('att|at&t')
}

BE_1_FORMS = ['is', 'are', 'am', 'was', 'were', 'being', 'be',
              'been', '\'s', '\'re', '\'m']
V_TENSE_FORMS = ['has', 'have', 'had', 'will', 'having', 'shall',
                 'would', '\'d', '\'ll', '\'ve']
W_FORMS = ['who', 'what', 'which', 'where', 'when', 'whom', 'that']
DO_FORMS = ['did', 'does']


def spans_overlapped(s1, s2):
    return s1[1] > s2[0] and s2[1] > s1[0]


def get_lemma(token: str, pos_tag: str):
    """ 使用 nltk 取找词的原型. """
    token = token.strip(PUNCTUATIONS)
    try:
        return get_wordnet().lemmatize(token, pos_tag)
    except Exception:
        return token


def get_word_bounds(string):
    s, e = 0, len(string)
    while s < e and not string[s].isalnum():
        s += 1
    while s < e and not string[e - 1].isalnum():
        e -= 1
    return s, e


def is_normal_node(label: str):
    return label.startswith('_') or label in SPECIAL_LABELS_AS_NORMAL_NODE


def is_aligned_node(label: str):
    return is_normal_node(label) or \
        label.endswith('_q') or \
        label in SPECIAL_LABELS_SHOULD_NOT_BE_ALIGNED


def is_prefix_node(label: str):
    match = re.match(REGEX_PREFIX, label)
    return match and match.group(1) != 'up-'


def _get_overlapped_nodes(sentence, nodes):
    selected_nodeids = []
    for nodeid in nodes:
        node = nodes[nodeid]
        beg, end = node.span
        string = sentence[beg:end].lower().strip(PUNCTUATIONS)
        if (is_normal_node(node.label) or node.label in COMP_NODES) \
           and (string.find('-') != -1
                or string.find('/') != -1
                or re.match(REGEX_SKR, string)
                or re.match(REGEX_SPECIAL_NAMED, string)):
            selected_nodeids.append(nodeid)

    groups = {nodeid: UnionNode(nodeid) for nodeid in selected_nodeids}
    for nodeid in selected_nodeids:
        node = nodes[nodeid]
        for other_nodeid in selected_nodeids:
            if nodeid != other_nodeid \
               and spans_overlapped(nodes[other_nodeid].span, node.span):
                groups[nodeid].union(groups[other_nodeid])

    clusters = {}
    for group in groups.values():
        clusters.setdefault(group.find().value, set()).add(group.value)

    def node_order(nodeid):
        node = nodes[nodeid]
        if node.label[0] == '_':
            return -1
        return node.span[0]

    return [sorted(cluster, key=node_order)
            for cluster in clusters.values() if len(cluster) > 1]


def fix_eds_prefix_node_span(graph):
    """处理前缀结点 (i.e. pre-, mid-, re-) """
    nodes = graph.nodes
    for nodeid in nodes:
        node = nodes[nodeid]
        match = re.match(REGEX_PREFIX, node.label)
        arg1 = node.outgoing_edges.get('ARG1')
        # up- 不是前缀
        if match and arg1 and match.group(1) != 'up-':  # 前缀
            # LOGGER.debug('%s: prefix: %s', graph.filename, match.group(0))
            prefix = match.group(1)
            if prefix.endswith('-'):
                prefix = prefix[:-1]
            arg1_node = nodes[arg1]
            if arg1_node.label == 'nominalization':
                arg1_node = nodes[arg1_node.outgoing_edges.get('ARG1')]
            # arg1 的 span 改为去掉前缀的
            beg = graph.sentence.lower().find(prefix, *node.span)
            if beg != -1:
                end = beg + len(prefix)
                # LOGGER.debug('    `%s` <-> `%s`: %s',
                #              node.span, (beg, end), node.label)
                node.span = beg, end
                if graph.sentence[end] == '-':
                    end += 1
                assert end < arg1_node.span[1]
                # LOGGER.debug('    `%s` <-> `%s`: %s',
                #              arg1_node.span, (end, arg1_node.span[1]),
                #              arg1_node.label)
                arg1_node.span = end, arg1_node.span[1]


def _fix_eds_dash_node_span(graph):
    clusters = _get_overlapped_nodes(graph.sentence, graph.nodes)

    for cluster in clusters:
        _fix_eds_dash_node_span_inner(graph, cluster)


def _get_dash_node_matched_tokens(node, sentence, nodes,
                                  is_number, is_name, is_much, is_comp):
    # 处理连词 '-' '/'
    span = node.span

    string = sentence[span[0]:span[1]].lower()
    # 清理两侧的标点, 但是记录位置
    s, e = get_word_bounds(string)
    striped_string = string[s:e]

    if striped_string.startswith('y-mp'):  # dirty work
        sep = '/'
    else:
        sep = '-'
    tokens = list(x for x in striped_string.split(sep) if x != '')

    # 处理 skr200
    match = re.match(REGEX_SKR, striped_string)
    if match:
        tokens = match.group(1), match.group(2)
        sep = ''
    elif striped_string == 'everytime':
        tokens = 'every', 'time'
        sep = ''

    # 处理 U.S, B.A.T 这种
    match = re.match(REGEX_SPECIAL_NAMED, striped_string)
    if match:
        tokens = list(x for x in striped_string.split('.') if x != '')
        sep = '.'

    # 数据集中出现 the like 10+ 次
    if striped_string == 'the like':
        tokens = 'the', 'like'
        sep = ' '

    # '/' 在后面考虑
    if len(tokens) <= 1:
        tokens = list(x for x in striped_string.split('/')
                      if x != '')
        sep = '/'

    if len(tokens) >= 2 and node.label == '_per_p':
        arg2 = node.outgoing_edges.get('ARG2')
        if arg2:
            # _per_p 结点 span 由 'xx/yy' 改为 '/'
            beg = nodes[arg2].span[0]
            node.span = beg - 1, beg
        return None
    elif len(tokens) >= 2 and node.pos == 'c' and sep == '/':  # _and_c 结点
        arg1 = node.outgoing_edges.get('L-INDEX') or \
            node.outgoing_edges.get('L-HNDL')
        if arg1:
            end = nodes[arg1].span[1]
            node.span = end, end + 1  # '/'
        return None

    if len(tokens) <= 1:
        return None

    if is_number:
        regex = CARD_REGEX
    elif is_comp:
        regex = COMP_REGEX
    elif is_name:
        regex = re.compile('.*')
    elif is_much:
        regex = MUCH_REGEX
    else:
        regex = re.compile(node.label[1:].split('_')[0])

    token_lens = list(len(x) for x in tokens)
    token_lens[0] += s
    token_lens[-1] += len(string) - e

    pos = span[0]
    matched_tokens = []
    for token, token_len in zip(tokens, token_lens):
        is_matched = regex.match(token)
        if not is_matched and isinstance(regex, str):
            is_matched = regex.match(get_wordnet().lemmatize(token, node.pos))
        if is_matched:
            matched_tokens.append((token, (pos, pos + token_len)))

        pos += token_len + len(sep)

    return matched_tokens, sep


def _fix_eds_dash_node_span_inner(graph, cluster):
    sentence = graph.sentence
    nodes = graph.nodes

    cluster_nodes = {nodeid: nodes[nodeid] for nodeid in cluster}
    covered_spans = set()
    for nodeid in cluster:
        node = nodes[nodeid]

        is_number = node.label in ['card', 'yofc', 'fraction', 'ord', 'numbered_hour']
        is_name = node.label in ['named', 'mofy', 'dofw', 'season', 'named_n', 'holiday']
        is_much = node.label == 'much-many_a'
        is_comp = node.label in COMP_NODES

        if not (is_number or is_name or is_much or is_comp or node.label.startswith('_')):
            continue

        matched_tokens = _get_dash_node_matched_tokens(
            node, sentence, nodes,
            is_number, is_name, is_much, is_comp)

        if matched_tokens is not None:
            matched_tokens, sep = matched_tokens

            matched_tokens = [(word, span)
                              for word, span in matched_tokens
                              if span not in covered_spans]
        if not matched_tokens:
            continue

        span = None
        if len(matched_tokens) == 1:
            span = matched_tokens[0][1]
            _set_node_span(node, span, graph)
        elif is_number:
            span = _fix_dash_for_number(graph, node, matched_tokens, cluster_nodes)
        elif is_name:
            span = _fix_dash_for_named(graph, node, sep, matched_tokens, cluster_nodes)

        if span is not None:
            covered_spans.add(span)


def fix_eds_dash_node_span(graph):
    """ 处理 '-' '/' 连接的词组. """
    return _fix_eds_dash_node_span(graph)


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


def _set_node_span(node, span, graph):
    sentence = graph.sentence
    start, end = span
    if node.span[1] >= end and end < len(sentence):
        if sentence[end] == '-':
            end += 1
    node.span = start, end


def _fix_dash_for_number(graph, node, matched_tokens, cluster_nodes):
    span = None
    if len(cluster_nodes) == 2 and len(matched_tokens) == 2:
        node1, node2 = list(cluster_nodes.values())
        other_node = node2 if node1 is node else node1
        plus1 = _get_parents_by_label(graph, node, 'plus', return_edge_label=True)
        plus2 = _get_parents_by_label(graph, other_node, 'plus', return_edge_label=True)
        if len(plus1) == 1 and len(plus2) == 1:
            label1, plus1 = plus1[0]
            label2, plus2 = plus2[0]
            if plus1 is plus2:
                if label1 < label2:  # order by edge label
                    _, span = matched_tokens[0]
                else:
                    _, span = matched_tokens[1]
                node.extra = 'plus-2-card'

        if span is None:
            if node.label == 'yofc' and other_node.label == 'card':
                _, span = matched_tokens[0]
                node.extra = 'yofc-card'
            elif other_node.label == 'yofc' and node.label == 'card':
                _, span = matched_tokens[1]
                node.extra = 'yofc-card'

    if span is None:
        if _get_parents_by_label(graph, node, 'interval_p_start'):
            _, span = matched_tokens[0]
            node.extra = 'interval-start'
        elif _get_parents_by_label(graph, node, 'interval_p_end'):
            _, span = matched_tokens[-1]
            node.extra = 'interval-end'
        elif node.outgoing_edges.get('ARG1'):  # TODO: year range
            _, span = matched_tokens[0]
            node.extra = 'number-start'
        else:
            _, span = matched_tokens[-1]
            node.extra = 'number-end'

    _set_node_span(node, span, graph)
    return span


def _fix_dash_for_named(graph, node, sep, matched_tokens, cluster_nodes):
    regex = '(single|double|triple)-[a-z]-'
    if re.match(regex, '-'.join(_[0] for _ in matched_tokens)):
        # NOTE: handle single-A-plus...
        _, (s, _) = matched_tokens[0]
        _, (_, e) = matched_tokens[1]
        matched_tokens[0] = None, (s, e)
        matched_tokens[1] = matched_tokens[2]
        matched_tokens.pop()

    span = node.span

    parents = []
    for parent in _get_parents_by_label(graph, node, 'compound'):
        start, end = parent.span
        if start <= span[0] and end >= span[1]:  # cover node.span
            parents.append(parent)

    left = 'ARG1'
    right = 'ARG2'
    reverse = False

    if len(parents) == 1:
        node.extra = '1-compound'
        parent = parents[0]
        reverse = sep == '-'
    elif len(parents) > 1:
        node.extra = '2-compounds'
        parents.sort(key=lambda x: x.span[1] - x.span[0])
        parent = parents[0]
        reverse = sep == '-'
    else:
        node.extra = '0-compound'
        ands = _get_parents_by_label(graph, node, '_and_c')
        parent = None
        if ands:
            left = 'L-INDEX'
            right = 'R-INDEX'
            parent = ands[0]

    if parent is not None:
        arg1 = parent.outgoing_edges.get(left)
        arg2 = parent.outgoing_edges.get(right)
        if arg1 is not None and graph.nodes[arg1] is node:
            _, span = matched_tokens[-1 if reverse else 0]
        elif arg2 is not None and graph.nodes[arg2] is node:
            _, span = matched_tokens[0 if reverse else -1]
        # TODO: special cases

    _set_node_span(node, span, graph)
    return span
