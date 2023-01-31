# -*- coding: utf-8 -*-

import re
import sys

TOKEN_REGEXP = re.compile(r'\s*([^\s()]+)')
SPAN_REGEXP = re.compile(r'#__#\[(\d+),(\d+)\]')
DEEPBANK_SPAN_REGEXP = re.compile(r'\(\d+, \d+, \d+, <(\d+):(\d+)>')
LABEL_SEP = '@'
PUNCT_LABEL = {'pt', 'punct'}

INDENT_STRING1 = '│   '
INDENT_STRING2 = '├──'


def print_tree(const_tree, indent=0, out=sys.stdout):
    for i in range(indent - 1):
        out.write(INDENT_STRING1)
    if indent > 0:
        out.write(INDENT_STRING2)
    out.write(const_tree.tag)
    if not isinstance(const_tree.children[0], ConstTree):
        out.write(f' {const_tree.children[0].string}\n')
    else:
        out.write('\n')
        for child in const_tree.children:
            print_tree(child, indent + 1, out)


def _next_token(string, pos):
    m = re.match(TOKEN_REGEXP, string[pos:])
    if not m:
        raise ConnectionError('??? at pos ' + str(pos))
    return pos + m.end(1), m.group(1)


def _make_tree(string):
    start, length = 0, len(string)
    stack = []
    lexicons = []

    root = None
    while start < length:
        c = string[start]
        if c == ')':
            if not stack:
                raise ConstTreeParserError('redundant ")" at pos ' + str(start))
            node = stack.pop()
            if not stack:
                root = node
            else:
                stack[-1].children.append(node)
            start += 1
        elif c == '(':
            start, tag = _next_token(string, start + 1)
            stack.append(ConstTree(tag))
        elif not c.isspace():
            if not stack:
                raise ConnectionError('??? at pos ' + str(start))

            start, lexicon = _next_token(string, start)
            lexicon = Lexicon(lexicon)
            lexicons.append(lexicon)
            stack[-1].children.append(lexicon)
        else:
            start += 1

    if not root or stack:
        raise ConstTreeParserError('missing ")".')

    return root, lexicons


class ConstTreeParserError(Exception):
    pass


class Lexicon:
    __slots__ = ('string', 'span', 'parent')

    def __init__(self, string, span=None):
        self.string = string
        self.span = span

    def __str__(self):
        return '<Lexicon {}>'.format(self.string)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.string == other.string

    def __hash__(self):
        return hash(self.string) + 2

    @property
    def tag(self):
        return self.string

    def to_string(self, quote_lexicon):
        if quote_lexicon:
            return '"{}"'.format(self.string)
        return self.string


class ConstTree:
    __slots__ = ('children', 'tag', 'span', 'word_span', 'has_semantics', 'index', 'parent')

    ROOT_LABEL = 'ROOT'

    def __init__(self, tag, children=None, span=None):
        self.tag = tag
        self.children = children if children is not None else []
        self.span = span
        self.word_span = None
        self.has_semantics = False
        self.index = None

    def __str__(self):
        child_string = ' + '.join(child.tag for child in self.children)
        return '{} {} => {}'.format(self.word_span, self.tag, child_string)

    def __repr__(self):
        return str(self)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.children[index]
        if isinstance(index, str):
            for child in self.children:
                if isinstance(child, ConstTree) and child.tag == index.upper():
                    return child
        raise KeyError

    def to_string(self, quote_lexicon=False):
        return '({} {})'.format(self.tag,
                                ' '.join(child.to_string(quote_lexicon)
                                         for child in self.children))

    @staticmethod
    def from_string(string):
        """ Construct ConstTree from parenthesis representation.

        :param string: string of parenthesis representation
        :return: ConstTree root and all leaf Lexicons
        """
        return _make_tree(string)

    @staticmethod
    def from_java_code_and_deepbank_1_1(tree_string, deepbank_data=None):
        tree, lexicons = _make_tree(tree_string)
        if deepbank_data is not None or 'NotExist' in tree_string:
            prev_span_map = {}
            if isinstance(deepbank_data, str):
                deepbank_data = deepbank_data.split('\n\n')
            span_data = deepbank_data[3]

            spans = sorted(set((int(span[0]), int(span[1]))
                               for span in DEEPBANK_SPAN_REGEXP.findall(span_data)))

            for index, span in enumerate(spans):
                prev_span_map[span] = spans[index - 1] if index > 0 else None
            for index, lexicon in enumerate(lexicons):
                if 'NotExist' in lexicon.string:
                    right_span = re.findall(SPAN_REGEXP, lexicon.string)[0]
                    prev_span = prev_span_map[int(right_span[0]), int(right_span[1])]
                    lexicon.string = lexicon.string.replace('NotExist', '{},{}'.format(*prev_span))

                    prev_lexicon = lexicons[index - 1]
                    prev_lexicon.string = re.sub(SPAN_REGEXP, '#__#[{0},{0}]'.format(prev_span[0]),
                                                 prev_lexicon.string)

        for lexicon in lexicons:
            spans = re.findall(SPAN_REGEXP, lexicon.string)
            start = min(int(span[0]) for span in spans)
            end = max(int(span[1]) for span in spans)
            lexicon.span = (start, end)
            lexicon.string = re.sub(SPAN_REGEXP, '', lexicon.string)
        return tree, lexicons

    def traverse_postorder(self):
        for child in self.children:
            if isinstance(child, ConstTree):
                yield from child.traverse_postorder()

        yield self

    def traverse_postorder_with_lexicons(self):
        for child in self.children:
            if isinstance(child, ConstTree):
                yield from child.traverse_postorder_with_lexicons()
            else:
                yield child

        yield self

    def generate_preterminals(self):
        for child in self.children:
            if isinstance(child, ConstTree):
                yield from child.generate_preterminals()

        for child in self.children:
            if isinstance(child, Lexicon):
                yield self

    def generate_lexicons(self, ignore_punct=False):
        for child in self.children:
            if isinstance(child, ConstTree):
                yield from child.generate_lexicons(ignore_punct)

        if not ignore_punct or self.tag not in PUNCT_LABEL:
            for child in self.children:
                if isinstance(child, Lexicon):
                    yield child

    def is_binary_tree(self):
        if isinstance(self.children[0], Lexicon):
            return True
        return len(self.children <= 2) and all(child.is_binary_tree()
                                               for child in self.children)

    def condensed_unary_chain(self, fold_duplicated=True):
        if len(self.children) > 1:
            return ConstTree(self.tag,
                             children=list(child.condensed_unary_chain()
                                           for child in self.children),
                             span=self.span)

        if isinstance(self.children[0], Lexicon):
            return ConstTree(self.tag, children=list(self.children), span=self.span)

        assert isinstance(self.children[0], ConstTree)
        node = self
        new_tag = self.tag
        last_tag = self.tag
        while len(node.children) == 1 and isinstance(node.children[0], ConstTree):
            node = node.children[0]
            if not fold_duplicated or node.tag != last_tag:
                new_tag += LABEL_SEP + node.tag
            last_tag = node.tag

        if len(node.children) == 1:
            children = list(node.children)
        else:
            children = list(child.condensed_unary_chain() for child in node.children)

        return ConstTree(new_tag, children=children, span=self.span)

    def expanded_unary_chain(self):
        if isinstance(self.children[0], Lexicon):
            children = list(self.children)
        else:
            children = list(child.expanded_unary_chain() for child in self.children)

        tags = self.tag.split(LABEL_SEP)
        for tag in reversed(tags):
            children = [ConstTree(tag, children=children, span=self.span)]

        return children[0]

    def calculate_span(self):
        self.span = self.children[0].span[0], self.children[-1].span[1]

    def populate_spans_internal(self):
        for child in self.children:
            if isinstance(child, ConstTree):
                child.populate_spans_internal()

        self.calculate_span()

    def add_postorder_index(self):
        for index, node in enumerate(self.traverse_postorder()):
            node.index = index
