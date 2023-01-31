# -*- coding: utf-8 -*-

import os
import re

from framework.common.logger import LOGGER
from framework.common.utils import MethodFactory, lazy_property

from .const_tree import ConstTree, Lexicon

HEAD_FILE = os.path.join(os.path.dirname(__file__), 'head-rules')

__ASSETS__ = [HEAD_FILE]

MODIFY_TREE_OPTIONS = MethodFactory(include_none=True)
FIX_HYPHEN_OPTIONS = MethodFactory(include_none=True)


@FIX_HYPHEN_OPTIONS.register('merge')
def fix_punct_hyphen(tree: ConstTree):
    span = tree.span
    lexicons = list(tree.generate_lexicons())
    if all(lexicon.span == span or lexicon.span[0] == lexicon.span[1] for lexicon in lexicons) \
       and sum(1 for lexicon in lexicons if lexicon.span == span) >= 2:
        lexicon = Lexicon(''.join(l.string for l in lexicons))
        lexicon.span = span
        tree.children = [lexicon]
        return

    if isinstance(tree.children[0], ConstTree):
        for child in tree.children:
            assert isinstance(child, ConstTree)
            fix_punct_hyphen(child)


def _remove_unary_chain(tree: ConstTree):
    while len(tree.children) == 1 \
            and isinstance(tree.children[0], ConstTree) \
            and tree.tag == tree.children[0].tag:
        tree.children = tree.children[0].children
    for child in tree.children:
        if isinstance(child, ConstTree):
            _remove_unary_chain(child)


@MODIFY_TREE_OPTIONS.register('no-suffix')
def remove_label_suffix(tree: ConstTree):
    if isinstance(tree, ConstTree):
        tree.tag = tree.tag.split('_')[0]
        for child in tree.children:
            remove_label_suffix(child)


@MODIFY_TREE_OPTIONS.register('no-suffix-internal')
def remove_label_suffix_internal(tree: ConstTree):
    if isinstance(tree, ConstTree) and isinstance(tree.children[0], ConstTree):
        tree.tag = tree.tag.split('_')[0]
        for child in tree.children:
            remove_label_suffix_internal(child)


@MODIFY_TREE_OPTIONS.register('no-label')
def remove_all_label(tree: ConstTree):
    tree.tag = 'X'
    for child in tree.children:
        if isinstance(child, ConstTree):
            remove_all_label(child)


class ToHeadModifier:
    # sometimes there are special labels such as "house_n2"
    LABEL_ALLOW = {'n', 'generic', 'punct', 'aj', 'v', 'p', 'av', 'cm', 'pp', 'd', 'c', 'pt', 'x'}
    SPECIAL_TAG_MAPPING = {'a': 'aj', 'adv': 'av'}
    NORMAL_TAG_MAPPING = {
        'genericname': 'generic',
        'hasnt': 'v', 'hadnt': 'v', 'havent': 'v', 'have': 'v', 'have-poss': 'v',
        'all': 'n', 'sharply': 'av'
    }

    def __init__(self, rule_file):
        self.rule_file = rule_file

    @lazy_property
    def rules(self):
        rules = {}
        for line in open(self.rule_file):
            line = line.strip()
            if not line:
                continue
            name, children_count, head = line.strip().split()
            rules[name] = int(children_count), int(head)
        return rules

    def modify(self, tree: ConstTree):
        num_children = len(tree.children)
        if isinstance(tree.children[0], Lexicon):
            original_tag = tree.tag
            assert num_children == 1, '??? num_children != 1'
            if tree.tag == 'but_np_not_conj':
                tree.tag = 'c'
                LOGGER.debug('rewrite but_np_not_conj to c')
            elif re.match(r'^.*\w{1,2}\d$', tree.tag):
                new_tag = tree.tag.rsplit('_', 1)[1].rstrip('0123456789')
                new_tag = self.SPECIAL_TAG_MAPPING.get(new_tag, new_tag)
                LOGGER.debug('rewrite %s to %s', tree.tag, new_tag)
                tree.tag = new_tag
            else:
                tree.tag = tree.tag.split('_')[0]
                tree.tag = self.NORMAL_TAG_MAPPING.get(tree.tag, tree.tag)

            if tree.tag not in self.LABEL_ALLOW:
                LOGGER.error('Invalid label %s -> %s for %s',
                             original_tag, tree.tag, tree.children[0])
            return tree

        if 'root' in tree.tag.lower() or num_children == 1:
            children_count = 1
            head_index = 0
        elif num_children == 2 and 'punct' in tree.children[0].tag:
            children_count = 2
            head_index = 1
        elif num_children == 2 and 'punct' in tree.children[1].tag:
            children_count = 2
            head_index = 0
        else:
            children_count, head_index = self.rules[tree.tag]
        assert num_children == children_count, '??? num_children != children_count'

        for child_index, child in enumerate(tree.children):
            child_head = self.modify(child)
            if child_index == head_index:
                head = child_head

        if 'root' in tree.tag.lower():
            tree.tag = 'ROOT-' + head.tagp
        else:
            tree.tag = head.tag
        return head


MODIFY_TREE_OPTIONS.register('head')(ToHeadModifier(HEAD_FILE).modify)


def modify_const_tree(tree: ConstTree, modify_option, fix_hyphen_option):
    MODIFY_TREE_OPTIONS.invoke(modify_option, tree)

    if modify_option != 'head':
        _remove_unary_chain(tree)

    tree = tree.condensed_unary_chain()
    tree.populate_spans_internal()

    return FIX_HYPHEN_OPTIONS.invoke(fix_hyphen_option, tree)
