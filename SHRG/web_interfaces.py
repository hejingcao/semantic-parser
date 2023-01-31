# -*- coding: utf-8 -*-

import gzip
import os
import pickle
from functools import lru_cache

from flask import Flask, abort, jsonify

from .const_tree import Lexicon
from .graph_draw import (draw_const_tree,
                         draw_derivation,
                         draw_hrg_rule,
                         draw_hyper_graph,
                         draw_tree_decomposition)
from .shrg import extract_shrg_rule
from .shrg_anonymize import anonymize_rule
from .shrg_extract import (ExtractionOptions,
                           compute_prefix_for_options,
                           generate_const_tree_and_hyper_graph)
from .tree_decomposition import tree_decomposition


class GrammarOptions(ExtractionOptions):
    anonymous: bool = False


def _format_cfg_rule_item(tag, edge):
    if isinstance(tag, Lexicon):
        return '"{}"'.format(tag.string)
    assert isinstance(tag, str)

    if edge is None:
        return tag

    if tag.startswith('<') and edge.is_terminal:
        carg = getattr(edge, 'carg', '???')
        return '{}({})'.format(tag, carg)
    return '{}#{:d}'.format(tag, len(edge.nodes))


class Grammar:
    def __init__(self, prefix, grammar_name, deepbank_data_dir, java_out_dir,
                 options: GrammarOptions):
        self._name = grammar_name
        self._deepbank_data_dir = deepbank_data_dir
        self._java_out_dir = java_out_dir

        self._rules = None
        self._cfg_strings = None
        self._params = options
        self._prefix = compute_prefix_for_options(prefix, grammar_name, options)

    def __str__(self):
        return '<Grammar @{}>'.format(self._prefix)

    def __repr__(self):
        return str(self)

    @property
    def name(self):
        name = os.path.basename(os.path.dirname(self._prefix))
        if self._params.anonymous:
            name += '.anonymous'
        return name

    @property
    def rules_filename(self):
        name = self._prefix + 'train.counter.p'
        if self._params.anonymous:
            name += '.anonymous'
        return name

    @property
    def cfg_strings(self):
        if self._cfg_strings is None:
            self._load_cfg_strings()
        return self._cfg_strings

    @property
    def rules(self):
        if self._rules is None:
            self._load_rules()
        return self._rules

    def _load_rules(self):
        try:
            rules_counter, _ = pickle.load(open(self.rules_filename, 'rb'))
            self._rules = rules_counter
        except FileNotFoundError:
            return abort(404)

    def _load_cfg_strings(self):
        cfg_strings = {}
        for tag in ('train', 'dev', 'test'):
            java_out_dir = self._java_out_dir + tag
            for bank in os.listdir(java_out_dir):
                if not bank.startswith('wsj'):
                    continue
                with open(os.path.join(java_out_dir, bank)) as fin:
                    while True:
                        sentence_id = fin.readline().strip()
                        if not sentence_id:
                            break
                        assert sentence_id.startswith('#')
                        sentence_id = sentence_id[1:]
                        tree_string = fin.readline().strip()
                        cfg_strings[sentence_id] = bank, tree_string

        self._cfg_strings = cfg_strings

    @lru_cache(maxsize=4096)
    def get_derivations(self, sentence_id):
        bank, tree_string = self.cfg_strings[sentence_id]
        filename = os.path.join(self._deepbank_data_dir, bank, sentence_id + '.gz')

        with gzip.open(filename, 'rb') as fin_bank:
            contents = fin_bank.read().decode()

        const_tree, (hyper_graph, _) = \
            generate_const_tree_and_hyper_graph(tree_string, contents, self._params)

        results = extract_shrg_rule(hyper_graph, const_tree,
                                    sentence_id=sentence_id,
                                    detect_function=self._params.detect_function,
                                    lexicalize_null_semantic=self._params.lexicalize_null_semantic,
                                    graph_type=self._params.graph_type,
                                    return_derivation_infos=True)
        if self._params.anonymous:
            shrg_rules, *rest = results
            for index, shrg_rule in enumerate(shrg_rules):
                new_rule = anonymize_rule(shrg_rule)
                if new_rule is not None:
                    shrg_rules[index] = new_rule
        const_tree.add_postorder_index()
        return results, const_tree, hyper_graph


class SHRGVirtualizationService(Flask):
    def __init__(self, grammars, static_folder):
        super().__init__(self.__call__.__name__,
                         static_url_path='/static', static_folder=static_folder)
        self.grammars = grammars

        self.add_url_rule('/',
                          view_func=self.index, methods=['GET'])
        self.add_url_rule('/<path:path>',
                          view_func=self.index, methods=['GET'])
        self.add_url_rule('/api/grammars',
                          view_func=self.get_grammars, methods=['GET'])
        self.add_url_rule('/api/rule/<string:grammar_name>/<int:rule_index>',
                          view_func=self.get_rule, methods=['GET'])
        self.add_url_rule('/api/search-rule/<string:grammar_name>/<string:condition>',
                          view_func=self.search_rule, methods=['GET'])
        self.add_url_rule('/api/sentence/<string:grammar_name>/<string:sentence_id>/<int:step>',
                          view_func=self.get_sentence, methods=['GET'])

    def index(self, path='index.html'):
        return self.send_static_file(path)

    def _get_rule(self, rule):
        hrg_rule = rule.hrg
        ep_count, hrg_source = 0, None
        if hrg_rule is not None:
            hrg_source = draw_hrg_rule(hrg_rule, output_format='source')
            ep_count = len(hrg_rule.lhs.nodes)

        cfg = '{}#{} ⇒ {}'.format(rule.cfg.lhs, ep_count,
                                  ' + '.join(_format_cfg_rule_item(*item) for item in rule.cfg.rhs))
        return {
            'cfg': cfg,
            'hrgSource': hrg_source,
            'label': hrg_rule.lhs.label if hrg_rule is not None else 'no semantic',
            'comment': hrg_rule.comment if hrg_rule is not None else {},
        }

    def get_grammars(self):
        return jsonify(list(self.grammars.keys()))

    def get_rule(self, grammar_name, rule_index):
        if grammar_name not in self.grammars:
            return abort(404)

        grammar = self.grammars[grammar_name]
        total_rule_count = len(grammar.rules)
        if rule_index < 0 or rule_index >= total_rule_count:
            return abort(404)
        rule, counter_item = grammar.rules[rule_index]

        decomposition = None
        if rule.hrg is not None:
            hyper_graph = rule.hrg.rhs
            external_nodes = rule.hrg.lhs.nodes
            tree_root = tree_decomposition(hyper_graph, external_nodes)
            decomposition = draw_tree_decomposition(tree_root, external_nodes,
                                                    output_format='source')

        return jsonify({
            'count': counter_item.count,
            'examples': counter_item.samples,
            'totalRuleCount': total_rule_count,
            'treeDecomposition': decomposition,
            **self._get_rule(rule)
        })

    def search_rule(self, grammar_name, condition):
        # TODO: Change to POST
        query = "[index for index, (rule, counter_item) in " \
            "enumerate(grammars[\"{grammar_name}\"].rules) if {condition}]"
        try:
            for key in ('import', 'open', 'for', 'while'):
                assert key not in condition, 'can not use "{}" keywords'.format(key)
            query = query.format(grammar_name=grammar_name, condition=condition)
            return jsonify(eval(query, {}, {'grammars': self.grammars}))
        except Exception as err:
            return jsonify({'errorMessage': str(err)})

    @lru_cache(maxsize=4096)
    def get_sentence(self, grammar_name, sentence_id, step):
        if grammar_name not in self.grammars:
            return abort(404)

        result, const_tree, hyper_graph = self.grammars[grammar_name].get_derivations(sentence_id)
        shrg_rules, (node_blame_dict, edge_blame_dict), derivation_infos = result

        if step >= len(shrg_rules):
            return abort(404)

        cfg_nodes = list(const_tree.traverse_postorder())
        cfg_node = cfg_nodes[step]
        node2index = {node: index for index, node in enumerate(cfg_nodes)}
        left_index, right_index, leftmost_index = -1, -1, -1
        if cfg_node.children:
            child = cfg_node.children[0]
            left_index = node2index.get(child, -1)
            if not isinstance(child, Lexicon):
                leftmost_index = node2index.get(next(child.traverse_postorder()), -1)
        if len(cfg_node.children) > 1:
            right_index = node2index.get(cfg_node.children[1], -1)

        attrs_map = {}

        def _fill_attrs_map(items, blame_dict):
            for item in items:
                blame_step = blame_dict.get(item, -1)
                color = None
                if blame_step == step:
                    color = 'red'
                elif leftmost_index <= blame_step <= left_index:
                    color = 'blue'
                elif left_index < blame_step <= right_index:
                    color = 'green'
                if color:
                    attrs_map[item] = {'color': color}

        _fill_attrs_map(hyper_graph.nodes, node_blame_dict)
        _fill_attrs_map(hyper_graph.edges, edge_blame_dict)

        shrg_rule = shrg_rules[step]
        panorama = draw_hyper_graph(hyper_graph, output_format='source', attrs_map=attrs_map)

        before = draw_derivation(derivation_infos[step], output_format='source')
        after = draw_derivation(derivation_infos[step + 1], output_format='source')

        nodes_attrs = {cfg_nodes[step]: {'fillcolor': 'red', 'style': 'filled'}}
        return jsonify({
            'constTree': draw_const_tree(const_tree, output_format='source',
                                         nodes_attrs=nodes_attrs),
            'before': before, 'after': after,
            'rule': self._get_rule(shrg_rule),
            'panorama': panorama,
            'totalStep': len(shrg_rules)
        })
