# -*- coding: utf-8 -*-H

import hashlib
from collections import defaultdict
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from dataclasses import dataclass

from framework.common.logger import LOGGER

from .const_tree import LABEL_SEP, ConstTree, Lexicon
from .hyper_graph import GraphNode, HyperEdge, HyperGraph
from .shrg_alignment import find_aligned_edge
from .shrg_compound_split import NodeDistributor
from .shrg_detect import DETECT_FUNCTIONS

# Internal node: rhs = Sequence[(cfg_rhs_1, corresponding hrg edge1), ...]
# Leaf node: rhs = Sequence[(lexicon, None)]
# Invalid rule: rhs = None
CFGRuleRHS = Optional[Sequence[Union[Tuple[str, Optional[HyperEdge]],
                                     Tuple[Lexicon, Sequence[HyperEdge]]]]]

LEXICALIZE_NULL_SEMANTIC_OPTIONS = ['merge-single', 'merge-both', 'delete', 'ignore_punct']


class CFGRule:
    __slots__ = ('lhs', 'rhs')

    def __init__(self, lhs: str, rhs: CFGRuleRHS):
        self.lhs = lhs
        self.rhs = tuple(rhs)

    def __hash__(self):
        return hash(self.lhs) ^ hash(self.rhs)

    def __eq__(self, other):
        return self.lhs == other.lhs and self.rhs == other.rhs

    def __str__(self):
        return '{} => {}'.format(self.lhs, self.rhs)

    def __repr__(self):
        return str(self)


class HRGRule:
    __slots__ = ('lhs', 'rhs', 'comment')

    def __init__(self, lhs: HyperEdge, rhs: HyperGraph, comment: Optional[Dict[str, str]]=None):
        self.lhs = lhs
        self.rhs = rhs
        self.comment = comment or {}

    def __hash__(self):
        return hash(self.lhs) ^ hash(self.rhs)

    def __eq__(self, other):
        return self.lhs == other.lhs and self.rhs == other.rhs

    def __str__(self):
        return '{} => \n{}\n'.format(self.lhs,
                                     '\n'.join(str(edge) for edge in self.rhs.edges))

    def __repr__(self):
        return str(self)


class SHRGRule:
    __slots__ = ('cfg', 'hrg')

    def __init__(self, cfg: CFGRule, hrg: HRGRule):
        self.cfg = cfg
        self.hrg = hrg

    def __hash__(self):
        return hash(self.cfg) ^ hash(self.hrg)

    def __eq__(self, other):
        return self.cfg == other.cfg and self.hrg == other.hrg

    def __str__(self):
        return 'SHRGRule({} ===> {})'.format(self.cfg, self.hrg)

    def __repr__(self):
        return str(self)


@dataclass
class DerivationInfo:
    hyper_graph: HyperGraph
    last_new_edge: HyperEdge
    all_edges: Set[HyperEdge]
    internal_nodes: Set[GraphNode]
    external_nodes: Set[GraphNode]


def extract_hrg_rule(edges: Set[HyperEdge],
                     internal_nodes: Set[GraphNode],
                     external_nodes: Set[GraphNode],
                     label: str, left_and_right_span=None):
    node_rename_map = {}
    nodes = internal_nodes | external_nodes

    edge_by_node = defaultdict(list)  # node -> (edge, index of this node in this edge)
    for edge in edges:
        for index, node in enumerate(edge.nodes):
            edge_by_node[node].append((edge, index))

    default_hash = hashlib.md5(b'13').digest()
    node_hashes = {node: default_hash for node in nodes}  # node -> hash

    def _calculate_edge_hashes(edge, index):
        """hash(edge) = hash(edge.label#external_count#hash(node_1)#hash(node_2)#...)"""
        md5_obj = hashlib.md5((edge.label + '#' + str(index)).encode())
        for adj_node in edge.nodes:
            md5_obj.update(node_hashes[adj_node] + b'#')
        return md5_obj.digest()

    def _calculate_sibling_hashes(node):
        """hash(node)=hash(set of hash(sibling_edges)#node_name if exist)"""
        md5_obj = hashlib.md5()
        edge_hashes = sorted(_calculate_edge_hashes(edge, index)
                             for edge, index in edge_by_node[node])
        for hash_ in edge_hashes:
            md5_obj.update(hash_)

        if node_rename_map.get(node) is not None:
            md5_obj.update(('#' + node_rename_map[node].name).encode('utf-8'))
        return md5_obj.digest()

    def _recalculate_hashes():
        new_node_hashes = {}
        # recalculate hashes
        for node in nodes:
            md5_obj = hashlib.md5()
            md5_obj.update(_calculate_sibling_hashes(node))
            md5_obj.update(b'\x01' if node in external_nodes else b'\x00')
            new_node_hashes[node] = md5_obj.digest()
        return new_node_hashes

    for cycle in range(len(nodes) + 10):
        node_hashes = _recalculate_hashes()

    node_hashes_original = dict(node_hashes)

    assert len(nodes) == len(node_hashes)
    node_count = len(node_hashes)
    while len(node_rename_map) < node_count:
        nodes_in_order = sorted(node_hashes.items(), key=lambda x: x[1])
        has_symmetric = False
        for index, (node, hash_value) in enumerate(nodes_in_order):
            if index != node_count - 1 and nodes_in_order[index + 1][1] == hash_value:
                # Detect symmetric
                has_symmetric = True
                assert node not in node_rename_map
                node_rename_map[node] = GraphNode(str(len(node_rename_map)))
                for cycle in range(len(nodes) + 10):
                    node_hashes = _recalculate_hashes()
                break

        if not has_symmetric:
            for node, hash_value in nodes_in_order:
                if node not in node_rename_map:
                    node_rename_map[node] = GraphNode(str(len(node_rename_map)))
            break

    rhs = HyperGraph(node_rename_map.values(),
                     (edge.new((node_rename_map[node] for node in edge.nodes))
                      for edge in edges))

    comment = None
    ep_permutation = None
    pending = []
    for permutation in permutations(external_nodes):
        if any(edge.nodes == permutation for edge in edges):
            pending.append(permutation)

    if len(pending) == 1:
        ep_permutation = [node_rename_map[node] for node in pending[0]]
        comment = {'EP permutation': 'Stick hyperedge to one edge'}
    elif len(external_nodes) == 2 and left_and_right_span is not None:
        left_span, right_span = left_and_right_span
        left_node = [edge.nodes[0] for edge in edges
                     if len(edge.nodes) == 1 and edge.span == left_span]
        right_node = [edge.nodes[0] for edge in edges
                      if len(edge.nodes) == 1 and edge.span == right_span]
        if left_node and right_node:
            left_node = left_node[0]
            right_node = right_node[0]
            if node_hashes_original[left_node] != node_hashes_original[right_node] \
               and {left_node, right_node} == external_nodes:
                comment = {'EP permutation':
                           'judge #EP2 edge direction by spans of left and right node'}
                ep_permutation = [node_rename_map[left_node], node_rename_map[right_node]]

    if ep_permutation is None:
        comment = {'EP permutation': 'arbitrary order'} if len(external_nodes) >= 2 else {}
        ep_permutation = sorted((node_rename_map[i] for i in external_nodes),
                                key=lambda x: int(x.name))

    lhs = HyperEdge(ep_permutation, label=label, is_terminal=False)
    return node_rename_map, HRGRule(lhs, rhs, comment)


def extract_shrg_rule(hyper_graph: HyperGraph, const_tree: ConstTree,
                      detect_function=None,
                      return_derivation_infos=False,
                      lexicalize_null_semantic=None,
                      graph_type='eds',
                      sentence_id=''):
    """ Extract rules from give hyper_graph and constituent tree. """
    detect_function = DETECT_FUNCTIONS.normalize(detect_function)

    edge_blame_dict: Dict[HyperEdge, int] = {}
    node_blame_dict: Dict[HyperEdge, int] = {}
    boundary_node_dict: Dict[GraphNode, int] = {}

    last_new_edge: Optional[HyperEdge] = None
    shrg_rules: List[Tuple[SHRGRule, HyperEdge]] = []
    derivation_infos: List[Optional[DerivationInfo]] = []

    node_distribution = NodeDistributor(hyper_graph, const_tree, graph_type,
                                        logger=LOGGER).solve()

    const_tree.add_postorder_index()  # give every tree node an index
    for step, cfg_node in enumerate(const_tree.traverse_postorder()):
        cfg_node.calculate_span()

        collected_pred_edges = node_distribution[cfg_node]
        for child_node in cfg_node.children:
            if isinstance(child_node, ConstTree):
                _, child_new_edge = shrg_rules[child_node.index]
                if child_new_edge is not None:
                    collected_pred_edges.add(child_new_edge)

        result = detect_function(hyper_graph, cfg_node, collected_pred_edges)

        if result is None:
            cfg_node.has_semantics = False
            cfg_rhs = tuple((child if isinstance(child, Lexicon) else child.tag, None)
                            for child in cfg_node.children)

            shrg_rules.append((SHRGRule(CFGRule(cfg_node.tag, cfg_rhs), None), None))
            if return_derivation_infos:
                derivation_infos.append(None)
            continue

        cfg_node.has_semantics = True
        all_edges, internal_nodes, external_nodes, detect_comment = result

        left_and_right_span = None
        if len(cfg_node.children) == 2:
            left_and_right_span = (cfg_node.children[0].span, cfg_node.children[1].span)

        node_rename_map, hrg_rule = extract_hrg_rule(all_edges, internal_nodes, external_nodes,
                                                     cfg_node.tag, left_and_right_span)

        hrg_rule_nodes = hrg_rule.rhs.nodes
        assert set(int(n.name) for n in hrg_rule_nodes) == set(range(len(hrg_rule_nodes)))

        if detect_comment is not None:
            hrg_rule.comment['DetectInner'] = detect_comment

        if not external_nodes and cfg_node is not const_tree:
            # If external nodes is empty and current node is not root, select first internal node
            node = None
            for internal_node in internal_nodes:
                if node_rename_map[internal_node].name == '0':
                    node = internal_node
                    break
            assert node

            internal_nodes.remove(node)
            external_nodes = set([node])
            hrg_rule.lhs.nodes = (node_rename_map[node], )
            hrg_rule.comment['Detect'] = 'Use first node as external node'

            # logger.warning('empty-external-nodes: %s/%d %s', sentence_id, step, cfg_node.tag)

        for edge in all_edges:
            if edge.is_terminal:
                assert edge not in edge_blame_dict
                edge_blame_dict[edge] = step

        for node in internal_nodes:
            assert node not in node_blame_dict
            node_blame_dict[node] = step

        if return_derivation_infos:
            derivation_infos.append(DerivationInfo(hyper_graph=hyper_graph,
                                                   last_new_edge=last_new_edge,
                                                   all_edges=all_edges,
                                                   internal_nodes=internal_nodes,
                                                   external_nodes=external_nodes))

        reversed_node_rename_map = {node: original_node
                                    for original_node, node in node_rename_map.items()}

        new_edge = HyperEdge((reversed_node_rename_map[node] for node in hrg_rule.lhs.nodes),
                             cfg_node.tag, False, cfg_node.span)
        boundary_node_dict[step] = tuple(new_edge.nodes)

        new_hyper_graph = HyperGraph(hyper_graph.nodes - internal_nodes,
                                     (hyper_graph.edges - all_edges) | {new_edge})

        hyper_graph = new_hyper_graph
        last_new_edge = new_edge

        if isinstance(cfg_node.children[0], Lexicon):
            assert len(cfg_node.children) == 1, 'Stange condition'
            cfg_rhs = find_aligned_edge(sentence_id, cfg_node.children[0], hrg_rule.rhs)
        else:
            assert all(isinstance(child_node, ConstTree) for child_node in cfg_node.children)
            cfg_rhs = []
            for child_node in cfg_node.children:
                if not child_node.has_semantics:
                    cfg_rhs.append((child_node.tag, None))
                    continue

                _, target_edge = shrg_rules[child_node.index]
                if target_edge.label != child_node.tag and target_edge.carg != child_node.tag:
                    LOGGER.warning('Non-consistent CFG and HRG: %s', sentence_id)
                    cfg_rhs = None
                    break
                target_edge = HyperEdge([node_rename_map[node] for node in target_edge.nodes],
                                        target_edge.label,
                                        target_edge.is_terminal)
                cfg_rhs.append((child_node.tag, target_edge))

        cfg_lhs = cfg_node.tag
        if cfg_lhs.startswith('ROOT'):  # merge all ROOT labels
            assert hrg_rule.lhs.label == cfg_lhs
            cfg_lhs = hrg_rule.lhs.label = 'ROOT'

        if cfg_rhs is not None:
            rule = SHRGRule(CFGRule(cfg_lhs, tuple(cfg_rhs)), hrg_rule)
        else:
            rule = SHRGRule(CFGRule(cfg_lhs, None), None)

        shrg_rules.append((rule, new_edge))

    shrg_rules = [rule for rule, _ in shrg_rules]

    if return_derivation_infos:
        derivation_infos.append(DerivationInfo(hyper_graph=hyper_graph,
                                               last_new_edge=last_new_edge,
                                               all_edges=all_edges,
                                               internal_nodes=internal_nodes,
                                               external_nodes=external_nodes))

    if lexicalize_null_semantic is not None:
        remove_null_semantic_rules(const_tree, shrg_rules, lexicalize_null_semantic)

    if return_derivation_infos:
        return shrg_rules, (node_blame_dict, edge_blame_dict), derivation_infos
    return shrg_rules, (node_blame_dict, edge_blame_dict), boundary_node_dict


def remove_null_semantic_rules(const_tree: ConstTree, shrg_rules: List[SHRGRule], option):
    use_delete = 'delete' in option
    use_merge_both = 'merge-both' in option
    use_merge_single = 'merge-single' in option
    ignore_punct = 'ignore_punct' in option
    assert not use_merge_both or not use_merge_single

    def _get_label(label, left_label, right_label, merge_left=True):
        if not use_merge_both and not use_merge_single:
            return label
        child_label = left_label if merge_left else right_label
        real_label = label.split(LABEL_SEP)[-1]  # label my be condensed
        real_child_label, *rest_labels = child_label.split('@', 1)
        if real_label != real_child_label:
            if use_merge_single:
                label = '{}!{}'.format(label, child_label)
            else:
                label = '{}!{}!{}'.format(label, left_label, right_label)
        elif rest_labels:
            label = '{}@{}'.format(label, rest_labels[0])
        return label

    cfg_nodes = list(const_tree.traverse_postorder())
    cfg_node2step = {cfg_node: step for step, cfg_node in enumerate(cfg_nodes)}
    for step, (cfg_node, rule) in enumerate(zip(cfg_nodes, shrg_rules)):
        if len(cfg_node.children) == 1:
            if isinstance(cfg_node.children[0], ConstTree):
                assert rule.hrg is not None, 'Strange condition'
            continue
        assert len(rule.cfg.rhs) == 2, 'Strange CFGRule '
        left, right = cfg_node.children
        left_rule = shrg_rules[cfg_node2step[left]]
        right_rule = shrg_rules[cfg_node2step[right]]
        if left_rule.hrg is None and right_rule.hrg is None:
            assert rule.hrg is None
            LOGGER.debug('null-semantic: both children of [%s]', cfg_node)
            continue
        if left_rule.hrg is not None and right_rule.hrg is not None:
            continue

        cfg = rule.cfg
        left_label = left_rule.cfg.lhs  # left child label
        right_label = right_rule.cfg.lhs  # right child label
        if left_rule.hrg is None:
            LOGGER.debug('null-semantic: left child of [%s]', cfg_node)
            label = _get_label(cfg.lhs, left_label, right_label, merge_left=False)

            # the nonterminal edge to be replaced with corresponding rules
            target_edge = cfg.rhs[1][1]
            assert right_rule.hrg.lhs.label.startswith(target_edge.label), 'Strange condition'

            # the left child has no semantics, we prepend all the lexicons in
            # left subtree to cfg items directly
            cfg_rhs = [] if use_delete else [(lexicon, None)
                                             for lexicon in left.generate_lexicons(ignore_punct)]
            cfg_rhs.extend(right_rule.cfg.rhs)

            child_external_nodes = right_rule.hrg.lhs.nodes
            child_nodes = right_rule.hrg.rhs.nodes
            child_edges = right_rule.hrg.rhs.edges
        else:  # right_rule.hrg is None
            LOGGER.debug('null-semantic: right child of [%s]', cfg_node)
            label = _get_label(cfg.lhs, left_label, right_label, merge_left=True)

            target_edge = cfg.rhs[0][1]
            assert left_rule.hrg.lhs.label.startswith(target_edge.label), 'Strange condition'

            # the right child has no semantics, we append all the lexicons in
            # right subtree to cfg items directly
            cfg_rhs = list(left_rule.cfg.rhs)
            if not use_delete:
                cfg_rhs.extend((lexicon, None) for lexicon in right.generate_lexicons(ignore_punct))

            child_external_nodes = left_rule.hrg.lhs.nodes
            child_nodes = left_rule.hrg.rhs.nodes
            child_edges = left_rule.hrg.rhs.edges

        assert len(target_edge.nodes) == len(child_external_nodes), 'Mismatch !!!'

        current_nodes = set(rule.hrg.rhs.nodes)
        current_edges = set(rule.hrg.rhs.edges)
        # first, remove edge which represents the subtree without semantics
        assert target_edge in current_edges, 'Target edge not in HRG graph !!!'
        current_edges.remove(target_edge)

        # node in child rule => node in current rule
        node_mapping = dict(zip(child_external_nodes, target_edge.nodes))
        extra_nodes = sorted([child_node
                              for child_node in child_nodes
                              if child_node not in child_external_nodes],
                             key=lambda x: int(x.name))
        # assign new name to node in child rule
        for index, child_node in enumerate(extra_nodes, len(current_nodes)):
            new_node = GraphNode(str(index))
            node_mapping[child_node] = new_node
            assert new_node not in current_nodes, 'Index of node in HRG graph is broken !!!'
            current_nodes.add(new_node)  # second, add nodes from child rule to current rule

        edge_mapping = {}
        for edge in child_edges:
            new_edge = edge.new((node_mapping[node] for node in edge.nodes), span=edge.span)
            edge_mapping[edge] = new_edge
            current_edges.add(new_edge)  # third, add edges from child rule to current rule

        for index, (_, edge) in enumerate(cfg_rhs):
            cfg_rhs[index] = (_, edge_mapping.get(edge))

        LOGGER.debug('rule-change: %s ### %s %s', rule.cfg, label, cfg_rhs)
        cfg.lhs, cfg.rhs = label, tuple(cfg_rhs)
        rule.hrg = HRGRule(lhs=HyperEdge(nodes=rule.hrg.lhs.nodes, label=label, is_terminal=False),
                           rhs=HyperGraph(nodes=current_nodes, edges=current_edges),
                           comment={'From': 'replace by its child'})
