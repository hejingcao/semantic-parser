# -*- coding: utf-8 -*-

from graphviz import Digraph

from .const_tree import ConstTree
from .hyper_graph import HyperGraph
from .tree_decomposition import TreeNode

SHAPES = {0: 'plaintext', 3: 'invtriangle', 4: 'diamond', 5: 'star',
          6: 'hexagon', 7: 'polygon', 8: 'octagon'}

FAKE_NODE_ATYTRS = {'width': '0.005', 'height': '0.005', 'fixedsize': 'true', 'color': 'white'}
EXTERNAL_NODE_ATTRS = {'color': 'red', 'shape': 'rectangle'}
NODE_ATTRS = {'width': '0.075', 'height': '0.075', 'fixedsize': 'true'}
LABELED_NODE_ATTRS = {'width': '0.35', 'height': '0.35'}


def _get_shape(number):
    return SHAPES.get(number, 'doublecircle')


def _draw_hyper_node(dot, node, attrs_map, nodes_labels_map, label=None):
    node_attrs = attrs_map.get(node, {})
    node_attrs.update(NODE_ATTRS)
    label = nodes_labels_map.get(node, label or '')
    if label != '':
        node_attrs.update(LABELED_NODE_ATTRS)

    dot.node(node.name, label=label, _attributes=node_attrs)


def _draw_hyper_edge(dot, edge, attrs_map={}, show_span=False):
    edge_attrs = attrs_map.get(edge, {})
    edge_attrs.update({'arrowsize': '0.5'})

    if edge.span is not None:
        fmt = '{}({},{})' if show_span else '{}'
        label = fmt.format(edge.label, *edge.span)
    else:
        label = edge.label

    if len(edge.nodes) == 1:
        # pred edge: add a invisible node as fake end
        fake_end = edge.nodes[0].name + label + '_end'
        dot.node(fake_end, label='', _attributes=FAKE_NODE_ATYTRS)
        # edge_attrs.update({'weight': '100'})
        dot.edge(edge.nodes[0].name, fake_end, label=label,
                 _attributes=edge_attrs)
    elif len(edge.nodes) == 2:
        # normal edge
        dot.edge(edge.nodes[0].name, edge.nodes[1].name, label,
                 _attributes=edge_attrs)
    else:
        # hyper edge:
        center_node = '{}_{}_hyperedge'.format(edge.label, edge.span)
        edge_attrs.pop('arrowsize')
        edge_attrs['shape'] = _get_shape(len(edge.nodes))
        dot.node(center_node, label=label, _attributes=edge_attrs)
        for index, end_point in enumerate(edge.nodes):
            dot.edge(center_node, end_point.name, label=str(index))


def draw_hyper_graph(hyper_graph: HyperGraph,
                     output_file=None, output_format='svg',
                     show_span=True, attrs_map=None, nodes_labels_map=None):
    if attrs_map is None:
        attrs_map = {}
    if nodes_labels_map is None:
        nodes_labels_map = {}

    dot = Digraph()
    for index, node in enumerate(hyper_graph.sorted_nodes):
        _draw_hyper_node(dot, node, attrs_map, nodes_labels_map, label=str(index))

    for edge in hyper_graph.edges:
        _draw_hyper_edge(dot, edge, attrs_map, show_span)

    if output_format == 'source':
        return dot.source
    dot.format = output_format
    dot.render(output_file, cleanup=True)


def draw_hrg_rule(hrg_rule, output_file=None, output_format='svg'):
    attrs_map = {}
    nodes_labels_map = {node: node.name for node in hrg_rule.rhs.nodes}
    for node in hrg_rule.lhs.nodes:
        attrs_map[node] = EXTERNAL_NODE_ATTRS
    return draw_hyper_graph(hrg_rule.rhs, output_file, output_format,
                            attrs_map=attrs_map, nodes_labels_map=nodes_labels_map)


def draw_derivation(derivation_info, output_file=None, output_format='svg'):
    if derivation_info is None and output_format == 'source':
        return 'digraph g {a [label="no semantics" shape="plaintext"];}'

    hyper_graph = derivation_info.hyper_graph
    all_edges = derivation_info.all_edges
    attrs_map = {}
    for edge in all_edges:
        attrs_map[edge] = {'color': 'red'}
    for node in derivation_info.internal_nodes:
        attrs_map[node] = {'color': 'red'}
    for node in derivation_info.external_nodes:
        attrs_map[node] = {'color': 'red', 'style': 'filled'}

    last_new_edge = derivation_info.last_new_edge
    if last_new_edge:
        attrs_map[last_new_edge] = {
            'color': 'blue' if last_new_edge not in all_edges else 'violet'
        }

    return draw_hyper_graph(hyper_graph, output_file, output_format, attrs_map=attrs_map)


def draw_const_tree(const_tree: ConstTree, output_file=None, output_format='svg',
                    nodes_attrs=None, show_span=True):
    dot = Digraph()
    dot_words = Digraph(name='words', graph_attr={'rank': 'same'})
    node2name = {}
    lexicon_index, inner_node_index = 0, 0
    for node in const_tree.traverse_postorder_with_lexicons():
        label = node.tag
        if isinstance(node, ConstTree):
            if show_span:
                label += str(node.span)
            name = 'T' + str(inner_node_index)
            inner_node_index += 1
            attrs = {'shape': 'rectangle'}
            attrs.update((nodes_attrs or {}).get(node, {}))
            dot.node(name, label=label, _attributes=attrs)
            for child in node.children:
                dot.edge(node2name[id(child)], name, arrowhead='none')
        else:
            name = 'L' + str(lexicon_index)
            lexicon_index += 1
            dot_words.node(name, label=label, shape='rectangle')

        node2name[id(node)] = name

    for i in range(max(lexicon_index - 1, 0)):
        dot_words.edge('L' + str(i), 'L' + str(i + 1), style='invis')

    dot.subgraph(dot_words)

    if output_format == 'source':
        return dot.source
    dot.format = output_format
    dot.render(output_file, cleanup=True)


def draw_tree_decomposition(tree_root: TreeNode, external_nodes,
                            output_file=None, output_format='svg'):
    def _get_fake_name(node):
        return '_' + str(node.index)

    dot = Digraph()
    stack = [(None, tree_root)]
    while stack:
        parent, tree_node = stack.pop()
        cluster_name = 'cluster' + str(tree_node.index)
        with dot.subgraph(name=cluster_name) as dot_nodes:
            fake_name = _get_fake_name(tree_node)
            dot_nodes.node(fake_name, label='', _attributes=FAKE_NODE_ATYTRS)

            for graph_node in tree_node.graph_nodes:
                node_attrs = dict(fixedsize='true', **LABELED_NODE_ATTRS)
                if graph_node in external_nodes:
                    node_attrs.update(EXTERNAL_NODE_ATTRS)

                node_name = '{}@{}'.format(cluster_name, graph_node.name)
                dot_nodes.node(node_name, label=graph_node.name,
                               _attributes=node_attrs)

            if tree_node.graph_edge:
                edge = tree_node.graph_edge
                label = '{}:{}'.format(edge.label, '--'.join(node.name for node in edge.nodes))
                dot_nodes.attr(label=label)

        if parent is not None:
            dot.edge(_get_fake_name(parent), _get_fake_name(tree_node))

        if tree_node.left_child:
            stack.append((tree_node, tree_node.left_child))
        if tree_node.right_child:
            stack.append((tree_node, tree_node.right_child))

    if output_format == 'source':
        return dot.source
    dot.format = output_format
    dot.render(output_file, cleanup=True)


def draw_panorama_with_steps(hyper_graph, const_tree, node_blame_dict, edge_blame_dict,
                             output_file=None, output_format='svg', show_span=True):
    cfg_nodes2step = {cfg_node: step for step,
                      cfg_node in enumerate(const_tree.traverse_postorder())}
    nodes_by_step = {}
    for eds_node, step in node_blame_dict.items():
        nodes_by_step.setdefault(step, []).append(eds_node)
    edges_by_step = {}
    for eds_edge, step in edge_blame_dict.items():
        edges_by_step.setdefault(step, []).append(eds_edge)

    main_dot = Digraph(name='G')

    def _draw_cfg_node(cfg_node, name=None, label=None):
        step = cfg_nodes2step[cfg_node]
        graph_attr = {}
        if label is None:
            label = 'step {}'.format(step)
        graph_attr['label'] = label
        dot = Digraph(name or 'cluster{}'.format(step), graph_attr=graph_attr)
        eds_nodes = nodes_by_step.get(step, [])
        for eds_node in eds_nodes:
            _draw_hyper_node(dot, eds_node, {}, {})
        for eds_edge in edges_by_step.get(step, []):
            _draw_hyper_edge(dot, eds_edge, {}, show_span)
        main_dot.subgraph(dot)
        if isinstance(cfg_node.children[0], ConstTree):
            for child in cfg_node.children:
                _draw_cfg_node(child)
        return dot

    _draw_cfg_node(const_tree, 'G')

    if output_format == 'source':
        return main_dot.source
    main_dot.format = output_format
    main_dot.render(output_file, cleanup=True)
