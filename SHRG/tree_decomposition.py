# -*- coding: utf-8 -*-

"""The code that follows comes from bolinas project. We use it to compute a
nice tree decomposition for the rule's graph.

@see https://www.isi.edu/licensed-sw/bolinas/index.html
"""

from itertools import chain


class TreeNode:
    __slots__ = ('graph_nodes', 'graph_edge', '_left_child', '_right_child', 'index', 'parent')

    def __init__(self, nodes=(), edge=None):
        self.parent = None
        self.graph_nodes = set(nodes)
        self.graph_edge = edge
        self._left_child = None
        self._right_child = None
        self.index = None

    @property
    def treewidth(self):
        width = len(self.graph_nodes) - 1
        if self._left_child:
            width = max(width, self._left_child.treewidth)
        if self._right_child:
            width = max(width, self._right_child.treewidth)
        return width

    @property
    def left_child(self):
        return self._left_child

    @property
    def right_child(self):
        return self._right_child

    @left_child.setter
    def left_child(self, child):
        self._left_child = child
        child.parent = self

    @right_child.setter
    def right_child(self, child):
        self._right_child = child
        child.parent = self

    def to_string(self, indent=0):
        left_string = self.left_child.to_string(indent + 4) if self.left_child else ''
        right_string = self.right_child.to_string(indent + 4) if self.right_child else ''
        return '{}#{}[{}] {}\n{}\n{}'.format(' ' * indent, self.index,
                                             self.graph_nodes, self.graph_edge,
                                             left_string, right_string)

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return str(self)

    def traverse_postorder(self):
        if self.left_child:
            yield from self.left_child.traverse_postorder()
        if self.right_child:
            yield from self.right_child.traverse_postorder()
        yield self


def run_intersection(tree_root, graph_node, graph_nodes_map):
    left = tree_root.left_child
    right = tree_root.right_child
    if right \
       and graph_node not in tree_root.graph_nodes \
       and graph_node in graph_nodes_map[left] \
       and graph_node in graph_nodes_map[right]:
        tree_root.graph_nodes.aad(graph_node)

    if graph_node in tree_root.graph_nodes:
        if left \
           and graph_node in graph_nodes_map[left] \
           and graph_node not in left.graph_nodes:
            left.graph_nodes.add(graph_node)
        if right \
           and graph_node in graph_nodes_map[right] \
           and graph_node not in right.graph_nodes:
            right.graph_nodes.add(graph_node)

    if left:
        run_intersection(left, graph_node, graph_nodes_map)
    if right:
        run_intersection(right, graph_node, graph_nodes_map)


def tree_decomposition(hyper_graph, external_nodes, completely=True):
    visited_edges = set()
    node_linked_edges = {}
    for node in hyper_graph.nodes:
        for edge in hyper_graph.edges:
            if node in edge.nodes:
                node_linked_edges.setdefault(node, []).append(edge)

    def _add_new_node(parent_node, child_node):
        if parent_node.left_child is None:
            parent_node.left_child = child_node
        else:
            binary_node = TreeNode()
            binary_node.left_child = parent_node.left_child
            binary_node.right_child = child_node
            parent_node.left_child = binary_node

    def _decompose(current_edge):
        current_node = TreeNode(current_edge.nodes, current_edge)
        visited_edges.add(current_edge)
        for node in current_edge.nodes:
            for edge in node_linked_edges[node]:
                if edge in visited_edges:
                    continue
                _add_new_node(current_node, _decompose(edge))

        if current_node.left_child is None:
            current_node.left_child = TreeNode()

        return current_node

    def _set_nodeid(tree, index=0):
        """
        @type tree: TreeNode
        """
        tree.index = index
        index += 1
        if tree.left_child:
            index = _set_nodeid(tree.left_child, index)
        if tree.right_child:
            index = _set_nodeid(tree.right_child, index)
        return index

    root = TreeNode(external_nodes)
    if external_nodes:
        initial_edges = node_linked_edges[next(iter(external_nodes))]
    else:
        initial_edges = ()
    for edge in chain(initial_edges, hyper_graph.edges):
        if edge in visited_edges:
            continue
        _add_new_node(root, _decompose(edge))

    if root.left_child is None:
        root.left_child = TreeNode()

    graph_nodes_map = {}
    for tree_node in root.traverse_postorder():
        graph_nodes = set(tree_node.graph_nodes)
        graph_nodes.update(graph_nodes_map.get(tree_node.left_child, ()))
        graph_nodes.update(graph_nodes_map.get(tree_node.right_child, ()))
        graph_nodes_map[tree_node] = graph_nodes

    if completely:
        for node in hyper_graph.nodes:
            run_intersection(root, node, graph_nodes_map)

    graph_nodes = root.graph_nodes
    if root.right_child is None \
       and (not graph_nodes or graph_nodes == root.left_child.graph_nodes):
        root = root.left_child

    _set_nodeid(root)

    return root
