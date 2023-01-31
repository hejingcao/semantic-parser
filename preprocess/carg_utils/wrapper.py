# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Tuple

from delphin.mrs.components import Pred

NodeID = str


class EdsGraph:
    """The internal data structure for EDS graphs."""

    class Node:
        """A Eds Node."""

        __slots__ = [
            'lemma', 'pos', 'sense', 'carg', 'label', 'span', 'properties',
            'outgoing_edges', 'incoming_edges',
            'extra'
        ]

        def __init__(self,
                     lemma: str,
                     pos: str,
                     sense: str,
                     carg: Optional[str],
                     label: str,
                     span: Tuple[int, int],
                     outgoing_edges: Dict[str, NodeID],
                     properties: List[str]):
            self.lemma = lemma
            self.pos = pos
            self.sense = sense
            self.carg = carg
            self.label = label
            self.outgoing_edges = outgoing_edges
            self.incoming_edges = []
            self.span = span
            self.properties = properties
            self.extra = None

        def __repr__(self):
            return str(self)

        def __str__(self):
            return '{}<{}:{}>'.format(self.label, self.span[0], self.span[1])

    def __init__(self, graph):
        self.top = graph['tops'][0]
        self.sentence = graph['input']

        incoming_edges = {}
        outgoing_edges = {}

        for edge in graph['edges']:
            source = edge['source']
            target = edge['target']
            elabel = edge['label']
            incoming_edges.setdefault(target, []).append((elabel, source))
            edges = outgoing_edges.setdefault(source, {})
            if elabel in edges:
                # print(f'{graph["id"]} duplicate out edges: {elabel}')
                continue

            edges[elabel] = target

        nodes = {}
        for node in graph['nodes']:
            anchors = node['anchors']
            assert len(anchors) == 1
            span = anchors[0]['from'], anchors[0]['to']

            properties = node.get('properties', [])
            values = node.get('values', [])
            properties = dict(zip(properties, values))

            label = node['label']
            node_pred = Pred.surface(label)
            node_id = node['id']
            node = nodes[node_id] = EdsGraph.Node(node_pred.lemma,
                                                  node_pred.pos,
                                                  node_pred.sense,
                                                  properties.get('carg'),
                                                  label,
                                                  span,
                                                  outgoing_edges.get(node_id, {}),
                                                  [])
            node.incoming_edges = incoming_edges.get(node_id, [])

        self.nodes = nodes  # type: Dict[NodeID, 'EdsGraph.Node']
