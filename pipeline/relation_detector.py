# -*- coding: utf-8 -*-

import itertools
import pickle
import tempfile
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.common.dataclass_options import ExistFile, argfield
from framework.common.logger import LOGGER, open_file
from framework.common.utils import DotDict
from framework.data.dataset import DataSample
from framework.data.sentence import SentenceFeatures
from framework.torch_extra.layers.biaffine import Biaffine, DiagonalBiaffine
from framework.torch_extra.layers.dropout import FeatureDropout
from framework.torch_extra.layers.sequential import make_mlp_layers
from framework.torch_extra.parser_base import ParserBase, ParserNetworkBase
from framework.torch_extra.utils import (broadcast_gather,
                                         cross_entropy_nd,
                                         pad_and_stack_1d,
                                         pad_and_stack_2d)

from .lemma_utils import read_lemma_mappings_file, recover_edge_label
from .smatch_utils import read_smatch_graphs, run_smatch


class EdsSample(DataSample):
    COMMONT_PREFIX = '###COMMENT### '

    length: int = 0
    words: Any = None

    def __len__(self):
        return self.length

    def to_string(self):
        return ' '.join(self.words)

    @classmethod
    def to_smatch_file(cls, fp, samples):
        for sample in samples:
            fp.write(f'# {sample.sample_id}\n')

            all_nodes = []
            for word_index, (nodes, word, lemmas) in \
                    enumerate(itertools.zip_longest(sample.nodes, sample.words, sample.lemmas)):
                for (node_name, node_label), lemma in zip(nodes, lemmas):
                    if lemma is not None:
                        node_label = lemma

                    all_nodes.append((node_name, node_label))

            fp.write(f'{len(all_nodes)}\n')
            for node_name, label in all_nodes:
                fp.write(f'{node_name} {label}\n')

            all_edges = []
            for start, end, labels in sample.edges:
                for label in labels.split('&&&'):
                    all_edges.append((start, end, label))

            fp.write(f'{len(all_edges)}\n')
            for start, end, label in all_edges:
                fp.write(f'{start} {end} {label}\n')

            fp.write('\n')

    @classmethod
    def from_smatch_file(cls, fp):
        samples = []
        for graph in read_smatch_graphs(fp):
            nodes = graph['nodes']
            graph['nodes'] = [nodes]
            graph['lemmas'] = [[None] * len(nodes)]
            graph['spans'] = [graph['spans']]
            samples.append(cls(words=['__empty__'], length=len(nodes), attrs=graph))
        return samples

    @classmethod
    def from_string(cls, _index, string, attrs):
        words = string.split()
        length = sum(map(len, attrs['nodes']))
        if length == 0:
            length = len(words)
            attrs['nodes'] = [[(str(index), 'None')] for index in range(length)]
            # assert not attrs['edges']
            LOGGER.warn('!!! empty graph %s', attrs.get('sample_id'))
        return cls(words=words, length=length)

    @classmethod
    def internal_evaluate(cls, gold_items, system_items, log_file, cleanup=True):
        tmp_gold = tempfile.NamedTemporaryFile()
        tmp_system = tempfile.NamedTemporaryFile()

        cls.to_file(tmp_gold.name, gold_items, output_format='smatch')
        cls.to_file(tmp_system.name, system_items, output_format='smatch')

        return run_smatch(tmp_gold.name, tmp_system.name, cleanup=cleanup)


class EdsFeatures(SentenceFeatures):
    node_word_indices: Any = None
    node_labels: Any = None
    edges: Any = None

    @classmethod
    def create(cls, original_index, original_object, plugins, statistics):
        sample = super().create(original_index, original_object, plugins, statistics,
                                lower_case=True)
        name_to_index = {}
        node_index = 0
        node_word_indices = []
        node_labels = []

        node_vocab = statistics.get('node')

        for word_index, nodes_info in enumerate(original_object.nodes):
            for node_name, node_label in nodes_info:
                name_to_index[node_name] = node_index
                node_index += 1

                node_word_indices.append(word_index)
                node_labels.append(node_label)

        sample.node_word_indices = torch.tensor(node_word_indices, dtype=torch.int64)
        sample.node_labels = torch.tensor(node_vocab.words_to_ids(node_labels), dtype=torch.int64)

        edges = original_object.attrs.get('edges')
        if edges is not None:
            edge_vocab = statistics.get('edge')

            # assert (sample.node_labels != node_vocab.unk_id).all()

            sample.edges = torch.zeros((len(node_labels), len(node_labels)), dtype=torch.int64)
            for start, end, label in edges:
                label = edge_vocab.word_to_id(label)
                sample.edges[name_to_index[start], name_to_index[end]] = label

        return sample

    @classmethod
    def pack_to_batch(cls, batch_samples, plugins, statistics):
        inputs = super().pack_to_batch(batch_samples, plugins, statistics)

        inputs.node_labels = pad_and_stack_1d([sample.node_labels for sample in batch_samples])
        inputs.node_word_indices = \
            pad_and_stack_1d([sample.node_word_indices for sample in batch_samples])

        if batch_samples[0].edges is not None:
            inputs.edges = pad_and_stack_2d([sample.edges for sample in batch_samples], pad=-100)

        return inputs


class MSGHyperParams(ParserBase.HyperParams):
    node_size: int = 256
    node_dropout: float = 0.5

    edge_hidden_size: int = 600
    label_hidden_size: int = 600

    use_hidden_layer_norm: bool = False

    label_loss_type: str = argfield('hinge', choices=['hinge', 'cross_entropy', 'label_smoothing'])
    label_smoothing_eps: float = 0.1

    edge_loss_type: str = argfield('hinge', choices=['hinge', 'cross_entropy'])

    activation: str = 'leaky_relu/0.1'


class MSGNetwork(ParserNetworkBase):
    def __init__(self, hyper_params, vocabs, plugins):
        super().__init__(hyper_params, vocabs, plugins)

        node_vocab = vocabs.get('node')
        edge_vocab = vocabs.get('edge')
        assert edge_vocab.pad_id == 0

        encoder_size = self.encoder.output_size
        node_size = hyper_params.node_size
        self.edge_hidden_size = edge_hidden_size = hyper_params.edge_hidden_size
        self.label_hidden_size = label_hidden_size = hyper_params.label_hidden_size

        final_size = 2 if hyper_params.edge_loss_type == 'cross_entropy' else 1

        self.edge_loss_type = hyper_params.edge_loss_type
        self.label_loss_type = hyper_params.label_loss_type
        self.label_smoothing_eps = hyper_params.label_smoothing_eps

        self.node_embeddings = \
            nn.Embedding(len(node_vocab), node_size, padding_idx=node_vocab.pad_id)
        self.node_dropout = FeatureDropout(hyper_params.node_dropout)

        reducer_size = 2 * (edge_hidden_size + label_hidden_size)
        self.node_reducer = make_mlp_layers(encoder_size + node_size, reducer_size, hidden_sizes=(),
                                            dropout=0,
                                            activation=hyper_params.activation,
                                            use_layer_norm=hyper_params.use_hidden_layer_norm,
                                            use_last_bias=True,
                                            use_last_activation=True)

        self.edge = Biaffine(edge_hidden_size, edge_hidden_size, final_size,
                             dropout=0.25,
                             input_bias1=True, input_bias2=True,
                             output_bias=(final_size == 2))
        self.edge_label = DiagonalBiaffine(edge_hidden_size, len(edge_vocab), dropout=0.33)

    def forward(self, batch_samples, inputs):
        input_embeddings = self.input_embeddings(inputs)[:, 1:-1]
        real_lengths = inputs.encoder_lengths - 2

        encoder_outputs, _ = self.encoder(input_embeddings, real_lengths)

        node_label_embeddings = self.node_dropout(self.node_embeddings(inputs.node_labels))
        # node_embeddings[b][l][*] = encoder_outputs[b][node_word_indices[b][l][*]][*]
        node_embeddings = broadcast_gather(encoder_outputs, 1, inputs.node_word_indices)

        encoder_outputs = torch.cat([node_embeddings, node_label_embeddings], dim=-1)

        edge_hidden_size = self.edge_hidden_size
        label_hidden_size = self.label_hidden_size

        sep0 = edge_hidden_size
        sep1 = sep0 + edge_hidden_size
        sep2 = sep1 + label_hidden_size

        node_features = self.node_reducer(encoder_outputs)

        source_features = node_features[:, :, :sep0]
        target_features = node_features[:, :, sep0:sep1]
        source_labels_features = node_features[:, :, sep1:sep2]
        target_labels_features = node_features[:, :, sep2:]

        # shape: [batch_size, num_nodes, final_size, num_nodes]
        edge_scores = self.edge(source_features, target_features)

        use_cross_entropy = edge_scores.size(2) == 2
        if use_cross_entropy:
            # shape: [batch_size, num_nodes, num_nodes, final_size]
            edge_scores = edge_scores.transpose(2, 3).contiguous()
        else:
            # shape: [batch_size, num_nodes, num_nodes]
            edge_scores = edge_scores.squeeze(2)

        # shape: [batch_size, num_nodes, num_nodes, num_labels]
        edge_label_scores = self.edge_label(source_labels_features, target_labels_features)
        edge_label_scores = edge_label_scores.transpose(2, 3).contiguous()

        outputs = DotDict()

        if self.training:
            edge_labels = inputs.edges

            # NOTE: 0 means no edges, -100 means padding
            # 1, 0 or -100
            gold_edges_unlabeled = torch.min(edge_labels, torch.ones_like(edge_labels))
            gold_edges_mask = gold_edges_unlabeled > 0  # 1, 0

            if use_cross_entropy:
                edge_loss = cross_entropy_nd(edge_scores, gold_edges_unlabeled, reduction='mean')
            else:
                # NOTE: try to make scores of correct edges above 0.6 and the
                # ones of wrong edges below -0.4
                edge_scores = edge_scores + 0.4 - gold_edges_mask.float()

                pred_edges = (edge_scores > 0) & (edge_labels != -100)

                wrong = pred_edges & ~gold_edges_mask
                missing = gold_edges_mask & (~pred_edges)

                total_count = max(wrong.sum().item() + missing.sum().item(), 1)
                edge_loss = (edge_scores[wrong].sum() - edge_scores[missing].sum()) / total_count

            # shape: [num_total_edges, num_labels]
            gold_edge_label_scores = edge_label_scores[gold_edges_mask]
            # shape: [num_total_edges]
            gold_edge_labels = edge_labels[gold_edges_mask]

            if self.label_loss_type == 'label_smoothing':
                eps = self.label_smoothing_eps
                num_labels = gold_edge_label_scores.size(1)

                one_hot = torch.zeros_like(gold_edge_label_scores)
                one_hot = one_hot.scatter(1, gold_edge_labels.view(-1, 1), 1)
                one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_labels - 1)

                log_prob = F.log_softmax(gold_edge_label_scores, dim=-1)

                edge_label_loss = -(one_hot * log_prob).sum(dim=-1).mean()
            elif self.label_loss_type == 'cross_entropy':
                edge_label_loss = cross_entropy_nd(gold_edge_label_scores,
                                                   gold_edge_labels, reduction='mean')
            else:
                # score of correct labels should be minus one first (margin)
                gold_edge_labels = gold_edge_labels.unsqueeze(1)
                gold_edge_label_scores = gold_edge_label_scores.scatter_add(
                    1,
                    gold_edge_labels,
                    -torch.ones_like(gold_edge_labels, dtype=torch.float)
                )

                # shape: [num_total_edges]
                gold_scores = gold_edge_label_scores.gather(1, gold_edge_labels).squeeze(1)

                pred_scores, _ = gold_edge_label_scores.max(dim=-1)
                edge_label_loss = (pred_scores - gold_scores).mean()

            outputs.loss = edge_loss + edge_label_loss
        else:
            if use_cross_entropy:
                edges = edge_scores.argmax(dim=-1)  # 1, 0
            else:
                edges = (edge_scores > 0).long()

            outputs.edges = edge_label_scores.argmax(dim=-1) * edges

            # NOTE: return scores
            outputs.edge_scores = edge_scores
            outputs.edge_label_scores = edge_label_scores

        return outputs


class MSGParser(ParserBase):
    METRICS_MODE = 'max'
    METRICS_NAME = 'SMATCH'

    FEATURES_CLASS = EdsFeatures
    SAMPLE_CLASS = EdsSample
    NETWORK_CLASS = MSGNetwork

    class Options(ParserBase.Options):
        lemma_dictionary_path: ExistFile
        dev_smatch_path: ExistFile

        hyper_params: MSGHyperParams

    def initialize(self, saved_state):
        super().initialize(saved_state)

        path = self.options.lemma_dictionary_path
        if path.endswith('pkl'):
            self.lemma_dictionary = pickle.load(open_file(path, 'rb'))
        else:
            self.lemma_dictionary = read_lemma_mappings_file(path)

    def split_outputs(self, batch_samples, outputs):
        edge_vocab = self.statistics.get('edge')

        for sample, edges in zip(batch_samples, outputs.edges.cpu().numpy()):
            num_nodes = len(sample.node_labels)
            node_names = [name for nodes in sample.original_object.nodes for name, _ in nodes]

            all_edges = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if edges[i, j] != 0:
                        all_edges.append((node_names[i], node_names[j],
                                          edge_vocab.id_to_word(edges[i, j])))

            lemmas = []

            for word_index, (nodes, word) in \
                enumerate(itertools.zip_longest(sample.original_object.nodes,
                                                sample.original_object.words)):
                current_lemmas = []
                for _, node_label in nodes:
                    lemma = None
                    if node_label.startswith('_'):
                        lemma = recover_edge_label(node_label, word, self.lemma_dictionary)
                    current_lemmas.append(lemma)

                lemmas.append(current_lemmas)

            yield sample.original_object.copy_attrs(edges=all_edges, lemmas=lemmas)

    def write_outputs(self, output_path, _, system_samples):
        with open_file(output_path, 'w') as fp:
            EdsSample.to_file(fp, system_samples)

        extra_file = output_path + '.smatch'
        with open_file(extra_file, 'w') as fp:
            EdsSample.to_file(fp, system_samples, output_format='smatch')

        return [extra_file]

    def run_evaluator(self, _, samples, outputs, output_files):
        output_files.pop(1)  # remove score file
        score, extra_files = run_smatch(self.options.dev_smatch_path, output_files[-1])
        output_files.extend(extra_files)  # add smatch outputs

        return score
