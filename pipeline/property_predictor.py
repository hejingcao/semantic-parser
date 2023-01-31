# -*- coding: utf-8 -*-

from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.common.dataclass_options import argfield
from framework.common.logger import LOGGER, open_file
from framework.common.utils import DotDict
from framework.torch_extra.layers.biaffine import Biaffine
from framework.torch_extra.layers.dropout import FeatureDropout
from framework.torch_extra.layers.sequential import make_mlp_layers
from framework.torch_extra.parser_base import ParserBase, ParserNetworkBase
from framework.torch_extra.utils import broadcast_gather, cross_entropy_nd, pad_and_stack_1d
from preprocess.tokenize import convert_char_span_to_word_span

from .relation_detector import EdsFeatures as EdsFeaturesBase
from .relation_detector import EdsSample as EdsSampleBase


class EdsSample(EdsSampleBase):
    @classmethod
    def from_string(cls, _index, string, attrs):
        words = string.split()
        length = sum(map(len, attrs['nodes']))
        if length == 0:
            length = len(words)
            attrs['nodes'] = [[(str(index), 'None')] for index in range(length)]
            attrs['spans'] = [[(-100, -100)]] * length
            attrs['word_index_spans'] = [(-100, -100)] * length

            assert not attrs['edges']
            LOGGER.warn('!!! empty graph %s', attrs.get('sample_id'))
        else:
            char_spans = attrs['token_spans']
            attrs['word_index_spans'] = [
                convert_char_span_to_word_span(span, char_spans, words)
                for spans in attrs['spans']
                for span in spans
            ]

        return cls(words=words, length=length)

    @classmethod
    def internal_evaluate(cls, gold_samples, system_samples, log_file):
        char_correct, word_correct = [0, 0, 0, 0], [0, 0, 0, 0]

        top_correct = [0, 0]
        for gold_sample, system_sample in zip(gold_samples, system_samples):
            if gold_sample.top == system_sample.top:
                top_correct[0] += 1
            top_correct[1] += 1

            gold_char_spans = np.array(sum(gold_sample.spans, []))
            system_char_spans = np.array(sum(system_sample.spans, []))

            gold_word_index_spans = np.array(gold_sample.word_index_spans)
            system_word_index_spans = np.array(system_sample.word_index_spans)

            word_mask = (gold_word_index_spans == system_word_index_spans) \
                & (gold_word_index_spans != -100)
            char_mask = (gold_char_spans == system_char_spans) & (gold_char_spans != -100)

            word_correct[0] += (word_mask[:, 0]).sum()
            word_correct[1] += (word_mask[:, 1]).sum()
            word_correct[2] += (word_mask[:, 0] & word_mask[:, 1]).sum()

            word_count = (gold_word_index_spans[:, 0] != -100).sum()
            assert word_count == (gold_word_index_spans[:, 1] != -100).sum()
            word_correct[3] += word_count

            char_correct[0] += (char_mask[:, 0]).sum()
            char_correct[1] += (char_mask[:, 1]).sum()
            char_correct[2] += (char_mask[:, 0] & char_mask[:, 1]).sum()

            char_count = (gold_char_spans[:, 0] != -100).sum()
            assert char_count == (gold_char_spans[:, 1] != -100).sum()
            char_correct[3] += char_count

        word_accuracy = [_ / word_correct[-1] * 100 for _ in word_correct[:-1]]
        char_accuracy = [_ / char_correct[-1] * 100 for _ in char_correct[:-1]]
        top_accuracy = 100 * top_correct[0] / top_correct[1]

        log_lines = sum(
            [
                [
                    f'# {tag} correct [start, end, span, total]: {correct}',
                    f'# {tag} accuracy [start, end, span]: {item}'
                ]
                for tag, correct, item in zip(['word', 'char'],
                                              [word_correct, char_correct],
                                              [word_accuracy, char_accuracy])
            ],
            [f'# [top, total]: {top_correct} {top_accuracy}'])

        correct = char_correct[-2] + top_correct[-2]
        count = char_correct[-1] + top_correct[-1]

        log_lines.append(f'total: {correct} {count} {correct/count}')

        if log_file is not None:
            with open(log_file, 'w') as fp:
                fp.write('\n'.join(log_lines))

        print(*log_lines, sep='\n')

        return correct / count


class EdsFeatures(EdsFeaturesBase):
    top: int = -1
    word_index_spans: Any = None

    @classmethod
    def create(cls, original_index, original_object, plugins, statistics):
        sample = super().create(original_index, original_object, plugins, statistics)

        name_to_index = {}
        node_index = 0
        for nodes_info in original_object.nodes:
            for node_name, _ in nodes_info:
                name_to_index[node_name] = node_index
                node_index += 1

        sample.word_index_spans = torch.tensor(original_object.word_index_spans)
        sample.top = name_to_index.get(original_object.top, -100)
        if sample.top == -100:
            LOGGER.warning('top node is removed %s', original_object.sample_id)

        return sample

    @classmethod
    def pack_to_batch(cls, batch_samples, plugins, statistics):
        inputs = super().pack_to_batch(batch_samples, plugins, statistics)

        inputs.word_index_spans = pad_and_stack_1d(
            [sample.word_index_spans for sample in batch_samples], pad=-100)
        inputs.tops = torch.tensor([sample.top for sample in batch_samples])

        return inputs


class PropertyPredictorHyperParams(ParserBase.HyperParams):
    node_size: int = 256
    span_size: int = 600

    node_dropout: float = 0.5
    top_dropout: float = 0.1

    top_hidden_sizes: List[int] = argfield(default_factory=lambda: [600])

    use_hidden_layer_norm: bool = False

    loss_type: str = argfield('hinge', choices=['hinge', 'cross_entropy'])

    activation: str = 'leaky_relu/0.1'


class PropertyPredictorNetwork(ParserNetworkBase):
    def __init__(self, hyper_params, vocabs, plugins):
        super().__init__(hyper_params, vocabs, plugins)

        node_vocab = vocabs.get('node')

        encoder_size = self.encoder.output_size
        node_size = hyper_params.node_size
        span_size = hyper_params.span_size

        self.loss_type = hyper_params.loss_type

        self.node_embeddings = \
            nn.Embedding(len(node_vocab), node_size, padding_idx=node_vocab.pad_id)
        self.node_dropout = FeatureDropout(hyper_params.node_dropout)

        self.node_project = \
            make_mlp_layers(encoder_size + node_size, span_size,
                            use_layer_norm=hyper_params.use_hidden_layer_norm,
                            activation=hyper_params.activation,
                            use_last_activation=True,
                            use_last_bias=True)
        self.word_project = \
            make_mlp_layers(encoder_size, span_size,
                            use_layer_norm=hyper_params.use_hidden_layer_norm,
                            activation=hyper_params.activation,
                            use_last_activation=True,
                            use_last_bias=True)

        self.top_project = \
            make_mlp_layers(encoder_size + node_size, 1,
                            hidden_sizes=hyper_params.top_hidden_sizes,
                            dropout=hyper_params.top_dropout,
                            activation=hyper_params.activation,
                            use_layer_norm=hyper_params.use_hidden_layer_norm,
                            use_last_bias=True)

        self.span_biaffine = Biaffine(span_size, span_size, 2,
                                      dropout=0.3,
                                      input_bias1=True, input_bias2=True)

    def forward(self, batch_samples, inputs):
        input_embeddings = self.input_embeddings(inputs)[:, 1:-1]
        real_lengths = inputs.encoder_lengths - 2

        encoder_outputs, _ = self.encoder(input_embeddings, real_lengths)

        node_label_embeddings = self.node_dropout(self.node_embeddings(inputs.node_labels))
        # node_embeddings[b][l][*] = encoder_outputs[b][node_word_indices[b][l][*]][*]
        node_embeddings = broadcast_gather(encoder_outputs, 1, inputs.node_word_indices)

        node_features = torch.cat([node_embeddings, node_label_embeddings], dim=-1)

        top_scores = self.top_project(node_features).squeeze(-1)
        # batch_size, node_count, 2, word_count
        span_scores = self.span_biaffine(self.node_project(node_features),
                                         self.word_project(encoder_outputs))

        outputs = DotDict()
        if self.training:
            word_index_spans = inputs.word_index_spans
            tops = inputs.tops

            if self.loss_type == 'cross_entropy':
                span_loss = cross_entropy_nd(span_scores, word_index_spans, reduction='mean')
                top_loss = cross_entropy_nd(top_scores, tops, reduction='mean')
            else:
                word_index_spans = word_index_spans.view(-1)

                weight = (word_index_spans != -100).float()
                word_index_spans = torch.max(word_index_spans, torch.zeros_like(word_index_spans))
                span_loss = F.multi_margin_loss(span_scores, word_index_spans, weight=weight,
                                                reduction='mean')

                weight = (tops != -100).float()
                tops = torch.max(tops, torch.zeros_like(tops))
                top_loss = F.multi_margin_loss(top_scores, tops, weight=weight,
                                               reduction='mean')

            outputs.loss = span_loss + top_loss
        else:
            outputs.word_index_spans = span_scores.argmax(dim=-1)
            outputs.tops = top_scores.argmax(dim=-1)

        return outputs


class PropertyPredictor(ParserBase):
    METRICS_MODE = 'max'
    METRICS_NAME = 'SMATCH'

    FEATURES_CLASS = EdsFeatures
    SAMPLE_CLASS = EdsSample
    NETWORK_CLASS = PropertyPredictorNetwork

    class Options(ParserBase.Options):
        hyper_params: PropertyPredictorHyperParams

    def split_outputs(self, batch_samples, outputs):
        for sample, top_index, word_index_spans in zip(batch_samples,
                                                       outputs.tops.cpu().numpy(),
                                                       outputs.word_index_spans.cpu().numpy()):
            original_object = sample.original_object

            num_words = len(original_object.words)
            num_nodes = len(sample.node_labels)

            word_index_spans = word_index_spans[:num_nodes].tolist()
            token_spans = original_object.token_spans

            top = None
            spans = []
            index = 0

            for nodes in original_object.nodes:
                current_spans = []
                for name, _ in nodes:
                    start, end = word_index_spans[index]
                    current_spans.append((token_spans[min(start, num_words - 1)][0],
                                          token_spans[min(end, num_words - 1)][1]))
                    if top_index == index:
                        top = name
                    index += 1
                spans.append(current_spans)

            yield original_object.copy_attrs(word_index_spans=word_index_spans,
                                             spans=spans, top=top)

    def write_outputs(self, output_path, _, system_samples):
        with open_file(output_path, 'w') as fp:
            EdsSample.to_file(fp, system_samples)
