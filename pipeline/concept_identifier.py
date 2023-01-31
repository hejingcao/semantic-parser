# -*- coding: utf-8 -*-

from typing import Any, List, Optional

import torch

from framework.common.dataclass_options import argfield
from framework.common.logger import open_file
from framework.common.utils import DotDict
from framework.data.dataset import DELETE, DataSample
from framework.data.sentence import SentenceFeatures, collect_sentence_vocabs
from framework.data.vocab import VocabularySet, lookup_words
from framework.evalute.labeling import LabelingFScorer
from framework.torch_extra.layers.crf import CRF
from framework.torch_extra.layers.sequential import ContextualOptions, make_mlp_layers
from framework.torch_extra.parser_base import ParserBase, ParserNetworkBase
from framework.torch_extra.utils import cross_entropy_nd, pad_and_stack_1d


class EdsSample(DataSample):
    COMMONT_PREFIX = '###COMMENT### '

    words: Any = None

    def __len__(self):
        return len(self.words)

    def to_string(self):
        return ' '.join(self.words)

    def new(self, labels, attachments):
        node_index = 0
        nodes = []
        for label, current_attachments in zip(labels, attachments):
            current = []
            if label != 'None':
                current.append((str(node_index), label))
                node_index += 1

            for attachment in current_attachments:
                current.append((str(node_index), attachment))
                node_index += 1

            nodes.append(current)

        return self.copy_attrs(nodes=nodes, edges=DELETE, labels=labels, attachments=attachments)

    @classmethod
    def from_string(cls, _index, string, attrs):
        words = string.split()
        attrs.setdefault('labels', ['None'] * len(words))
        attrs.setdefault('attachments', [[]] * len(words))

        return cls(words=words)

    @classmethod
    def internal_evaluate(cls, gold_items, system_items, log_file):
        labels_scorer = LabelingFScorer('Labels')
        attachments_scorer = LabelingFScorer('Attachments')
        total_scorer = LabelingFScorer('ALL')

        for gold_item, system_item in zip(gold_items, system_items):
            labels_scorer.update(gold_item.labels, system_item.labels)
            attachments_scorer.update_sets(gold_item.attachments, system_item.attachments)
            total_scorer.update(gold_item.labels, system_item.labels, except_tag='None')
            total_scorer.update_sets(gold_item.attachments, system_item.attachments)

        labels_str, labels_f1 = labels_scorer.get_report()
        attachments_str, attachments_f1 = attachments_scorer.get_report()
        total_str, total_f1 = total_scorer.get_report()

        print(f'Labels: {labels_f1:.2f}')
        print(f'Attachments: {attachments_f1:.2f}')
        print(f'Total: {total_f1:.2f}')

        if log_file is not None:
            with open_file(log_file, 'w') as fp:
                fp.write(labels_str)
                fp.write(attachments_str)
                fp.write(total_str)

        return total_f1


class EdsFeatures(SentenceFeatures):
    labels: Any = None
    attachment_bags: Any = None

    @classmethod
    def create(cls, original_index, original_object, plugins, statistics):
        sample = super().create(original_index, original_object, plugins, statistics,
                                lower_case=True)

        sample.labels = lookup_words(original_object.labels, statistics.get('label'),
                                     sos_and_eos=False)

        sample.attachment_bags = lookup_words(
            [';'.join(sorted(attachments)) for attachments in original_object.attachments],
            statistics.get('attachment_bag'),
            sos_and_eos=False)

        return sample

    @classmethod
    def pack_to_batch(cls, batch_samples, plugins, statistics):
        inputs = super().pack_to_batch(batch_samples, plugins, statistics)

        inputs.labels = pad_and_stack_1d(
            [torch.from_numpy(sample.labels) for sample in batch_samples],
            pad=-100)
        inputs.attachment_bags = pad_and_stack_1d(
            [torch.from_numpy(sample.attachment_bags) for sample in batch_samples],
            pad=-100)

        return inputs


class ConceptIdentifierHyperParams(ParserBase.HyperParams):
    use_crf: bool = True
    mlp_dropout: float = 0.2

    hidden_sizes: List[int] = argfield(default_factory=lambda: [100])

    attachment_encoder: Optional[ContextualOptions]
    attachment_hidden_sizes: List[int] = argfield(default_factory=lambda: [100])

    hidden_layer_norm: bool = False

    activation: str = 'leaky_relu/0.1'


class ConceptIdentifierNetwork(ParserNetworkBase):
    def __init__(self, hyper_params, vocabs, plugins):
        super().__init__(hyper_params, vocabs, plugins)

        attachment_encoder = hyper_params.attachment_encoder
        if attachment_encoder is not None:
            self.attachment_encoder = attachment_encoder.create(self.input_embeddings.output_size)
            attachment_encoder_size = self.attachment_encoder.output_size
        else:
            self.attachment_encoder = None
            attachment_encoder_size = self.encoder.output_size

        self.label_count = len(vocabs.get('label'))
        self.attachment_bag_count = len(vocabs.get('attachment_bag'))

        self.projection = \
            make_mlp_layers(self.encoder.output_size,
                            self.label_count,
                            hidden_sizes=hyper_params.hidden_sizes,
                            dropout=hyper_params.mlp_dropout,
                            activation=hyper_params.activation,
                            use_layer_norm=hyper_params.hidden_layer_norm,
                            use_last_bias=True)

        self.attachment_projection = \
            make_mlp_layers(attachment_encoder_size,
                            self.attachment_bag_count,
                            hidden_sizes=hyper_params.attachment_hidden_sizes,
                            dropout=hyper_params.mlp_dropout,
                            activation=hyper_params.activation,
                            use_layer_norm=hyper_params.hidden_layer_norm,
                            use_last_bias=True)

        if hyper_params.use_crf:
            self.crf_unit = CRF(self.label_count)
        else:
            self.crf_unit = None

    def forward(self, batch_samples, inputs):
        input_embeddings = self.input_embeddings(inputs)[:, 1:-1]
        real_lengths = inputs.encoder_lengths - 2

        encoder_outputs, _ = self.encoder(input_embeddings, real_lengths)
        if self.attachment_encoder is None:
            attachment_encoder_outputs = encoder_outputs
        else:
            attachment_encoder_outputs, _ = self.attachment_encoder(input_embeddings, real_lengths)

        label_logits = self.projection(encoder_outputs)
        attachment_logits = self.attachment_projection(attachment_encoder_outputs)

        outputs = DotDict(label_logits=label_logits,
                          attachment_logits=attachment_logits)

        if self.training:
            labels = inputs.labels
            if self.crf_unit is not None:
                # padding of labels is -100
                labels = torch.max(labels, torch.zeros_like(labels))
                label_loss = -self.crf_unit(label_logits, labels, real_lengths, batch_first=True)
            else:
                label_loss = cross_entropy_nd(label_logits, labels)

            attachment_loss = cross_entropy_nd(attachment_logits, inputs.attachment_bags)

            outputs.loss = label_loss + attachment_loss
        else:
            if self.crf_unit is not None:
                outputs.labels = \
                    self.crf_unit.viterbi_decode(label_logits, real_lengths, batch_first=True)
            else:
                outputs.labels = label_logits.argmax(dim=-1)
            outputs.attachment_bags = attachment_logits.argmax(dim=-1)

        return outputs


class ConceptIdentifier(ParserBase):
    METRICS_MODE = 'max'
    METRICS_NAME = 'F1'

    FEATURES_CLASS = EdsFeatures
    SAMPLE_CLASS = EdsSample
    NETWORK_CLASS = ConceptIdentifierNetwork

    class Options(ParserBase.Options):
        hyper_params: ConceptIdentifierHyperParams

    def build_vocabs(self):
        vocabs = VocabularySet()

        label_vocab = vocabs.new('label')
        attachment_vocab = vocabs.new('attachment')
        attachment_bag_vocab = vocabs.new('attachment_bag')
        edges = vocabs.new('edge')
        nodes = vocabs.new('node')

        for train_path in self.options.train_paths:
            buckets = self.create_buckets(train_path, 'train')

            collect_sentence_vocabs(buckets.original_objects, vocabs, sos_and_eos=True)

            for sample in buckets.original_objects:
                label_vocab.update(sample.labels)
                for attachments in sample.attachments:
                    attachment_vocab.update(attachments)
                    attachment_bag_vocab.add(';'.join(sorted(attachments)))

                edges.update(edge_label for _, _, edge_label in sample.edges)
                nodes.update(node_label
                             for nodes_info in sample.nodes
                             for _, node_label in nodes_info)

        return vocabs

    def split_outputs(self, batch_samples, outputs):
        label_vocab = self.statistics.get('label')
        attachment_bag_vocab = self.statistics.get('attachment_bag')

        for sample, labels, attachment_bags in zip(batch_samples,
                                                   outputs.labels, outputs.attachment_bags):
            length = len(sample.original_object)

            labels = label_vocab.ids_to_words(labels[:length].cpu().numpy())
            attachments = [
                [_ for _ in bag.split(';') if _]
                for bag in attachment_bag_vocab.ids_to_words(attachment_bags[:length].cpu().numpy())
            ]

            yield sample.original_object.new(labels, attachments)

    def write_outputs(self, output_path, _, system_samples):
        with open_file(output_path, 'w') as fp:
            EdsSample.to_file(fp, system_samples)
