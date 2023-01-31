# -*- coding: utf-8 -*-

from typing import Any, List

import torch

from framework.common.dataclass_options import argfield
from framework.common.logger import open_file
from framework.common.utils import DotDict
from framework.data.dataset import DELETE, DataSample
from framework.data.sentence import SentenceFeatures, collect_sentence_vocabs
from framework.data.vocab import VocabularySet, lookup_words
from framework.evalute.labeling import LabelingFScorer
from framework.torch_extra.layers.crf import CRF
from framework.torch_extra.layers.sequential import make_mlp_layers
from framework.torch_extra.parser_base import ParserBase, ParserNetworkBase
from framework.torch_extra.utils import cross_entropy_nd, pad_and_stack_1d
from preprocess.tokenize import tokenize_sentence


class EdsSample(DataSample):
    COMMONT_PREFIX = '###COMMENT### '

    words: Any = None
    labels: Any = None

    def __len__(self):
        return len(self.words)

    def to_string(self):
        return ' '.join(self.words)

    def new(self, labels):
        node_index = 0

        nodes = []
        for label in labels:
            current_nodes = []
            for node in label.split(';'):
                if node != 'None':
                    current_nodes.append((str(node_index), node))
                    node_index += 1

            nodes.append(current_nodes)

        return self.copy_attrs(nodes=nodes, edges=DELETE, labels=labels)

    @classmethod
    def from_raw_file(cls, fp):
        samples = []
        for index, line in enumerate(fp):
            tokens, spans = tokenize_sentence(line.strip())
            samples.append(cls(words=tokens, attrs={'token_spans': spans, 'sample_id': str(index)}))
        return samples

    @classmethod
    def from_string(cls, _index, string, attrs):
        labels = [';'.join(sorted(label for _, label in nodes)) if nodes else 'None'
                  for nodes in attrs['nodes']]
        return cls(words=string.split(), labels=labels)

    @classmethod
    def internal_evaluate(cls, gold_samples, system_samples, log_file):
        total_scorer = LabelingFScorer('ALL')

        for gold_sample, system_sample in zip(gold_samples, system_samples):
            if gold_sample.attrs.get('ignore'):  # ignore this example
                continue

            gold_nodes = [[node for _, node in nodes] for nodes in gold_sample.nodes]
            system_nodes = [[node for _, node in nodes] for nodes in system_sample.nodes]
            total_scorer.update_sets(gold_nodes, system_nodes)

        report, f1 = total_scorer.get_report()
        print(f'Total: {f1:.2f}')

        if log_file is not None:
            with open_file(log_file, 'w') as fp:
                fp.write(report)

        return f1


class EdsFeatures(SentenceFeatures):
    labels: Any = None

    @classmethod
    def create(cls, original_index, original_object, plugins, statistics):
        sample = super().create(original_index, original_object, plugins, statistics,
                                lower_case=True)

        if original_object.labels is not None:
            sample.labels = lookup_words(original_object.labels, statistics.get('label'),
                                         sos_and_eos=False)
        return sample

    @classmethod
    def pack_to_batch(cls, batch_samples, plugins, statistics):
        inputs = super().pack_to_batch(batch_samples, plugins, statistics)

        if batch_samples[0].labels is not None:
            inputs.labels = pad_and_stack_1d(
                [torch.from_numpy(sample.labels) for sample in batch_samples],
                pad=-100)

        return inputs


class ConceptIdentifierHyperParams(ParserBase.HyperParams):
    use_crf: bool = True
    mlp_dropout: float = 0.2

    hidden_sizes: List[int] = argfield(default_factory=lambda: [100])

    hidden_layer_norm: bool = False

    activation: str = 'leaky_relu/0.1'


class ConceptIdentifierNetwork(ParserNetworkBase):
    def __init__(self, hyper_params, vocabs, plugins):
        super().__init__(hyper_params, vocabs, plugins)

        self.label_count = len(vocabs.get('label'))

        self.projection = \
            make_mlp_layers(self.encoder.output_size,
                            self.label_count,
                            hidden_sizes=hyper_params.hidden_sizes,
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
        label_logits = self.projection(encoder_outputs)
        outputs = DotDict(label_logits=label_logits)

        if self.training:
            labels = inputs.labels
            if self.crf_unit is not None:
                # padding of labels is -100
                labels = torch.max(labels, torch.zeros_like(labels))
                outputs.loss = -self.crf_unit(label_logits, labels, real_lengths, batch_first=True)
            else:
                outputs.loss = cross_entropy_nd(label_logits, labels)
        else:
            if self.crf_unit is not None:
                outputs.labels = \
                    self.crf_unit.viterbi_decode(label_logits, real_lengths, batch_first=True)
            else:
                outputs.labels = label_logits.argmax(dim=-1)

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

        labels = vocabs.new('label')
        edges = vocabs.new('edge')
        nodes = vocabs.new('node')

        for train_path in self.options.train_paths:
            buckets = self.create_buckets(train_path, 'train')

            collect_sentence_vocabs(buckets.original_objects, vocabs, sos_and_eos=True)

            for sample in buckets.original_objects:
                labels.update(sample.labels)

                edges.update(edge_label for _, _, edge_label in sample.edges)
                nodes.update(node_label
                             for nodes_info in sample.nodes
                             for _, node_label in nodes_info)

        return vocabs

    def split_outputs(self, batch_samples, outputs):
        label_vocab = self.statistics.get('label')

        for sample, labels in zip(batch_samples, outputs.labels):
            length = len(sample.original_object)
            labels = label_vocab.ids_to_words(labels[:length].cpu().numpy())

            yield sample.original_object.new(labels)

    def write_outputs(self, output_path, _, system_samples):
        with open_file(output_path, 'w') as fp:
            EdsSample.to_file(fp, system_samples)
