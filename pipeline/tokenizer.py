# -*- coding: utf-8 -*-

import math
import os
import re
import subprocess as sp
import tempfile
from typing import Any, List, Optional

import torch
import torch.nn as nn

from framework.common.logger import open_file
from framework.common.utils import DotDict
from framework.data.dataset import DataFeatures, DataSample
from framework.data.vocab import VocabularySet, lookup_words
from framework.torch_extra.parser_base import ParserBase, ParserNetworkBase
from framework.torch_extra.utils import cross_entropy_nd, pad_and_stack_1d

__ASSETS__ = ['bin/tokenizer_score.perl']


class BISample(DataSample):
    sample_id: str = ''
    words: Optional[List[str]] = None
    tags: Optional[List[str]] = None

    def __len__(self):
        return len(self.words)

    @classmethod
    def from_string(cls, index, string, _attrs):
        sample_id = index
        string = string.strip()

        parts = list(map(str.strip, string.split('|||')))
        if len(parts) == 1:
            sample_id = str(index)
            words = parts[0].split()
            tags = None
        else:
            sample_id = parts[0]
            words = parts[1].split()
            if len(parts) > 2:
                tags = parts[2].split()
                assert len(words) == len(tags)

        return cls(sample_id=sample_id, words=words, tags=tags)

    def retrieve_tokens(self, tags=None):
        if tags is None:
            tags = self.tags

        word_buffer = []
        tokens = []
        for word, tag in zip(self.words, tags):
            if tag == 'B' and word_buffer:
                tokens.append('_'.join(word_buffer))
                word_buffer.clear()
            word_buffer.append(word)

        if word_buffer:
            tokens.append('_'.join(word_buffer))
        return tokens

    @classmethod
    def extract_performance(cls, output):
        results = {}
        performance_pattern = re.compile(r'^=== (.+?):\t([\d.]+)', re.MULTILINE)
        for k, v in performance_pattern.findall(output):
            v = float(v)
            results[k] = (-1 if math.isnan(v) else v)

        return results

    @classmethod
    def external_evaluate(cls, gold_file, system_file, log_file):
        script_file = os.path.join(os.path.dirname(__file__), __ASSETS__[0])

        tmp_dict_file = tempfile.NamedTemporaryFile()
        tmp_gold_file = tempfile.NamedTemporaryFile()
        proc = None
        try:
            cls.to_file(tmp_gold_file.name, cls.from_file(gold_file))

            args = [script_file, tmp_dict_file.name, tmp_gold_file.name, system_file]

            proc = sp.Popen(args=args,
                            universal_newlines=True,
                            stdout=sp.PIPE, stderr=sp.PIPE)
            output, error = proc.communicate()

            with open(log_file, 'w') as fp:
                fp.write(output)

            return cls.extract_performance(output).get('F MEASURE', 0.0) * 100
        finally:
            if proc is not None:
                proc.kill()
            tmp_dict_file.close()
            tmp_gold_file.close()

        return -1

    def to_string(self):
        return ' '.join(self.retrieve_tokens())


class BIFeatures(DataFeatures):
    words: Any = None
    tags: Any = None
    length: int = -1

    @classmethod
    def create(cls, original_index, original_object, plugins, statistics):
        sample = cls(original_index=original_index, original_object=original_object)

        sample.length = len(original_object.words)
        sample.words = lookup_words(original_object.words, statistics.get('word'),
                                    sos_and_eos=False)
        if original_object.tags:
            sample.tags = lookup_words(original_object.tags, statistics.get('tag'),
                                       sos_and_eos=False,
                                       default_id=100)

        cls.run_plugins_for_sample(sample, plugins, sos_and_eos=False)

        return sample

    @classmethod
    def pack_to_batch(cls, batch_samples, plugins, statistics):
        inputs = DotDict(
            words=pad_and_stack_1d([torch.from_numpy(sample.words) for sample in batch_samples]),
            encoder_lengths=torch.tensor([sample.length for sample in batch_samples])
        )

        if batch_samples[0].tags is not None:
            inputs.tags = \
                pad_and_stack_1d([torch.from_numpy(sample.tags) for sample in batch_samples],
                                 pad=-100)

        cls.run_plugins_for_batch(batch_samples, inputs, plugins)
        return inputs


class TokenizerNetwork(ParserNetworkBase):
    def __init__(self, hyper_params, vocabs, plugins):
        super().__init__(hyper_params, vocabs, plugins)

        self.projection = nn.Linear(self.encoder.output_size, len(vocabs.get('tag')))

    def forward(self, batch_samples, inputs):
        input_embeddings = self.input_embeddings(inputs)
        encoder_outputs, _ = self.encoder(input_embeddings, inputs.encoder_lengths)

        logits = self.projection(encoder_outputs)
        pred_tags = logits.argmax(dim=-1)

        outputs = DotDict(tags=pred_tags)
        if self.training:
            outputs.loss = cross_entropy_nd(logits, inputs.tags)

        return outputs


class Tokenizer(ParserBase):
    METRICS_MODE = 'max'
    METRICS_NAME = 'F1'

    FEATURES_CLASS = BIFeatures
    SAMPLE_CLASS = BISample
    NETWORK_CLASS = TokenizerNetwork

    class Options(ParserBase.Options):
        hyper_params: ParserBase.HyperParams

    def build_vocabs(self):
        options = self.options
        vocabs = VocabularySet()

        word_vocab = vocabs.new('word')
        tag_vocab = vocabs.new('tag', initial=())
        for train_path in options.train_paths:
            buckets = self.create_buckets(train_path, 'train')
            for sample in buckets.original_objects:
                word_vocab.update(sample.words)
                tag_vocab.update(sample.tags)

        return vocabs

    def split_outputs(self, batch_samples, outputs):
        tag_vocab = self.statistics.get('tag')
        for sample, tags in zip(batch_samples, outputs.tags.cpu().numpy()):
            yield tag_vocab.ids_to_words(tags[:len(sample.original_object)])

    def write_outputs(self, output_path, samples, outputs):
        with open_file(output_path, 'w') as fp:
            for sample, tags in zip(samples, outputs):
                fp.write(' '.join(sample.original_object.retrieve_tokens(tags)) + '\n')
