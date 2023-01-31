# -*- coding: utf-8 -*-

import re
from collections import Counter, defaultdict

from framework.common.logger import open_file

PUNCTUATION = '.,\'"();?!{}` '
SEP = '!!!'
LEXICON_REs = [re.compile(r'ORTH\s+<\s+((?:"[^"]+",\s+)*"[^"]+")\s+>,'
                          r'\s+SYNSEM.*?CARG\s+("[^"]+")'),
               re.compile(r'ORTH\s+<\s+((?:"[^"]+",\s+)*"[^"]+")\s+>,'
                          r'\s+SYNSEM.*?\n.*?ALTKEYREL.CARG\s+("[^"]+")')]


def _try_list(fn_lst, label, span, sentence):
    from_, to_ = span
    if sentence[to_ - 1] == '-' and sentence[from_:to_] != '--':
        to_ -= 1

    word = sentence[from_:to_]
    for fn in fn_lst:
        result = fn(label, (from_, to_), word, sentence)
        if result is not None:
            break

    if span[1] != to_:
        if result == word and label not in ['named_n']:
            result += '-'

    return result


def read_dictionay(path, label=None):
    dictionary = {}
    for line in open_file(path, 'r'):
        if not line.strip() or line.startswith('#'):
            continue
        key, value = line.split(' - ')
        key = key.strip()
        value = value.strip().replace(' ', '+')
        if label is not None:
            key = label, key
        dictionary[key] = value
    return dictionary


class CARGTransformer:
    def __init__(self,
                 enabled_labels_file,
                 abbrevs_file,
                 label_mappings_file,
                 label_word_mappings_file=None,
                 training=False):
        self.enabled_labels = open_file(enabled_labels_file, 'r').read().split()
        self.abbrevs = read_dictionay(abbrevs_file)
        self.label_mappings = read_dictionay(label_mappings_file)

        if training:
            self.label_word_mappings = {}
            self._trainable_mappings = defaultdict(Counter)
            self._correct_cargs = set()
            self._label_word_mappings_file = label_word_mappings_file
        else:
            self.label_word_mappings = read_dictionay(label_word_mappings_file)

        self.training = training

    def carg_try_mappings(self, label, span, word, sentence):
        carg = self.label_mappings.get(label)
        if carg is not None:
            return carg

        word = word.strip(PUNCTUATION)
        candidates = [label + SEP + word]
        if label in ['ord', 'card']:
            candidates.append(label + SEP + word.lower())

        for candidate in candidates:
            carg = self.label_word_mappings.get(candidate)
            if carg is not None:
                return carg

    def carg_try_escape_quote(self, label, span, word, sentence):
        word = word.strip(PUNCTUATION)
        pos = word.find('\'')
        if pos != -1 and (pos == len(word) - 1 or word[pos + 1] != 's'):
            return word.replace('\'', '’')

    def carg_try_map_numbers(self, label, span, word, sentence):
        if label == 'card':
            try:  # use original word
                value = float(word)
                if value < 1 and value > 0:
                    return '1'
            except Exception:
                pass

    def carg_default(self, label, span, word, sentence):
        puncts = PUNCTUATION
        if re.match(r'([a-zA-Z]+\.){2,}', word):  # NOTE: A.T. => A.T.
            if span[1] != len(sentence):
                puncts = puncts[1:]

        return word.strip(puncts).replace(' & ', ' and ') \
                                 .replace(' ', '+') \
                                 .replace('\'s', 's')

    def _get_carg(self, label, span, sentence, correct_carg=None):
        fn_list = [
            self.carg_try_mappings,
            self.carg_try_map_numbers,
            self.carg_try_escape_quote,
            self.carg_default
        ]

        if not self.training or correct_carg is None:  # predict mode
            return _try_list(fn_list, label, span, sentence)

        carg = _try_list(fn_list[1:], label, span, sentence)

        self._add_mapping(label, sentence[span[0]:span[1]], correct_carg,
                          is_correct=(carg == correct_carg))

        return carg

    def __call__(self, labels, spans, sentence, correct_cargs=None):
        if correct_cargs is None:
            correct_cargs = [None] * len(labels)

        cargs = []
        for label, span, correct_carg in zip(labels, spans, correct_cargs):
            carg = None
            if label in self.enabled_labels:
                carg = self._get_carg(label, span, sentence, correct_carg=correct_carg)

            cargs.append(carg)

        return cargs

    def _add_mapping(self, label, word, carg, is_correct):
        if label in ['ord', 'card']:
            word = word.strip(PUNCTUATION).lower()
        else:
            word = word.strip(PUNCTUATION)
        if is_correct:
            self._correct_cargs.add((label, word, carg))

        self._trainable_mappings[label, word][carg] += 1

    def save(self):
        mappings = self._trainable_mappings

        with open(self._label_word_mappings_file, 'w') as fp:
            for label, word in sorted(mappings):
                counter = mappings[label, word]
                most_common_carg = max(counter, key=lambda x: counter[x])
                if word != most_common_carg \
                   and (label, word, most_common_carg) not in self._correct_cargs:
                    fp.write(f'{label}{SEP}{word} - {most_common_carg}\n')
