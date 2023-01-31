# -*- coding: utf-8 -*-

from collections import Counter

from framework.common.logger import open_file
from SHRG.utils.lexicon import get_lemma_and_pos, get_wordnet


def read_lemma_mappings_file(path):
    mappings = {}

    for line in open_file(path, 'r'):
        key, value = line.split(':::')
        key = tuple(key.strip().split('\t'))
        mappings[key] = counter = Counter()
        for item in value.strip().split('\t'):
            label, count = item.rsplit('@@', 1)
            counter[label] = int(count)

    return mappings


def recover_edge_label(label, word, lemma_dictionary):
    _, pos, lemma_start, lemma_end = get_lemma_and_pos(label, True)
    lemma = None
    lemmatizer = get_wordnet()
    if pos in ('n', 'v', 'a'):
        for try_word in [word, word.lower()]:  # NOTE: additional try
            lemmas = lemma_dictionary.get((lemmatizer.lemmatize(try_word, pos), label))
            if lemmas is not None:
                break
    else:
        lemmas = lemma_dictionary.get((word, label))

    if lemmas is not None:
        lemma = sorted(lemmas.items(), key=lambda x: x[1], reverse=True)[0][0]

    if lemma is None:
        lemma = word.lower()  # NOTE: lowr label

    new_label = label.replace('{NEWLEMMA}', lemma)
    return new_label
