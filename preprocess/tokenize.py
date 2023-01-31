# -*- coding: utf-8 -*-

import re

from nltk.tokenize.moses import MosesTokenizer

from framework.common.logger import LOGGER

TOKENIZER = MosesTokenizer()


def convert_char_span_to_word_span(source_span, char_spans, words, strict=True):
    source_start, source_end = source_span

    start_index = 0
    for (start, end), word in zip(char_spans, words):
        if end > source_start:
            if start < source_start:
                LOGGER.warning('start of source has an endpoint '
                               'inside one of the char_spans: %s', word)
            break
        start_index += 1

    assert start_index < len(char_spans) or not strict
    end_index = start_index
    for (start, end), word in zip(char_spans[start_index:], words):
        if end > source_end:
            if start < source_end:
                LOGGER.warning('end of source has an endpoint '
                               'inside one of the char_spans: %s', word)
            break
        end_index += 1

    assert end_index > start_index or not strict
    return start_index, end_index - 1


def tokenize_sentence(sentence):
    dash_spliter = r'([-/])', r' \1 '

    tokens = []
    for token in TOKENIZER.tokenize(sentence, escape=False):
        tokens.extend(re.sub(dash_spliter[0], dash_spliter[1], token).split())

    spans = []

    start = 0
    for token in tokens:
        pos = sentence.find(token, start)
        assert pos != -1
        end = pos + len(token)
        spans.append((pos, end))

        start = end

    return tokens, spans
