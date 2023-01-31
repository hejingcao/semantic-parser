# -*- coding: utf-8 -*-

WORDNET_LEMMATIZER = None


def get_wordnet():
    global WORDNET_LEMMATIZER
    if WORDNET_LEMMATIZER is None:
        from nltk.stem import WordNetLemmatizer

        WORDNET_LEMMATIZER = WordNetLemmatizer()
        try:
            WORDNET_LEMMATIZER.lemmatize('does', 'v')
        except LookupError:
            import nltk
            nltk.download('wordnet')

            assert WORDNET_LEMMATIZER.lemmatize('does', 'v') == 'do'

    return WORDNET_LEMMATIZER


def get_lemma_and_pos(edge_label, get_start_end=False):
    label = edge_label
    lemma_start = label.find('_') + 1
    lemma_end = label.find('_', lemma_start)
    lemma_end_slash = label.rfind('/', lemma_start, lemma_end)
    if lemma_end_slash != -1:
        lemma_end = lemma_end_slash
    old_lemma = label[lemma_start:lemma_end]

    tag_end = label.find('_', lemma_end + 1)
    if tag_end != -1:
        pos = label[lemma_end + 1:tag_end]
    else:
        pos = None

    if get_start_end:
        return old_lemma, pos, lemma_start, lemma_end
    return old_lemma, pos
