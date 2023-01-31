# -*- coding: utf-8 -*-


def is_empty(span):
    return span[0] >= span[1]


def is_disjoint(span1, span2):
    return span1[0] >= span2[1] or span1[1] < span2[0]


def is_overlapped(span1, span2):
    return span1[0] < span2[1] and span1[1] >= span2[0]


def contains(span1, span2):
    return span1[0] <= span2[0] <= span2[1] <= span1[1]


def is_nested_spans(spans):
    outer = min(s[0] for s in spans), max(s[1] for s in spans)
    if outer in spans:
        return True
