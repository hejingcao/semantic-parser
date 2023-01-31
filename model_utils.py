# -*- coding: utf-8 -*-

import pprint
import sys

MODELS = {
    'tok': 'pipeline.tokenizer.Tokenizer',
    'ci': 'pipeline.concept_identifier.ConceptIdentifier',
    'ci2': 'pipeline.concept_identifier_v2.ConceptIdentifier',
    'rd': 'pipeline.relation_detector.MSGParser',
    'pp': 'pipeline.property_predictor.PropertyPredictor'
}


def get_model_class(argv=None):
    if argv is None:
        argv = sys.argv

    model_class = ''
    if len(argv) >= 2:
        model_class = argv.pop(1).lower()

    model_class = MODELS.get(model_class)
    if model_class is None:
        print('Select model first:')
        pprint.pprint(MODELS)
        sys.exit(1)

    return model_class
