# -*- coding: utf-8 -*-

from framework.torch_extra.predict_session import PredictSession
from model_utils import get_model_class


def predict(model_bytes=None, argv=None, entry_class=None):
    abbrevs = {'test_paths': '-i', 'output_prefix': '-p'}
    PredictSession.entry_point(entry_class, model_bytes=model_bytes,
                               argv=argv,
                               abbrevs=abbrevs,
                               entry_point='predict')


if __name__ == '__main__':
    predict(entry_class=get_model_class())
