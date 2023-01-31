# -*- coding: utf-8 -*-

import sys

from framework.torch_extra.utils import import_class
from model_utils import get_model_class

# sys.argv[1] = gold
# sys.argv[2] = system
model_class = get_model_class()
import_class(model_class).SAMPLE_CLASS.external_evaluate(sys.argv[1], sys.argv[2], None)
