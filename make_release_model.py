# -*- coding: utf-8 -*-

import torch

from framework.common.dataclass_options import ExistFile, OptionsBase, SingleOptionsParser
from framework.common.logger import LOGGER


class Options(OptionsBase):
    input_path: ExistFile
    output_path: ExistFile


if __name__ == '__main__':
    parser = SingleOptionsParser()
    parser.setup(Options, abbrevs={'input_path': 'input_path', 'output_path': 'output_path'})
    options = parser.parse_args()

    LOGGER.info('load < %s', options.input_path)
    saved_state = torch.load(options.input_path)
    LOGGER.info('save > %s', options.output_path)
    torch.save(
        {
            'options': saved_state['options'],
            'object': {key: value for key, value in saved_state['object'].items()
                       if key not in ['scheduler', 'optimizer']}
        },
        options.output_path
    )
