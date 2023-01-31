# -*- coding: utf-8 -*-

import glob
import gzip
import multiprocessing as mp
import os
from abc import ABCMeta, abstractclassmethod

from framework.common.dataclass_options import OptionsBase, argfield
from framework.common.logger import LOGGER
from framework.common.utils import ProgressReporter

from .graph_io import READERS

# TODO: modify shrg_extract.py to use ReaderBase


class ReaderBase(metaclass=ABCMeta):
    class Options(OptionsBase):
        graph_type: str = argfield(default='eds')

    def __init__(self, options: Options, data_path, split_patterns, logger=LOGGER, **extra_args):
        self.logger = logger
        self.options = options

        self.extra_args = extra_args

        self._data = {}
        self._splits = {}

        for split, pattern in split_patterns:
            if isinstance(pattern, str):
                pattern = [pattern]
            dirs = sum((glob.glob(os.path.join(data_path, p)) for p in pattern), [])
            files = [glob.glob(os.path.join(dir, '*.gz')) for dir in dirs]

            logger.info('SPLIT %s: %d directories, %d files',
                        split, len(dirs), sum(map(len, files)))
            self._splits[split] = files

    def on_error(self, filename, error):
        pass

    def get_split(self, split, num_workers=-1):
        def _iter_results():
            if num_workers == 1:
                yield from map(self._worker, all_options)
            else:
                pool = None
                try:
                    pool = mp.Pool(num_workers)
                    yield from pool.imap_unordered(self._worker, all_options)
                finally:
                    if pool is not None:
                        pool.terminate()

        if num_workers == -1:
            num_workers = max(8, mp.cpu_count())

        training = (split == 'train')

        data = self._data.get(split)
        if data is None:
            all_options = [(files, self.options, training, self.extra_args)
                           for files in self._splits[split]]
            data = {}
            progress = ProgressReporter(len(all_options), step=1)
            for outputs in progress(_iter_results()):
                for is_ok, filename, output in outputs:
                    if not is_ok:
                        self.on_error(filename, output)
                        continue

                    sample_id = os.path.basename(filename).split('.')[0]
                    data[sample_id] = output

            self._data[split] = data
        return data

    @abstractclassmethod
    def build_graph(cls, reader_output, filename, options, training):
        pass

    @classmethod
    def _worker(cls, args):
        files, options, training, extra_args = args
        read_fn = READERS.normalize(options.graph_type)

        outputs = []
        for filename in files:
            with gzip.open(filename, 'rb') as fp:
                try:
                    fields = fp.read().decode().strip().split('\n\n')
                    reader_output = read_fn(fields, options)
                    output = cls.build_graph(reader_output, filename, options, training, extra_args)
                    outputs.append((True, filename, output))
                except Exception as err:
                    LOGGER.exception('%s', filename)
                    outputs.append((False, filename, err))

        return outputs
