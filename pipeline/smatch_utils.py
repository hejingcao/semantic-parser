# -*- coding: utf-8 -*-

import os
import re
import subprocess as sp
import tempfile

from framework.common.logger import LOGGER, smart_open_file

__ASSETS__ = ['bin/smatch_multi', 'bin/smatch.hpp', 'bin/smatch_utils.cpp', 'bin/smatch_multi.cpp']


def read_smatch_graphs(path_or_fp):
    with smart_open_file(path_or_fp, 'r') as fp:
        for block in fp.read().split('\n\n'):
            block = block.strip()
            if not block:
                continue

            lines = block.split('\n')
            assert lines[0].startswith('#')
            sentence_id = lines[0][1:].strip()

            node_count = int(lines[1])
            edge_count = int(lines[node_count + 2])

            assert len(lines) == node_count + edge_count + 3
            offset = 2
            nodes = []
            spans = []

            for index, line in enumerate(lines[offset:offset + node_count]):
                try:
                    node_id, node_label = line.strip().split()
                except Exception:
                    node_id = node_label = '???'
                    LOGGER.warning('Empty node !!! %s:%d', sentence_id, index)

                node_id, *extra_info = node_id.split('@')
                spans.append(tuple(map(int, extra_info[0].split(','))) if extra_info else (-1, -1))
                nodes.append((node_id, node_label))

            offset += node_count + 1
            edges = []
            for line in lines[offset:offset + edge_count]:
                source_id, target_id, edge_label = line.strip().split()

                source_id = source_id.split('@')[0]
                target_id = target_id.split('@')[0]
                edges.append((source_id, target_id, edge_label))

            yield {'nodes': nodes, 'edges': edges, 'sample_id': sentence_id, 'spans': spans}


def run_smatch(gold_path, system_path, cleanup=False):
    script_file = os.path.join(os.path.dirname(__file__), __ASSETS__[0])

    proc = None
    try:
        if cleanup:
            tmp_file = tempfile.NamedTemporaryFile()
            output_files = [tmp_file.name] * 2
        else:
            output_files = [system_path + '.score',
                            system_path + '.nodemapoutput']

        args = [script_file, system_path, gold_path] + output_files
        LOGGER.info("%s", args)

        score = 0.0
        proc = sp.Popen(args=args, universal_newlines=True, stdout=sp.PIPE)
        for line in iter(proc.stdout.readline, ''):
            print('>>>', line.rstrip(),
                  end=('\r' if re.match(r'[0-9]+\n', line) else '\n'))
            match = re.findall(r'Average SMATCH f=(.*)', line)
            if match:
                score = float(match[-1])

        if cleanup:
            return score

        return score, output_files
    finally:
        if proc is not None:
            proc.kill()
