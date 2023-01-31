# -*- coding: utf-8 -*-

import gzip
import os
import pickle
from collections import Counter, OrderedDict
from multiprocessing import Pool, cpu_count
from typing import List, Optional

from framework.common.dataclass_options import OptionsBase, argfield
from framework.common.logger import LOGGER, open_wrapper

from .const_tree import ConstTree
from .const_tree_preprocess import modify_const_tree
from .graph_io import READERS, TEXT_WRITRES
from .graph_transformers import TRANSFORMS
from .shrg import LEXICALIZE_NULL_SEMANTIC_OPTIONS, extract_shrg_rule
from .utils.container import IndexedCounter


class IgnoreException(Exception):
    pass


class ExtractionOptions(OptionsBase):
    graph_type: str = argfield(default='eds')
    detect_function: str = argfield(default='small')

    modify_tree: str = argfield(default='no-suffix')
    modify_label: List[str] = argfield(default_factory=list,
                                       choices=['strip_d', 'strip-hndl', 'strip-lemma'],
                                       nargs='+')

    graph_transformers: List[str] = argfield(default_factory=list)
    remove_disconnected: bool = False

    lexicalize_null_semantic: Optional[List[str]] = \
        argfield(default=None,
                 choices=LEXICALIZE_NULL_SEMANTIC_OPTIONS,
                 nargs='*')

    fix_hyphen: str = argfield(default='none')


class MainOptions(OptionsBase):
    grammar_name: str

    prefix: str
    java_code_data_prefix: str
    deepbank_data_dir: str

    extraction: ExtractionOptions

    debug: bool = False


def compute_prefix_for_options(prefix, grammar_name, options: ExtractionOptions):
    suffix = '.{}.{}'.format(options.detect_function, options.modify_tree)
    if options.modify_label:
        suffix += '.' + '+'.join(sorted(options.modify_label))

    if options.remove_disconnected:
        suffix += '.conn'

    if options.lexicalize_null_semantic is not None:
        suffix += '.condensed(' + '+'.join(options.lexicalize_null_semantic) + ')'

    if options.graph_transformers:
        suffix += '.trans(' + '+'.join(options.graph_transformers) + ')'

    return prefix.format(grammar=grammar_name, graph_type=options.graph_type, suffix=suffix)


def generate_const_tree_and_hyper_graph(tree_string, deepbank_data,
                                        extraction_options: ExtractionOptions):
    const_tree = ConstTree.from_java_code_and_deepbank_1_1(tree_string, deepbank_data)[0]
    const_tree = modify_const_tree(const_tree,
                                   extraction_options.modify_tree,
                                   extraction_options.fix_hyphen)

    fields = deepbank_data.strip().split('\n\n')
    hyper_graph, eds_graph = READERS.invoke(extraction_options.graph_type, fields,
                                            extraction_options)

    for transformer in extraction_options.graph_transformers:
        hyper_graph = TRANSFORMS.invoke(transformer, hyper_graph)

    lexicalize_options = extraction_options.lexicalize_null_semantic
    ignore_punct = lexicalize_options and 'ignore_punct' in lexicalize_options

    eds_graph.lemma_sequence = \
        ' '.join(x.string for x in const_tree.generate_lexicons(ignore_punct))

    return const_tree, (hyper_graph, eds_graph)


def _extract_worker(args):
    java_out_dir, deepbank_export_path, bank, is_training, extraction_options = args
    results = []

    with open(os.path.join(java_out_dir, bank)) as fin:
        while True:
            sentence_id = fin.readline().strip()
            if not sentence_id:
                break
            assert sentence_id.startswith('#')
            sentence_id = sentence_id[1:]
            tree_string = fin.readline().strip()
            sentence_path = os.path.join(deepbank_export_path, bank, sentence_id + '.gz')
            try:
                with gzip.open(sentence_path, 'rb') as fin_bank:
                    contents = fin_bank.read().decode()
                const_tree, (hyper_graph, eds_graph) = \
                    generate_const_tree_and_hyper_graph(tree_string, contents, extraction_options)

                if extraction_options.remove_disconnected and not hyper_graph.is_connected():
                    LOGGER.debug('%s disconnected', sentence_id)
                    continue

                shrg_rules = nodes_info = edges_info = None
                if is_training:
                    shrg_rules, (_, edge_blame_dict), boundary_node_dict = \
                        extract_shrg_rule(
                            hyper_graph, const_tree,
                            detect_function=extraction_options.detect_function,
                            lexicalize_null_semantic=extraction_options.lexicalize_null_semantic,
                            graph_type=extraction_options.graph_type,
                            sentence_id=sentence_id)
                    nodes_info = [None] * len(shrg_rules)
                    edges_info = [set() for _ in range(len(shrg_rules))]
                    for edge, step in edge_blame_dict.items():
                        assert edge.is_terminal, f'{edge} is not a terminal ???'
                        edges_info[step].add(edge.to_tuple())
                    for step, nodes in boundary_node_dict.items():
                        nodes_info[step] = tuple(_.name for _ in nodes)
                results.append((sentence_id, shrg_rules, (edges_info, nodes_info),
                                const_tree, eds_graph))
            except IgnoreException as err:
                LOGGER.debug('%s %s', err, sentence_id)
            except Exception:
                LOGGER.exception("%s", sentence_id)
    LOGGER.info('Finish %d valid graphs (%s)', len(results), bank)
    return bank, results


def save_rules(output_prefix, rules_counter, params, extra_suffix=''):
    _open = open_wrapper(lambda x: output_prefix + x + extra_suffix)

    hrg2cfg_mapping = OrderedDict()
    head_counter = Counter()
    for index, (shrg_rule, counter_item) in enumerate(rules_counter):
        hrg = shrg_rule.hrg
        if hrg is None:
            hrg = shrg_rule.cfg.lhs
            head_counter[hrg] += counter_item.count
        else:
            head_counter[hrg.lhs.unique_label] += counter_item.count
        hrg2cfg_mapping.setdefault(hrg, []).append((index, shrg_rule.cfg, counter_item.count))

    LOGGER.info('All hrg rules: %s', len(hrg2cfg_mapping))
    LOGGER.info('All rules: %s', len(rules_counter))

    with _open('.mapping.txt', 'w') as mapping_out:
        TEXT_WRITRES.invoke('mapping', mapping_out, hrg2cfg_mapping, head_counter,
                            write_detail=False)
    del hrg2cfg_mapping
    del head_counter

    with _open('.rules.detail.txt', 'w') as rule_detail_out:
        TEXT_WRITRES.invoke('shrg', rule_detail_out, rules_counter, write_detail=True)

    with _open('.counter.p', 'wb') as out:
        pickle.dump((rules_counter, params), out)


def _extract_writer(all_results, output_prefix, is_trianing, params):
    _open = open_wrapper(lambda x: output_prefix + x)

    rules_counter = IndexedCounter(5)
    derivations = {}

    tree_writer = TEXT_WRITRES.normalize('tree')
    eds_writer = TEXT_WRITRES.normalize('eds')

    modify_label = params.modify_label

    total_count = sum(len(results) for _, results in all_results)
    with _open('.graphs.txt', 'w') as graph_out, _open('.trees.txt', 'w') as tree_out:
        graph_out.write(str(total_count) + '\n')
        tree_out.write(str(total_count) + '\n')
        for bank, results in all_results:
            for sentence_id, shrg_rules, (edges_info, nodes_info), const_tree, eds_graph in results:
                if is_trianing:
                    shrg_rules = [
                        rules_counter.add(shrg_rule, (sentence_id, step))
                        for step, shrg_rule in enumerate(shrg_rules)
                    ]
                if is_trianing:
                    derivation = []
                    assert len(shrg_rules) == len(edges_info), 'Strange ???'
                    for node_index, (rule_index, node, *info) in \
                        enumerate(zip(shrg_rules,
                                      const_tree.traverse_postorder(),
                                      edges_info, nodes_info)):
                        children = filter(lambda x: isinstance(x, ConstTree),
                                          getattr(node, 'children', []))
                        derivation.append((rule_index, *info, *(child.index for child in children)))
                else:
                    derivation = shrg_rules
                derivations[sentence_id] = derivation
                sentence_id = bank + os.path.sep + sentence_id
                eds_writer(graph_out, sentence_id, eds_graph, modify_label)
                tree_writer(tree_out, sentence_id, const_tree, shrg_rules)

    LOGGER.info('All graphs: %s', len(derivations))

    if not is_trianing:
        return

    with _open('.derivations.p', 'wb') as out:
        pickle.dump(derivations, out)
    del derivations

    save_rules(output_prefix, rules_counter, params)


def extract_shrg_from_dataset(java_out_dir, deepbank_export_path, output_prefix, extraction_options,
                              num_processes=-1, is_training=False):

    if num_processes == -1:
        num_processes = min(cpu_count(), 8)
    banks = [bank for bank in os.listdir(java_out_dir)
             if bank.startswith('wsj')
             and os.path.exists(os.path.join(deepbank_export_path, bank))]

    LOGGER.info('use %d processes', num_processes)
    LOGGER.info('begin to extract rules ...')

    pool = Pool(processes=num_processes)
    all_options = [(java_out_dir, deepbank_export_path, bank, is_training, extraction_options)
                   for bank in banks]
    try:
        all_results = sorted(pool.imap_unordered(_extract_worker, all_options), key=lambda x: x[0])
    finally:
        pool.terminate()

    LOGGER.info('begin to write results ...')
    _extract_writer(all_results, output_prefix, is_training, extraction_options)
