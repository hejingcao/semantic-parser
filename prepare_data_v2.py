# -*- coding: utf-8 -*-

import os

from preprocess.data_reader import DeepBankReader, MRPReader, ReaderOptions
from preprocess.data_writer import Writer

# from SHRG.graph_transformers import TRANSFORMS

# TODO attach udef_q
# TODO anonymize_label problem: shipsets
# TODO recovered_label f1


# TRANSFORMS.register('attach-quantifier', fn=attach_quantifier)

if __name__ == '__main__':
    options = ReaderOptions()
    options.graph_transformers = [
        'remove-isolated',
        # 'attach-quantifier'
    ]

    reader = DeepBankReader(options, '../../data/deepbank1.1/export',
                            [('dev', 'wsj20*'), ('train', ['wsj1*', 'wsj0*'])])
    # reader = MRPReader(options, 'data/mrp2019/training/eds/wsj.mrp',
    #                    [('dev', {os.path.basename(filename).strip('.gz')
    #                              for files in reader._splits['dev']
    #                              for filename in files}),
    #                     ('train', True)])
    writer = Writer()

    dev_data = reader.get_split('dev', num_workers=4)
    train_data = reader.get_split('train', num_workers=4)
    writer.to_file(dev_data, 'data/deepbank1.1.v2.dev',
                   write_smatch=True, training=False)
    writer.to_file(train_data,
                   'data/deepbank1.1.v2.train',
                   'data/deepbank1.1.lemma_mappings.v2.txt',
                   training=True)

    # train_data = reader.get_split('train')
    # dev_data = reader.get_split('dev')
    # writer.to_file(dev_data, 'data/mrp2019.v2.dev', write_smatch=True)
    # writer.to_file(train_data, 'data/mrp2019.v2.train',
    #                'data/mrp2019.lemma_mappings.v2.txt')
