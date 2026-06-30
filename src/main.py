# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start, green_quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':

    config = {
        'field_separator': '\t',

        # hyper-params
    }

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', '-m', type=str,
                        default='MMGCF', help='name of models')
    parser.add_argument('--dataset', '-d', type=str,
                        default='movielens_1m', help='name of datasets')

    args, _ = parser.parse_known_args()
    quick_start(model=args.model, dataset=args.dataset,
                config_dict=config, save_model=True)
