#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils import ProcessBigDataFrame, add_deepcoy_properties


def main(args):
    
    if args.command == 'feature':
        if args.type == 'deepcoy':
            print('Add DeepCoy feature.')
            ProcessBigDataFrame(args.infile, add_deepcoy_properties, 
                                args.outfile)

if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    
    feature_parser = subparser.add_parser("feature")
    feature_parser.add_argument('type')
    feature_parser.add_argument('infile')
    feature_parser.add_argument('outfile')
    
    args = parser.parse_args()
    
    main(args)