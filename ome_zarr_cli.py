#!/usr/bin/env python

import argparse
from ome_zarr import info as zarr_info
from ome_zarr import download as zarr_download


def info(args):
    zarr_info(args.path)


def download(args):
    zarr_download(args.path, args.output, args.name)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True

    # foo
    parser_info = subparsers.add_parser('info')
    parser_info.add_argument('path')
    parser_info.set_defaults(func=info)

    # download
    parser_download = subparsers.add_parser('download')
    parser_download.add_argument('path')
    parser_download.add_argument('--output', default='')
    parser_download.add_argument('--name', default='')
    parser_download.set_defaults(func=download)

    args = parser.parse_args()
    args.func(args)
