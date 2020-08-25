#!/usr/bin/env python

import argparse
import logging
import sys
from typing import List

from .data import astronaut, coins, create_zarr
from .utils import download as zarr_download
from .utils import info as zarr_info


def config_logging(loglevel: int, args: argparse.Namespace) -> None:
    loglevel = loglevel - (10 * args.verbose) + (10 * args.quiet)
    logging.basicConfig(level=loglevel)
    # DEBUG logging for s3fs so we can track remote calls
    logging.getLogger("s3fs").setLevel(logging.DEBUG)


def info(args: argparse.Namespace) -> None:
    config_logging(logging.INFO, args)
    zarr_info(args.path)


def download(args: argparse.Namespace) -> None:
    config_logging(logging.WARN, args)
    zarr_download(args.path, args.output, args.name)


def create(args: argparse.Namespace) -> None:
    config_logging(logging.INFO, args)
    if args.method == "coins":
        method = coins
    elif args.method == "astronaut":
        method = astronaut
    else:
        raise Exception(f"unknown method: {args.method}")
    create_zarr(args.path, method=method)


def main(args: List[str] = None) -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase loglevel for each use, e.g. -vvv",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=0,
        help="decrease loglevel for each use, e.q. -qqq",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    # info
    parser_info = subparsers.add_parser("info")
    parser_info.add_argument("path")
    parser_info.set_defaults(func=info)

    # download
    parser_download = subparsers.add_parser("download")
    parser_download.add_argument("path")
    parser_download.add_argument("--output", default="")
    parser_download.add_argument("--name", default="")
    parser_download.set_defaults(func=download)

    # coin
    parser_create = subparsers.add_parser("create")
    parser_create.add_argument(
        "--method", choices=("coins", "astronaut"), default="coins"
    )
    parser_create.add_argument("path")
    parser_create.set_defaults(func=create)

    if args is None:
        ns = parser.parse_args(sys.argv[1:])
    else:
        ns = parser.parse_args(args)
    ns.func(ns)
