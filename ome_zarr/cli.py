"""Entrypoint for the `ome_zarr` command-line tool."""
import argparse
import logging
import sys
from typing import List

from .data import astronaut, coins, create_zarr
from .scale import Scaler
from .utils import download as zarr_download
from .utils import info as zarr_info


def config_logging(loglevel: int, args: argparse.Namespace) -> None:
    """Configure logging taking the `verbose` and `quiet` arguments into account.

    Each `-v` increases the `loglevel` by 10 and each `-q` reduces the loglevel by 10.
    For example, an initial loglevel of `INFO` will be converted to `DEBUG` via `-qqv`.
    """
    loglevel = loglevel - (10 * args.verbose) + (10 * args.quiet)
    logging.basicConfig(level=loglevel)
    # DEBUG logging for s3fs so we can track remote calls
    logging.getLogger("s3fs").setLevel(logging.DEBUG)


def info(args: argparse.Namespace) -> None:
    """Wrap the :func:`~ome_zarr.utils.info` method."""
    config_logging(logging.WARN, args)
    list(zarr_info(args.path))


def download(args: argparse.Namespace) -> None:
    """Wrap the :func:`~ome_zarr.utils.download` method."""
    config_logging(logging.WARN, args)
    zarr_download(args.path, args.output)


def create(args: argparse.Namespace) -> None:
    """Chooses between data generation methods in :module:`ome_zarr.utils` like.

    :func:`~ome_zarr.data.coins` or :func:`~ome_zarr.data.astronaut`.
    """
    config_logging(logging.WARN, args)
    if args.method == "coins":
        method = coins
        label_name = "coins"
    elif args.method == "astronaut":
        method = astronaut
        label_name = "circles"
    else:
        raise Exception(f"unknown method: {args.method}")
    create_zarr(args.path, method=method, label_name=label_name)


def scale(args: argparse.Namespace) -> None:
    """Wrap the :func:`~ome_zarr.scale.Scaler.scale` method."""
    scaler = Scaler(
        copy_metadata=args.copy_metadata,
        downscale=args.downscale,
        in_place=args.in_place,
        labeled=args.labeled,
        max_layer=args.max_layer,
        method=args.method,
    )
    scaler.scale(args.input_array, args.output_directory)


def main(args: List[str] = None) -> None:
    """Run appropriate function with argparse arguments, handling errors."""
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
    parser_download.add_argument("--output", default=".")
    parser_download.set_defaults(func=download)

    # create
    parser_create = subparsers.add_parser("create")
    parser_create.add_argument(
        "--method", choices=("coins", "astronaut"), default="coins"
    )
    parser_create.add_argument("path")
    parser_create.set_defaults(func=create)

    parser_scale = subparsers.add_parser("scale")
    parser_scale.add_argument("input_array")
    parser_scale.add_argument("output_directory")
    parser_scale.add_argument(
        "--labeled",
        action="store_true",
        help="assert that the list of unique pixel values doesn't change",
    )
    parser_scale.add_argument(
        "--copy-metadata",
        action="store_true",
        help="copies the array metadata to the new group",
    )
    parser_scale.add_argument(
        "--method", choices=list(Scaler.methods()), default="nearest"
    )
    parser_scale.add_argument(
        "--in-place", action="store_true", help="if true, don't write the base array"
    )
    parser_scale.add_argument("--downscale", type=int, default=2)
    parser_scale.add_argument("--max_layer", type=int, default=4)
    parser_scale.set_defaults(func=scale)

    ns = parser.parse_args(args)

    if args is None:
        ns = parser.parse_args(sys.argv[1:])
    else:
        ns = parser.parse_args(args)

    try:
        ns.func(ns)
    except AssertionError as error:
        logging.getLogger("ome_zarr.cli").error(error)
        sys.exit(2)
