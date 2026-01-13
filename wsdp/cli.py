import argparse

from .core import pipeline


def _run_pipeline(args):
    pipeline(input_path=args.input_path, output_folder=args.output_folder, dataset=args.dataset)


def main_cli():
    parser = argparse.ArgumentParser(description="wsdp CLI")
    subparser = parser.add_subparsers(dest="command", required=True, help="available commands")

    parser_run = subparser.add_parser("run", help="run pipeline")
    parser_run.add_argument("input_path", type=str, help="input data path")
    parser_run.add_argument("output_folder", type=str, help="output path")
    parser_run.add_argument("dataset", type=str, help="dataset name")
    parser_run.set_defaults(func=_run_pipeline)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()