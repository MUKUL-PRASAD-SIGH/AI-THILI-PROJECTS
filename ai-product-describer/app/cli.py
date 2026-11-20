"""Command-line entry point."""
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ai-product-describer")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that the tool runs")
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Hello from ai-product-describer!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
