#! /usr/bin/env python

from argparse import ArgumentParser
import shutil


def run(args):
    copy = bool(args.copy)

    if copy:
        shutil.copyfile(args.source, args.dest)
    else:
        shutil.move(args.source, args.dest)


if __name__ == "__main__":
    parser = ArgumentParser(description='Moves or copies a file.')

    parser.add_argument('source', help="SOURCE file.",  metavar="SOURCE")
    parser.add_argument('dest', help="DESTination file.",  metavar="DEST")

    parser.add_argument('--copy', type=int, default=0, help="COPY the file",  metavar="COPY")

    args = parser.parse_args()
    run(args)
    exit(0)
