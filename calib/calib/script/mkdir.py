#! /usr/bin/env python

from argparse import ArgumentParser
import os


def run(args):
    if not os.path.exists(args.directory):
        os.mkdir(args.directory, int(args.mode, 8))


if __name__ == "__main__":
    parser = ArgumentParser(description='Create a directory.')

    parser.add_argument('directory', help="Name of the DIRECTORY.",  metavar="DIRECTORY")
    parser.add_argument('mode', help="Permission MODE.",  metavar="MODE")

    args = parser.parse_args()
    run(args)
    exit(0)
