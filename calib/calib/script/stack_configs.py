#! /usr/bin/env python

from argparse import ArgumentParser

import calib.gen as gen
import calib.data as data


def run(args):
    path_context = data.PathContext.relative()

    configurations = []
    for conf in args.configurations:
        conf_location = data.DataLocation(path_context, args.location, conf, "json")
        print(" saving {}...".format(conf_location.computed_path))
        configurations.append(data.load(gen.ConfigurationSequence, conf_location))

    stacked = gen.ConfigurationSequence.Stack(*configurations)

    stacked_location = data.DataLocation(path_context, args.location, args.dest, "json")
    print(" saving {}...".format(stacked_location.computed_path))
    data.save(stacked, stacked_location)


if __name__ == "__main__":
    parser = ArgumentParser(description='Create a configuration sequence.')

    parser.add_argument('location', help="Path to the input/output LOCATION.",  metavar="LOCATION")
    parser.add_argument('dest', help="Name of the DESTination file (without extension).",  metavar="DEST")
    parser.add_argument('configurations', nargs="+", help="CONFiguration files (without extension).",  metavar="CONF")

    args = parser.parse_args()

    run(args)
    exit(0)
