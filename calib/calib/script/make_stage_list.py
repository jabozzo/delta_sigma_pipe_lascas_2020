#! /usr/bin/env python

from argparse import ArgumentParser

import calib.gen as gen
import calib.data as data


def run(args):
    path_context = data.PathContext.relative()

    memo = {}
    stages = []

    for stage_name in args.stages:
        location = data.DataLocation(path_context, args.location, stage_name, "json")
        print(" loading {}...".format(location.computed_path))
        stages.append(data.load(gen.StageParameters, location, memo))

    stage_list = data.list_of(gen.StageParameters)(stages * args.repeat)

    list_location = data.DataLocation(path_context, args.location, args.dest, "json")
    print(" saving {}...".format(list_location.computed_path))
    data.save(stage_list, list_location)


if __name__ == "__main__":
    parser = ArgumentParser(description='Create an stage list appending stages files.')

    parser.add_argument('location', help= "Path to the output LOCATION.",  metavar="LOCATION")
    parser.add_argument('dest', help= "Name of the DESTination file (without extension).",  metavar="DEST")
    parser.add_argument('stages', nargs='+', default=[], help= "STAGES files to append to the list (without extension).",  metavar="STAGES")

    parser.add_argument('--repeat', type=int, default=1, help= "Times to REPEAT the list.",  metavar="REPEAT")

    args = parser.parse_args()
    run(args)
    exit(0)
