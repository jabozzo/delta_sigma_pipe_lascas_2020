#! /usr/bin/env python

from argparse import ArgumentParser

import calib.gen as gen
import calib.data as data

from calib.misc import json_args, push_random_state

import numpy as np


def create_single(args, metas, modify):
    stages = [meta.generate_gaussian(args.seff, args.scap, args.srefs, args.sthres, args.scm)
        for meta in metas]
    tail = np.random.normal(gen.compute_thres(args.tailbits, *stages[-1].meta.fsr), args.sthres)
    adc = gen.PipeParameters(stages, tail)
    adc = adc.create_modified(modify)
    return adc


def run(args):
    modify = json_args(args.modify)
    path_context = data.PathContext.relative()

    memo = {}
    metas = []

    for meta in args.meta:
        meta_location = data.DataLocation(path_context, args.location, meta, "json")
        print(" loading {}...".format(meta_location.computed_path))
        meta = data.load(gen.StageMeta, meta_location, memo)
        metas.append(meta)

    assert args.n >= 0

    if args.seed is not None:
        np.random.seed(int(args.seed))

    if args.n == 0:
        adcs = create_single(args, metas, modify)
        ideals = adcs.as_ideal()

    else:
        adcs = [create_single(args, metas, modify) for _ in range(args.n)]
        adcs = data.list_of(gen.PipeParameters)(adcs)
        ideals = [adc.as_ideal() for adc in adcs]
        ideals = data.list_of(gen.PipeParameters)(ideals)

    adc_location = data.DataLocation(path_context, args.location, args.dest, "json")
    print(" saving {}...".format(adc_location.computed_path))
    data.save(adcs, adc_location)

    ideal_location = data.DataLocation(path_context, args.location, args.dest + ".ideal", "json")
    print(" saving {}...".format(ideal_location.computed_path))
    data.save(ideals, ideal_location)

if __name__ == "__main__":
    parser = ArgumentParser(description='Create an adc.')
    parser.add_argument('location', help= "Path to the output LOCATION.",  metavar="LOCATION")
    parser.add_argument('tailbits', help= "TAIL Bits.",  metavar="TAIL_B")
    parser.add_argument('dest', help= "Name of the DESTination file (without extension).",  metavar="DEST")

    parser.add_argument('--meta', nargs='+', default=[], help= "Stage meta SOURCE.",  metavar="SOURCE")

    parser.add_argument('--scm', type=float, default=0, help= "Standard deviation of Common Mode. In volts.",  metavar="SCM")
    parser.add_argument('--seff', type=float, default=0, help= "Standard deviation of charge transfer Efficiency factor. in time constants.",  metavar="SEFF")
    parser.add_argument('--scap', type=float, default=0, help= "Standard deviation of each CAPacitance. In farads.",  metavar="SCAP")
    parser.add_argument('--srefs', type=float, default=0, help= "Standard deviation of each REFerence volatage.",  metavar="SREFS")
    parser.add_argument('--sthres', type=float, default=0, help= "Standard deviation of each THREShold voltage.",  metavar="SLOSS")

    modify_help = (  "Manually MODIFY an attribute of the adc. The format is a json dict with the following properties:\n"
                    "- 'parameter': The parameter name to modify (eff, cs, ref, thres).\n"
                    "- 'stage'    : The stage number (starting from 0) of the parameter to modify.\n"
                    "- 'index'    : The index of the parameter, use a json list. Empty list for eff.\n"
                    "- 'value'    : New value of the parameter.")
    parser.add_argument('--modify', nargs='+', default=[], help= modify_help,  metavar="MODIFY")

    parser.add_argument('--n', default=0, type=int, help="Number of adcs, 0 for scalar, > 0 for array",  metavar="N")
    parser.add_argument('--seed', default=None, help="SEED for tail thresholds generation",  metavar="SEED")

    args = parser.parse_args()
    run(args)
    exit(0)
