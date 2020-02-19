#! /usr/bin/env python

from argparse import ArgumentParser

import calib.gen as gen
import calib.data as data
import calib.simulation as sims

from calib.misc import json_args
from calib.script.make_stage_testbench import save_stages


def run(args):
    sweeps = json_args(args.sweep)
    assert len(sweeps) > 0 or len(args.sweep_config)

    path_context = data.PathContext.relative()
    testbench_location = data.DataLocation(path_context, args.location, args.source, "json")

    print(" loading {}...".format(testbench_location.computed_path))
    testbench = data.load(sims.StageTestbench, testbench_location)

    sweep_config = []
    for conf_source in args.sweep_config:
        conf_loc = data.DataLocation(path_context, args.location, conf_source, "json")
        print(" loading {}...".format(conf_loc.computed_path))
        sweep_config.append(data.load(gen.ConfigurationSequence, conf_loc))

    if len(sweep_config) > 0:
        print("Sweeping configurations...")
        testbench = sims.StageTestbench(testbench.stages, testbench.ins, sweep_config)

    if len(sweeps) > 0:
        print("Sweeping parameters ...")
        testbench = testbench.sweep_parameters(sweeps)

    testbench_location = data.DataLocation(path_context, args.location, args.dest, "json")
    print(" saving {}...".format(testbench_location.computed_path))
    data.save(testbench, testbench_location)

    save_stages(testbench, path_context, args.location, args.dest)


if __name__ == "__main__":
    parser = ArgumentParser(description='Sweep a testbench parameters.')

    parser.add_argument('location', help= "Path to the input/output LOCATION.",  metavar="LOCATION")
    parser.add_argument('source', help= "Name of the SOURCE file (without extension).",  metavar="SOURCE")
    parser.add_argument('dest', help= "Name of the DESTination file (without extension).",  metavar="DEST")

    sweep_help = (  "Make a SWEEP. The format is a json dict with the following properties:\n"
                    "- 'parameter': The parameter name for sweeping (eff, cf, ss, ref, thres, test).\n"
                    "- 'stage'    : The stage number (starting from 0) of the parameter to sweep. Ignored for test.\n"
                    "- 'index'    : The index of the parameter, use a json list. Ignored for loss.\n"
                    "- 'start'    : Start value of the sweep.\n"
                    "- 'end'      : End value of the sweep.\n"
                    "- 'samples'  : Number of samples.")

    parser.add_argument('--sweep', nargs='+', default=[], help=sweep_help,  metavar="SWEEP")
    parser.add_argument('--sweep-config', nargs='+', default=[], help="Test with different configurations",  metavar="CONFIG")

    args = parser.parse_args()
    run(args)
    exit(0)
