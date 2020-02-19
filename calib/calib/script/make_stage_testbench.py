#! /usr/bin/env python

from argparse import ArgumentParser

import numpy as np

import calib.gen as gen
import calib.data as data
import calib.simulation as sims


def save_stages(tb, path_context, location, dest):
    shape = tb.conf_shape + tb.shape
    array = len(shape) > 0

    stages = np.full(shape, None)
    stages_ideal = np.full(shape, None)

    for idx in tb.iter_idx():
        conf_idx, stage_idx = idx
        idx = conf_idx + stage_idx

        real = tb.stages[stage_idx]
        stages[idx] = real

        ideal = real.meta.generate_ideal()
        tail_bits, tail_half = gen.infer_thres_bits(real.thres)
        thres_ideal = gen.compute_thres(tail_bits, *real.meta.fsr, half_bit=tail_half)
        ideal._thres = thres_ideal
        stages_ideal[idx] = ideal

    NestedStages = data.nested_lists_of(gen.StageParameters)
    dims = len(shape)
    stages = NestedStages(stages.tolist() if array else stages[tuple()], dims)
    stages_ideal = NestedStages(stages_ideal.tolist() if array else stages_ideal[tuple()], dims)

    stages_location = data.DataLocation(path_context, location, dest + ".stages", "json")
    print(" saving {}...".format(stages_location.computed_path))
    data.save(stages if array else stages[tuple()], stages_location)

    ideal_location = data.DataLocation(path_context, location, dest + ".stages.ideal", "json")
    print(" saving {}...".format(ideal_location.computed_path))
    data.save(stages_ideal if array else stages_ideal[tuple()], ideal_location)


def run(args):
    path_context = data.PathContext.relative()
    adcs_location = data.DataLocation(path_context, args.location, args.adc_source, "json")
    config_location = data.DataLocation(path_context, args.location, args.config_source, "json")

    array = bool(args.array)
    AdcsType = gen.PipeParameters
    AdcsType = data.list_of(AdcsType) if array else AdcsType

    print(" loading {}...".format(adcs_location.computed_path))
    adcs = data.load(AdcsType, adcs_location)
    print(" loading {}...".format(config_location.computed_path))
    config = data.load(gen.ConfigurationSequence, config_location)

    if not array:
        adcs = [adcs]

    adcs_stages = [adc.as_delta_sigma() for adc in adcs]
    stages = [stages[args.stage_idx] for stages in adcs_stages]
    shape = (len(stages),) if array else tuple()

    testbench = sims.StageTestbench(stages, args.tests, config, shape=shape)
    testbench_location = data.DataLocation(path_context, args.location, args.dest, "json")
    print(" saving {}...".format(testbench_location.computed_path))
    data.save(testbench, testbench_location)

    save_stages(testbench, path_context, args.location, args.dest)


if __name__ == "__main__":
    parser = ArgumentParser(description='Create an stage testbench.')
    parser.add_argument('location', help= "Path to the output LOCATION.",  metavar="LOCATION")
    parser.add_argument('adc_source', help= "Name of the source ADC file (without extension).",  metavar="ADC")
    parser.add_argument('config_source', help= "Name of the source CONFiguration file (without extension).",  metavar="CONFIG")
    parser.add_argument('stage_idx', type=int, help= "Index of the stage to test.",  metavar="INDEX")
    parser.add_argument('dest', help= "Name of the DESTination file (without extension).",  metavar="DEST")

    parser.add_argument('--tests', type=float, default=[], nargs="+", help= "External tests voltages.",  metavar="TESTS")
    parser.add_argument('--array', type=int, default=0, help= "Input adc is an ARRAY of adcs.",  metavar="ARRAY")
    args = parser.parse_args()

    run(args)
    exit(0)
