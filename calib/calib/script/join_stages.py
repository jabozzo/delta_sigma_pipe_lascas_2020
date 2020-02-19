#! /usr/bin/env python

from argparse import ArgumentParser

import numpy as np

import calib.gen as gen
import calib.data as data


def run(args):
    path_context = data.PathContext.relative()
    stages_location = [data.DataLocation(path_context, args.location, source, "json") for source in getattr(args, "stages-sources")]

    NestedStages = data.nested_lists_of(gen.StageParameters)

    all_stages = []
    for stage_loc in stages_location:
        print(" loading {}...".format(stage_loc.computed_path))
        stage = data.load(NestedStages, stage_loc)
        all_stages.append(stage)

    assert len(all_stages) > 0
    shape = np.shape(all_stages[0].data)
    assert all(shape == np.shape(stages.data) for stages in all_stages[1:])

    adcs = np.full(shape, None)
    for idx in all_stages[0].iter_idx():
        stages = tuple(stages[idx] for stages in all_stages)
        thresholds = [stage.thres for stage in stages]
        meta0 = stages[0].meta
        thresholds = [gen.compute_thres(meta0.n_bits, *meta0.fsr, half_bit=meta0.half_bit)] + thresholds
        thresholds, tail = thresholds[:-1], thresholds[-1]

        assert all(stage.meta.n_codes == np.size(thres) + 1 for stage, thres in zip(stages, thresholds))
        stages = [gen.StageParameters(s.meta, s.eff, s.caps, s.refs, thres, s.common_mode)
                    for s, thres in zip(stages, thresholds)]

        adcs[idx] = gen.PipeParameters(stages, tail)

    adcs = data.nested_lists_of(gen.PipeParameters)(adcs.tolist(), len(np.shape(adcs)))

    adcs_location = data.DataLocation(path_context, args.location, args.dest, "json")
    print(" saving {}...".format(adcs_location.computed_path))
    data.save(adcs, adcs_location)


if __name__ == "__main__":
    parser = ArgumentParser(description='Join delta-sigma stages back into an ADC.')

    parser.add_argument('location', help= "Path to the output LOCATION.",  metavar="LOCATION")
    parser.add_argument('dest', help= "Name of the DESTination file (without extension).",  metavar="DEST")
    parser.add_argument('stages-sources', nargs='+', help= "SOURCES of stages, in order.",  metavar="SOURCES")
    args = parser.parse_args()

    run(args)
    exit(0)
