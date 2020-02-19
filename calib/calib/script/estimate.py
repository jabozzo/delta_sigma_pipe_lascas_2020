#! /usr/bin/env python

from argparse import ArgumentParser

import numpy as np

import calib.simulation as sims
import calib.data as data
import calib.calibration as cal
import calib.gen as gen

import calib.misc as misc


def run_single(start, testbench, codes, mask, lsb_scale):
    assert testbench.is_scalar
    assert isinstance(start, gen.StageParameters)
    assert isinstance(codes, np.ndarray)

    conf_seq = testbench.configuration_sequence[tuple()]
    mask = cal.json_dicts_to_mask(start, mask)
    calibration = cal.CalibrationSystem(start, conf_seq, codes, mask=mask)

    n_ins = conf_seq.n_ins
    n_diff = start.meta.n_diff
    ins = np.zeros((n_ins, n_diff,))
    estimation = start

    nfev = None if args.nfev is None else int(args.nfev)
    maxiter = None if args.maxiter is None else int(args.maxiter)

    if not bool(args.skip):
        if args.mode == 0 or args.mode == 1:
            estimation, ins = calibration.run_calibration(estimation, ins, switching_nfev=nfev, lsb_scale=lsb_scale)

        if args.mode == 0 or args.mode == 2:
            # Limit to 4Gb
            n_items = conf_seq.n_conf * conf_seq.samples

            LIMIT = 127422 # ITEMS FOR 4Gb
            max_vectorial = max(1, LIMIT // n_items)
            estimation, ins = calibration.refine(
                estimation, ins, n_iter=maxiter, max_vectorial=max_vectorial, lsb_scale=lsb_scale)

        real_stage = testbench.stages[tuple()]
        print(cal.build_report(real_stage, estimation, "REAL", "EST"))

    assert isinstance(estimation, gen.StageParameters)
    return estimation


def run(args):
    mask = misc.json_args(args.mask)
    path_context = data.PathContext.relative()

    testbench_location = data.DataLocation(path_context, args.location, args.testbench, "json")
    codes_location = data.DataLocation(path_context, args.location, args.codes, "json")
    start_location = data.DataLocation(path_context, args.location, args.start, "json")
    dest_location = data.DataLocation(path_context, args.location, args.dest, "json")

    NestedNum = data.nested_lists_of(data.NumpyData)
    NestedStage = data.nested_lists_of(gen.StageParameters)

    print(" loading {}...".format(testbench_location.computed_path))
    testbench = data.load(sims.StageTestbench, testbench_location)
    print(" loading {}...".format(codes_location.computed_path))
    codes = data.load(NestedNum, codes_location)
    print(" loading {}...".format(start_location.computed_path))
    starts = data.load(NestedStage, start_location)

    def codes_rec_shape(element):
        if isinstance(element, np.ndarray):
            return tuple()
        else:
            return (len(element),) + codes_rec_shape(element[0])

    iter_shape = testbench.conf_shape + testbench.shape

    assert codes_rec_shape(codes.data) == iter_shape, "Expected {}, recieved {}".format(codes_rec_shape(codes.data), iter_shape)
    assert np.shape(starts.data) == iter_shape, "Expected {}, recieved {}".format(np.shape(starts.data), iter_shape)

    estimations = np.full(iter_shape, None)

    scalar_testbenches = testbench.as_scalars()
    iter_idx = tuple(testbench.iter_idx())

    for ii, idx in enumerate(iter_idx):
        idx_conf, idx_stage = idx
        idx = idx_conf + idx_stage
        start = starts[idx]
        tb = scalar_testbenches[idx]
        code = codes[idx]

        print("  estimating {} {}/{}".format(idx, ii+1, len(iter_idx)))
        estimation = run_single(start, tb, code, mask, args.lsb_scale)
        estimations[idx] = estimation

    estimations = NestedStage(estimations.tolist(), len(iter_shape))

    print(" saving {}...".format(dest_location.computed_path))
    data.save(estimations, dest_location)

    reports = np.full(iter_shape, None)

    for idx in iter_idx:
        idx_conf, idx_stage = idx
        idx = idx_conf + idx_stage
        stage = testbench.stages[idx_stage]
        estimation = estimations[idx]
        reports[idx] = data.TextFile(cal.build_report(stage, estimation))

    reports_location = data.DataLocation(path_context, args.location, args.dest, "txt")
    NestedTexts = data.nested_lists_of(data.TextFile)
    reports = NestedTexts(reports.tolist(), len(np.shape(reports)))
    reports.set_children_data_location(reports_location)

    for idx in iter_idx:
        idx_conf, idx_stage = idx
        idx = idx_conf + idx_stage
        report = reports[idx]
        path = report.data_location.computed_path
        print(" saving {}...".format(path))
        data.save(report)


if __name__ == "__main__":
    parser = ArgumentParser(description='Make an estimation.')

    parser.add_argument('location', help="Path to the input/output LOCATION.",  metavar="LOCATION")
    parser.add_argument('testbench', help="Name of the TESTBENCH file (without extension).",  metavar="TESTBENCH")
    parser.add_argument('codes', help="Name of the CODES array file (without extension).",  metavar="CODES")
    parser.add_argument('start', help="Stage array to use as START position (without extension).",  metavar="START")
    parser.add_argument('dest', help="Name of the DESTination file (without extension).",  metavar="DEST")

    parser.add_argument('--mode', type=int, choices=(0, 1, 2,), default=0,
        help="Estimation MODE: 0. LSQ and refine, 1. LSQ, 2. Refine.",  metavar="MODE")
    parser.add_argument('--nfev', default=None, help="Number of evaluations for LSQ.",  metavar="NFEV")
    parser.add_argument('--maxiter', default=None, help="Maximum number of iterations for refine",  metavar="NFEV")
    parser.add_argument('--skip', default=0, type=int, help="SKIP estimation, can be used to dry-test a script",  metavar="SKIP")

    mask_help = (   "Ignore variables estimation in MASK:\n"
                    "- 'parameter': The parameter name to mask (eff, cs, ref, cm).\n"
                    "- 'index'    : The index of the parameter, use a json list. Empty list for eff.")
    parser.add_argument('--mask', default=[], nargs='+', help=mask_help,  metavar="MASK")
    parser.add_argument('--lsb-scale', type=float, default=1, help="Scale LSB before appliying bands",  metavar="LSB")

    args = parser.parse_args()

    run(args)
    exit(0)
