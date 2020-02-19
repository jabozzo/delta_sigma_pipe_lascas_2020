#! /usr/bin/env python

from argparse import ArgumentParser
from itertools import product as cartesian

import numpy as np

import calib.simulation as sims
import calib.data as data

import warnings


def run(args):
    seed = None if args.seed is None else int(args.seed)

    path_context = data.PathContext.relative()
    testbench_location = data.DataLocation(path_context, args.location, args.source, "json")

    cache = bool(args.cache)
    array = bool(args.array)
    serial = bool(args.serial)

    print(" loading {}...".format(testbench_location.computed_path))
    testbench = data.load(sims.StageTestbench, testbench_location)

    simulator = sims.Simulator(seed, args.ref_snr, args.thres_snr, args.test_snr)
    testbenches = testbench.as_scalars() if serial else testbench

    iter_shape = testbench.conf_shape + testbench.shape if serial else tuple()

    codes = np.full(iter_shape, None, dtype=object)
    us = np.full(iter_shape, None, dtype=object)

    iter_idx = tuple(cartesian(*tuple(range(ss) for ss in iter_shape))) if serial else (tuple(),)

    for ii, idx in enumerate(iter_idx):
        bench = testbenches[idx] if serial else testbenches
        if cache:
            print(" building cache...")
            for conf_idx in bench.iter_conf_idx():
                bench.configuration_sequence[conf_idx].build_cache()

        print(" simulating...")
        code, u = bench.simulate(simulator, raise_=bool(getattr(args, "raise")))

        indexes = list(bench.iter_idx())
        fsr = [bench.stages[idx_stage].meta.fsr for _, idx_stage in indexes]
        out_of_range = []
        for ff, fsr_idx in zip(fsr, indexes):
            idx_conf, idx_stage = fsr_idx
            iidx = idx_conf + idx_stage
            outs = np.logical_or(u[iidx + (Ellipsis,)] > ff[1], u[iidx + (Ellipsis,)] < ff[0])
            if outs.any():
                out_of_range.append((np.sum(outs), np.size(outs), ff))

        for outs in out_of_range:
            sum_, size_, fsr = outs
            warnings.warn("{}/{} samples are out of range (fsr: {}, {})".format(
                sum_, size_, *fsr))

        codes[idx] = code
        us[idx] = u

    c_name = "{}.c".format(args.dest)
    u_name = "{}.u".format(args.dest)

    if array:
        ext = "json"

        NestedNum = data.nested_lists_of(data.NumpyData)
        data_code = np.full(iter_shape, None)
        data_u = np.full(iter_shape, None)

        for idx in testbench.iter_idx():
            idx = idx[0] + idx[1]
            data_code[idx] = data.at_least_numpydata(codes[idx])
            data_u[idx] = data.at_least_numpydata(us[idx])

        codes = NestedNum(data_code.tolist(), dims=len(np.shape(data_code)))
        us = NestedNum(data_u.tolist(), dims=len(np.shape(data_u)))

        codes.set_children_data_location(data.DataLocation(path_context, args.location, c_name, "npy"))
        us.set_children_data_location(data.DataLocation(path_context, args.location, u_name, "npy"))

    else:
        codes = np.array(codes.tolist())
        us = np.array(us.tolist())

        samples_index = len(iter_shape)
        if samples_index > 0:
            transpose_indexes = ((samples_index,) + tuple(range(samples_index))
                                + tuple(range(samples_index+1, len(np.shape(codes)))))

            codes = np.transpose(codes)
            us = np.transpose(us)

        ext = "npy"

    code_location = data.DataLocation(path_context, args.location, c_name, ext)
    u_location = data.DataLocation(path_context, args.location, u_name, ext)

    print(" saving {}...".format(code_location.computed_path))
    data.save(codes, code_location)

    if bool(args.store_u):
        print(" saving {}...".format(u_location.computed_path))
        data.save(us, u_location)


if __name__ == "__main__":
    parser = ArgumentParser(description='Simulate a testbench.')

    parser.add_argument('location', help="Path to the input/output LOCATION.",  metavar="LOCATION")
    parser.add_argument('source', help="Name of the SOURCE testbench file (without extension).",  metavar="SOURCE")
    parser.add_argument('dest', help="Name of the DESTination file (without extension).",  metavar="DEST")

    parser.add_argument('--serial', type=int, default=0, help="Process the stages arrays SERIALly.",  metavar="SERIAL")
    parser.add_argument('--array', type=int, default=0, help="Put the simulation output in an ARRAY",  metavar="ARRAY")

    parser.add_argument('--seed',  default=None, help="The simulator SEED.", metavar="SEED")
    parser.add_argument('--u-history', type=bool, default=True, help="Save the residual (U) history.", metavar="U")
    parser.add_argument('--ref-snr', type=float, default=0, help="Snr of REF values.", metavar="REF")
    parser.add_argument('--test-snr', type=float, default=0, help="Snr of TEST values.", metavar="TEST")
    parser.add_argument('--thres-snr', type=float, default=0, help="Snr of THRES values.", metavar="THRES")

    parser.add_argument('--cache', type=int, default=1, help="Build CACHE for faster simulation, requires more ram.", metavar="CACHE")
    parser.add_argument('--raise', type=int, default=1, help="RAISE exception if residual out of range.", metavar="RAISE")
    parser.add_argument('--store-u', type=int, default=0, help="STORE residual value.", metavar="STORE")

    args = parser.parse_args()
    run(args)
    exit(0)
