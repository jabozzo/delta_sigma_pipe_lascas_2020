#! /usr/bin/env python

from argparse import ArgumentParser

from itertools import product as cartesian

import calib.gen as gen
import calib.data as data
import calib.misc as misc

import numpy as np


def single(meta, args):
    n_cs = (meta.n_caps - 1)//2
    n_cf = meta.n_caps - n_cs
    n_refs = meta.n_refs

    ds_samples = np.sum(args.samples)*args.loop

    if args.n_test == 0:
        input = gen.InternalDC(meta, [n_refs//2]*n_cs)
    elif args.n_test == 1:
        input = gen.ExternalDC(meta, [0]*n_cs)
    else:
        raise ValueError("Single supports at most 1 test.")

    if args.ic == "clear":
        ic = gen.InitialCondition.Discharged(meta, n_cf)
    elif args.ic == "precharge":
        ic = gen.InitialCondition(meta, [n_refs//2]*n_cf)
    else:
        raise ValueError("IC '{}' is not supported.".format(args.ic))

    configuration = gen.Configuration(meta, list(range(n_cs)))
    c_set = gen.ConfigurationSet(ds_samples, [input], [configuration])
    c_seq = gen.ConfigurationSequence([ic], [c_set])

    return c_seq


def calib(meta, args, interlace, use_full_range=None):
    n_caps = meta.n_caps
    n_refs = meta.n_refs
    n_diff = meta.n_diff

    n_cs = (n_caps - 1)//2
    n_cf = n_caps - n_cs

    use_full_range = misc.default(use_full_range, n_cs < 2)
    ds_samples = args.samples

    if args.n_test > 0:
        raise ValueError("Minimal does not support test inputs.")

    comb_cs = misc.iterate_combinations(n_caps, n_cs)
    if args.full:
        comb_cs = [tuple(misc.iterate_permutations(cs)) for cs in comb_cs]
        comb_cs = [elem for tlp in comb_cs for elem in tlp]

    slice_ = slice(None) if use_full_range else slice(1, -1)

    comb_refs = gen.ds_map(n_cs, n_refs, n_cs*(n_refs-1) + 1)
    comb_refs = np.transpose(comb_refs[:, slice_], (1, 0, 2,))
    comb_refs = comb_refs.tolist()
    comb_refs = [(comb_refs[ii], comb_refs[jj],)
                    for ii in range(len(comb_refs))
                    for jj in range(ii+1, len(comb_refs)) ]

    comb_cs = list(comb_cs)
    comb_refs = list(comb_refs)

    even_configs = []
    even_ins = []

    ics = []

    with misc.push_random_state():
        seed = None if args.seed is None else int(args.seed)
        np.random.seed(seed)

        for cs_ii, refs_ii in cartesian(comb_cs, comb_refs):
            even_configs.append(gen.Configuration(meta, cs_ii))

            top_ii, bot_ii = refs_ii
            if args.inputs == "":
                sub_seed = np.random.randint(0, 4294967296)
                even_ins.append(gen.InternalRandom(meta, np.size(cs_ii), sub_seed))

            else:
                top = gen.InternalDC(meta, top_ii)
                bot = gen.InternalDC(meta, bot_ii)

                even_ins.append(gen.ACCombinator(meta, top, bot, args.period))

            inv = [[n_refs - 1 - iii for iii in ii] for ii in top_ii]
            inv = inv + [[n_refs//2, n_refs - n_refs//2][:n_diff]]*(n_cf - n_cs)
            ics.append(gen.InitialCondition(meta, inv))

    if interlace:
        n_cs_h = n_cs // 2
        assert n_cs_h > 0, "Not enough capacitors to decrease bits."

        odd_configs = []
        odd_ins = []

        for conf, in_ in zip(even_configs, even_ins):
            left = (n_cs - n_cs_h) // 2
            cs_range = range(left, left+n_cs_h)
            mask = np.zeros((n_cs,), dtype=bool)
            mask[cs_range] = 1

            odd_configs.append(gen.Configuration(conf.meta, conf.cs[cs_range, :]))
            odd_ins.append(gen.InputMask(in_.meta, in_, mask))

    else:
        odd_ins = even_ins
        odd_configs = even_configs

    conf_sets = []
    parity = 0
    for samples in ds_samples:
        if parity == 0:
            configs = even_configs
            inputs = even_ins

        else:
            configs = odd_configs
            inputs = odd_ins

        conf_sets.append(gen.ConfigurationSet(samples, inputs, configs))
        parity = (parity + 1) % 2

    if args.ic == "clear":
        ics = [gen.InitialCondition.Discharged(meta, n_cf)] * len(odd_ins)
    elif args.ic == "precharge":
        pass
    else:
        raise ValueError("ic type {} not supported".format(args.ic))

    return gen.ConfigurationSequence(ics, conf_sets * args.loop)


def run(args):
    args.full = bool(args.full)

    path_context = data.PathContext.relative()
    meta_location = data.DataLocation(path_context, args.location, args.source, "json")
    print(" loading {}...".format(meta_location.computed_path))
    meta = data.load(gen.StageMeta, meta_location)

    if args.type == "single":
        c_seq = single(meta, args)
    elif args.type == "standard":
        c_seq = calib(meta, args, False)
    elif args.type == "interlaced":
        c_seq = calib(meta, args, True)
    else:
        raise ValueError("Unexpected type '{}'.".format(args.type))

    print(" Made {} unique configurations.".format(c_seq.n_conf))

    seq_location = data.DataLocation(path_context, args.location, args.dest, "json")
    print(" saving {}...".format(seq_location.computed_path))
    data.save(c_seq, seq_location)


if __name__ == "__main__":
    parser = ArgumentParser(description='Create a configuration sequence.')

    parser.add_argument('location', help="Path to the input/output LOCATION.",  metavar="LOCATION")
    parser.add_argument('source', help="Name of the SOURCE file (without extension).",  metavar="SOURCE")
    parser.add_argument('dest', help="Name of the DESTination file (without extension).",  metavar="DEST")
    parser.add_argument('type', type=str, choices=("single", "standard", "interlaced",), help="The calibration scenario.",  metavar="TYPE")
    parser.add_argument('samples', type=int, nargs='+', help="Delta sigma SAMPLES.",  metavar="SAMPLES")

    parser.add_argument('--n_test', type=int, default=0, help="TEST voltage values",  metavar="TEST")
    parser.add_argument('--ic', type=str, choices=("clear", "precharge",), help="Sets the Inital Condition to clear or precharge.",  metavar="IC")
    parser.add_argument('--loop', type=int, default=1, help="Number of times to LOOP the configuration.",  metavar="LOOP")
    parser.add_argument('--period', type=int, default=16, help="Input PERIOD, if applicable.",  metavar="PERIOD")
    parser.add_argument('--full', type=int, default=0, help="Perform FULL iteration.",  metavar="FULL")
    parser.add_argument('--inputs', choices=("ac", "random",), default="ac", help="Type of INPUTS.",  metavar="INPUTS")
    parser.add_argument('--seed', default=None, help="SEED for random generation.",  metavar="SEED")

    args = parser.parse_args()

    run(args)
    exit(0)
