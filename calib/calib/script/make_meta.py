#! /usr/bin/env python

from argparse import ArgumentParser

import calib.gen as gen
import calib.data as data

from calib.misc import default


def run(args):
    n_bits = args.nbits
    n_refs = default(args.nrefs, 3)

    seed = None if args.seed is None else int(args.seed)

    meta = gen.StageMeta(
        args.nbits,
        n_refs,
        eff=args.eff,
        cap=args.cap,
        common_mode=args.common_mode,
        fsr=(args.min, args.max,),
        differential=args.differential,
        seed=seed )

    data_location = data.DataLocation(data.PathContext.relative(), args.location, args.dest, "json")
    print(" saving {}...".format(data_location.computed_path))
    data.save(meta, data_location)


if __name__ == "__main__":
    parser = ArgumentParser(description='Create an adc metadata.')
    parser.add_argument('location', help= "Path to the output LOCATION.",  metavar="LOCATION")
    parser.add_argument('dest', help= "DESTINATION of the file to save.",  metavar="DESTINATION")

    parser.add_argument('nbits', help= "Number of BITS of each stage.",  metavar="BITS")

    parser.add_argument('--nrefs', type=int, const=None, default=None, help= "Number of REFERENCES of each stage capacitor.",  metavar="REFERENCES")

    parser.add_argument('--min', type=float, default=-0.5, help= "MINimum input voltage of the ADC.",  metavar="MIN")
    parser.add_argument('--max', type=float, default=0.5, help= "MAXimum input voltage of the ADC.",  metavar="MAX")

    parser.add_argument('--eff', type=float, default=0.95, help= "Mean of charge transfer EFFiciency.",  metavar="EFF")
    parser.add_argument('--cap', type=float, default=1, help= "Mean of each CAPacitance.",  metavar="CAP")

    parser.add_argument('--differential', type=bool, default=False, help= "If the stage architecture is DIFFERENTIAL.",  metavar="DIFFERENTIAL")
    parser.add_argument('--seed', default=None, help= "SEED for random generation.",  metavar="SEED")
    parser.add_argument('--common-mode', default=None, help= "Common Mode value.",  metavar="CM")

    args = parser.parse_args()
    run(args)
    exit(0)
