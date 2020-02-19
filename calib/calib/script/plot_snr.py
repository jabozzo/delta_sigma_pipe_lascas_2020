#! /usr/bin/env python

from argparse import ArgumentParser
from itertools import product as cartesian

import numpy as np

import calib.data as data
import calib.gen as gen
import calib.plot as cplot
import calib.analysis as an
import calib.simulation as sims


def enob_improvement(args, real_snr, ideal_snr, est_snr, extras_snr, x_axis, x_label):
    y_label = "ENOB improvement"

    real_mean, real_std = real_snr
    ideal_mean, ideal_std = ideal_snr
    est_mean, est_std = est_snr
    extra_mean, extra_std = tuple(zip(*extras_snr))

    for mean in (real_mean, ideal_mean, est_mean,) + tuple(extra_mean):
        assert len(np.shape(mean)) == 1

    real_mean_im = real_mean - ideal_mean
    est_mean_im = est_mean - ideal_mean
    extra_mean_im = [extra - ideal for extra, ideal in zip(extra_mean, ideal_mean)]

    canvas = cplot.Canvas1d()

    equal_plot = cplot.Plot1d([0, 0,], x=[np.min(x_axis), np.max(x_axis)],
        axes_labels=(x_label, y_label,), title=args.title, style="plot",
        plot_kwargs={"c": 'k', 'linestyle': '--'})

    canvas.add(equal_plot)

    estimated_plot = cplot.Plot1d(est_mean_im, x=x_axis,
        axes_labels=(x_label, y_label,), title=args.title, style="scatter",
        plot_kwargs={"c": 'orange'})

    canvas.add(estimated_plot, "Calibrated")

    if est_std is not None:
        for sign in (1, -1,):
            value = est_mean_im + sign * est_std
            estimated_std = cplot.Plot1d(value, x=x_axis,
                axes_labels=(x_label, y_label,), title=args.title, style="scatter",
                plot_kwargs={"c": 'orange', 'marker': '+'})

            canvas.add(estimated_std)

    perfect_plot = cplot.Plot1d(real_mean_im, x=x_axis,
        axes_labels=(x_label, y_label,), title=args.title, style="scatter",
        plot_kwargs={"c": 'b'})

    canvas.add(perfect_plot, "Perfect knowledge")

    if real_std is not None:
        for sign in (1, -1,):
            value = real_mean_im + sign * real_std
            perfect_std = cplot.Plot1d(value, x=x_axis,
                axes_labels=(x_label, y_label,), title=args.title, style="scatter",
                plot_kwargs={"c": 'b', 'marker': '+'})

            canvas.add(perfect_std)

    for ii, mean in enumerate(extra_mean_im):
        std = extra_std[ii]
        extra_plot = cplot.Plot1d(mean, x=x_axis,
            axes_labels=(x_label, y_label,), title=args.title, style="scatter",
            plot_kwargs={"c": 'cyan'})

        canvas.add(extra_plot)

        if std is not None:
            for sign in (1, -1,):
                value = mean + sign * std
                extra_std = cplot.Plot1d(value, x=x_axis,
                    axes_labels=(x_label, y_label,), title=args.title, style="scatter",
                    plot_kwargs={"c": 'cyan', 'marker': '+'})

                canvas.add(extra_std)

    return canvas


def enob_compare(args, real_snr, ideal_snr, est_snr, extras_snr):
    x_label = "Naive ENOB"
    y_label = "ENOB"

    def flat(arr):
        return None if arr is None else arr.flatten()

    real_mean, real_std = tuple(flat(arr) for arr in real_snr)
    ideal_mean, ideal_std = tuple(flat(arr) for arr in ideal_snr)
    est_mean, est_std = tuple(flat(arr) for arr in est_snr)
    extra_mean, extra_std = tuple(zip(*extras_snr))
    extra_mean = [flat(arr) for arr in extra_mean]
    extra_std = [flat(arr) for arr in extra_std]

    assert ideal_std is None or (ideal_std == 0).all()

    canvas = cplot.Canvas1d()

    ranges = [np.min(ideal_mean), np.max(ideal_mean)]
    equal_plot = cplot.Plot1d(ranges, x=ranges,
        axes_labels=(x_label, y_label,), title=args.title, style="plot",
        plot_kwargs={"c": 'k', 'linestyle': '--'})

    canvas.add(equal_plot)

    assert np.shape(est_mean) == np.shape(ideal_mean)
    estimated_plot = cplot.Plot1d(est_mean, x=ideal_mean,
        axes_labels=(x_label, y_label,), title=args.title, style="scatter",
        plot_kwargs={"c": 'orange'})

    canvas.add(estimated_plot, "Calibrated ENOB")

    if est_std is not None:
        for sign in (1, -1,):
            value = est_mean + sign * est_std
            assert np.shape(value) == np.shape(ideal_mean)
            estimated_std = cplot.Plot1d(value, x=ideal_mean,
                axes_labels=(x_label, y_label,), title=args.title, style="scatter",
                plot_kwargs={"c": 'orange', 'marker': '+'})

            canvas.add(estimated_std)

    assert np.shape(real_mean) == np.shape(ideal_mean)
    perfect_plot = cplot.Plot1d(real_mean, x=ideal_mean,
        axes_labels=(x_label, y_label,), title=args.title, style="scatter",
        plot_kwargs={"c": 'b'})

    canvas.add(perfect_plot, "Perfect knowledge ENOB")

    if real_std is not None:
        for sign in (1, -1,):
            assert np.shape(value) == np.shape(ideal_mean)
            value = real_mean + sign * real_std
            perfect_std = cplot.Plot1d(value, x=ideal_mean,
                axes_labels=(x_label, y_label,), title=args.title, style="scatter",
                plot_kwargs={"c": 'b', 'marker': '+'})

            canvas.add(perfect_std)

    for ii, mean in enumerate(extra_mean):
        std = extra_std[ii]
        print(mean)
        assert np.shape(mean) == np.shape(ideal_mean)
        extra_plot = cplot.Plot1d(mean, x=ideal_mean,
            axes_labels=(x_label, y_label,), title=args.title, style="scatter",
            plot_kwargs={"c": 'cyan'})

        canvas.add(extra_plot)

        if std is not None:
            for sign in (1, -1,):
                value = mean + sign * std
                assert np.shape(value) == np.shape(ideal_mean)
                extra_std = cplot.Plot1d(value, x=ideal_mean,
                    axes_labels=(x_label, y_label,), title=args.title, style="scatter",
                    plot_kwargs={"c": 'cyan', 'marker': '+'})

                canvas.add(extra_std)

    return canvas


def run(args):
    path_context = data.PathContext.relative()
    adcs_location = data.DataLocation(path_context, args.location, args.adcs, "json")
    estimations_location = (adcs_location if args.estimation is None else
        data.DataLocation(path_context, args.location, args.estimation, "json") )

    NestedAdcs = data.nested_lists_of(gen.PipeParameters)

    print(" loading {}...".format(adcs_location.computed_path))
    adcs = data.load(NestedAdcs, adcs_location)

    print(" loading {}...".format(estimations_location.computed_path))
    estimations = data.load(NestedAdcs, estimations_location)

    extras = []
    for extra in args.extra:
        extra_location = data.DataLocation(path_context, args.location, extra, "json")
        print(" loading {}...".format(extra_location.computed_path))
        extras.append(data.load(NestedAdcs, extra_location))

    def snr(real_adc, est_adc):
        magnitude = an.snr(real_adc, est_adc, sampling_factor=args.sampling_factor)
        return an.snr_to_enob(magnitude)

    shape = np.shape(adcs.data)
    assert shape == np.shape(estimations.data)
    for extra in extras:
        assert shape == np.shape(extra.data)

    real_snr = np.empty(shape)
    ideal_snr = np.empty(shape)
    est_snr = np.empty(shape)
    extras_snr = [np.empty(shape) for _ in range(len(extras))]

    iter_idx = (tuple(),) if len(shape) == 0 else cartesian(*tuple(range(ss) for ss in shape))

    for idx in iter_idx:
        real = adcs[idx]
        est = estimations[idx]

        real_snr[idx] = snr(real, real)
        ideal_snr[idx] = snr(real, real.as_ideal())
        est_snr[idx] = snr(real, est)

        for ii, extra in enumerate(extras):
            extras_snr[ii][idx] = snr(real, extra[idx])

    keep_axis = None if args.keep_axis is None else int(args.keep_axis)

    if keep_axis is not None:
        axes = list(range(len(shape)))
        axes.pop(keep_axis)
        axes = tuple(axes)

        def reduce_(arr):
            return (np.mean(arr, axis=axes), np.std(arr, axis=axes),)

    else:
        def reduce_(arr):
            return (arr, None,)

    real_snr = reduce_(real_snr)
    ideal_snr = reduce_(ideal_snr)
    est_snr = reduce_(est_snr)
    extras_snr = [reduce_(e_snr) for e_snr in extras_snr]

    x_param = args.x_param
    if x_param is None:
        canvas = enob_compare(args, real_snr, ideal_snr, est_snr, extras_snr)

    else:
        assert keep_axis is not None
        samples = shape[keep_axis]

        if x_param == 'eff':
            def element(idx):
                return (0,)*keep_axis + (idx,) + (0,)*(len(shape) - 1 - keep_axis)

            samples_idx = tuple(element(ii) for ii in range(samples))

            x_axis = [adcs[idx].stages[0].eff.item() for idx in samples_idx]
            x_label = "Charge transfer efficiency"

        else:
            assert args.testbench is not None, "Testbench required for x-param"

            testbench_loc = data.DataLocation(path_context, args.location, args.testbench, "json")
            testbench = data.load(sims.StageTestbench, testbench_loc)
            conf_shape = testbench.conf_shape

            def element(idx):
                return (0,)*keep_axis + (idx,) + (0,)*(len(conf_shape) - 1 - keep_axis)

            samples_idx = tuple(element(ii) for ii in range(samples))

            def conf_seq(idx):
                return testbench.configuration_sequence[idx]

            if x_param == 'samples':
                x_axis = [conf_seq(idx).samples for idx in samples_idx]
                x_label = "Samples"

            if x_param == 'switches':
                x_axis = [len(conf_seq(idx).configuration_sets) for idx in samples_idx]
                x_label = "Switching"

        canvas = enob_improvement(args, real_snr, ideal_snr, est_snr, extras_snr, x_axis, x_label)

    dest_location = data.DataLocation(
        path_context, args.location, args.dest, "json")
    print(" saving {}...".format(dest_location.computed_path))
    data.save(canvas, dest_location)


if __name__ == "__main__":
    parser = ArgumentParser(description='Plot estimation improvements.')

    parser.add_argument('location', help="Path to the input/output LOCATION.",  metavar="LOCATION")
    parser.add_argument('adcs', help="Name of the ADCS list file (without extension).",  metavar="ADCS")
    parser.add_argument('dest', help="Name of the DESTination file (without extension).",  metavar="DEST")

    parser.add_argument('--estimation', default=None, help="Name of the adcs ESTIMATION list file (without extension).",  metavar="ESTIMATION")
    parser.add_argument('--extra', nargs="*", default=[], help="Name of the EXTRA adcs list file (without extension).",  metavar="EXTRA")
    parser.add_argument('--title', default=None, help="TITLE of the plot",  metavar="TITLE")
    parser.add_argument('--sampling-factor', type=int, default=16, help="Factor of number of SAMPLES to use.",  metavar="SAMPLING")

    parser.add_argument("--keep-axis", default=None, help= "Reduce all AXIS except the selected one",  metavar="AXIS")
    parser.add_argument("--testbench", default=None , help="Reference TESTBENCH",  metavar="TESTBENCH")
    parser.add_argument("--x-param", default=None, choices=(None, 'eff', 'samples', 'switches',),
        help= "Which PARAMeter to use for the x axis.",  metavar="PARAM")

    args = parser.parse_args()

    run(args)
    exit(0)
