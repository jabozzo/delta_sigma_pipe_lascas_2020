#! /usr/bin/env python

import copy
import pickle

from itertools import count
from itertools import product as cartesian

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plots
from scipy.special import erfinv

import calib.gen as gen
import calib.data as data
import calib.simulation as sims
import calib.calibration as cal
import calib.analysis as an

import calib.misc as misc

from calib.script.make_config import calib as make_config
from calib.script.mkdir import run as mkdir

######## VARIABLES ########

# Set to true if calibration has already done and just want to replot
JUST_PLOT = False

FULL_SIZE = True
TOGETHER = True  # Set to false to have individual graphs per test.
TESTS = (0, 1, 2,)
S_TAU = 0.5
PLOT_STD = False  # Set to true to plot standard deviation of aggregation.

# To quickly check everything is in order, will yield wrong results
DRY_RUN = False

###########################


def avg_mismatch_to_std(avg_mm, cap_value=1):
    # Last term is integal of |x| * phi(x), with phi standard normal function.
    return cap_value * avg_mm * (2/np.sqrt(2*np.pi))


def bits_to_required_eff(n_bits, half_bit=None, lsb_req=0.5):
    lsb = gen.compute_lsb(n_bits, 0, 1, half_bit=half_bit)
    eff = 1 - lsb * lsb_req
    return eff


def gen_adc(bits, seed, args, n_adcs=1):
    assert len(bits) > 1

    adcs = []
    with misc.push_random_state():
        np.random.seed(seed)
        seeds = [np.random.randint(0, 4294967296) for _ in range(n_adcs)]

    for seed in seeds:
        with misc.push_random_state():
            np.random.seed(seed)
            sub_seeds = [np.random.randint(0, 4294967296) for _ in range(len(bits))]

        stages = []
        for ii, cbits, sseed in zip(count(), bits[:-1], sub_seeds):
            _, half_bit = gen.parse_bits(cbits)
            n_refs = 3 if half_bit else 2
            eff = bits_to_required_eff(cbits)
            eff = max(eff, 0.95)
            meta = gen.StageMeta(cbits, n_refs, eff=eff, seed=sseed)

            s_ref = args.s_ref
            if s_ref is None:
                s_ref = 0.5 * meta.lsb / 3

            stage = meta.generate_gaussian(S_TAU,
                s_cap=args.s_cap,
                s_refs=s_ref,
                s_thres=0,
                s_cm=0)
            stages.append(stage)

        cbits, half_bit = gen.parse_bits(bits[-1])
        n_refs = 3 if half_bit else 2
        tail_meta = gen.StageMeta(cbits, n_refs, half_bit=half_bit, seed=sub_seeds[-1])
        tail = meta.generate_ideal().thres

        adc = gen.PipeParameters(stages, tail)
        adcs.append(adc)

    return adcs


def calibrate_single(args, adc, lsb_scale, seed, snr_ref, snr_thres):
    assert len(args.samples) == 1
    assert args.n_test == 0

    delta_sigma = adc.as_delta_sigma()
    adc_ideal = adc.as_ideal()
    ideal_delta_sigma = adc.as_delta_sigma()

    calibrated_stages = []
    trully_calibrated = True

    with misc.push_random_state():
        np.random.seed(seed)
        seeds = [np.random.randint(0, 4294967296) for _ in range(len(delta_sigma))]

    for_iter = zip(seeds, delta_sigma, ideal_delta_sigma, adc_ideal.stages)
    for seed, ds_stage, ideal, pipe_ideal in for_iter:
        meta = ds_stage.meta
        ins = np.zeros((0, meta.n_diff,))
        cargs = copy.deepcopy(args)
        cargs.seed = seed

        conf = make_config(meta, cargs, False)
        tb = sims.StageTestbench.Scalar(ds_stage, ins, conf)
        simulator = sims.Simulator(seed, snr_ref, snr_thres)

        if DRY_RUN:
            calibrated = copy.deepcopy(ds_stage)

        else:
            codes, _ = tb.simulate(simulator)

            system = cal.CalibrationSystem(ideal, conf, codes,
                mask=cal.json_dicts_to_mask(ideal, [{"parameter": "thres"}]),
                sum_conf=True,
                use_bands=True)

            calibrated, _ = system.run_calibration(samples_step=args.samples[0]+1, lsb_scale=lsb_scale)

            x = system.map_in(calibrated, ins)
            trully_calibrated = (trully_calibrated
                and system.system(x, scalar=True, lsb_scale=lsb_scale) == 0)

        calibrated._thres = pipe_ideal.thres
        calibrated_stages.append(calibrated)

    calibrated = gen.PipeParameters(calibrated_stages, adc_ideal.tail)
    return calibrated, trully_calibrated


def characterize_point(args, adcs, seed, lsb_scale, snr_ref, snr_thres):
    calibrated = [calibrate_single(args, adc, lsb_scale, seed, snr_ref, snr_thres)[0] for adc in adcs]
    per_snr = [an.snr(adc) for adc in adcs]
    cal_snr = [an.snr(adc, cc) for adc, cc in zip(adcs, calibrated)]
    nav_snr = [an.snr(adc, adc.as_ideal()) for adc in adcs]

    return calibrated, per_snr, cal_snr, nav_snr


def characterize_uncertanty(args, adcs_sweep, seed, est_thres_s):
    shape = np.shape(adcs_sweep)
    shape = (shape[0], np.size(est_thres_s), shape[1],)

    real = np.stack((adcs_sweep,)*shape[1], axis=1)
    calibrated = np.copy(real)
    per_snr = np.zeros(shape)
    cal_snr = np.zeros(shape)
    nav_snr = np.zeros(shape)

    confidence = args.confidence
    sigmas = np.sqrt(2)*erfinv(confidence)

    # Sweep thres
    for idx in cartesian(*tuple(range(s) for s in shape[:-1])):
        e_thres_s = est_thres_s[idx[1]]

        idx = idx + (slice(None,),)

        lsb_scale = 1 + 2*(e_thres_s*sigmas)
        mod_adcs = real[idx].tolist()

        calibrated_, per_snr_, cal_snr_, nav_snr_ = \
            characterize_point(args, mod_adcs, seed, lsb_scale, 0, 0)

        calibrated[idx] = np.array(calibrated_, dtype=object)
        per_snr[idx] = np.array(per_snr_)
        cal_snr[idx] = np.array(cal_snr_)
        nav_snr[idx] = np.array(nav_snr_)

    return real, calibrated, per_snr, cal_snr, nav_snr


def characterize_noise(args, adcs_sweep, seed, snr_noise, est_thres_s, ref, match_noise=False):
    shape = np.shape(adcs_sweep)
    shape = (shape[0], np.size(snr_noise), shape[1],)

    real = np.stack((adcs_sweep,)*shape[1], axis=1)
    calibrated = np.copy(real)
    per_snr = np.zeros(shape)
    cal_snr = np.zeros(shape)
    nav_snr = np.zeros(shape)

    confidence = args.confidence
    sigmas = np.sqrt(2)*erfinv(confidence)

    # Sweep thres
    for idx in cartesian(*tuple(range(s) for s in shape[:-1])):
        e_thres_s = est_thres_s[idx[0]]
        noise = snr_noise[idx[1]]

        idx = idx + (slice(None,),)

        lsb_scale = (1 + 2*(e_thres_s*sigmas)) if match_noise else 1
        mod_adcs = real[idx].tolist()

        aargs = (noise, 0,) if ref else (0, noise,)
        calibrated_, per_snr_, cal_snr_, nav_snr_ = \
            characterize_point(args, mod_adcs, seed, lsb_scale, *aargs)

        calibrated[idx] = np.array(calibrated_, dtype=object)
        per_snr[idx] = np.array(per_snr_)
        cal_snr[idx] = np.array(cal_snr_)
        nav_snr[idx] = np.array(nav_snr_)

    return real, calibrated, per_snr, cal_snr, nav_snr


def characterize(args, adcs, seed):

    with misc.push_random_state():
        np.random.seed(seed)
        seeds = [np.random.randint(0, 4294967296) for _ in range(3)]

    assert not args.relative_snr_ref, "TODO implement relative"
    assert not args.relative_snr_thres, "TODO implement relative"

    min_snr_ref_v = args.min_snr_ref_v
    min_snr_thres_v = args.min_snr_thres_v

    fsr = adcs[0].stages[0].meta.fsr
    n_bits = int(np.sum([np.floor(stage.meta.n_bits) for stage in adcs[0].stages]))
    n_bits += int(gen.infer_thres_bits(adcs[0].tail)[0])

    lsb = gen.compute_lsb(n_bits, *fsr)

    if min_snr_ref_v is None:
        min_snr_ref_v = lsb/2

    if min_snr_thres_v is None:
        min_snr_thres_v = lsb/2

    snr_ref_inv = np.linspace(0, min_snr_ref_v, args.samples_snr_ref)
    snr_thres_inv = np.linspace(0, min_snr_thres_v, args.samples_snr_thres)

    snr_ref = np.power((fsr[1] - fsr[0]), 2)/np.power(snr_ref_inv, 2)
    snr_thres = np.power((fsr[1] - fsr[0]), 2)/np.power(snr_thres_inv, 2)

    snr_ref[0] = 0
    snr_thres[0] = 0

    real_thres_s = data.at_least_ndarray(args.real_thres_s)

    shape = (np.size(real_thres_s), np.size(adcs),)
    adcs_sweep = np.tile(np.array(adcs, dtype=object), shape[:-1] + (1,))

    # Sweep thres
    with misc.push_random_state():
        np.random.seed(seed)
        for idx in cartesian(*tuple(range(s) for s in shape)):
            r_thres_s = real_thres_s[idx[0]]
            adcs_sweep[idx] = copy.deepcopy(adcs_sweep[idx])
            for stage in adcs_sweep[idx].stages:
                r_thres_s_local = r_thres_s * stage.meta.lsb
                stage._thres = np.random.normal(stage.thres, r_thres_s_local)
            adcs_sweep[idx]._tail = np.random.normal(adcs_sweep[idx].tail, r_thres_s)

    uncertain = characterize_uncertanty(args, adcs_sweep, seed, real_thres_s)
    ref = characterize_noise(args, adcs_sweep, seed, snr_ref, real_thres_s, ref=True)
    thres = characterize_noise(args, adcs_sweep, seed, snr_thres, real_thres_s, ref=False)

    return uncertain, ref, thres


def prepickle_data(dat, path_context, directory, name):
    real, calibrated, per_snr, cal_snr, nav_snr = dat

    real_str = np.empty_like(real)
    calibrated_str = np.empty_like(calibrated)

    for idx in cartesian(*tuple(range(s) for s in np.shape(real))):
        lname = "{}[{}]".format(name, ','.join(str(ii) for ii in idx))

        real_location = data.DataLocation(path_context, directory, lname + ".real", ".json")
        real_str[idx] = data.save(real[idx], real_location)

        calibrated_location = data.DataLocation(path_context, directory, lname + ".calibrated", ".json")
        calibrated_str[idx] = data.save(calibrated[idx], calibrated_location)

    return real_str, calibrated_str, per_snr, cal_snr, nav_snr


def postpickle_data(dat, path_context):
    if dat is None:
        return None

    real_str, calibrated_str, per_snr, cal_snr, nav_snr = dat

    real = np.empty_like(real_str)
    calibrated = np.empty_like(calibrated_str)

    memo = {}
    for idx in cartesian(*tuple(range(s) for s in np.shape(real_str))):
        real_path = real_str[idx]
        real_dl = data.DataLocation.Parse(real_path, path_context, None_on_fail=False)
        real[idx] = data.load(gen.PipeParameters, real_dl, memo)

        calibrated_path = calibrated_str[idx]
        calibrated_dl = data.DataLocation.Parse(calibrated_path, path_context, None_on_fail=False)
        calibrated[idx] = data.load(gen.PipeParameters, calibrated_dl, memo)

    return real, calibrated, per_snr, cal_snr, nav_snr


def plot2d(name, dat, name_axes, x_axes=None, y_axes=None, noise=False, hide_constants=False, ax=None, color=None, first_black=False, legend_handlers=None, index=0, def_color=(0.0,)*3):
    if color is None:
        color = def_color

    real, calibrated, per_snr, cal_snr, nav_snr = dat

    per_snr = an.snr_to_enob(per_snr)
    cal_snr = an.snr_to_enob(cal_snr)
    nav_snr = an.snr_to_enob(nav_snr)

    per_snr_m = np.mean(per_snr, axis=-1)
    cal_snr_m = np.mean(cal_snr, axis=-1)
    nav_snr_m = np.mean(nav_snr, axis=-1)

    per_snr_s = np.std(per_snr, axis=-1)
    cal_snr_s = np.std(cal_snr, axis=-1)
    nav_snr_s = np.std(nav_snr, axis=-1)

    savefig = ax is None

    if ax is None:
        f = plots.figure()
        ax = f.add_subplot(111)

    else:
        f = ax.get_figure()

    n = np.size(per_snr_m, 1)

    color_per = def_color
    color_nav = def_color

    def pplot(ax, mean, std, idx, color, label, ranges=False, apply_off=False, hatch=None, linestyle=None):
        if apply_off:
            styles = (('////', None,), ('///', '--'), ('//', '-.'), ('/', ':'))

            if index == 1:
                styles = (('///', None,), ('//', '--'), ('/', '-.'),)
            elif index == 2:
                styles = (('\\\\\\', None,), ('\\\\', '--'), ('\\', '-.'),)

            hatch, linestyle = styles[idx[1] % len(styles)]

        # l_color = whithen(color, idx[1], max_=0.8, apply_off=apply_off)
        lx_axes_none = x_axes is None
        lx_axes = np.arange(np.size(mean[idx])) if lx_axes_none else x_axes
        if ranges:
            lx_axes = [-1] if lx_axes_none else (lx_axes[0] - (0.05)*(lx_axes[-1] - lx_axes[0]),)
            h = ax.scatter(lx_axes, [mean[idx][0]], color=l_color, label=label)
            ax.scatter(lx_axes, [mean[idx][0] + std[idx][0]], color=l_color, marker='+')
            ax.scatter(lx_axes, [mean[idx][0] - std[idx][0]], color=l_color, marker='+')

        else:
            h = ax.plot(lx_axes, mean[idx], color=l_color, label=label, linestyle=linestyle, zorder=3)

#            x_mean_diff = np.mean(np.diff(lx_axes))*0.05
#            x_dup = np.array([lx_axes - x_mean_diff, lx_axes + x_mean_diff])
#
#            ax.plot(x_dup, np.array([mean[idx] + std[idx]]*2), color=l_color, linestyle=linestyle, zorder=1)
#            ax.plot(x_dup, np.array([mean[idx] - std[idx]]*2), color=l_color, linestyle=linestyle, zorder=1)

            if PLOT_STD:
                ll_color = l_color + (0.2,)
                facecolor = None if hatch is None else 'none'

                high = mean[idx] + std[idx]
                low = mean[idx] - std[idx]
                hatch = None # FIXME
                polys = ax.fill_between(lx_axes, low, high, color=ll_color, hatch=hatch, zorder=1, linewidth=0.0)
                if hatch is not None:
                    polys.set_facecolor('none')

                ll_color = l_color + (0.4,)
                ax.plot(lx_axes, high, color=ll_color, zorder=2, linestyle=linestyle)
                ax.plot(lx_axes, low, color=ll_color, zorder=2, linestyle=linestyle)


            # ax.plot(*aargs, mean[idx] + std[idx], color=color, linestyle='--')
            # ax.plot(*aargs, mean[idx] - std[idx], color=color, linestyle='--')

            h = h[0]

        return h, label

    plot_legend = legend_handlers is None
    legend_handlers = [] if legend_handlers is None else legend_handlers
    for ii in reversed(range(n)):
        if ii == 0 and first_black is None:
            continue

        idx = (slice(None), ii,)
        if noise and y_axes is not None:
            y_value = y_axes[ii]
            if y_value == 0:
                y_value = "noiseless"
            else:
                y_value = "{:0.1f} dB".format(an.magnitude_to_db(1/y_value))
        elif y_axes is not None:
            y_value = "{:0.2f}".format(y_axes[ii])
        else:
            y_value = ii

        legend = "{}: {}".format(name_axes[1], y_value)
        l_color = color
        if ii == 0 and first_black is True:
            legend = "Modeled std 0, no noise"
            l_color = def_color
        h = pplot(ax, cal_snr_m, cal_snr_s, idx, l_color, legend, apply_off=True)
        legend_handlers.append(h)

        if ii == 0:
            if not hide_constants:
                h0 = pplot(ax, nav_snr_m, nav_snr_s, idx, color_nav, "No calibration", hatch='//', linestyle='--')
                h1 = pplot(ax, per_snr_m, per_snr_s, idx, color_per, "Perfect knowledge", hatch='/', linestyle='-.')

                legend_handlers.append(h0)
                legend_handlers.append(h1)

    ax.set_xlabel(name_axes[0])
    ax.set_ylabel("ENOB")

    if plot_legend:
        legend_handlers = reversed(legend_handlers)
        paths, names = tuple(zip(*legend_handlers))
        f.legend(paths, names)

    if savefig:
        f.savefig("{}.png".format(name))
        plots.show()

    return f, ax, legend_handlers


def run(args):
    assert len(args.bits) == len(args.seed)

    directory = args.location
    mkdir(misc.Namespace(directory=directory, mode="750"))
    path_context = data.PathContext.relative()

    names = ["paper_set_{}{}".format(ii, "_DRY" if DRY_RUN else "") for ii in range(len(args.bits))]

    if not JUST_PLOT:
        for ii, bits, seed, name in zip(count(), args.bits, args.seed, names):
            print("SET {}/{}".format(ii+1, len(args.bits)))
            adcs = gen_adc(bits, seed, args, args.n_adcs)
            uncertain, ref, thres = characterize(args, adcs, seed)

            if 0 in TESTS:
                uncertain = prepickle_data(uncertain, path_context, directory, "{}.{}".format(name, "uncertain"))
            else:
                uncertain = None

            if 1 in TESTS:
                ref = prepickle_data(ref, path_context, directory, "{}.{}".format(name, "ref"))
            else:
                ref = None

            if 2 in TESTS:
                thres = prepickle_data(thres, path_context, directory, "{}.{}".format(name, "thres"))
            else:
                thres = None

            with open(name, 'wb') as f:
                pickle.dump((uncertain, ref, thres,), f, protocol=pickle.HIGHEST_PROTOCOL)

    for name in names:
        with open(name, 'rb') as f:
            dat = pickle.load(f)
            uncertain, ref, thres = dat

            uncertain = postpickle_data(uncertain, path_context)
            ref = postpickle_data(ref, path_context)
            thres = postpickle_data(thres, path_context)

            dat = (uncertain, ref, thres,)

            real_thres_s = data.at_least_ndarray(args.real_thres_s)

            sample_adc = dat[TESTS[0]][0][0,0,0]
            fsr = sample_adc.stages[0].meta.fsr
            n_bits = int(np.sum([np.floor(stage.meta.n_bits) for stage in sample_adc.stages]))
            n_bits += int(gen.infer_thres_bits(sample_adc.tail)[0])

            lsb = gen.compute_lsb(n_bits, *fsr)

            min_snr_ref_v = args.min_snr_ref_v
            min_snr_thres_v = args.min_snr_thres_v

            if min_snr_ref_v is None:
                min_snr_ref_v = lsb/2

            if min_snr_thres_v is None:
                min_snr_thres_v = lsb/2

            snr_ref_inv = np.linspace(0, min_snr_ref_v, args.samples_snr_ref)
            snr_thres_inv = np.linspace(0, min_snr_thres_v, args.samples_snr_thres)

            u_args = ("{}.uncertain".format(name), uncertain, ("Real threshold voltage standard deviation (LSB)", "Modeled std (LSB)",),)
            u_kwargs = {"x_axes": real_thres_s, "y_axes": real_thres_s}

            t_args = ("{}.thres".format(name), thres, ("Real threshold voltage standard deviation (LSB)", "Thres. noise (SNR)",),)
            t_kwargs = {"x_axes": real_thres_s, "y_axes": snr_thres_inv, "noise": True}

            r_args = ("{}.ref".format(name), ref, ("Real threshold voltage standard deviation (LSB)", "Ref. noise (SNR)",),)
            r_kwargs = {"x_axes": real_thres_s, "y_axes": snr_ref_inv, "noise": True}

            if TOGETHER:
                f = plots.figure(figsize=(10, 4,) if FULL_SIZE else (5*1.5, 2*1.5,))
                ax = f.add_subplot(111)
                h0 = []
                h1 = []
                h2 = []

                brightness = 1

                if 0 in TESTS:
                    f, ax, h0 = plot2d(*u_args, **u_kwargs, ax=ax, color=(brightness,0,0), first_black=True, index=0, legend_handlers=h0, hide_constants=False)
                    h0 = reversed(h0)

                if 2 in TESTS:
                    f, ax, h2 = plot2d(*t_args, **t_kwargs, ax=ax, color=(0,0,brightness), first_black=None, index=2, legend_handlers=h2)
                    h2 = reversed(h2)

                if 1 in TESTS:
                    f, ax, h1 = plot2d(*r_args, **r_kwargs, ax=ax, color=(0, brightness,0), first_black=None, index=1, legend_handlers=h1)
                    h1 = reversed(h1)

                h = []
                for hh in (h0, h1, h2):
                    h.extend(hh)

                proportion = 0.65
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * proportion, box.height])
                ax.set_facecolor((0.85,)*3)

                legend_handlers = h
                paths, names = tuple(zip(*legend_handlers))

                f.legend(paths, names, loc='center left', bbox_to_anchor=(proportion, 0.5))
                prev = mpl.rcParams['hatch.linewidth']
                mpl.rcParams['hatch.linewidth'] = 5.0
                f.savefig("{}.png".format(name), dpi=300)

                plots.show()
                mpl.rcParams['hatch.linewidth'] = prev

            else:
                if 0 in TESTS:
                    plot2d(*u_args, **u_kwargs, color=(0.8,0,1), hide_constants=False)

                if 1 in TESTS:
                    plot2d(*r_args, **r_kwargs, color=(0.8,0,1), hide_constants=False)

                if 2 in TESTS:
                    plot2d(*t_args, **t_kwargs, color=(0.8,0,1), hide_constants=False)

    print("DONE")


if __name__ == "__main__":
    # This was supoused to be built using argparse, but time contraints!
    N_FOR_STD = 4
    SAMPLES = 4
    SAMPLES_THRES = 3
    SAMPLES_REF = 3

    args = misc.Namespace()

    # Generation args
    args.bits = [[3.5, 3.5, 2.5, 4]]
    # args.bits = [4.5, 3.5, 3.5, 4]
    args.seed = [486582]

    args.s_cap = 0.0025
    args.s_ref = None

    # Characterization args
    args.real_thres_s = np.linspace(0, 0.5/3, SAMPLES) # In lsb
    args.confidence = 0.95

    args.n_adcs = N_FOR_STD

    args.min_snr_ref_v = gen.compute_lsb(8, 0, 1)/2
    args.min_snr_thres_v = gen.compute_lsb(8, 0, 1)/2

    args.samples_snr_ref = SAMPLES_REF
    args.samples_snr_thres = SAMPLES_THRES

    args.relative_snr_ref = False
    args.relative_snr_thres = False

    # Configuration args
    args.samples = [64]
    args.inputs = "random"
    args.ic = "precharge"
    args.full = False

    args.loop = 1
    args.period = 16
    args.n_test = 0

    # other
    args.location = "paper"

    run(args)
    exit(0)
