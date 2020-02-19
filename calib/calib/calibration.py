#! /usr/bin/env python

import copy
import warnings

from itertools import product as cartesian

import numpy as np

from scipy.optimize import least_squares
from scipy.signal import lfilter

import calib.gen as gen
import calib.data as data
import calib.simulation as sims

from calib.misc import default, Namespace
from calib.misc import multiget as get


PRINT_ERROR = False
STRICT = False
REBUNDANT_CHECKS = False
DEBUG = False


def json_dicts_to_mask(stage_reference, dcts, invert=True):
    result = {}
    attr_map = {"cm": "common_mode"}

    for dct in dcts:
        param = dct["parameter"]

        if param not in result:
            attr = getattr(stage_reference, attr_map.get(param, param))
            result[param] = np.ones_like(attr, dtype=bool)

        try:
            index = tuple(dct["index"])
            result[param][index] = False

        except KeyError:
            assert len(dct) == 1
            result[param][...] = False

    if not invert:
        for param in CalibrationSystem.MASK_KEYS:
            if param not in result:
                attr = getattr(stage_reference, attr_map.get(param, param))
                result[param] = np.zeros_like(attr, dype=bool)

            else:
                result[param] = np.logical_not(result[param])

    return result


class CalibrationSystem(object):

    MASK_KEYS = ("eff", "caps", "refs", "thres", "cm",)

    @property
    def codes(self):
        return self._codes

    @property
    def configuration_sequence(self):
        return self._configuration_sequence

    def __init__(self, start_stage, configuration_sequence, codes, mask=None, sum_conf=False, use_bands=False):
        assert isinstance(configuration_sequence, gen.ConfigurationSequence), "Check scalar testbench is being used."
        assert isinstance(start_stage, gen.StageParameters), "Check scalar testbench is being used."

        self._start = start_stage
        self._codes = codes
        self._configuration_sequence = configuration_sequence

        self._mask = default(mask, {})
        assert all(key in self.MASK_KEYS for key in self._mask.keys())

        self.sum_conf = sum_conf
        self.use_bands = use_bands

    def _compute_u_center(self, thres):
        assert len(np.shape(thres)) == 2, "Need multiple thres instances."

        codes = self.codes

        fsr = self._start.meta.fsr
        half_bit = gen.infer_thres_bits(thres[0, ...])[1]
        lsb_ext = np.array([[gen.infer_thres_lsb(thres[ii, ...], fsr)] for ii in range(np.size(thres, 0))])

        half_factor = 3/2 if half_bit else 1

        extremes = (thres[:, :1, ...] - half_factor * (lsb_ext/2),
                    thres[:, -1:, ...] + half_factor * (lsb_ext/2),)

        lsb = np.diff(thres, axis=1)
        mid = thres[:, :-1, ...] + lsb/2

        center_map = np.concatenate((extremes[0], mid, extremes[1],), axis=1)
        lsb = np.concatenate((lsb_ext * half_factor, lsb, lsb_ext * half_factor,), axis=1)

        def evaluate(arr):
            return np.transpose(arr[:, codes], (1,0,2,))

        return evaluate(center_map), evaluate(lsb)

    def _multi_mask(self):
        cm_shape = np.shape(self._start.common_mode)
        eff_shape = np.shape(self._start.eff)
        caps_shape = np.shape(self._start.caps)
        refs_shape = np.shape(self._start.refs)
        thres_shape = np.shape(self._start.thres)

        masks = self._mask

        cm_mask   = np.broadcast_to(masks.get("cm", True), cm_shape)
        eff_mask  = np.broadcast_to(masks.get("eff", True), eff_shape)
        caps_mask = np.broadcast_to(masks.get("caps", True), caps_shape)
        refs_mask = np.broadcast_to(masks.get("refs", True), refs_shape)
        thres_mask = np.broadcast_to(masks.get("thres", True), thres_shape)

        return cm_mask, eff_mask, caps_mask, refs_mask, thres_mask

    def _flat_mask(self):
        cm_mask, eff_mask, caps_mask, refs_mask, thres_mask = self._multi_mask()

        result = []
        result.extend(cm_mask.flatten().tolist())
        result.extend(eff_mask.flatten().tolist())
        result.extend(caps_mask.flatten().tolist())
        result.extend(refs_mask.flatten().tolist())
        result.extend(thres_mask.flatten().tolist())

        return result

    def bounds(self, ins):
        cm_mask, eff_mask, caps_mask, refs_mask, thres_mask = self._multi_mask()
        ins_mask = np.ones_like(ins, dtype=bool)

        ni = -np.inf
        pi = np.inf

        low = []
        high = []

        def broadcast_and_mask(value, mask):
            return np.full_like(mask, value, dtype=float)[mask].flatten().tolist()

        low.extend( broadcast_and_mask(ni, cm_mask))
        high.extend(broadcast_and_mask(pi, cm_mask))

        low.extend( broadcast_and_mask(0, eff_mask))
        high.extend(broadcast_and_mask(1, eff_mask))

        low.extend( broadcast_and_mask(0,  caps_mask))
        high.extend(broadcast_and_mask(pi, caps_mask))

        low.extend( broadcast_and_mask(ni, refs_mask))
        high.extend(broadcast_and_mask(pi, refs_mask))

        low.extend( broadcast_and_mask(ni, thres_mask))
        high.extend(broadcast_and_mask(pi, thres_mask))

        low.extend( broadcast_and_mask(ni, ins_mask))
        high.extend(broadcast_and_mask(pi, ins_mask))

        return (low, high,)

    def map_in(self, stage, ins):
        cm = stage.common_mode
        eff = stage.eff
        caps = stage.caps
        refs = stage.refs
        thres = stage.thres

        result = []

        assert stage.meta == self._start.meta
        assert len(np.shape(ins)) == 2
        assert stage.meta.n_diff == np.size(ins, -1)

        result.extend(cm.flatten().tolist())
        result.extend(eff.flatten().tolist())
        result.extend(caps.flatten().tolist())
        result.extend(refs.flatten().tolist())
        result.extend(thres.flatten().tolist())

        if len(self._mask) > 0:
            mask = self._flat_mask()
            result = [r for r, m in zip(result, mask) if m]

        result.extend(ins.flatten().tolist())

        return result

    def _x_offsets(self):
        s_stage = self._start
        mask = self._mask

        def mask_or_default(param, arr):
            return np.sum(mask[param]) if param in mask else np.size(arr)

        eff_off = mask_or_default("cm", s_stage.common_mode)
        caps_off = mask_or_default("eff", s_stage.eff) + eff_off
        refs_off = mask_or_default("caps", s_stage.caps) + caps_off
        thres_off = mask_or_default("refs", s_stage.refs) + refs_off
        ins_off = mask_or_default("thres", s_stage.thres) + thres_off

        return eff_off, caps_off, refs_off, thres_off, ins_off

    def map_out(self, x):
        assert np.size(x, 0) == 1

        cm, eff, caps, refs, thres, ins = self._map_internal(x)

        cm = cm[0, ...]
        eff = eff[0, ...]
        caps = caps[0, ...]
        refs = refs[0, ...]
        thres = thres[0, ...]
        ins = ins[0, ...]

        s_stage = self._start
        stage = gen.StageParameters(s_stage.meta, eff, caps, refs, thres, cm)

        return stage, ins

    def _map_internal(self, x):
        x = data.at_least_ndarray(x)
        assert len(np.shape(x)) == 2

        eff_off, caps_off, refs_off, thres_off, ins_off = self._x_offsets()

        n_x = np.size(x, 0)
        s_stage = self._start

        if len(self._mask) > 0:

            cm = np.stack((np.copy(s_stage.common_mode),)*n_x, axis=0)
            eff = np.stack((np.copy(s_stage.eff),)*n_x, axis=0)
            caps = np.stack((np.copy(s_stage.caps),)*n_x, axis=0)
            refs = np.stack((np.copy(s_stage.refs),)*n_x, axis=0)
            thres = np.stack((np.copy(s_stage.thres),)*n_x, axis=0)

            cm_mask, eff_mask, caps_mask, refs_mask, thres_mask = self._multi_mask()
            for ii in range(n_x):
                if cm_mask.item():
                    cm  [ii]        = x[ii,         :eff_off ]
                if eff_mask.item():
                    eff [ii]        = x[ii,  eff_off:caps_off]
                caps[ii, ...][caps_mask] = x[ii, caps_off:refs_off]
                refs[ii, ...][refs_mask] = x[ii, refs_off:thres_off ]
                thres[ii, ...][thres_mask] = x[ii, thres_off:ins_off ]

        else:
            cm   = np.reshape(x[:,         :eff_off ], (n_x,) + np.shape(s_stage.common_mode))
            eff  = np.reshape(x[:, eff_off :caps_off], (n_x,) + np.shape(s_stage.eff))
            caps = np.reshape(x[:, caps_off:refs_off], (n_x,) + np.shape(s_stage.caps))
            refs = np.reshape(x[:, refs_off:thres_off ], (n_x,) + np.shape(s_stage.refs))
            thres = np.reshape(x[:, thres_off:ins_off ], (n_x,) + np.shape(s_stage.thres))

        n_diff = s_stage.meta.n_diff
        ins = np.array(x[:, ins_off:])
        ins = np.reshape(ins, (np.size(ins, 0), np.size(ins, 1)//n_diff, n_diff,))

        return cm, eff, caps, refs, thres, ins

    def system(self, x, scalar=False, use_bands=None, sum_conf=None, limit_samples=None, lsb_scale=1):
        if scalar:
            x = [x]

        cm, eff, caps, refs, thres, ins = self._map_internal(x)
        ext_codes = self._codes[:, np.newaxis, ...]

        if not hasattr(self, "_cache"):
            self._cache = {}

        base_shape = np.shape(x)[:-1]
        if base_shape not in self._cache:
            print("Generating cache for {}".format(base_shape))
            self._cache[base_shape] = self.recreate_cache(
                eff, caps, refs, thres, ins, cm,
                self._configuration_sequence, ext_codes)

        use_bands = default(use_bands, self.use_bands)
        sum_conf = default(sum_conf, self.sum_conf)

        ext_codes = self._codes[:, np.newaxis, ...]

        u = self.recreate(
            eff, caps, refs, thres, ins, cm,
            self._configuration_sequence, self._cache[base_shape],
            limit_samples=limit_samples)

        meta = self._configuration_sequence.meta
        # Ignore last u since it is not converted to code
        u = np.diff(u[:-1, :, ...], axis=-1) if meta.differential else u[:-1, :, ..., -1]
        samples = np.size(u, 0)

        u_center, lsb = self._compute_u_center(thres)
        thres_bits, thres_half = gen.infer_thres_bits(thres[0, ...])
        ideal_lsb = gen.compute_lsb(thres_bits, *self._start.meta.fsr, thres_half)

        lsb = lsb + ideal_lsb * (lsb_scale - 1) if use_bands else np.zeros((samples, 1, 1,))
        result = np.maximum(np.abs(u - u_center[:samples, ...]) - lsb[:samples, ...]/2, 0)

        if DEBUG: # FIXME
            import matplotlib.pyplot as plots
            try:
                print(np.shape(result))
                sum_axis = tuple(range(len(np.shape(result)) - 1))
                cut = (np.sum(result, axis=sum_axis) > 0).tolist().index(True)
                print("Found cut {}".format(cut))

            except:
                cut = 10
                print("Not found cut")

            x_thres = np.stack((np.zeros_like(thres[0, :]), np.full_like(thres[0, :], samples),), axis=0)
            y_thres = np.stack((thres[0, :],)*2, axis=0)
            plots.plot(x_thres, y_thres, c='orange', linestyle='-.')
            plots.plot(u[:, 0 , cut,...], 'r')
            internal, _ = self.configuration_sequence.configuration_sets[0].generate_in(self.configuration_sequence.configuration_sets[0].ds_samples)
            internal = np.sum(internal, axis=(-2, -1,))
            plots.plot(0.25*(internal[:, cut, ...] - 1.5), 'cyan')
            #plots.plot(internal[:samples, cut, ...], 'k')

            adj_codes = (0.5/(np.max(self._codes)))*(self._codes - np.max(self._codes)/2)
            plots.plot(adj_codes[:samples, cut, ...], 'k')
            plots.plot(u_center[:samples, 0, cut, ...], 'b')
            plots.plot(u_center[:samples, 0, cut, ...] + lsb[:samples, 0, cut, ...]/2, 'g', linestyle='--')
            plots.plot(u_center[:samples, 0, cut, ...] - lsb[:samples, 0, cut, ...]/2, 'g', linestyle='--')

            plots.show()

        if scalar:
            result = result[:, 0, ...]

        else:
            result = np.transpose(result, (1, 0,) + tuple(range(2, len(np.shape(result)))))

        sum_axis = (-2, -1,) if sum_conf else (-2,)
        result = np.sum(result, axis=sum_axis)

        if PRINT_ERROR:
            print("System error: {}".format(np.sum(np.power(result, 2))))

        return result

    @staticmethod
    def recreate_cache(eff, caps, refs, thres, ins, common_mode, c_seq, codes, scalar=None):
        fun = sims.Simulator

        scalar = default(scalar, len(np.shape(eff)) == 0)
        if scalar:
            codes = codes[:, np.newaxis, ...]

        data = fun.simulate_setup(
            eff,
            caps,
            refs,
            thres,
            ins,
            common_mode,
            c_seq,
            scalar=scalar )

        ext_dct = get(data, "extended")[0]
        eff, thres, cm = get(ext_dct, "eff", "thres", "cm")

        # shape (..., n_conf, n_diff,)
        seq_idx = fun.init_seq_idx(c_seq, data)
        set_data = None
        du_idx = None
        trans_idx = None

        sets_cache = []
        cache = {"data": data, "seq_idx": seq_idx, "sets": sets_cache}

        for ii_set, c_set in enumerate(c_seq.configuration_sets):
            set_data = fun.init_set(c_set, data, set_data, du_idx)

            if set_data["previous"] is not None:
                ds_offset = data["ds_offset"]
                code = codes[ds_offset:ds_offset+1, ...]
                trans_idx = fun.transition_step_idx(c_set, set_data, data, code)

            ds_offset = data["ds_offset"]
            code = codes[ds_offset:ds_offset+c_set.ds_samples, ...]
            du_idx = fun.du_indexes(code, c_set, set_data, data)
            sets_cache.append((set_data, trans_idx, du_idx,))

        return cache

    @staticmethod
    def recreate(eff, caps, refs, thres, ins, common_mode, c_seq, cache, scalar=None, limit_samples=None):
        fun = sims.Simulator
        scalar = default(scalar, len(np.shape(eff)) == 0)
        limit_samples = default(limit_samples, c_seq.samples) + 1
        samples = 0

        cache = dict(cache)
        cache["data"] = fun.simulate_setup(
            eff,
            caps,
            refs,
            thres,
            ins,
            common_mode,
            c_seq,
            scalar=scalar )
        data = cache["data"]
        seq_idx = cache["seq_idx"]

        u_history = []

        # shape (..., n_conf, n_diff,)
        u = fun.init_seq(seq_idx, data)
        u_history.append(u)
        samples += 1

        # Simulate each configuration set
        dct_idx, dct_ext, dct_n = get(data, "indexing", "extended", "n")

        n_conf, n_diff = get(dct_n, "n_conf", "n_diff")
        eff, cm = get(dct_ext, "eff", "cm")
        base_shape, base_len = get(dct_idx, "base_shape", "base_len")

        c_sets = c_seq.configuration_sets

        def multi_idx_filter(idx_tuple, sub_idx):
            return tuple(ii[sub_idx] if hasattr(ii, "__getitem__") else ii
                for ii in idx_tuple)

        for ii_set, c_set, data_trans_du in zip(range(len(c_sets)), c_sets, cache["sets"]):
            set_data, trans_idx, du_idx = data_trans_du

            if set_data["previous"] is not None:
                u = fun.transition_step(trans_idx, u[-1:, ...], set_data, data)
                u_history.append(u)
                samples += 1

            r_du = (1-eff)*cm
            n_u = np.empty((c_set.ds_samples,) + base_shape + (n_conf, n_diff,))

            local_du_idx = dict(du_idx)
            # used in u
            du = fun.du_compute(du_idx, c_set, set_data, data)

            for idx in cartesian(*tuple(range(ss) for ss in base_shape)):
                local_idx = tuple(slice(ii, ii+1) for ii in idx) + (Ellipsis,)
                ext_local_idx = (slice(None),) + local_idx

                local_du_idx["r_ref"] = local_du_idx["r_ref"][ext_local_idx]
                local_du_idx["m_in_ref"] = multi_idx_filter(local_du_idx["m_in_ref"], ext_local_idx)
                local_du_idx["m_in_ins"] = multi_idx_filter(local_du_idx["m_in_ins"], ext_local_idx)

                zi = u[(slice(-1, None),) + local_idx]*eff[local_idx]
                local_u = lfilter([1], [1, -eff[local_idx].item()],
                    du[ext_local_idx] + r_du[(np.newaxis,) + local_idx], zi=zi, axis=0)[0]
                n_u[ext_local_idx] = local_u

            u = n_u
            u_history.append(u)
            samples += np.size(u, 0)

            if samples >= limit_samples:
                break

        u_history = np.concatenate(u_history, axis=0)
        u_history = u_history[:limit_samples, ...]

        if scalar:
            assert np.size(u_history, 1) == 1
            u_history = u_history[:, 0, ...]

        return u_history

    def clear_cache(self):
        if hasattr(self, "_cache"):
            del self._cache

    def default_ins(self, stage, ins):
        if ins is None:
            n_test = self._configuration_sequence.n_ins
            n_diff = stage.meta.n_diff
            ins = np.zeros((n_test, n_diff,))

        return ins

    def run_calibration(self, start_stage=None, start_ins=None, n_switches=5, switching_nfev=30, samples_step=512, lsb_scale=1):
        stage = copy.deepcopy(default(start_stage, self._start))
        ins = self.default_ins(stage, start_ins)

        x = self.map_in(stage, ins)
        bounds = self.bounds(ins)

        nfev = switching_nfev
        steps = self._configuration_sequence.samples // samples_step
        steps = [ss*samples_step for ss in range(1, steps)] + [None]

        for limit_samples in steps:
            res = Namespace()
            res.cost = 1

            for try_ in range(n_switches):
                if res.cost == 0:
                    break

                print(" No bands, no bounds {}/{}, limit {}".format(try_+1, n_switches, limit_samples))
                res = least_squares(lambda x: self.system(x, scalar=True, use_bands=False, limit_samples=limit_samples, lsb_scale=lsb_scale),
                    x, max_nfev=nfev, verbose=2) # FIXME: verbose
                x = res.x

                # Clip
                x = [max(min(xx, M), m) for xx, m, M in zip(x, bounds[0], bounds[1])]

                print(" Bands {}/{}, limit {}".format(try_+1, n_switches, limit_samples))
                res = least_squares(lambda x: self.system(x, scalar=True, use_bands=True, limit_samples=limit_samples, lsb_scale=lsb_scale),
                    x, bounds=bounds, max_nfev=nfev, verbose=2) # FIXME: verbose
                    #x, max_nfev=nfev, verbose=2) # FIXME: bounds
                x = res.x

        if res.cost != 0:
            print(" Final no bounds")
            res = least_squares(lambda x: self.system(x, scalar=True, use_bands=True, limit_samples=limit_samples, lsb_scale=lsb_scale),
                x, verbose=2) # FIXME: verbose
            x = res.x

            # Clip
            x = [max(min(xx, M), m) for xx, m, M in zip(x, bounds[0], bounds[1])]

            print(" Final")
            res = least_squares(lambda x: self.system(x, scalar=True, use_bands=True, limit_samples=limit_samples, lsb_scale=lsb_scale),
                x, bounds=bounds, verbose=2) # FIXME: verbose
            x = res.x

        stage, ins = self.map_out([x])
        return stage, ins

    @staticmethod
    def limit_system(system, max_vectorial=None):
        if max_vectorial is not None:
            old_system = system

            def system(x, *args, scalar=None, **kwargs):
                assert scalar is not None
                if scalar:
                    return old_system(x, *args, scalar=scalar, **kwargs)

                else:
                    n_vec = np.size(x, 0)
                    n_sections = max((n_vec - 1) // max_vectorial, 0)

                    sections = [slice(ii*max_vectorial, (ii+1)*max_vectorial)
                                for ii in range(n_sections)]
                    sections = sections + [slice(n_sections*max_vectorial, None)]
                    results = []
                    for sec in sections:
                        results.append(old_system(x[sec, ...], *args, scalar=scalar, **kwargs))

                    return np.concatenate(tuple(results), axis=0)

        return system

    def refine(self, start_stage=None, start_ins=None, n_iter=None, max_vectorial=None, lsb_scale=1):
        stage = copy.deepcopy(default(start_stage, self._start))
        ins = self.default_ins(stage, start_ins)

        cm = stage.common_mode
        eff = stage.eff
        caps = stage.caps

        caps_off = np.size(cm) + np.size(eff)
        refs_off = caps_off + np.size(caps)

        norm_range = (caps_off, refs_off,)

        x = self.map_in(stage, ins)

        def system(x, scalar):
            return self.system(x, scalar, use_bands=True, sum_conf=True, lsb_scale=lsb_scale)

        system = self.limit_system(system, max_vectorial)

        x = refine(system, x, norm_range=norm_range, n_iter=n_iter)
        stage, ins = self.map_out([x])
        return stage, ins

    def get_slack(self, stage=None, ins=None, max_vectorial=None, ftol=None, lsb_scale=1):
        stage = copy.deepcopy(default(stage, self._start))
        ins = self.default_ins(stage, ins)

        def system(x, scalar):
            return self.system(x, scalar, use_bands=True, sum_conf=True, lsb_scale=lsb_scale)

        system = self.limit_system(system, max_vectorial)

        tol = default(ftol, 1e-12)
        x_scale = 1e-3

        x = self.map_in(stage, ins)
        xap, xan, xbp, xbn, xc = find_ab(system, x, x_scale, tol)

        resp = [self.map_out(xap[ii:ii+1, :])[0] for ii in range(np.size(xap, 0))]
        resn = [self.map_out(xan[ii:ii+1, :])[0] for ii in range(np.size(xan, 0))]
        return list(zip(resp, resn))


def find_not_zero(system, x, direction, tol, d_fact=1.5, max_steps=512):
    if REBUNDANT_CHECKS:
        assert (system(x, scalar=False) - tol <= 0).all()
    prev_x, x = find(system, x, direction, tol, d_fact, max_steps, zero=False)

    if REBUNDANT_CHECKS:
        if STRICT:
            assert (system(x, scalar=False) - tol > 0).all()
        elif not (system(x, scalar=False) - tol > 0).all():
            warnings.warn("Could not find border")

    return prev_x, x


def find(system, x, direction, tol, d_fact, max_steps, zero):
    x = np.copy(x)
    step = 0

    assert len(np.shape(x)) == 2
    x, direction = np.broadcast_arrays(x, direction)

    x = np.copy(x)
    direction = np.copy(direction)
    prev_x = np.copy(x)

    indexes = np.arange(np.size(x, 0))

    cont_cond = (lambda x: x > 0) if zero else (lambda x: x <= 0)

    if REBUNDANT_CHECKS:
        f_sys = system(x, scalar=False) - tol
        cont = cont_cond(f_sys)
        assert cont.all()

    while(np.size(indexes) > 0):
        prev_x[indexes, ...] = x[indexes, ...]
        x[indexes, ...] = x[indexes, ...] + direction[indexes, ...]
        direction[indexes, ...] *= d_fact

        f_sys = system(x[indexes, ...], scalar=False) - tol
        cont = cont_cond(f_sys)

        indexes = indexes[cont]

        step += 1
        if step == max_steps:
            break

    return prev_x, x


def bin_search(system, xa, xb, tol, max_dist=1e-10):
    fa = system(xa, scalar=False) - tol
    fb = system(xb, scalar=False) - tol

    if REBUNDANT_CHECKS:
        if STRICT:
            assert (fa > 0).all()
        assert (fb <= 0).all()

    indexes = np.arange(np.size(xa, 0))

    def cond(xa, xb, indexes):
        return np.linalg.norm(xa[indexes, ...] - xb[indexes, ...], axis=-1) > max_dist

    while(len(indexes) > 0):
        x = 0.5 * (xa[indexes, ...] + xb[indexes, ...])
        fx = system(x, scalar=False) - tol

        mask = fx > 0
        xa[indexes[mask], ...] = x[mask, ...]

        mask = np.logical_not(mask)
        xb[indexes[mask], ...] = x[mask, ...]

        indexes = indexes[cond(xa, xb, indexes)]

    return xa, xb


# @profile
# Returns a, the exterior points and b, the interior points
def find_ab(system, x, x_scale, tol):
    x_len = len(x)
    if REBUNDANT_CHECKS:
        assert system(x, scalar=True) - tol <= 0

    x_scale = np.broadcast_to(x_scale, np.shape(x))

    def gen_dir(idx, val):
        dir_ = np.zeros((x_len,))
        dir_[idx] = val
        return dir_

    xc = np.array(x)

    dirs = np.array([gen_dir(ii,  ss/1.9) for ii, ss in enumerate(x_scale)])
    bp, ap = find_not_zero(system, xc[np.newaxis, ...], dirs, tol)
    bn, an = find_not_zero(system, xc[np.newaxis, ...], -dirs, tol)

    max_dist = min(np.min(x_scale) / 10, 1e-6)

    xap, xbp = bin_search(system, ap, bp, tol, max_dist=max_dist)
    xan, xbn = bin_search(system, an, bn, tol, max_dist=max_dist)

    return xap, xan, xbp, xbn, xc


def compute_movement(system, xap, xan, xbp, xbn, xc, tol, damping=0.15):

    move_fact = (1 - damping)
    dp = xbp - xc[np.newaxis, :]
    dn = xbn - xc[np.newaxis, :]

    displacement = np.sum(dp + dn, axis=0)
    xn = xc + displacement*move_fact

    fn = system(xn, scalar=True) - tol

    while fn > 0:
        assert move_fact < 1
        xp, xn = bin_search(system, xn[np.newaxis, ...], xc[np.newaxis, ...], tol)

        xp = xp[0, ...]
        xn = xn[0, ...]

        xn = xc + (xn - xc) * move_fact
        fn = system(xn, scalar=True) - tol

    return xn


# @profile
def refine(system, x, norm_range=None, n_iter=None, xtol=1e-8, ftol=None):
    tol_fact = 1.0
    tol = ftol

    x_len = len(x)
    n_iter = default(n_iter, x_len*x_len)

    prev_x = x
    x_scale = 1e-1

    norm = np.linalg.norm

    if ftol is None:
        tol = system(x, scalar=True).item() * tol_fact

    for ii in range(n_iter):
        print("-" * 30)
        print("Iter {}, tol: {}".format(ii, tol))
        print("-" * 30)

        xap, xan, xbp, xbn, xc = find_ab(system, x, x_scale, tol)
        prev_x = x
        x = compute_movement(system, xap, xan, xbp, xbn, xc, tol)

        print(x_scale)
        print(x - prev_x)

        if REBUNDANT_CHECKS:
            assert system(x, scalar=True).item() - tol <= 0

        if norm_range is not None:
            slc = slice(norm_range[0], norm_range[1])
            x[slc] /= np.sum(x[slc])

        if REBUNDANT_CHECKS:
            assert system(x, scalar=True).item() - tol <= 0

        x_scale = 0.5 * (norm(xbp - x[np.newaxis, ...], axis=1) + norm(xbn - x[np.newaxis, ...], axis=1))
        print("Tol before: {}".format(tol))
        tol = min(tol, system(x, scalar=True).item() * tol_fact)
        print("Tol after : {}".format(tol))

        if np.linalg.norm(x - prev_x) < xtol:
            print("xtol reached")
            break

    return x


def compare_adcs(ideal, real):
    assert ideal.meta == real.meta

    def prop_error(ideal, real):
        return (real - ideal)/ideal

    def lsb_error(ideal, real, lsb):
        return (real - ideal) / lsb

    cm = lsb_error(ideal.common_mode, real.common_mode, ideal.meta.lsb)
    eff = prop_error(ideal.eff, real.eff)
    cap = prop_error(ideal.caps, real.caps)
    cs_CF = prop_error(ideal.caps/np.sum(ideal.caps), real.caps/np.sum(real.caps))
    ref = lsb_error(ideal.refs, real.refs, ideal.meta.lsb)

    return cm, eff, cap, cs_CF, ref


def build_report(ideal, real, ideal_title="IDEAL", real_title="REAL"):
    assert ideal.meta == real.meta

    title_len = 24
    index_len = 6
    ideal_len = 10
    real_len = 10
    error_len = 10

    def title(header):
        return "- {} ".format(header) + '-'*max(title_len - 3 - len(header), 0)

    def index(idx):
        assert len(idx) > 0
        if len(idx) == 1:
            return "{}".format(idx)
        else:
            return "({})".format(','.join([str(ii) for ii in idx]))

    head = " {{:^{}}} {{:^{}}} {{:^{}}} {{:^{}}}".format(
        index_len, ideal_len, real_len, error_len)
    line = " {{:^{}}} {{:{}.{}f}} {{:{}.{}f}} {{:{}.{}f}}".format(
        index_len,
        ideal_len, ideal_len-3,
        real_len, real_len-3,
        error_len, error_len-3 )

    cm, eff, cap, cs_CF, ref = compare_adcs(ideal, real)

    result = []
    result.append(title("CHARGE TRANSFER"))
    result.append(head.format("INDEX", ideal_title, real_title, "ERROR (%)"))
    for idx in cartesian(*tuple(range(ss) for ss in np.shape(eff))):
        result.append(line.format('-', ideal.eff, real.eff, 100*eff))

    result.append('')
    result.append(title("COMMON MODE"))
    result.append(head.format("INDEX", ideal_title, real_title, "ERROR (LSB)"))
    for idx in cartesian(*tuple(range(ss) for ss in np.shape(cm))):
        r_cm = real.common_mode[idx]
        i_cm = ideal.common_mode[idx]
        result.append(line.format('-', i_cm, r_cm, cm[idx]))

    result.append('')
    result.append(title("CAPACITOR"))
    result.append(head.format("INDEX", ideal_title, real_title, "ERROR (%)"))
    for idx in cartesian(*tuple(range(ss) for ss in np.shape(cap))):
        r_cap = real.caps[idx]
        i_cap = ideal.caps[idx]
        result.append(line.format(index(idx), i_cap, r_cap, 100*cap[idx]))

    result.append('')
    result.append(title("CAPACITOR (Cs/CF)"))
    result.append(head.format("INDEX", ideal_title, real_title, "ERROR (%)"))
    for idx in cartesian(*tuple(range(ss) for ss in np.shape(cs_CF))):
        r_cap = real.caps[idx]/np.sum(real.caps)
        i_cap = ideal.caps[idx]/np.sum(ideal.caps)
        result.append(line.format(index(idx), i_cap, r_cap, 100*cs_CF[idx]))

    result.append('')
    result.append(title("REFERENCE"))
    result.append(head.format("INDEX", ideal_title, real_title, "ERROR (LSB)"))
    for idx in cartesian(*tuple(range(ss) for ss in np.shape(ref))):
        r_ref = real.refs[idx]
        i_ref = ideal.refs[idx]
        result.append(line.format(index(idx), i_ref, r_ref, ref[idx]))

    return '\n'.join(result)
