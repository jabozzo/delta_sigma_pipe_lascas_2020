#! usr/bin/env python

import copy
from itertools import product as cartesian
import warnings

import numpy as np
import numexpr as ne

import calib.gen as gen
import calib.data as data

import calib.misc as misc
from calib.misc import default, push_random_state
from calib.misc import multiget as get


def transition_cx_change(prev_cs, prev_cf, next_cs, next_cf):
    """
    Computes the capacitors that changed from a feedback position to a
    feedfoward position, the ones that did the oposite during a
    configuration transition and the ones that kept their positions.

    :param prev_cs: Feedfoward capacitors indexes of the previous
        configuration.
    :type prev_cs: :class:`numpy.array`
    :param prev_cf: Feedback capacitors indexes of the previous configuration.
    :type prev_cf: :class:`numpy.array`
    :param next_cs: Feedfoward capacitors indexes of the next configuration.
    :type next_cs: :class:`numpy.array`
    :param next_cf: Feedback capacitors indexes of the next configuration.
    :type next_cf: :class:`numpy.array`
    :returns: The indexes of the capacitors that changed from the feedfoward to
        feedback position (cs_cf). The indexes of the capacitors that
        changed from the feedfoward to feedback position (cf_cs). The indexes
        of the feedfoward capacitors that kept their positions (cs_cs) and the
        indexes of the feedback capacitors that kept their positions (cf_cf).
        The result order is (cs_cf, cf_cs, cs_cs, cf_cf,).

        The shape of each variable is:
        * shape(cs_cf) = (n_conf, n_cs_prev, n_diff)
        * shape(cf_cs) = (n_conf, n_cs_next, n_diff)
        * shape(cs_cs) = (n_conf, n_cs_next, n_diff)
        * shape(cf_cf) = (n_conf, n_cf_next, n_diff)

    :rtype: (:class:`numpy.ndarray`, :class:`numpy.ndarray`,
        :class:`numpy.ndarray`, :class:`numpy.ndarray`,)
    """

    def intersect(source, reference, axis):
        result = np.copy(source)
        ogrid = tuple(np.ogrid[tuple(slice(ss) for ss in np.shape(source))])

        def ogrid_idx(idx):
            idx = np.reshape(idx, (1,)*len(np.shape(source)))
            return ogrid[:axis] + (idx,) + ogrid[(axis+1):]

        for ii in range(np.size(source, axis)):
            idx = ogrid_idx(ii)

            mask = source[idx] == reference
            mask = np.any(mask, axis=axis, keepdims=True)
            mask = np.logical_not(mask)  # Elements not in the row

            sub_view = result[idx]
            sub_view[mask] = -1
            result[idx] = sub_view

        return result

    cs_cf = intersect(prev_cs, next_cf, -2)
    cf_cs = intersect(next_cs, prev_cf, -2)
    cs_cs = intersect(next_cs, prev_cs, -2)
    cf_cf = intersect(next_cf, prev_cf, -2)

    return cs_cf, cf_cs, cs_cs, cf_cf


class StageTestbench(data.JsonData):

    @property
    def stages(self):
        return self._stages

    @property
    def ins(self):
        return self._ins

    @property
    def shape(self):
        return self._shape

    @property
    def conf_shape(self):
        return np.shape(self.configuration_sequence.data)

    @property
    def is_scalar(self):
        return len(self.shape) + len(self.conf_shape) == 0

    @property
    def configuration_sequence(self):
        return self._configuration_sequence

    @classmethod
    def Scalar(cls, stage, ins, configuration_sequence, data_location=None):
        return cls(stage, ins, configuration_sequence, shape=tuple(), data_location=data_location)

    def __init__(self, stages, ins, configuration_sequence, shape=None, data_location=None):
        super().__init__(data_location)

        NestedStages = data.nested_lists_of(gen.StageParameters)
        NestedSequences = data.nested_lists_of(gen.ConfigurationSequence)

        if not isinstance(configuration_sequence, NestedSequences):
            conf_shape = np.shape(configuration_sequence)
            configuration_sequence = NestedSequences(configuration_sequence, len(conf_shape))

        conf_shape = np.shape(configuration_sequence.data)

        if not isinstance(stages, NestedStages):
            shape = default(shape, np.shape(stages))
            dims = len(shape)
            cur = stages
            root_arr = stages

            # Check a valid 0,0,0...
            for dd in range(dims):
                assert isinstance(cur, (tuple, list,))
                cur = cur[0]

            # Check elements
            for idx in cartesian(*tuple(range(ss) for ss in shape)):
                assert isinstance(misc.getitem(stages, idx), gen.StageParameters)

            # Check regularity
            def rec_check(lst, shape):
                valid = (not isinstance(lst, list) and len(shape) == 0) or len(lst) == shape[0]
                if len(shape) > 1:
                    sub_shape = shape[1:]
                    valid = valid and all([rec_check(llst, sub_shape) for llst in lst])
                return valid

            assert rec_check(stages, shape)

        else:
            stages_shape = np.shape(stages.data)
            shape = default(shape, stages_shape)

            assert shape == stages_shape

            dims = len(shape)
            root_arr = stages.data

        ref_element = misc.getitem(root_arr, (0,)*dims)
        ins = data.at_least_ndarray(ins)
        if len(np.shape(ins)) == 1:
            ins = ins[..., np.newaxis]

        # Broadcast ins
        if len(np.shape(ins)) == 2:
            ins = ins[(np.newaxis,)*dims + (Ellipsis,)]
            ins = np.tile(ins, shape + (1, 1,))

        assert len(np.shape(ins)) == dims + 2

        if np.size(ins, -1) != ref_element.meta.n_diff:
            cm = ref_element.meta.common_mode
            ins = np.concatenate((cm-ins, cm+ins,), axis=1)

        # All meta the same
        for idx in cartesian(*tuple(range(ss) for ss in shape)):
            c_element = misc.getitem(root_arr, idx)
            assert c_element.meta == ref_element.meta

        self._stages = NestedStages.EnsureIsInstance(stages)
        self._shape = shape
        self._ins = ins
        self._configuration_sequence = configuration_sequence

    def _to_json_dict(self, path_context, memo=None):
        dct = {}

        dct["stages"] = self.stages.save(path_context, memo=memo)
        if np.size(self.ins) == 0:
            dct["ins_shape"] = np.shape(self.ins)
        dct["ins"] = data.at_least_numpydata(self.ins).save(path_context, memo=memo)
        dct["configuration_sequence"] = self.configuration_sequence.save(path_context, memo=memo)
        dct["shape"] = self.shape

        return dct

    @classmethod
    def _JsonDictToArgs(cls, path_context, data_location, dct, memo=None):
        _, args, kwargs = super()._JsonDictToArgs(path_context, data_location, dct, memo=memo)

        StagesType = data.nested_lists_of(gen.StageParameters)
        ConfType = data.nested_lists_of(gen.ConfigurationSequence)
        args.append(StagesType.Load(path_context, dct["stages"], memo=memo))
        ins = data.NumpyData.Load(path_context, dct["ins"], memo=memo)

        try:
            shape = dct["ins_shape"]
            ins = np.reshape(ins, shape)
        except KeyError:
            pass

        args.append(ins)
        args.append(ConfType.Load(path_context, dct["configuration_sequence"], memo=memo))

        kwargs["shape"] = None if dct["shape"] is None else tuple(dct["shape"])

        return cls, args, kwargs

    def iter_idx(self):
        return cartesian(self.iter_conf_idx(), self.iter_stages_idx())

    def iter_conf_idx(self):
        if len(self.conf_shape) == 0:
            yield tuple()

        else:
            for idx in cartesian(*tuple(range(ss) for ss in self.conf_shape)):
                yield idx

    def iter_stages_idx(self):
        if len(self.shape) == 0:
            yield tuple()

        else:
            for idx in cartesian(*tuple(range(ss) for ss in self.shape)):
                yield idx

    def as_scalars(self):
        result = np.zeros(self.conf_shape + self.shape, dtype=object)
        for idx_conf, idx_stage in self.iter_idx():
            ins = self.ins[idx_stage + (Ellipsis,)]
            stage = self.stages[idx_stage]
            conf = self.configuration_sequence[idx_conf]
            result[idx_conf + idx_stage] = StageTestbench.Scalar(stage, ins, conf)

        ResultType = data.nested_lists_of(StageTestbench)

        return ResultType(result.tolist(), dims=len(np.shape(result)))

    def simulation_args(self, conf_idx):

        shape = self.shape
        sample_idx = next(self.iter_stages_idx())
        sample_stage = self.stages[sample_idx]

        eff_shape = np.shape(sample_stage.eff)
        caps_shape = np.shape(sample_stage.caps)
        refs_shape = np.shape(sample_stage.refs)
        thres_shape = np.shape(sample_stage.thres)
        cm_shape = np.shape(sample_stage.common_mode)

        eff = np.zeros(shape + eff_shape)
        caps = np.zeros(shape + caps_shape)
        refs = np.zeros(shape + refs_shape)
        thres = np.zeros(shape + thres_shape)
        cm = np.zeros(shape + cm_shape)

        for idx in self.iter_stages_idx():
            stage = self.stages[idx]

            eff[idx + (Ellipsis,)] = stage.eff
            caps[idx + (Ellipsis,)] = stage.caps
            refs[idx + (Ellipsis,)] = stage.refs
            thres[idx + (Ellipsis,)] = stage.thres
            cm[idx + (Ellipsis,)] = stage.common_mode

        ins = self.ins
        cal_seq = self.configuration_sequence[conf_idx]

        return eff, caps, refs, thres, ins, cm, cal_seq

    def simulate(self, simulator, raise_=False):
        conf_shape = self.conf_shape
        if len(conf_shape) == 0:
            return simulator.simulate(*self.simulation_args(tuple()), raise_=raise_)

        else:
            codes = np.full(conf_shape, None)
            us = np.full(conf_shape, None)

            for conf_idx in self.iter_conf_idx():
                code, u = simulator.simulate(*self.simulation_args(conf_idx), raise_=raise_)
                codes[conf_idx] = code
                us[conf_idx] = u

            codes = np.array(codes.tolist(), dtype=int)
            us = np.array(us.tolist())

            transpose_idx = len(conf_shape)
            transpose_idx = ( (transpose_idx,)
                            + tuple(range(0, transpose_idx))
                            + tuple(range(transpose_idx + 1, len(np.shape(codes)))) )

            codes = np.transpose(codes, transpose_idx)
            us = np.transpose(us, transpose_idx)

            return codes, us

    def sweep_parameters(self, sweep_dicts):

        def sweep(dct):
            sw_type = dct.get("type", "linear")

            if sw_type == "linear":
                values = np.linspace(dct["start"], dct["end"], dct["samples"])

            elif sw_type == "log":
                values = np.logspace(dct["start"], dct["end"], dct["samples"])

            else:
                raise ValueError("sweep type {} not recognized.".format(sw_type))

            def gen_dict(value):
                copy_keys = ("parameter", "index",)
                result = {key: dct[key] for key in copy_keys}
                result["value"] = value
                return result

            return [gen_dict(value) for value in values]

        values_axes = tuple(sweep(dct) for dct in sweep_dicts)
        shape = tuple(len(axis) for axis in values_axes)

        ins = np.zeros(shape + self.shape, dtype=int)
        stages = np.zeros(shape + self.shape, dtype=object)

        for idx in cartesian(*tuple(range(ss) for ss in shape)):
            val = tuple(vals[ii] for ii, vals in zip(idx, values_axes))

            this_stages = copy.deepcopy(self.stages)
            in_ = np.array(self.ins)

            new_val = tuple()
            for vall in val:
                if vall["parameter"] == "test":
                    in_[(Ellipsis, vall["index"], slice(None),)] = vall["value"]

                else:
                    new_val = (vall,) + new_val

            for sub_idx in this_stages.iter_idx():
                this_stages[idx] = this_stages[idx].create_modified(new_val)

            ins[idx + (Ellipsis,)] = in_
            stages[idx + (Ellipsis,)] = this_stages

        return values_axes, StageTestbench(stages.tolist(), ins,
            self.configuration_sequence, shape=shape + self.shape)


class Simulator(data.JsonData):

    @property
    def seed(self):
        return self._seed

    @property
    def ref_snr(self):
        return self._ref_snr

    @property
    def thres_snr(self):
        return self._thres_snr

    @property
    def in_snr(self):
        return self._in_snr

    @property
    def u_history(self):
        return self._u_history

    def __init__(self, seed, ref_snr=0, thres_snr=0, in_snr=0, u_history=True, data_location=None):
        super().__init__(data_location)

        self._seed = seed
        self._u_history = u_history
        self._ref_snr = ref_snr
        self._thres_snr = thres_snr
        self._in_snr = in_snr

        with push_random_state() as state_store:
            np.random.seed(self._seed)
        self._random_state = state_store

    def _to_json_dict(self, path_context, memo=None):
        dct = {}

        dct["u_history"] = self.u_history
        dct["seed"] = self.seed
        dct["in_snr"] = data.at_least_numpydata(self.in_snr).save(path_context, memo=memo)
        dct["ref_snr"] = data.at_least_numpydata(self.ref_snr).save(path_context, memo=memo)
        dct["thres_snr"] = data.at_least_numpydata(self.thres_snr).save(path_context, memo=memo)

        return dct

    @classmethod
    def _JsonDictToArgs(cls, path_context, data_location, dct, memo=None):
        _, args, kwargs = super()._JsonDictToArgs(path_context, data_location, dct, memo=memo)

        args.append(dct["seed"])
        for attr in ("in_snr", "ref_snr", "thres_snr",):
            kwargs[attr] = data.NumpyData.Load(path_context, dct[attr], memo=memo)

        kwargs["u_history"] = dct["u_history"]

        return cls, args, kwargs

    def _standard_deviations(self, meta, data):
        # Noise magnitude, computed using https://en.wikipedia.org/wiki/Signal-to-noise_ratio
        ref_snr = self.ref_snr
        in_snr = self.in_snr
        thres_snr = self.thres_snr

        fsr = meta.fsr
        fsr = fsr[1] - fsr[0]

        s_ref = 0 if ref_snr == 0 else np.sqrt(np.power(fsr, 2)/ref_snr)
        s_thres = 0 if thres_snr == 0 else np.sqrt(np.power(fsr, 2)/thres_snr)
        s_in = 0 if in_snr == 0 else np.sqrt(np.power(fsr, 2)/in_snr)

        s_dict = {  "s_ref": s_ref,
                    "s_thres": s_thres,
                    "s_in": s_in}

        data["std"] = s_dict

    @staticmethod
    def simulate_setup(eff, caps, refs, thres, ins, common_mode, conf_seq, scalar):
        if scalar:
            eff = eff[np.newaxis, ...]
            caps = caps[np.newaxis, ...]
            refs = refs[np.newaxis, ...]
            thres = thres[np.newaxis, ...]
            ins = ins[np.newaxis, ...]
            common_mode = common_mode[np.newaxis, ...]

        base_shape = np.shape(eff)
        base_len = len(base_shape)

        assert len(np.shape(common_mode)) == base_len, "Expected same dimensions as eff."
        assert len(np.shape(caps)) == base_len + 2, "Expected 2 extra dimensions."
        assert len(np.shape(refs)) == base_len + 3, "Expected 3 extra dimensions."
        assert len(np.shape(thres)) == base_len + 1, "Expected 1 extra dimensions."
        assert len(np.shape(ins)) == base_len + 2, "Expected 2 extra dimensions."

        common_mode = np.reshape(common_mode, np.shape(common_mode) + (1, 1,))

        meta = conf_seq.meta

        n_diff = meta.n_diff
        n_conf = conf_seq.n_conf

        assert n_diff == np.size(caps, -1), "Inconsistent data with meta."
        assert n_diff == np.size(refs, -1), "Inconsistent data with meta."
        assert n_diff == np.size(ins, -1), "Inconsistent data with meta."

        n_caps = meta.n_caps
        n_refs = meta.n_refs
        # n_ins = np.size(refs, -2)
        n_thres = np.size(thres, -1)
        n_codes = n_thres + 1

        assert n_caps == np.size(caps, -2), "Inconsistent data with meta."
        assert n_refs == np.size(refs, -2), "Inconsistent data with meta."
        assert n_caps == np.size(refs, -3), "Inconsistent data with meta."

        # Create extended versions
        zl = np.zeros_like
        ins_shape = np.shape(ins)
        ins_extra_shape = ins_shape[:-2] + (1, ins_shape[-1],)

        eff = eff[..., np.newaxis, np.newaxis]

        ins = np.concatenate((ins, np.zeros(ins_extra_shape),), axis=-2)
        caps = np.concatenate((caps, zl(caps[..., 0:1, :]),), axis=-2)
        refs = np.concatenate((refs, zl(refs[..., 0:1, :]),), axis=-2)
        refs = np.concatenate((refs, zl(refs[..., 0:1, :, :]),), axis=-3)

        diff_ii = misc.ogrid(2, n_diff, 3)
        diff_ii_ext = misc.ogrid(base_len + 2, n_diff, base_len + 3)
        base_ii = tuple(misc.ogrid(ii, ss, base_len + 3) for ii, ss in enumerate(base_shape))

        cap_axis = base_len + 1

        ext_dct = { "eff": eff,
                    "caps": caps,
                    "refs": refs,
                    "thres": thres,
                    "ins": ins,
                    "cm": common_mode }

        n_dct = {   "n_conf": n_conf,
                    "n_caps": n_caps,
                    "n_refs": n_refs,
                    "n_thres": n_thres,
                    "n_codes": n_codes,
                    "n_diff": n_diff }

        idx_dct = { "base_shape": base_shape,
                    "base_len": base_len,
                    "diff_ii": diff_ii,
                    "diff_ii_ext": diff_ii_ext,
                    "base_ii": base_ii,
                    "cap_axis": cap_axis }

        s_dict = {"s_ref": 0, "s_thes": 0, "s_in": 0}

        data_dict = {   "extended": ext_dct,
                        "n": n_dct,
                        "indexing": idx_dct,
                        "std": s_dict,
                        "ds_offset": 0 }

        return data_dict

    @staticmethod
    def init_seq_idx(conf_seq, data):
        idx_dct, ext_dct, n_dct = get(data, "indexing", "extended", "n")

        base_shape, diff_ii = get(idx_dct, "base_shape", "diff_ii")
        caps, refs, cm = get(ext_dct, "caps", "refs", "cm")
        n_conf, n_diff = get(n_dct, "n_conf", "n_diff")

        ic = conf_seq.initial_conditions

        # Set initial condition

        # shape (n_conf, n_cf, n_diff,)
        ic_ii = np.array(list(iic.ref_ii for iic in ic), dtype=int)
        # shape (n_conf, n_cf, n_diff,)
        cf_ii = conf_seq.configuration_sets[0].cf

        m_refs_idx = (Ellipsis, cf_ii, ic_ii, diff_ii,)
        m_caps_idx = (Ellipsis, cf_ii, diff_ii,)

        init_idxs = {
            "m_refs_idx": m_refs_idx,
            "m_caps_idx": m_caps_idx }
        return init_idxs

    @staticmethod
    def init_seq(indexes, data):
        idx_dct, ext_dct, n_dct, std = get(
            data, "indexing", "extended", "n", "std")

        base_shape = get(idx_dct, "base_shape")[0]
        caps, refs, cm = get(ext_dct, "caps", "refs", "cm")
        n_conf, n_diff = get(n_dct, "n_conf", "n_diff")
        s_ref = get(std, "s_ref")[0]

        m_refs_idx, m_caps_idx = get(indexes, "m_refs_idx", "m_caps_idx")

        u = np.zeros(base_shape + (n_conf, n_diff,))

        # shape (base_shape, n_conf, n_cf, n_diff,)
        ic_refs = refs[m_refs_idx]
        if s_ref > 0:
            ic_refs = np.random.normal(ic_refs, s_ref, size=np.shape(ic_refs))

        ic_cf = caps[m_caps_idx]
        ic_g = ic_cf / np.sum(ic_cf, axis=-2, keepdims=True)
        u += np.sum(ic_g * ic_refs, axis=-2)

        if n_diff == 2:
            u += cm - np.mean(u, axis=-1, keepdims=True)

        u = u[np.newaxis, ...]

        return u

    @staticmethod
    def init_set(conf_set, data, prev_set_data, prev_du_idx):
        idx_dct, ext_dct, n_dct = get(data, "indexing", "extended", "n")

        n_refs, n_codes = get(n_dct, "n_refs", "n_codes")
        base_len, diff_ii = get(idx_dct, "base_len", "diff_ii")

        n_cs = conf_set.n_cs
        cs_ii = conf_set.cs
        cf_ii = conf_set.cf

        cs_ii_base = cs_ii[(np.newaxis,)*base_len + (Ellipsis,)]

        meta = conf_set.meta
        ds_map = gen.ds_map(n_cs, n_refs, n_codes, meta.differential)

        if prev_du_idx is None:
            prev_dct = None
        else:
            prev_dct = {"r_ref": prev_du_idx["r_ref"],
                        "cs_ii": prev_set_data["indexing"]["cs_ii"],
                        "cf_ii": prev_set_data["indexing"]["cf_ii"] }

        return {    "indexing" : {
                        "cs_ii" : cs_ii,
                        "cf_ii": cf_ii,
                        "cs_ii_base": cs_ii_base, },
                    "n": {"n_cs": n_cs },
                    "previous": prev_dct,
                    "ds_map": ds_map}

    @staticmethod
    def transition_step_idx(conf_set, set_data, data, code):
        assert np.size(code, 0) == 1

        idx_dct, ext_dct, n_dct = get(data, "indexing", "extended", "n")
        idx_set, pre_set, n_set, ds_map = get(set_data,
            "indexing", "previous", "n", "ds_map")

        cs_ii, cf_ii, cs_ii_base = get(idx_set, "cs_ii", "cf_ii", "cs_ii_base")
        prev_cs_ii, prev_cf_ii = get(pre_set, "cs_ii", "cf_ii")
        prev_r_ref = get(pre_set, "r_ref")[0][-1, ...]

        diff_ii, diff_ii_ext, cap_axis = get(idx_dct,
            "diff_ii", "diff_ii_ext", "cap_axis")

        base_shape, base_len, base_ii = get(idx_dct,
            "base_shape", "base_len", "base_ii")

        eff, caps, refs, thres, ins, cm = get(ext_dct,
            "eff", "caps", "refs", "thres", "ins", "cm")

        n_conf, n_diff = get(n_dct, "n_conf", "n_diff")

        # Compute transition
        cs_cf_ii, cf_cs_ii, cs_cs_ii, cf_cf_ii = transition_cx_change(
            prev_cs_ii, prev_cf_ii, cs_ii, cf_ii)

        # shape(n_cs, ..., n_conf, n_diff)
        this_ref_ii = ds_map[:, code, :]
        # shape(..., n_conf, n_cs, n_diff)
        ds_map_transpose = tuple(range(1, base_len + 1 + 2)) + (0, -1,)
        # shape = (n_samples, base_shape, n_conf, n_cs, n_diff) (before idx)
        this_ref_ii = np.transpose(this_ref_ii, ds_map_transpose)[0, ...]

        ds_offset = data["ds_offset"]
        in_ref_ii, in_ins_ii = conf_set.generate_in(1, ds_offset)
        data["ds_offset"] = ds_offset + 1

        m_this_ref_idx = base_ii + (cs_ii_base, this_ref_ii, diff_ii_ext,)
        r_this_ref_idx = np.ravel_multi_index(m_this_ref_idx, refs.shape)

        transition_idx = {
            "m_cf_cs_idx": (Ellipsis, cf_cs_ii, diff_ii,),
            "m_cs_cf_idx": (Ellipsis, cs_cf_ii, diff_ii,),
            "m_cs_cs_idx": (Ellipsis, cs_cs_ii, diff_ii,),
            "m_cf_cf_idx": (Ellipsis, cf_cf_ii, diff_ii,),
            "r_this_r_ref": r_this_ref_idx,
            "r_prev_r_ref": prev_r_ref,
            "m_this_in_ref": (Ellipsis, cs_ii, in_ref_ii[0, ...], diff_ii,),
            "m_this_in_ins": (Ellipsis, in_ins_ii[0, ...], diff_ii,) }

        return transition_idx

    @staticmethod
    def transition_step(indexes, u, set_data, data):
        idx_dct, ext_dct, n_dct = get(data, "indexing", "extended", "n")

        cap_axis = get(idx_dct, "cap_axis")[0]
        n_diff = get(n_dct, "n_diff")[0]

        eff, caps, refs, thres, ins, cm = get(ext_dct,
            "eff", "caps", "refs", "thres", "ins", "cm")

        m_cf_cs_idx, m_cs_cf_idx, m_cs_cs_idx, m_cf_cf_idx = get(indexes,
            "m_cf_cs_idx", "m_cs_cf_idx", "m_cs_cs_idx", "m_cf_cf_idx")

        r_this_r_ref, r_prev_r_ref = get(indexes,
            "r_this_r_ref", "r_prev_r_ref")

        m_this_in_ref, m_this_in_ins = get(indexes,
            "m_this_in_ref", "m_this_in_ins")

        std = get(data, "std")[0]
        s_ref, s_in = get(std, "s_ref", "s_in")

        cf_cs = caps[m_cf_cs_idx]
        cs_cf = caps[m_cs_cf_idx]
        cs_cs = caps[m_cs_cs_idx]  # used in du
        cf_cf = caps[m_cf_cf_idx]

        prev_ref = refs.ravel().take(r_prev_r_ref)
        if s_ref > 0:
            prev_ref = np.random.normal(prev_ref, s_ref, size=np.shape(prev_ref))

        this_ref = refs.ravel().take(r_this_r_ref)
        if s_ref > 0:
            this_ref = np.random.normal(this_ref, s_ref, size=np.shape(this_ref))

        this_in_ref = refs[m_this_in_ref] # used in du
        if s_ref > 0:
            this_in_ref = np.random.normal(this_in_ref, s_ref, size=np.shape(this_in_ref))

        this_in_ins = ins[m_this_in_ins] # used in du
        if s_in > 0:
            this_in_ins = np.random.normal(this_in_ins, s_in, size=np.shape(this_in_ins))

        # used in du
        u_gain = (np.sum(cf_cf, axis=-2) + np.sum(cf_cs, axis=-2)
             ) / (np.sum(cf_cf, axis=-2) + np.sum(cs_cf, axis=-2))

        # Sum on next_cs shaped
        du_stmt = ("sum(cs_cs*(this_in_ref + this_in_ins)"
                   " - (cf_cs + cs_cs)*(this_ref), axis={})").format(cap_axis)
        du = ne.evaluate(du_stmt)

        # Sum on prev_cs shaped
        du_stmt = ("sum(cs_cf*prev_ref, axis={})").format(cap_axis)
        du += ne.evaluate(du_stmt)

        assert np.size(u, 0) == 1, "Only one sample."

        CF = np.sum(cf_cf, axis=cap_axis) + np.sum(cs_cf, axis=cap_axis)

        # Apply gain and charge loss
        u = ne.evaluate("u*u_gain*eff + (1-eff)*cm + du/CF")

        # common mode feedback
        if n_diff == 2:
            u += cm - np.mean(u, axis=-1, keepdims=True)[np.newaxis, ...]

        return u

    @staticmethod
    # @profile
    def du_indexes(code, conf_set, set_data, data):
        n_samples = np.size(code, 0)

        idx_dct, ext_dct, n_dct = get(data, "indexing", "extended", "n")
        idx_set, n_set, ds_map = get(set_data, "indexing", "n", "ds_map")

        refs, ins = get(ext_dct, "refs", "ins")
        diff_ii, diff_ii_ext, cap_axis, base_ii, base_len = get(idx_dct,
            "diff_ii", "diff_ii_ext", "cap_axis", "base_ii", "base_len")

        cs_ii, cs_ii_base, = get(idx_set, "cs_ii", "cs_ii_base")

        ds_offset = data["ds_offset"]
        in_ref_ii, in_ins_ii = conf_set.generate_in(n_samples, ds_offset)
        data["ds_offset"] = ds_offset + n_samples

        # Using take instead of indexing because for some reason it's faster
        ref_ii = ds_map.take(code, axis=1)
        ds_map_transpose = tuple(range(1, base_len + 1 + 2)) + (0, -1,)
        # shape = (n_samples, base_shape, n_conf, n_cs, n_diff)
        ref_ii = np.transpose(ref_ii, ds_map_transpose)
        set_data["indexing"]["ref_ii"] = ref_ii[-1, ...]

        ext_idx = (np.newaxis, Ellipsis,)
        ext_base_ii = tuple(bb[ext_idx] for bb in base_ii)

        m_ref = ext_base_ii + (cs_ii_base[ext_idx], ref_ii, diff_ii_ext[ext_idx],)
        m_in_ref = (Ellipsis,) + (cs_ii[ext_idx], in_ref_ii, diff_ii[ext_idx],)
        m_in_ins = (Ellipsis,) + (in_ins_ii, diff_ii[ext_idx],)

        ravel = np.ravel_multi_index
        r_ref = ravel(m_ref, refs.shape, mode='wrap')

        return {"r_ref": r_ref, "m_in_ref": m_in_ref, "m_in_ins": m_in_ins }

    @staticmethod
    # @profile
    def du_compute(indexes, conf_set, set_data, data):
        idx_dct, ext_dct, n_dct = get(data, "indexing", "extended", "n")

        caps, refs, ins, cm = get(ext_dct, "caps", "refs", "ins", "cm")
        cap_axis, base_len, base_ii, diff_ii = get(idx_dct,
            "cap_axis", "base_len", "base_ii", "diff_ii")

        n_diff = get(n_dct, "n_diff")[0]
        r_ref, m_in_ref, m_in_ins = get(indexes, "r_ref", "m_in_ref", "m_in_ins")

        cs_ii, cf_ii = get(get(set_data, "indexing")[0], "cs_ii", "cf_ii")

        std = get(data, "std")[0]
        s_ref, s_in = get(std, "s_ref", "s_in")

        cs = caps[np.newaxis, ..., cs_ii, diff_ii] # used in du
        cf = caps[np.newaxis, ..., cf_ii, diff_ii]
        CF = np.sum(cf, axis=-2, keepdims=True) # used in du

        # Using take instead of indexing because for some reason it's faster
        ref = refs.ravel().take(r_ref)
        if s_ref > 0:
            ref = np.random.normal(ref, s_ref, size=np.shape(ref))

        in_ref = refs[m_in_ref]
        if s_ref > 0:
            in_ref = np.random.normal(in_ref, s_ref, size=np.shape(in_ref))

        in_ins = ins[m_in_ins]
        if s_in > 0:
            in_ins = np.random.normal(in_ins, s_in, size=np.shape(in_ins))

        in_ref = np.transpose(in_ref, (-4,) + tuple(range(base_len)) + (-3, -2, -1,))
        in_ins = np.transpose(in_ins, (-4,) + tuple(range(base_len)) + (-3, -2, -1,))
        # sum done un numpy to prevent issue 79 of numexpr
        du = np.sum(ne.evaluate("(cs/CF)*(in_ref + in_ins - ref)"), cap_axis+1)

        if n_diff == 2:
            du += cm[np.newaxis, ...] - np.mean(du, axis=-1, keepdims=True)

        return du

    @staticmethod
    def u_thres_to_code(u, thres, data):
        s_thres = get(get(data, "std")[0], "s_thres")[0]
        if s_thres > 0:
            thres = np.random.normal(thres, s_thres, np.shape(thres))
        # reinterpret shape (..., n_diff) to (..., n_thres, n_diff)
        code = np.diff(u, axis=-1) if np.size(u, -1) == 2 else u
        code = np.sum(code >= thres[..., np.newaxis, :], axis=-1)
        return code

    def simulate(self, eff, caps, refs, thres, ins, common_mode, c_seq, scalar=None, raise_=False):
        scalar = default(scalar, len(np.shape(eff)) == 0)

        meta = c_seq.meta

        data = self.simulate_setup(
            eff,
            caps,
            refs,
            thres,
            ins,
            common_mode,
            c_seq,
            scalar=scalar )

        self._standard_deviations(meta, data)

        idx_dct, ext_dct, n_dct = get(data, "indexing", "extended", "n")

        eff, thres, cm = get(ext_dct, "eff", "thres", "cm")
        n_diff = get(n_dct, "n_diff")[0]

        u_history = []

        with self._random_state as _:
            # shape (..., n_conf, n_diff,)
            seq_idx = self.init_seq_idx(c_seq, data)
            u = self.init_seq(seq_idx, data)

            if self.u_history:
                u_history.append(u)

            # Simulate each configuration set
            codes = []

            du_idx = None
            set_data = None
            n_sets = len(c_seq.configuration_sets)

            for ii_set, c_set in enumerate(c_seq.configuration_sets):
                set_data = self.init_set(c_set, data, set_data, du_idx)

                if set_data["previous"] is not None:
                    code = self.u_thres_to_code(u[-1, ...], thres, data)
                    code = code[np.newaxis, ...]

                    trans_idx = self.transition_step_idx(c_set, set_data, data, code)
                    u = self.transition_step(trans_idx, u[-1:, ...], set_data, data)

                    codes.append(code)

                    if self.u_history:
                        u_history.append(u)

                for sample in range(c_set.ds_samples):
                    # print(" {}/{} Sample {}/{} ({:0.2f}%)".format(
                    #     ii_set+1, n_sets, sample+1, c_set.ds_samples, 100*sample/c_set.ds_samples),
                    #     end='\r')
                    code = self.u_thres_to_code(u[-1, ...], thres, data)
                    code = code[np.newaxis, ...]
                    codes.append(code)

                    du_idx = self.du_indexes(code, c_set, set_data, data)
                    # used in u
                    du = self.du_compute(du_idx, c_set, set_data, data)
                    u = ne.evaluate("u*eff + (1-eff)*cm + du")

                    # common mode feedback
                    if n_diff == 2:
                        u += cm[np.newaxis, ...] - np.mean(u, axis=-1, keepdims=True)

                    if self.u_history:
                        u_history.append(u)

            codes = np.concatenate(codes, axis=0)

            if self.u_history:
                u_history = np.concatenate(u_history, axis=0)

            if scalar:
                assert np.size(codes, 1) == 1
                assert np.size(u_history, 1) == 1

                codes = codes[:, 0, ...]
                u_history = u_history[:, 0, ...]

            if ((u_history < meta.fsr[0] - meta.lsb).any() or
                (u_history > meta.fsr[1] + meta.lsb).any()):

                message = "Residual out of range."

                if raise_:
                    raise ValueError(message)
                else:
                    warnings.warn(message)

            return codes, u_history


def dc_decimate(loop, codes, configuration_sequence):
    assert len(configuration_sequence.configuration_sets) == 1, "Chained sets not supported."

    c_set = configuration_sequence.configuration_sets[0]

    meta = loop.meta
    ds_map = gen.ds_map(c_set.n_cs, meta.n_refs, loop.n_thres+1, meta.differential)
    code_to_lsb = np.sum(ds_map[:, :, 0], axis=0)
    lsb = code_to_lsb[codes]

    # sum done un numpy to prevent issue 79 of numexpr
    return np.sum(lsb, axis=len(np.shape(lsb)) - 1)
