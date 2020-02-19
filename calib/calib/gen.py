#! /usr/bin/env python

import abc
import copy
from itertools import islice

import numpy as np

import calib.gen as gen
import calib.data as data
from calib.misc import default, push_random_state


def pipe_map(n_caps, n_refs, differential=False):
    """
    Creates the code->reference map for pipeline operation. Each column in this
    map contains the reference index to be used for the feedfoward capacitors.

    :param n_caps: Number of feedfoward capacitors with references,
        usually the number of stage capacitors -1.
    :type n_caps: :class:`int`
    :param n_refs: Number of references per capacitor.
    :type n_refs: :class:`int`
    :param differential: Generate a differential or single-ended map.
    :type differential: :class:`bool`
    :returns: The map with shape :code:`(n_caps, n_codes, n_diff)`
    :rtype: :class:`numpy.ndarray`
    """
    res = np.array([0]*n_caps + [1]*(n_caps-1), dtype=int)
    res = np.array(list(res[ii:ii+n_caps] for ii in range(n_caps)))
    res = res[::-1, :]
    res = (tuple(res + ii for ii in range(n_refs-1))
        + (np.reshape([n_refs-1]*n_caps, (n_caps, 1,)),) )
    res = np.concatenate(res, axis=1)

    if differential:
        res = np.stack((res, res[:, ::-1],), axis=2)
    else:
        res = np.reshape(res, np.shape(res) + (1,))

    return res


def adjust_map(map_, n_codes):
    """
    Expands a pipeline map to work with an sub-ADC with the same or more codes
    than the original sub-ADC.

    :param map_: The pipeline map.
    :type map_: :class:`numpy.ndarray`
    :param n_codes: The number of codes the sub-ADC support.
    :type n_codes: :class:`int`
    :returns: The expanded pipeline map.
    :rtype: :class:`numpy.ndarray`
    :raises AssertionError: If the map is bigger than the target number of
        codes.

    .. seealso :: :func:`pipe_map`
    """
    assert np.size(map_, 1) <= n_codes, "Map does not fit in number of codes"
    margin = n_codes - np.size(map_, 1)
    left = margin // 2
    right = margin - left
    return np.concatenate((map_[:, 0:1, :],)*left + (map_,) + (map_[:,-1:, :],)*right, axis=1)


def ds_map(n_caps, n_refs, n_codes, differential=False):
    """
    Creates the code->reference map for delta-sigma operation. Each column in
    this map contains the reference index to be used for the feedfoward
    capacitors.

    :param n_caps: Number of feedfoward capacitors with references, usually.
    :type n_caps: :class:`int`
    :param n_refs: Number of references per capacitor.
    :type n_refs: :class:`int`
    :param n_codes: Number of codes from the loop sub-ADC.
    :type n_codes: :class:`int`
    :param differential: Generate a differential or single-ended map.
    :type differential: :class:`bool`
    :returns: The map with shape :code:`(n_caps, n_codes, n_diff)`
    :rtype: :class:`numpy.ndarray`
    """

    p_map = pipe_map(n_caps, n_refs, differential)
    return adjust_map(p_map, n_codes)


def parse_bits(n_bits, half_bit=None):
    """
    Recieves either string, float or integer representations of a number of
    bits and returns the int + bool respresentation.

    :param n_bits: Number of bits
    :type n_bits: :class:`str`, :class:`float`, :class:`int`
    :param half_bit: If the representation is half bit or not.
    :type half_bit: :class:`bool`
    :returns: The int + bool representation of n_bits
    :rtype: (:class:`int`, :class:`bool`,)
    :raises ValueError: If the type of n_bits is not suported or n_bits and
        half_bit conflict with each other (eg: "2.5" and False)
    """
    if isinstance(n_bits, str):
        try:
            n_bits = int(n_bits)
        except ValueError:
            n_bits = float(n_bits)

    if isinstance(n_bits, float):
        int_bits = int(n_bits)
        c_half_bit = n_bits != int_bits

    elif isinstance(n_bits, int):
        int_bits = n_bits
        c_half_bit = default(half_bit, False)
        half_bit = c_half_bit

    else:
        raise ValueError("n_bits cannot be of type {}".format(type(n_bits)))

    half_bit = default(half_bit, c_half_bit)

    if half_bit != c_half_bit:
        raise ValueError(("Inconsistent parameters "
            "(n_bits: {}, half_bit: {})").format(n_bits, half_bit))

    return int_bits, half_bit


def format_bits(n_bits, half_bit=None):
    """
    Converts from int + bool representation to str representation

    :param n_bits: Number of bits
    :type n_bits: :class:`str`, :class:`float`, :class:`int`
    :param half_bit: If the representation is half bit or not.
    :type half_bit: :class:`bool`
    :returns: The str representation of n_bits
    :rtype: :class:`str`

    .. seealso :: :func:`parse_bits`, for n_bits and half_bit specification.
    """
    n_bits, half_bit = parse_bits(n_bits, half_bit)
    return "{}.5".format(n_bits) if half_bit else n_bits


def compute_n_codes(n_bits, half_bit=None):
    """
    Computes the number of codes a pipeline stage can process.

    :param n_bits: Number of bits of the stage
    :type n_bits: :class:`str`, :class:`float`, :class:`int`
    :param half_bit: If the number of bits is half-bit or not
    :type half_bit: :class:`bool`
    :returns: The number of codes the pipeline stage can handle
    :rtype: :class:`int`

    .. seealso :: :func:`parse_bits`, for n_bits and half_bit specification.
    """
    n_bits, half_bit = parse_bits(n_bits, half_bit)
    if half_bit:
        return 2**(n_bits + 1) - 1
    else:
        return 2**n_bits


def compute_n_caps(n_bits, n_refs, half_bit=None):
    """
    Computes the number of capacitors a stage needs to achieve the desired
    resolution.

    :param n_bits: Number of bits of the stage
    :type n_bits: :class:`str`, :class:`float`, :class:`int`
    :param n_refs: Number of references each feedfoward capacitor can use.
    :type half_bit: :class:`int`
    :param half_bit: If the number of bits is half-bit or not
    :type half_bit: :class:`bool`
    :returns: The number of capacitors the stage needs.
    :rtype: :class:`int`

    :raises AssertionError: If the number of references is invalid for the
        number of bits.

    .. seealso :: :func:`parse_bits`, for n_bits and half_bit specification.
    """
    n_bits, half_bit = parse_bits(n_bits, half_bit)
    n_codes = compute_n_codes(n_bits, half_bit)

    assert n_refs >= 2, "Need at least 2 references."
    assert (n_codes - 1) % (n_refs - 1) == 0, ("Cannot match {} "
        "refs with {} codes.".format(n_refs, n_codes))

    return ((n_codes - 1) // (n_refs - 1)) + 1


def compute_lsb(n_bits, fsr_min, fsr_max, half_bit=None):
    """
    Computes the least significant bit (LSB) magnitude in the stage MDAC and
    sub-ADC to achieve a desired full scale range (FSR). The input FSR and
    output FSR are assumed to be the same so only one value is returned.

    :param n_bits: Number of bits of the stage
    :type n_bits: :class:`str`, :class:`float`, :class:`int`
    :param fsr_min: Minimum value of the full voltage scale
    :type fsr_min: :class:`float`
    :param fsr_min: Maximum value of the full voltage scale
    :type fsr_min: :class:`float`
    :param half_bit: If the number of bits is half-bit or not
    :type half_bit: :class:`bool`
    :returns: The voltage value of the LSB.
    :rtype: :class:`float`

    .. seealso :: :func:`parse_bits`, for n_bits and half_bit specification.
    """
    n_bits, half_bit = parse_bits(n_bits, half_bit)
    n_codes = compute_n_codes(n_bits, half_bit)

    diff = fsr_max - fsr_min

    if half_bit:
        lsb = diff/(n_codes + 1)
    else:
        lsb = diff/n_codes

    return lsb


def infer_thres_bits(thres):
    d_bits = 2
    n_thres = np.size(thres)
    n_codes = compute_n_codes(d_bits // 2, half_bit=bool(d_bits % 2))
    while n_codes < n_thres + 1:
        d_bits += 1
        n_codes = compute_n_codes(d_bits // 2, half_bit=bool(d_bits % 2))

    assert n_codes == n_thres + 1, "No match found"
    return d_bits // 2, bool(d_bits % 2)


def infer_stage_lsb(stage):
    n_bits, half_bit = infer_thres_bits(stage.thres)
    # HACK: tail fsr asumes stage fsr
    return compute_lsb(n_bits, *stage.meta.fsr, half_bit=half_bit)


def infer_thres_lsb(thres, fsr):
    n_bits, half_bit = infer_thres_bits(thres)
    # HACK: tail fsr asumes stage fsr
    return compute_lsb(n_bits, *fsr, half_bit=half_bit)


def compute_ref(fsr_min, fsr_max, n_refs):
    """
    Computes the capacitor references votage for a pipeline stage.

    :param n_bits: Number of bits of the stage
    :type n_bits: :class:`str`, :class:`float`, :class:`int`
    :param fsr_min: Minimum value of the full voltage scale
    :type fsr_min: :class:`float`
    :param fsr_min: Maximum value of the full voltage scale
    :type fsr_min: :class:`float`
    :param n_refs: Number of references each feedfoward capacitor can use.
    :type n_refs: :class:`int`
    :param half_bit: If the number of bits is half-bit or not
    :type half_bit: :class:`bool`
    :returns: The values of the feedfoward capacitor references.
    :rtype: :class:`numpy.ndarray`

    .. seealso :: :func:`parse_bits`, for n_bits and half_bit specification.
    """
    return np.linspace(fsr_min, fsr_max, n_refs)


def compute_thres(n_bits, fsr_min, fsr_max, half_bit=None):
    """
    Computes the threshold voltages for a flash sub-ADC of a pipeline stage.

    :param n_bits: Number of bits of the stage
    :type n_bits: :class:`str`, :class:`float`, :class:`int`
    :param fsr_min: Minimum value of the full voltage scale
    :type fsr_min: :class:`float`
    :param fsr_min: Maximum value of the full voltage scale
    :type fsr_min: :class:`float`
    :param half_bit: If the number of bits is half-bit or not
    :type half_bit: :class:`bool`
    :returns: The values of the theshold voltages.
    :rtype: :class:`numpy.ndarray`

    .. seealso :: :func:`parse_bits`, for n_bits and half_bit specification.
    """
    n_bits, half_bit = parse_bits(n_bits, half_bit)
    lsb = compute_lsb(n_bits, fsr_min, fsr_max, half_bit)
    n_codes = compute_n_codes(n_bits, half_bit)

    range_2 = (lsb * (n_codes - 2))/2
    avg = (fsr_max + fsr_min) / 2

    if n_codes > 2:
        return np.linspace(-range_2, range_2, n_codes-1) + avg
    else:
        return np.reshape(avg, (1,))


def eff_random(eff_mean, tau_std):
    """
    Generates random number following a lognormal distribution, appropiate for
    the charge efficiency transfer parameter (eff). The lognormal distribution
    has mean in the mean parameter, but the standard deviation is computed by
    proyecting the mean in logspace, generating a normal distribution with that
    standard deviation and then exponeating again. This emulates the linear
    settling time given by equation:

    .. math ::
        \\mathit{eff} = e^{\\frac{-1}{2 \\pi \\tau}}

    :param eff_mean: Mean of the distribution. A factor between 0 and 1.
    :type eff_mean: :class:`numpy.ndarray`
    :param tau_std: Standard deviation in logspace, in time constants units.
    :type tau_std: :class:`numpy.ndarray`
    :returns: The random numbers following the lognormal distribution. The
        shape is the shape of the mutual boradcast of eff_mean and tau_std.
    :rtype: :class:`numpy.ndarray`
    """
    tau = (-1.0/np.log(eff_mean))/(2*np.pi)
    tau = np.random.normal(tau, tau_std)
    return np.exp(-1/(2*np.pi*tau))

## STAGE OBJECTS ##


class StageMeta(data.JsonData):
    """
    This class holds the metadata for a pipeline stage. The metadata is
    comprised of the design constraints that generate the ADC, these are:
    number of bits, number of references per capacitor and full scale range
    (FSR). The rest of the metadata is either deduced from that (LSB, number
    of capacitors, ...) or is arbitrary (capacitor magnitude, charge transfer
    efficiency, ...)
    """

    @property
    def n_refs(self):
        """
        Number of references each capacitor can access.
        """
        return self._n_refs

    @property
    def n_caps(self):
        """
        Total number of capacitors in the stage.
        """
        return compute_n_caps(self.n_bits, self.n_refs, half_bit=self.half_bit)

    @property
    def n_codes(self):
        """
        Number of codes this stage handles (the sub-ADC and MDAC)
        """
        return compute_n_codes(self.n_bits, self.half_bit)

    @property
    def n_bits(self):
        """
        Integer number of bits of this stage.

        .. seealso :: :py:attr:`half_bit`, for information about
            half-bit stages.
        """
        return self._n_bits

    @property
    def n_diff(self):
        """
        This is the same as 1 + int(self.differential), used for arrays that
        need a dimension with either len 1 for single-ended circuits or len 2
        for differential circuits.
        """
        return 1 + int(self.differential)

    @property
    def half_bit(self):
        """
        Boolean indicating if the stage is half_bit. If n_bits is N and
        half_bit is True, then the stage is N-5 bits, otherwise is N bits.
        """
        return self._half_bit

    @property
    def cap(self):
        """
        Ideal value of a capacitor.
        """
        return self._cap

    @property
    def eff(self):
        """
        Ideal of the charge transfer efficiency.
        """
        return self._eff

    @property
    def seed(self):
        """
        Seed for generating random stage instances.
        """
        return self._seed

    @property
    def lsb(self):
        """
        Least significant bit (LSB) value, given the number of references per
        capacitor, stage bits and FSR of this adc.
        """
        return compute_lsb(self.n_bits, *self.fsr, half_bit=self.half_bit)

    @property
    def differential(self):
        """
        True if the stage have a differential architecture.
        """
        return self._differential

    @property
    def fsr(self):
        """
        Full scale range (FSR). It's a tuple with the minimum and maximum
        voltages the stage can handle without saturation.
        """
        return self._fsr

    @property
    def common_mode(self):
        """
        The common_mode of the stage amplifier.
        """
        return self._common_mode

    @classmethod
    def _JsonDictToArgs(cls, path_context, data_location, dct, memo=None):
        n_bits, half_bit = parse_bits(dct.pop("n_bits"), dct.pop("half_bit", None))
        _, args, kwargs = super()._JsonDictToArgs(path_context, data_location, dct, memo=memo)
        args.append(n_bits)
        kwargs.update(dct)
        kwargs["half_bit"] = half_bit
        return cls, args, kwargs

    def __init__(self, n_bits, n_refs,
                eff=0.995, cap=1, common_mode=None, fsr=(-0.5, 0.5,),
                differential=False, seed=None, half_bit=None, data_location=None):
        """
        :param n_bits: Number of bits
        :type n_bits: :class:`str`, :class:`float`, :class:`int`
        :param n_refs: Number of references per capacitor.
        :type n_refs: :class:`int`
        :param eff: Charge transfer efficiency.
        :type eff: :class:`float`
        :param cap: Capacitors' capacitance.
        :type cap: :class:`float`
        :param common_mode: Common mod of the stage amplifier.
        :type common_mode: :class:`float`
        :param fsr: Full scale range (FSR) limits.
        :type fsr: (float, float,)
        :param differential: If the stage architecture is differential.
        :type differential: :class:`bool`
        :type seed: The seed used for random instances generation.
        :param seed: :class:`int`
        :param half_bit: If the number of bits is half-bit or not
        :type half_bit: :class:`bool`
        """
        super().__init__(data_location)
        assert n_refs >= 2, "Need at least 2 references."
        n_bits, half_bit = parse_bits(n_bits, half_bit)

        n_ref_half = n_refs % 2 == 1
        if half_bit and not n_ref_half:
            raise ValueError("Need odd n_refs for half bit, recieved {}.".format(n_refs))
        elif not half_bit and n_ref_half:
            raise ValueError("Need even n_refs for integer bit, recieved {}.".format(n_refs))

        self._n_bits = n_bits
        self._n_refs = n_refs
        self._half_bit = half_bit

        self._seed = seed
        self._state = None

        self._common_mode = default(common_mode, np.mean(fsr))

        self._eff = eff
        self._cap = cap
        self._fsr = tuple(fsr)

        self._differential = differential

        with push_random_state() as state_store:
            np.random.seed(self._seed)
        self._random_state = state_store

    def _to_json_dict(self, path_context, memo=None):
        copy_attrs = ("n_refs", "eff", "cap", "fsr", "seed", "differential", "common_mode",)
        result = {attr: getattr(self, attr) for attr in copy_attrs}
        result["n_bits"] = format_bits(self.n_bits, self.half_bit)
        return result

    def _ideal_values(self):
        n_bits = self.n_bits
        half_bit = self.half_bit

        n_caps = self.n_caps
        n_refs = self.n_refs
        n_diff = self.n_diff
        fsr = self.fsr

        eff = np.array(self.eff)
        caps = np.reshape([self.cap]*n_caps*n_diff, (n_caps, n_diff))
        refs = np.empty((n_caps, n_refs, n_diff,))

        ideal_ref = compute_ref(*fsr, n_refs)
        refs[:, :, 0] = ideal_ref
        if self.differential:
            refs[:, :, 1] = ideal_ref[::-1]

        thres = compute_thres(n_bits, *fsr, half_bit=half_bit)

        return eff, caps, refs, thres, self.common_mode

    def generate_ideal(self):
        """
        Instances an ideal stage.

        :returns: An ideal stage instance based on this metadata.
        :rtype: :class:`StageParameters`
        """
        return StageParameters(self, *self._ideal_values())

    def generate_gaussian(self, s_eff=0, s_cap=0, s_refs=0, s_thres=0, s_cm=0):
        """
        Instances a variation of the stage, with mostly gaussian perturbations.

        :param s_eff: Standard deviation of the charge transfer efficiency. The
            standard deviation is expressed in time constants as explained in
            :func:`eff_random`.
        :type s_eff: :class:`numpy.array`
        :param s_cap: Standard deviation of capacitors, in farads.
        :type s_cap: :class:`numpy.array`
        :param s_refs: Standard deviation of references, in volts.
        :type s_refs: :class:`numpy.array`
        :param s_thres: Standard deviation of sub-ADC thresholds, in volts.
        :type s_thres: :class:`numpy.array`
        :param s_cm: Standard deviation of the stage amplifier common mode,
            in volts.
        :type s_cm: :class:`numpy.array`

        :returns: An ideal stage instance based on this metadata
        :rtype: :class:`StageParameters`

        .. seealso :: :func:`eff_random`, for information about the random
            distribution of the charge transfer efficiency parameter.
        """
        eff, caps, refs, thres, cm = self._ideal_values()

        def fill_shape(arr, target):
            shape = np.shape(arr)
            assert all(s == 1 or s == t for s, t in zip(shape, target)), (
                "Shape {} cannot be extended to target {}".format(shape, target))
            arr = np.reshape(arr, shape + (1,)*(len(target) - len(shape)))
            return arr

        s_eff = fill_shape(s_eff, np.shape(eff))
        s_cap = fill_shape(s_cap, np.shape(caps))
        s_refs = fill_shape(s_refs, np.shape(refs))
        s_thres = fill_shape(s_thres, np.shape(thres))
        s_cm = fill_shape(s_cm, np.shape(cm))

        with self._random_state as _:
            eff = eff_random(eff, s_eff)
            caps = np.random.normal(caps, s_cap)
            refs = np.random.normal(refs, s_refs)
            thres = np.random.normal(thres, s_thres)
            cm = np.random.normal(cm, s_cm)

        return StageParameters(self, eff, caps, refs, thres, cm)

    def __copy__(self):
        return type(self)(  self.n_bits,
                            self.n_refs,
                            eff=self.eff,
                            cap=self.cap,
                            common_mode=self.common_mode,
                            fsr=self.fsr,
                            differential=self.differential,
                            seed=self.seed,
                            half_bit=self.half_bit,
                            data_location=self.data_location )

    def __deepcopy__(self, memo):
        return type(self)(  self.n_bits,
                            self.n_refs,
                            eff=self.eff,
                            cap=self.cap,
                            common_mode=self.common_mode,
                            fsr=copy.deepcopy(self.fsr, memo),
                            differential=self.differential,
                            seed=self.seed,
                            half_bit=self.half_bit,
                            data_location=copy.deepcopy(self.data_location, memo) )

    def __eq__(self, other):
        if isinstance(other, StageMeta):
            to_compare = ("n_bits", "half_bit", "n_refs", "common_mode", "seed", "eff", "cap", "fsr", "differential",)
            eq = all(getattr(self, attr) == getattr(other, attr) for attr in to_compare)
            return eq

        else:
            return NotImplemented


class StageParameters(data.JsonData):

    """
    This class holds the instanced data for a stage. The interpretation of this
    class is double. If interpreted as a pipeline stage, the thresholds
    represent the sub-ADC of this stage. It can be also used to represent a
    delta-sigma loop, with thresholds representing the sub-ADC of the next
    stage or the last sub-ADC in a pipeline converter.
    """

    @property
    def meta(self):
        """
        The metadata of this stage.

        :type: :class:`StageMeta`
        """
        return self._meta

    @property
    def eff(self):
        """
        The charge transfer efficiency parameter.

        :type: :class:`numpy.ndarray`
        """
        return self._eff

    @property
    def caps(self):
        """
        The capacitors value, in farads.

        :type: :class:`numpy.ndarray`
        """
        return self._caps

    @property
    def refs(self):
        """
        The references value for each capacitor, in volts.

        :type: :class:`numpy.ndarray`
        """
        return self._refs

    @property
    def thres(self):
        """
        The threshold values, in volts. Depending on the interpretation, those
        are associated with the stage sub-ADC (pipeline) or the next stage
        sub-ADC (delta-sigma).

        :type: :class:`numpy.ndarray`
        """
        return self._thres

    @property
    def n_thres(self):
        """
        Number of thresholds values.

        :type: :class:`int`
        """
        return np.size(self.thres)

    @property
    def common_mode(self):
        """
        The common_mode, derived from the references.
        """
        return self._common_mode

    @classmethod
    def _JsonDictToArgs(cls, path_context, data_location, dct, memo=None):
        _, args, kwargs = super()._JsonDictToArgs(path_context, data_location, dct, memo=memo)

        arr_attr = ("eff", "caps", "refs", "thres", "common_mode")
        meta = StageMeta.Load(path_context, dct["meta"])

        args.extend([meta] + [data.NumpyData.Load(path_context, dct[key]) for key in arr_attr])
        return cls, args, kwargs

    def __init__(self, meta, eff, caps, refs, thres, common_mode, data_location=None):
        super().__init__(data_location)

        self._meta = meta

        self._eff = data.at_least_ndarray(eff)
        self._caps = data.at_least_ndarray(caps)
        self._refs = data.at_least_ndarray(refs)
        self._thres = data.at_least_ndarray(thres)
        self._common_mode = data.at_least_ndarray(common_mode)

    def _to_json_dict(self, path_context, memo=None):
        meta = self.meta.save(path_context, memo=memo)
        arr_attr = ("eff", "caps", "refs", "thres", "common_mode")
        result = {attr: data.at_least_numpydata(getattr(self, attr)).save(path_context, memo=memo)
            for attr in arr_attr}
        result["meta"] = meta
        return result

    def __copy__(self):
        return type(self)(  self.meta,
                            self.eff,
                            self.caps,
                            self.refs,
                            self.thres,
                            self.common_mode,
                            data_location=self.data_location )

    def __deepcopy__(self, memo):
        return type(self)(  copy.deepcopy(self.meta, memo),
                            np.copy(self.eff),
                            np.copy(self.caps),
                            np.copy(self.refs),
                            np.copy(self.thres),
                            np.copy(self.common_mode),
                            data_location=copy.deepcopy(self.data_location, memo) )

    def create_modified(self, mod_dicts):
        """
        Creates a modified version of this stage. The mod_dicts is a iterable
        of dictionaries where each dict contains the following items:

        * parameter; the name of the parameter to modify
            (eff, cap, ref, thres, common_mode,).
        * index; A tuple with the index of the parameter to modify. An empty
            tuple for 'eff'.
        * value; Value on the modified parameter.

        :param mod_dicts: The parameters to modify.
        :type s_eff: Iterable of :class:`dict`
        :returns: A modified version of this stage.
        :rtype: :class:`StageParameters`
        """
        result = copy.deepcopy(self)

        for mod_dict in mod_dicts:
            param = mod_dict["parameter"]
            idx = mod_dict["index"]
            value = mod_dict["value"]

            if param == "eff":
                result._eff[idx] = value
            elif param == "cap":
                result._caps[idx] = value
            elif param == "ref":
                result._refs[idx] = value
            elif param == "cm":
                result._common_mode[idx] = value
            else:
                raise ValueError("Parameter '{}' not recognized.".format(param))

        return result

    def __eq__(self, other):
        if isinstance(other, StageParameters):
            to_compare = ("eff", "caps", "refs", "thres",)
            eq = all((getattr(self, attr) == getattr(other, attr)).all() for attr in to_compare)
            return eq

        else:
            return NotImplemented


class PipeParameters(data.JsonData):

    """
    This class just holds multiple stages together and the thresholds of the
    final sub-ADC that converts the last stage residual.
    """

    @property
    def stages(self):
        """
        The stages that make this adc.

        :type: :class:`data.tuple_of(StageParameters)`
        """
        return self._stages

    @property
    def tail(self):
        """
        The thresholds of the final sub-ADC that converts the last stage's
        residual.

        :type: :class:`numpy.array`
        """
        return self._tail

    @classmethod
    def _JsonDictToArgs(cls, path_context, data_location, dct, memo=None):
        _, args, kwargs = super()._JsonDictToArgs(path_context, data_location, dct, memo=memo)

        stages = data.tuple_of(StageParameters).Load(path_context, dct['stages'], memo=memo)
        tail = data.NumpyData.Load(path_context, dct['tail'], memo=memo)

        args.append(stages)
        args.append(tail)

        return cls, args, kwargs

    def __init__(self, stages, tail, data_location=None):
        super().__init__(data_location)
        self._stages = data.tuple_of(StageParameters).EnsureIsInstance(stages)
        self._tail = tail

    def _to_json_dict(self, path_context, memo=None):
        stages = self._stages.save(path_context, memo=memo)
        tail = data.at_least_numpydata(self._tail).save(path_context, memo=memo)

        dct = {}
        dct["stages"] = stages
        dct["tail"] = tail

        return dct

    def __copy__(self):
        return type(self)(
            self.stages,
            self.tail,
            data_location=self.data_location)

    def __deepcopy__(self, memo):
        return type(self)(
            copy.deepcopy(self.stages, memo),
            np.copy(self.tail),
            data_location=copy.deepcopy(self.data_location, memo))

    def create_modified(self, mod_dicts):
        """
        Creates a modified version of this adc. The mod_dicts is a iterable of
        dictionaries where each dict contains the following items:

        * parameter; the name of the parameter to modify
            (eff, cap, ref, thres).
        * stage; The index of stage to modify.
        * index; A tuple with the index of the parameter to modify. An empty
            tuple for 'eff'.
        * value; Value on the modified parameter.

        :param mod_dicts: The parameters to modify.
        :type s_eff: Iterable of :class:`dict`
        :returns: A modified version of this stage.
        :rtype: :class:`PipeParameters`
        """
        n_stages = len(self.stages)

        stages_mods = [mod_dict for mod_dict in mod_dicts if mod_dict["stage"] < n_stages]
        tail_mods = [mod_dict for mod_dict in mod_dicts if mod_dict["stage"] == n_stages]

        assert len([mod_dict for mod_dict in mod_dicts if mod_dict["stage"] > n_stages]) == 0

        stages_dicts = [[] for _ in range(n_stages)]

        for mod in stages_mods:
            stages_dicts[mod["stage"]].append(mod)

        new_stages = tuple(stage.create_modified(mods)
            for stage, mods in zip(self.stages, stages_dicts))

        new_tail = np.copy(self.tail)

        for mod in tail_mods:
            assert mod["parameter"] == "thres"
            new_tail[mod["index"]] = mod["value"]

        return type(self)(new_stages, new_tail)

    def as_delta_sigma(self):
        """
        Returns the stages with the thresholds associated with the following
        stage.

        :returns: A list with the delta-sigma representation of each stage.
        :rtype: :class:`list`
        """
        thres = [ss.thres for ss in islice(self.stages, 1, None)] + [self.tail]
        return [StageParameters(ss.meta, ss.eff, ss.caps, ss.refs, tt, ss.common_mode)
            for ss, tt in zip(self.stages, thres)]

    def __eq__(self, other):
        if isinstance(other, PipeParameters):
            return (len(self.stages) == len(other.stages)
                    and all(s == o for s, o in zip(self.stages, other.stages))
                    and (self.tail == other.tail).all())
        else:
            return NotImplemented

    def convert(self, data, cf_cap=None):
        """
        Performs data conversion on input data.

        :param data: Voltages to transform, shape must comply base_shape + (n_diff,)
        :type data: :class:`numpy.array`
        :param cf_cap: Index of the capacitor to assume as the feedback capacitor.
        :type cf_cap: :class:`tuple`

        :returns: The references used by each MDAC: a tuple integer arrays with
            shape = shape(data) + (n_caps - 1, n_diff,). Also the code of the tail.
        :rtype: :class:`tuple`, :class:`numpy.array`
        """

        cf_cap = default(cf_cap, tuple(stage.meta.n_caps-1 for stage in self.stages))
        assert len(cf_cap) == len(self.stages)

        assert np.size(data, -1) == self.stages[0].meta.n_diff
        base_shape = np.shape(data)[:-1]
        base_len = len(base_shape)

        result = []
        colors = ['g','c','b']

        for stage, cf_ii in zip(self.stages, cf_cap):
            meta = stage.meta
            cs_ii = list(range(0, cf_ii)) + list(range(cf_ii+1, meta.n_caps))
            p_map = pipe_map(meta.n_caps - 1, meta.n_refs,
                differential=meta.differential)

            thres = stage.thres[(np.newaxis,) * (base_len + 1) + (Ellipsis,)]
            code = np.diff(data, axis=-1, keepdims=True) if meta.differential else data
            code = np.sum(code[..., np.newaxis] >= thres, axis=-1)

            diff_idx = (1,) * base_len + (meta.n_diff,)
            diff_idx = np.reshape(list(range(meta.n_diff)), diff_idx)
            refs_idx = p_map[:, code, diff_idx]

            refs_idx = np.transpose(refs_idx, tuple(range(1, base_len + 1)) + (0, -1,))
            result.append(refs_idx)

            g = stage.caps[cs_ii, ...] / stage.caps[[cf_ii], ...]
            g = g[(np.newaxis,) * base_len + (Ellipsis,)]

            cs_ii = np.reshape(cs_ii, (1,) * base_len + (len(cs_ii), 1,))
            refs = stage.refs[cs_ii, refs_idx, diff_idx[np.newaxis, ...]]

            data = data + np.sum(g*(data[..., np.newaxis, :] - refs), axis=-2)
            data = data * stage.eff + (1 - stage.eff) * stage.common_mode

        thres = self.tail[(np.newaxis,) * base_len + (Ellipsis,)]
        code = np.diff(data, axis=-1, keepdims=True) if meta.differential else data
        code = np.sum(code[..., np.newaxis] >= thres, axis=(-1, -2,))

        return tuple(result), code

    def rebuild(self, conversion, tail_code, cf_cap=None):
        """
        Recreates the voltage from a conversion

        :param conversion: Tuples of references of convesion,
            shape = base_shape + (n_caps-1, n_diff,)
        :type conversion: :class:`tuple`
        :param conversion: Codes generated by the tail.
        :type conversion: :class:`numpy.array`
        :param cf_cap: Index of the capacitor to assume as the feedback capacitor.
        :type cf_cap: :class:`tuple`

        :returns: The converted voltage
            shape = shape(data) + (n_caps - 1, n_diff,)
        :rtype: :class:`ndarray`
        """
        cf_cap = default(cf_cap, tuple(stage.meta.n_caps-1 for stage in self.stages))
        assert len(cf_cap) == len(self.stages)
        assert len(conversion) == len(self.stages)

        base_shapes = tuple(np.shape(arr)[:-2] for arr in conversion)
        assert all(base_shape == base_shapes[0] for base_shape in base_shapes)
        base_shape = base_shapes[0]
        base_len = len(base_shape)

        tail_mid = (self.tail[1:, ...] + self.tail[:-1, ...])/2

        if np.size(self.tail, 0) > 1:
            extremes = (tail_mid[0], tail_mid[-1],)
            lsb = np.mean(np.diff(self.tail), axis=0)

        else:
            extremes = (0, 0,)
            fsr = self.stages[-1].meta.fsr
            lsb = (fsr[1] - fsr[0]) / 4

        tail_mid = np.concatenate(
            ([extremes[0] - lsb], tail_mid, [extremes[1] + lsb],), axis=0)

        result = tail_mid[tail_code]

        for stage, conv, cf_ii in reversed(list(zip(self.stages, conversion, cf_cap))):
            meta = stage.meta
            assert meta.n_diff == 1, "Differential not supported"
            cs_ii = list(range(0, cf_ii)) + list(range(cf_ii+1, meta.n_caps))
            cs_ii = np.reshape(cs_ii, (1,) * base_len + (len(cs_ii), 1,))
            diff_ii = np.reshape(list(range(meta.n_diff)), (1,) * base_len + (1, meta.n_diff,))

            refs = stage.refs[cs_ii, conv, diff_ii]
            g = stage.caps[cs_ii, diff_ii] / stage.caps[[cf_ii], diff_ii]

            sum_axis = (-1, -2,)
            eff = stage.eff
            cm = stage.common_mode

            result = ( (np.sum(g * refs, axis=sum_axis) * eff + result - (1 - eff) * cm)
                    / (eff * (1 + np.sum(g, axis=sum_axis))) )

        return result

    def as_ideal(self):
        stages = [stage.meta.generate_ideal() for stage in self.stages]

        half_bit_tail = np.size(self.tail) % 2 == 0
        n_bits_tail = int(np.log2(np.size(self.tail)))
        tail = gen.compute_thres(n_bits_tail, *stages[-1].meta.fsr,
            half_bit=half_bit_tail)

        return PipeParameters(stages, tail)


## IC OBJECT ##


class InitialCondition(data.JsonData):
    """
    Holds the information about the pre-charging of the delta-sigma loop before
    operation.
    """
    @classmethod
    def _JsonDictToArgs(cls, path_context, data_location, dct, memo=None):
        _, args, kwargs = super()._JsonDictToArgs(path_context, data_location, dct, memo=memo)

        args.append(StageMeta.Load(path_context, dct['meta'], memo=memo))
        args.append(data.NumpyData.Load(path_context, dct['ref_ii'], memo=memo))

        return cls, args, kwargs

    @classmethod
    def Discharged(cls, meta, n_cf):
        """
        Generate a discharged initial condition.

        :param meta: Delta-sigma metadata
        :type meta: :class:`StageMeta`
        :param n_cf: Number of feedback capacitors
        :type n_cf: :class:`int`
        """
        n_diff = meta.n_diff
        ref_ii = np.array([-1]*n_cf, dtype=int)
        ref_ii = np.stack((ref_ii,)*n_diff, axis=1)
        return cls(meta, ref_ii)

    @property
    def ref_ii(self):
        """
        Indexes of the references to precharge. The output shape is
        (n_cf, n_diff,). Index -1 to not use that capacitor.

        :type: :class:`numpy.array`
        """
        return self._ref_ii

    @property
    def n_cf(self):
        """
        Number of feedback capacitors.

        :type: :class:`int`
        """
        return np.size(self._ref_ii, 0)

    @property
    def meta(self):
        """
        Delta-sigma metadata

        :type: :class:`StageMeta`
        """
        return self._meta

    def __init__(self, meta, ref_ii, data_location=None):
        super().__init__(data_location)

        assert np.size(ref_ii, 1) == 1 + meta.differential

        self._meta = meta
        self._ref_ii = data.at_least_ndarray(ref_ii, dtype=int)

    def _to_json_dict(self, path_context, memo=None):
        dct = {}
        dct['meta'] = self._meta.save(path_context, memo=memo)
        dct['ref_ii'] = data.at_least_numpydata(self._ref_ii, dtype=int).save(path_context, memo=memo)
        return dct

    def __eq__(self, other):
        if isinstance(other, InitialCondition):
            return self.meta == other.meta and (self.ref_ii == other.ref_ii).all()
        else:
            return NotImplemented


## INPUT OBJECTS ##


class InputGenerator(data.JsonData):
    """
    Abstract class for generating the indexes of the input signals.
    """

    @classmethod
    def register(cls, class_):
        """
        Register a sub-class to the json tag system.

        .. seealso :: :func:`_JsonDictToArgs` for information about the json
            tag system.
        """
        cls._registered[class_.tag()] = class_

    @abc.abstractclassmethod
    def tag(cls):
        """
        Tag of the sub-class to be used when reading / writing json.

        .. seealso :: :func:`_to_json_dict` for information about the json
            tag system.
        """
        pass

    @property
    @abc.abstractmethod
    def n_cs(self):
        """
        Number of feedfoward capacitors.

        .. seealso :: :func:`_to_json_dict` for information about the json
            tag system.
        """
        pass

    @property
    @abc.abstractmethod
    def n_ins(self):
        """
        Minimum number of extenral input to support this generator.

        :type: :class:`int`
        """
        pass

    @property
    def meta(self):
        """
        Delta-sigma metadata

        :type: :class:`StageMeta`
        """
        return self._meta

    @classmethod
    def _JsonDictToArgs(cls, path_context, data_location, dct, memo=None):
        """
        Reads a dictionary and parses it into constructor positional and
        keyword arguments.

        Sub-classes of this class have to read the 'data' entry of
        this dictionary instead of the root dictionary when implementing either
        :func:`_to_json_dict` or :func:`_JsonDictToArgs`.

        :param data_location: Location where the dictionary has been read.
        :type data_location: :class:`data.DataLocation`
        :param dct: A dictionary representation of the instance.
        :type dct: :class:`dict`
        :param memo: For detecting already loaded files, same as memo for
            :func:`copy.deepcopy`.
        :type memo: :class:`dict`
        :returns: The arguments and keyword arguments needed or ready to be
            further populated to instance an :class:`InputGenerator`.
        :rtype: (:class:`list`, :class:`dict`)

        .. seealso :: :func:`_to_json_dict`, for information about the
            modifications on the usual :class:`data.JsonData` interface.
        """
        if(cls is InputGenerator):
            tag = dct['tag']
            data = dct['data']
            return cls._registered[tag]._JsonDictToArgs(path_context, data_location, data, memo=memo)
        else:
            _, args, kwargs = super()._JsonDictToArgs(path_context, data_location, dct, memo=memo)
            args.append(StageMeta.Load(path_context, dct['meta'], memo=memo))
            return cls, args, kwargs

    def __init__(self, meta, data_location=None):
        super().__init__(data_location)
        self._meta = meta

    def _to_json_dict(self, path_context, memo=None):
        """
        Dump the instance information to a json dictionary. This method breaks
        the :class:`data.JsonData` interface, since the instance information is
        nested in the entry 'data' of another dictionary. This root dictionary
        contains also the 'tag' entry. This entry enables this class to select
        which of the registered sub-class to load when the dictionary is then
        loaded.

        Sub-classes of this class have to populate and read the 'data' entry of
        this dictionary instead of the root dictionary when implementing either
        :func:`_to_json_dict` or :func:`_JsonDictToArgs`.

        :param path_context: Path from where the save operation is being
            invoked.
        :type path_context: :class:`data.PathContext`
        :returns: A dictionary representation of the instance.
        :rtype: :class:`dict`

        .. seealso :: :func:`_JsonDictToArgs`, the complementary operation.
        """
        return {'tag': self.tag(), 'data':
            {'meta': self._meta.save(path_context, memo=memo)}}

    # Returns refs, ins
    # shape of ref: (n_samples, n_cs)
    # shape of in: (n_samples, n_cs)
    def generate_in(self, n_samples, start_sample=0):
        """
        Get the indexes of either the references or input values to charge the
        feedfoward capacitors on :math:`\\phi_0.`

        :param n_samples: Number of clock cycles to generate.
        :type n_samples: :class:`int`
        :param n_samples: Number of clock cycles to skip before generation.
        :type n_samples: :class:`int`
        :returns: The indexes of references and inputs, in that order. The
            shape of both outputs is (n_samples, n_cs, n_diff)
        :rtype: :class:`numpy.ndarray`
        """
        pass


InputGenerator._registered = dict()


class ExternalDC(InputGenerator):
    """
    Generate a constant input from the pipeline input pin.
    """

    @classmethod
    def tag(cls):
        return "external-dc"

    @property
    def n_cs(self):
        return np.size(self.ins_ii, 0)

    @property
    def n_ins(self):
        return np.max(self._ins_ii) + 1

    @property
    def ins_ii(self):
        """
        Index of the external input.

        :type: :class:`numpy.ndarray`
        """
        return self._ins_ii

    def __init__(self, meta, ins_ii, data_location=None):
        super().__init__(meta, data_location)
        ins_ii = data.at_least_ndarray(ins_ii, dtype=int)
        if len(np.shape(ins_ii)) == 1:
            ins_ii = np.stack((ins_ii,)*meta.n_diff, axis=1)
        assert len(np.shape(ins_ii)) == 2
        assert (ins_ii == ins_ii.flatten()[0]).all(), ("Different inputs for "
            "different capacitors is not possible by architecture.")
        self._ins_ii = ins_ii

    @classmethod
    def _JsonDictToArgs(cls, path_context, data_location, dct, memo=None):
        _, args, kwargs = super()._JsonDictToArgs(path_context, data_location, dct, memo=memo)
        ins_ii = data.NumpyData.Load(path_context, dct['ins_ii'], memo=memo)

        args.append(ins_ii)

        return cls, args, kwargs

    def _to_json_dict(self, path_context, memo=None):
        dct = super()._to_json_dict(path_context, memo=memo)
        ins_ii = data.at_least_numpydata(self.ins_ii, dtype=int).save(path_context, memo=memo)
        dct['data']['ins_ii'] = ins_ii
        return dct

    def generate_in(self, n_samples, start_sample=0):
        n_diff = self.meta.n_diff
        shape = (1, self.n_cs, n_diff,)
        full_shape = (n_samples, self.n_cs, n_diff,)

        refs = np.broadcast_to(np.reshape([-1] * self.n_cs * n_diff, shape), full_shape)
        ins = np.broadcast_to(np.reshape(self.ins_ii, shape), full_shape)

        return refs, ins

    def __eq__(self, other):
        if isinstance(other, ExternalDC):
            return (self.meta == other.meta
                and self.n_cs == other.n_cs
                and (self.ins_ii, other.ins_ii).all() )
        else:
            return NotImplemented


InputGenerator.register(ExternalDC)


class InternalDC(InputGenerator):
    """
    Generate a constant input from the references of the feedfoward capacitors.
    """

    @classmethod
    def tag(cls):
        return "internal-dc"

    @property
    def n_cs(self):
        return np.size(self.refs_ii, 0)

    @property
    def n_ins(self):
        return 0

    @property
    def refs_ii(self):
        """
        Index of the internal references.

        :type: :class:`numpy.ndarray`
        """
        return self._refs_ii

    def __init__(self, meta, refs_ii, data_location=None):
        super().__init__(meta, data_location)
        assert len(np.shape(refs_ii)) == 2
        self._refs_ii = data.at_least_ndarray(refs_ii, dtype=int)

    @classmethod
    def _JsonDictToArgs(cls, path_context, data_location, dct, memo=None):
        _, args, kwargs = super()._JsonDictToArgs(path_context, data_location, dct, memo=memo)
        refs_ii = data.NumpyData.Load(path_context, dct['refs_ii'], memo=memo)
        args.append(refs_ii)
        return cls, args, kwargs

    def _to_json_dict(self, path_context, memo=None):
        dct = super()._to_json_dict(path_context, memo=memo)
        refs_ii = data.at_least_numpydata(self.refs_ii, dtype=int).save(path_context, memo=memo)
        dct['data']['refs_ii'] = refs_ii
        return dct

    def generate_in(self, n_samples, start_sample=0):
        n_diff = self.meta.n_diff
        shape = (1, self.n_cs, n_diff,)
        full_shape = (n_samples, self.n_cs, n_diff,)

        ins = np.broadcast_to(np.reshape([-1] * self.n_cs * n_diff, shape), full_shape)
        refs = np.broadcast_to(np.reshape(self.refs_ii, shape), full_shape)

        return refs, ins

    def __eq__(self, other):
        if isinstance(other, InternalDC):
            return (self.meta == other.meta
                and self.n_cs == other.n_cs
                and (self.refs_ii, other.refs_ii).all() )
        else:
            return NotImplemented


InputGenerator.register(InternalDC)


class ACCombinator(InputGenerator):

    """
    Switches between two inputs generators, alternating them in a periodic way.
    """

    @classmethod
    def tag(cls):
        return "ac-combinator"

    @property
    def n_cs(self):
        return self.top.n_cs

    @property
    def n_ins(self):
        return max(self.top.n_ins, self.bot.n_ins)

    @property
    def top(self):
        """
        First input generator to select in the periodic switching.

        :type: :class:`InputGenerator`
        """
        return self._top

    @property
    def bot(self):
        """
        Second input generator to select in the periodic switching.

        :type: :class:`InputGenerator`
        """
        return self._bot

    @property
    def period(self):
        """
        Switching period.

        :type: :class:`int`
        """
        return self._period

    def __init__(self, meta, top, bot, period, data_location=None):
        super().__init__(meta, data_location=data_location)

        assert period > 0, "Period must be greater than 0"
        assert top.n_cs == bot.n_cs, "Top and bottom generators' n_cs must be the same."
        assert top.meta == meta
        assert bot.meta == meta

        self._top = top
        self._bot = bot
        self._period = period

    @classmethod
    def _JsonDictToArgs(cls, path_context, data_location, dct, memo=None):
        _, args, kwargs = super()._JsonDictToArgs(path_context, data_location, dct, memo=memo)

        top = InputGenerator.Load(path_context, dct["top"], memo=memo)
        bottom = InputGenerator.Load(path_context, dct["bot"], memo=memo)

        args.append(top)
        args.append(bottom)
        args.append(dct["period"])

        return cls, args, kwargs

    def _to_json_dict(self, path_context, memo=None):
        dct = super()._to_json_dict(path_context, memo=memo)
        dct['data'].update({
            'top': self.top.save(path_context, memo=memo),
            'bot': self.bot.save(path_context, memo=memo),
            'period': self.period })
        return dct

    def generate_in(self, n_samples, start_sample=0):
        r_top, i_top = self.top.generate_in(n_samples, start_sample)
        r_bot, i_bot = self.bot.generate_in(n_samples, start_sample)

        refs = np.empty_like(r_top)
        ins = np.empty_like(i_top)

        refs[...] = -1
        ins[...] = -1

        p = self.period

        mask = np.arange(start_sample, n_samples + start_sample)
        mask = mask % p < p//2

        refs[mask, ...] = r_top[mask, ...]
        ins[mask, ...] = i_top[mask, ...]

        mask = np.logical_not(mask)

        refs[mask, ...] = r_bot[mask, ...]
        ins[mask, ...] = i_bot[mask, ...]

        return refs, ins

    def __eq__(self, other):
        if isinstance(other, ACCombinator):
            return all(getattr(self, attr) == getattr(other, attr) for attr in
                ("period", "top", "bot", "meta",))
        else:
            return NotImplemented


InputGenerator.register(ACCombinator)


class InternalRandom(InputGenerator):
    """
    Generate a random input from the references of the feedfoward capacitors.
    """

    @classmethod
    def tag(cls):
        return "internal-random"

    @property
    def n_cs(self):
        return self._n_cs

    @property
    def n_ins(self):
        return 0

    @property
    def seed(self):
        """
        Seed of the generation.

        :type: :class:`int`
        """
        return self._seed

    def __init__(self, meta, n_cs, seed, data_location=None):
        super().__init__(meta, data_location)
        self._n_cs = n_cs
        self._seed = seed

    @classmethod
    def _JsonDictToArgs(cls, path_context, data_location, dct, memo=None):
        _, args, kwargs = super()._JsonDictToArgs(path_context, data_location, dct, memo=memo)
        args.append(dct["n_cs"])
        args.append(dct["seed"])
        return cls, args, kwargs

    def _to_json_dict(self, path_context, memo=None):
        dct = super()._to_json_dict(path_context, memo=memo)
        dct['data']['n_cs'] = self.n_cs
        dct['data']['seed'] = self.seed
        return dct

    def generate_in(self, n_samples, start_sample=0):
        tot_samples = start_sample + n_samples
        n_diff = self.meta.n_diff
        shape = (1, self.n_cs, n_diff,)
        full_shape = (tot_samples, self.n_cs, n_diff,)
        crop_idx = (slice(start_sample, start_sample+n_samples), slice(None), slice(None),)

        ins = np.broadcast_to(np.reshape([-1] * self.n_cs * n_diff, shape), full_shape)
        ins = ins[crop_idx]

        with push_random_state() as _:
            np.random.seed(self.seed)
            refs = np.random.randint(0, self.meta.n_refs, size=full_shape[:-1] + (1,))
            if n_diff == 2:
                o_refs = (self.meta.n_refs - 1) - refs
                refs = np.concatenate((o_refs, refs,), axis=-1)

        refs = refs[crop_idx]

        return refs, ins

    def __eq__(self, other):
        if isinstance(other, InternalRandom):
            return (self.meta == other.meta
                and self.n_cs == other.n_cs
                and self.seed == other.seed )
        else:
            return NotImplemented


InputGenerator.register(InternalRandom)


class InputMask(InputGenerator):
    """
    Masks an input-generator capacitor.
    """

    @classmethod
    def tag(cls):
        return "input-mask"

    @property
    def n_cs(self):
        return np.sum(self.mask)

    @property
    def n_ins(self):
        return 0

    @property
    def mask(self):
        """
        Mask applied to the input generator.

        :type: :class:`np.ndarray`
        """
        return self._mask

    @property
    def generator(self):
        """
        Original generator.

        :type: :class:`InputGenerator`
        """
        return self._generator

    def __init__(self, meta, generator, mask, data_location=None):
        super().__init__(meta, data_location)
        assert generator.n_ins == 0, "Only masks internal generators"
        self._generator = generator

        assert mask.dtype == np.dtype(bool)
        assert len(np.shape(mask)) == 1
        assert np.size(mask) == generator.n_cs

        self._mask = mask

    @classmethod
    def _JsonDictToArgs(cls, path_context, data_location, dct, memo=None):
        _, args, kwargs = super()._JsonDictToArgs(path_context, data_location, dct, memo=memo)
        args.append(InputGenerator.Load(path_context, dct["generator"], memo=memo))
        args.append(data.NumpyData.Load(path_context, dct["mask"], memo=memo))
        return cls, args, kwargs

    def _to_json_dict(self, path_context, memo=None):
        dct = super()._to_json_dict(path_context, memo=memo)
        dct['data']['generator'] = self.generator.save(path_context, memo=memo)
        dct['data']['mask'] = data.at_least_numpydata(self.mask).save(path_context, memo=memo)
        return dct

    def generate_in(self, n_samples, start_sample=0):
        refs, ins = self.generator.generate_in(n_samples, start_sample=start_sample)
        mask = self.mask
        return refs[:, mask, :], ins[:, mask, :]

    def __eq__(self, other):
        if isinstance(other, InternalRandom):
            return (self.meta == other.meta
                and self.mask == other.mask
                and self.generator == other.generator )
        else:
            return NotImplemented


InputGenerator.register(InputMask)


## CONFIGURATION OBJECT ##


class Configuration(data.JsonData):

    """
    Specifies a particular feedback and feedfoward capacitor selection.
    """

    @property
    def n_cs(self):
        """
        Number of feedfoward capacitors.

        :type: :class:`int`
        """
        return np.size(self._cs, 0)

    @property
    def n_cf(self):
        """
        Number of feedback capacitors.

        :type: :class:`int`
        """
        return self.meta.n_caps - self.n_cs

    # shape (n_cs, n_diff,)
    @property
    def cs(self):
        """
        Indexes of the feedfoward capacitors. The output shape is
            (n_cs, n_diff,).

        :type: :class:`numpy.ndarray`
        """
        return self._cs

    # shape (n_cf, n_diff,)
    @property
    def cf(self):
        """
        Indexes of the feedback capacitors. The output shape is
            (n_cf, n_diff,).

        :type: :class:`numpy.ndarray`
        """
        all_caps = set(range(self.meta.n_caps))
        cf = []
        cs = self.cs
        for diff in range(np.size(cs, 1)):
            cf.append(list(all_caps - set(cs[:, diff])))

        cf = np.transpose(cf, axes=(1,0,))
        return cf

    @property
    def meta(self):
        """
        Delta-sigma metadata

        :type: :class:`StageMeta`
        """
        return self._meta

    def __init__(self, meta, cs, data_location=None):
        super().__init__(data_location)

        if len(np.shape(cs)) == 1:
            cs = np.reshape(cs, np.shape(cs) + (1,))

        assert np.size(cs, 1) == meta.n_diff

        self._meta = meta
        self._cs = data.at_least_ndarray(cs)

    @classmethod
    def _JsonDictToArgs(cls, path_context, data_location, dct, memo=None):
        _, args, kwargs = super()._JsonDictToArgs(path_context, data_location, dct, memo=memo)

        args.append(StageMeta.Load(path_context, dct['meta'], memo=memo))
        args.append(data.NumpyData.Load(path_context, dct['cs'], memo=memo))

        return cls, args, kwargs

    def _to_json_dict(self, path_context, memo=None):
        dct = {}

        dct["meta"] = self.meta.save(path_context, memo=memo)
        dct["cs"] = data.at_least_numpydata(self._cs).save(path_context, memo=memo)

        return dct

    def __eq__(self, other):
        if isinstance(other, Configuration):
            return self.meta == other.meta and (self._cs == other._cs).all()
        else:
            return NotImplemented


class ConfigurationSet(data.JsonData):

    """
    Represents a number of configuration and inputs pairs.
    """

    @classmethod
    def Stack(cls, conf_set0, *conf_sets):
        assert all(conf_set0.ds_samples == conf_set.ds_samples for conf_set in conf_sets)
        conf_sets = (conf_set0,) + conf_sets

        inputs = tuple(e for conf_set in conf_sets for e in conf_set.inputs)
        configurations = tuple(e for conf_set in conf_sets for e in conf_set.configurations)

        return cls(conf_set0.ds_samples, inputs, configurations)

    @property
    def meta(self):
        """
        Delta-sigma metadata

        :type: :class:`StageMeta`
        """
        return self._configurations[0].meta

    @property
    def n_conf(self):
        """
        Number of configuration-input pairs

        :type: :class:`int`
        """
        return max(len(self._inputs), len(self._configurations))

    @property
    def inputs(self):
        """
        Stage input generators

        :type: :class:`InputGenerator`
        """
        inputs = self._inputs
        return inputs * self.n_conf if len(inputs) == 1 else inputs

    @property
    def configurations(self):
        """
        Stage configurations

        :type: :class:`Configuration`
        """
        conf = self._configurations
        return conf * self.n_conf if len(conf) == 1 else conf

    @property
    def n_cs(self):
        """
        Number of feedfoward capacitors in all configurations and inputs.

        :type: :class:`int`
        """
        return self._configurations[0].n_cs

    @property
    def n_cf(self):
        """
        Number of feedback capacitors in all configurations and inputs.

        :type: :class:`int`
        """
        return self._configurations[0].n_cf

    @property
    def cs(self):
        """
        Indexes of the feedfoward capacitors. The output shape is
            (n_conf, n_cs, n_diff,).

        :type: :class:`numpy.ndarray`
        """
        if hasattr(self, "_cs_cache"):
            return self._cs_cache
        return np.array([conf.cs for conf in self.configurations], dtype=int)

    @property
    def cf(self):
        """
        Indexes of the feedback capacitors. The output shape is
            (n_conf, n_cf, n_diff,).

        :type: :class:`numpy.ndarray`
        """
        if hasattr(self, "_cf_cache"):
            return self._cf_cache
        return np.array([conf.cf for conf in self.configurations], dtype=int)

    @property
    def ds_samples(self):
        """
        Number of samples each configuration is operated.

        :type: :class:`int`
        """
        return self._ds_samples

    def clear_cache(self):
        if hasattr(self, "_cs_cache"):
            del self._cs_cache
        if hasattr(self, "_cf_cache"):
            del self._cf_cache
        if hasattr(self, "_in_cache"):
            del self._in_cache

    def build_cache(self, samples=None):
        self._cs_cache = self.cs
        self._cf_cache = self.cf
        self._in_cache = self.generate_in(default(samples, self.ds_samples))

    def expand_cache(self, samples=None):
        if not hasattr(self, "_in_cache"):
            self.build_cache(samples)
        elif np.size(self._in_cache[0], 0) < samples:
            del self._in_cache
            self._in_cache = self.generate_in(default(samples, self.ds_samples))

    # shape of ref: (n_samples, n_conf, n_cs)
    # shape of in: (n_samples, n_conf, n_cs)
    def generate_in(self, n_samples, start_sample=0):
        """
        The result of generating all the inputs. The output shape is
        (n_samples, n_conf, n_cs, n_diff)

        :param n_samples: Number of clock cycles to generate.
        :type n_samples: :class:`int`
        :param n_samples: Number of clock cycles to skip before generating.
        :type n_samples: :class:`int`
        :returns: The references and input indexes, in that order, that
            generate the delta-sigma input.
        :rtype: (:class:`numpy.array`, :class:`numpy.array`,)

        .. seealso :: :func:`InputGenerator.generate_in`
        """
        if hasattr(self, "_in_cache"):
            refs, ins = self._in_cache
            assert start_sample + n_samples <= np.size(refs, 0), ("Asked {}:{}"
                " of a total of {}".format(start_sample, n_samples, np.size(refs, 0)))

            idx = (slice(start_sample, start_sample + n_samples), Ellipsis,)
            return refs[idx], ins[idx]

        refs_ins = tuple(input.generate_in(n_samples, start_sample) for input in self.inputs)
        refs, ins = tuple(zip(*refs_ins))

        refs = np.array(list(refs), dtype=int)
        ins = np.array(list(ins), dtype=int)

        refs = np.transpose(refs, axes=(1, 0, 2, 3,))
        ins = np.transpose(ins, axes=(1, 0, 2, 3,))

        return refs, ins

    def __init__(self, ds_samples, inputs, configurations, data_location=None):
        super().__init__(data_location)

        assert len(configurations) > 0
        assert len(inputs) == 1 or len(configurations) == 1 or len(configurations) == len(inputs)
        assert ds_samples > 0

        self._inputs = data.tuple_of(InputGenerator).EnsureIsInstance(inputs)
        self._configurations = data.tuple_of(Configuration).EnsureIsInstance(configurations)
        self._ds_samples = int(ds_samples)

        assert all(c_.n_cs == i_.n_cs for c_, i_ in zip(configurations, inputs))

        meta = self.meta
        assert all(c_.meta == meta and i_.meta == meta
            for c_, i_ in zip(configurations, inputs))

    @classmethod
    def _JsonDictToArgs(cls, path_context, data_location, dct, memo=None):
        _, args, kwargs = super()._JsonDictToArgs(path_context, data_location, dct, memo=memo)

        args.append(dct["ds_samples"])
        args.append(data.tuple_of(InputGenerator).Load(path_context, dct["inputs"], memo=memo))
        args.append(data.tuple_of(Configuration).Load(path_context, dct["configurations"], memo=memo))

        return cls, args, kwargs

    def _to_json_dict(self, path_context, memo=None):
        dct = {}

        dct["inputs"] = self._inputs.save(path_context, memo=memo)
        dct["configurations"] = self._configurations.save(path_context, memo=memo)
        dct["ds_samples"] = self._ds_samples

        return dct

    def __eq__(self, other):
        if isinstance(other, ConfigurationSet):
            return (self.n_conf == other.n_conf
                and all(s == o for s, o in zip(self.configurations, other.configurations)) )
        else:
            return NotImplemented


class ConfigurationSequence(data.JsonData):

    """
    Represents a series of chained configuration sets preceded by the initial
    condition to start the simulation.
    """

    @classmethod
    def Stack(cls, conf_seq0, *conf_seqs):
        assert(len(conf_seq0.configuration_sets) == len(conf_seq.configuration_sets) for conf_seq in conf_seqs)
        conf_seqs = (conf_seq0,) + conf_seqs

        conf_sets = []
        for all_conf_sets in zip(*tuple(conf_seq.configuration_sets for conf_seq in conf_seqs)):
            conf_sets.append(ConfigurationSet.Stack(*all_conf_sets))

        ics = tuple(e for conf_seq in conf_seqs for e in conf_seq.initial_conditions)
        return cls(ics, conf_sets)

    @property
    def meta(self):
        """
        Delta-sigma metadata

        :type: :class:`StageMeta`
        """
        return self._configuration_sets[0].meta

    @property
    def configuration_sets(self):
        """
        Chained sequence of configuration sets.

        :type: :class:`Sequence`
        """
        return self._configuration_sets

    @property
    def initial_conditions(self):
        """
        The initial conditions for each configuration set chain.

        :type: :class:`Sequence`
        """
        return self._initial_conditions

    @property
    def n_conf(self):
        """
        Number of configurations for each configuration set.

        :type: :class:`int`
        """
        return self._configuration_sets[0].n_conf

    @property
    def n_ins(self):
        """
        Minimum number of external inputs

        :type: :class:`int`
        """
        return np.max([[input.n_ins for input in c_set.inputs]
                            for c_set in self.configuration_sets])

    def clear_cache(self):
        for c_set in self.configuration_sets:
            c_set.clear_cache()

        # for ic in self.initial_conditions:
        #     ic.clear_cache()

    def build_cache(self):
        samples = 0
        for c_set in self.configuration_sets:
            samples += c_set.ds_samples
            c_set.expand_cache(samples)
            samples += 1

        # for ic in self.initial_conditions:
        #     ic.build_cache()

    @property
    def samples(self):
        return (   sum([c_set.ds_samples for c_set in self.configuration_sets])
                + (len(self.configuration_sets) - 1) )

    def __init__(self, initial_conditions, configuration_sets, data_location=None):
        super().__init__(data_location)

        # sets consistency
        assert all(c_set.meta == configuration_sets[0].meta for c_set in configuration_sets)
        assert all(c_set.ds_samples == configuration_sets[0].ds_samples for c_set in configuration_sets)

        # initial condition and sets consistency
        assert all(len(initial_conditions) == c_set.n_conf for c_set in configuration_sets)
        assert all(i_cond.meta == conf.meta for conf, i_cond in
            zip(configuration_sets[0].configurations, initial_conditions))

        assert all(i_cond.n_cf == conf.n_cf for conf, i_cond in
            zip(configuration_sets[0].configurations, initial_conditions))

        self._configuration_sets = data.tuple_of(ConfigurationSet).EnsureIsInstance(configuration_sets)
        self._initial_conditions = data.tuple_of(InitialCondition).EnsureIsInstance(initial_conditions)

    @classmethod
    def _JsonDictToArgs(cls, path_context, data_location, dct, memo=None):
        _, args, kwargs = super()._JsonDictToArgs(path_context, data_location, dct, memo=memo)

        args.append(data.tuple_of(InitialCondition).Load(path_context, dct["initial_conditions"], memo=memo))
        args.append(data.tuple_of(ConfigurationSet).Load(path_context, dct["configuration_sets"], memo=memo))

        return cls, args, kwargs

    def _to_json_dict(self, path_context, memo=None):
        dct = {}
        dct["initial_conditions"] = self._initial_conditions.save(path_context, memo=memo)
        dct["configuration_sets"] = self._configuration_sets.save(path_context, memo=memo)
        return dct

    def __eq__(self, other):
        if isinstance(other, ConfigurationSequence):
            return (len(self.initial_conditions) == len(other.initial_conditions)
                and all(s == o for s, o in zip(self.configuration_sets, other.configuration_sets))
                and all(s == o for s, o in zip(self.initial_conditions, other.initial_conditions)) )
        else:
            return NotImplemented
