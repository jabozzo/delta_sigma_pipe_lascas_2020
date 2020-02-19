#! /usr/bin/env python

import numpy as np

from calib.misc import default


def snr(adc_real, adc_assumed=None, sampling_factor=16):
    """
    Computes the signal to noise ratio (SNR) of an ADC. The output is in
    absolute magnitude (not decibels or bits).

    :param adc_real: Adc to be simulated
    :type n_caps: :class:`gen.PipeParameters`
    :param adc_assumed: Parameters used to calibrate the conversion, None to
        assume perfect knowledge.
    :type adc_assumed: :class:`gen.PipeParameters`
    :param sampling_factor: The number of samples used to calculate the SNR is
        `(2**adc_n_bits + 1) * sampling_factor`, more samples increases
        precision.
    :type sampling_factor: :class:`int`
    :returns: The signal to noise ratio of the adc assuming its parameters, in
        absoulte magnitude.
    :rtype: :class:`float`
    """
    adc_assumed = default(adc_assumed, adc_real)

    assert all(real.meta == assumed.meta for real, assumed in
        zip(adc_real.stages, adc_assumed.stages))
    assert len(adc_real.stages) == len(adc_assumed.stages)
    assert np.size(adc_real.tail) == np.size(adc_assumed.tail)

    tot_bits = ( sum([stage.meta.n_bits for stage in adc_real.stages])
                + int(np.log2(np.size(adc_real.tail))) )
    samples = int((2**tot_bits + 1) * sampling_factor)
    original_signal = np.linspace(*adc_real.stages[0].meta.fsr, samples)

    conversion, tail_code = adc_real.convert(original_signal[..., np.newaxis])
    converted_signal = adc_assumed.rebuild(conversion, tail_code)

    signal_power = np.power(original_signal - np.mean(original_signal), 2)
    error_power = np.power(original_signal - converted_signal, 2)

    return np.sum(signal_power) / np.sum(error_power)


def magnitude_to_db(value, is_power=True):
    pow_fact = 10 if is_power else 20
    return pow_fact * np.log10(value)


def snr_to_enob(value):
    return (magnitude_to_db(value) - 10 * np.log10(3/2)) / (20*np.log10(2))
