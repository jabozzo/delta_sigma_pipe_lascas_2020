#! /usr/bin/env python

import unittest

import numpy as np

import calib.gen as gen
import calib.analysis as an


class TestAnalysis(unittest.TestCase):

    def test_snr(self):
        TOLERANCE = 0.03
        for bits in range(2, 10):
            for half_bit in (False, True,):
                n_refs = 3 if half_bit else 2
                meta = gen.StageMeta(bits, n_refs, eff=1, fsr=(-1,1,), half_bit=half_bit)
                stage = meta.generate_ideal()
                adc = gen.PipeParameters([stage], gen.compute_thres(1, *meta.fsr))
                snr = an.snr(adc, sampling_factor=40)
                enob = an.snr_to_enob(snr)

                self.assertTrue(np.abs((bits + 1) - (enob + 0.293)) - TOLERANCE <= 0)
