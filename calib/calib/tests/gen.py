#! /usr/bin/env python

import unittest
import tempfile
import shutil

import numpy as np

import calib.gen as gen
import calib.data as data


class TestGen(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path_context = data.PathContext.cwd()

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_pipe_map(self):
        expected = (
            [[0, 1, 2]],
            [[0, 1, 1, 1, 2, 2, 2],
             [0, 0, 1, 1, 1, 2, 2],
             [0, 0, 0, 1, 1, 1, 2]],
            [[0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
             [0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
             [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
             [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2],
             [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2],
             [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2]], )

        expected = tuple(np.reshape(ex, np.shape(ex) + (1,)) for ex in expected)

        for h_bits, exp in zip(range(1, 1+len(expected)), expected):
            n_caps = 2**h_bits - 1
            generated = gen.pipe_map(n_caps, 3)
            self.assertTrue(np.all(np.array(exp) == generated),
                "Pipe map invalid for n_caps {}, n_ref {}".format(n_caps, 3) )

        expected = (
            [[0, 1]],
            [[0, 1, 1, 1],
             [0, 0, 1, 1],
             [0, 0, 0, 1]],
            [[0, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 1]], )

        expected = tuple(np.reshape(ex, np.shape(ex) + (1,)) for ex in expected)

        for h_bits, exp in zip(range(1, 1+len(expected)), expected):
            n_caps = 2**h_bits - 1
            generated = gen.pipe_map(n_caps, 2)
            self.assertTrue(np.all(np.array(exp) == generated),
                "Pipe map invalid for n_caps {}, n_ref {}".format(n_caps, 2) )

    def test_parse_bits(self):

        test_pairs = [  [("1",   False,), (1, False,)],
                        [(1  ,   False,), (1, False,)],
                        [(1.0,   False,), (1, False,)],
                        [("1",   None,),  (1, False,)],
                        [(1  ,   None,),  (1, False,)],
                        [(1.0,   None,),  (1, False,)],
                        [("1",   True,),  (1, True,)],
                        [(1  ,   True,),  (1, True,)],
                        [("1.5", True,),  (1, True,)],
                        [(1.5  , True,),  (1, True,)],
                        [("1.5", None,),  (1, True,)],
                        [(1.5  , None,),  (1, True,)],
                        [("2",   False,), (2, False,)],
                        [(2  ,   False,), (2, False,)],
                        [(2.0,   False,), (2, False,)],
                        [("2",   None,),  (2, False,)],
                        [(2  ,   None,),  (2, False,)],
                        [(2.0,   None,),  (2, False,)],
                        [("2",   True,),  (2, True,)],
                        [(2  ,   True,),  (2, True,)],
                        [("2.5", True,),  (2, True,)],
                        [(2.5  , True,),  (2, True,)],
                        [("2.5", None,),  (2, True,)],
                        [(2.5  , None,),  (2, True,)],
                        [("3",   False,), (3, False,)],
                        [(3  ,   False,), (3, False,)],
                        [(3.0,   False,), (3, False,)],
                        [("3",   None,),  (3, False,)],
                        [(3  ,   None,),  (3, False,)],
                        [(3.0,   None,),  (3, False,)],
                        [("3",   True,),  (3, True,)],
                        [(3  ,   True,),  (3, True,)],
                        [("3.5", True,),  (3, True,)],
                        [(3.5  , True,),  (3, True,)],
                        [("3.5", None,),  (3, True,)],
                        [(3.5  , None,),  (3, True,)] ]

        for args, expected in test_pairs:
            real = gen.parse_bits(*args)
            self.assertEqual(expected, real,
                ("Could not match parse_bits({}, {}) with ({}, {},), "
                "recieved ({}, {},)").format(*args, *expected, *real))

        test_raise = [
            (1.0, True,),
            ("1.5", False,),
            (1.5  , False,),
            (2.0, True,),
            ("2.5", False,),
            (2.5  , False,),
            (3.0, True,),
            ("3.5", False,),
            (3.5  , False,) ]

        for args in test_raise:
            with self.assertRaises(ValueError):
                _ = gen.parse_bits(*args)

    def test_compute_n_codes(self):
        test_pairs = [(1, 2,), (1.5, 3,), (2, 4,), (2.5, 7,), (3, 8,), (3.5, 15,)]

        for n_bits, e_n_codes in test_pairs:
            self.assertEqual(e_n_codes, gen.compute_n_codes(n_bits))

    def test_meta_save_load(self):
        name = "meta.{}"
        name_h = "meta.{}h"
        dot = data.PathContext.relative()

        for n_bits in range(1, 4):
            data_location = data.DataLocation(dot, self.directory, name.format(n_bits), "json")
            data_location_h = data.DataLocation(dot, self.directory, name_h.format(n_bits), "json")

            meta = gen.StageMeta(n_bits, n_refs=2)
            meta_h = gen.StageMeta(n_bits, n_refs=3, half_bit=True)

            data.save(meta, data_location)
            data.save(meta_h, data_location_h)

            meta_l = data.load(gen.StageMeta, data_location)
            meta_lh = data.load(gen.StageMeta, data_location_h)

            self.assertEqual(meta, meta_l)
            self.assertEqual(meta_h, meta_lh)

    def test_stage_save_load(self):
        name = "stage.{}"
        name_h = "stage.{}h"

        for n_bits in range(1, 4):
            data_location = data.DataLocation(self.path_context, self.directory, name.format(n_bits), "json")
            data_location_h = data.DataLocation(self.path_context, self.directory, name_h.format(n_bits), "json")

            meta = gen.StageMeta(n_bits, n_refs=2)
            meta_h = gen.StageMeta(n_bits, n_refs=3, half_bit=True)

            stage = meta.generate_ideal()
            stage_h = meta_h.generate_ideal()

            data.save(stage, data_location)
            data.save(stage_h, data_location_h)

            stage_l = data.load(gen.StageParameters, data_location)
            stage_lh = data.load(gen.StageParameters, data_location_h)

            self.assertEqual(stage, stage_l)
            self.assertEqual(stage_h, stage_lh)

    def test_stage_thres_code_consistency(self):
        for n_bits in range(1, 12):
            meta = gen.StageMeta(n_bits, n_refs=2)
            meta_h = gen.StageMeta(n_bits, n_refs=3, half_bit=True)

            for m in (meta, meta_h,):
                thres = gen.compute_thres(m.n_bits, *m.fsr, half_bit=m.half_bit)
                self.assertEqual(m.n_codes - 1, len(thres))

    def test_convert_rebuild_cycle(self):
        N_TEST = 3
        SAMPLES_FACT = 50
        for n_bits_0, n_bits_1, n_bits_tail in \
            zip(range(3, 3 + N_TEST), range(2, 2 + N_TEST), range(1, 1 + N_TEST)):

            tot_bits = n_bits_0 + n_bits_1 + n_bits_tail
            samples = int((2**tot_bits + 1)*SAMPLES_FACT)

            for half_bit in (False, True,):
                n_refs = 3 if half_bit else 2
                meta_0 = gen.StageMeta(n_bits_0, n_refs=n_refs, half_bit=half_bit, eff=1)
                meta_1 = gen.StageMeta(n_bits_1, n_refs=n_refs, half_bit=half_bit, eff=1)

                stages = [meta_0.generate_ideal(), meta_1.generate_ideal()]
                thres_tail = gen.compute_thres(n_bits_tail, *meta_1.fsr, half_bit=half_bit)

                adc = gen.PipeParameters(stages, thres_tail)
                data = np.linspace(*meta_0.fsr, samples) # TODO:randomize

                conversion, tail_code = adc.convert(data[..., np.newaxis])
                re_data = adc.rebuild(conversion, tail_code)
                lsb = (meta_0.fsr[1] - meta_0.fsr[0]) / 2 ** tot_bits

                self.assertTrue((np.abs(data - re_data) <= lsb / 2).all())

    def test_internal_random_deterministic(self):
        N_TEST = 100
        TEST_LEN = 100
        meta = gen.StageMeta(3.5, 3)
        for _ in range(N_TEST):
            seed = np.random.randint(0, 6425243)
            generator = gen.InternalRandom(meta, meta.n_caps - 1 // 2, seed)
            full_refs, _ = generator.generate_in(TEST_LEN)
            for ii in range(TEST_LEN):
                refs, _ = generator.generate_in(1, ii)
                self.assertTrue((full_refs[ii:ii+1, ...] == refs).all())
