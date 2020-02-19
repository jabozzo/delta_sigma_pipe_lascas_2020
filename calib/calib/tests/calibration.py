#! /usr/bin/env python

import unittest
import tempfile
import shutil

import numpy as np

import calib.gen as gen
import calib.data as data
import calib.simulation as sims
import calib.calibration as cal

import calib.misc as misc
import calib.script.make_config as mconf


SEED = 567892131
INPUTS = "random"


class TestCalibration(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path_context = data.PathContext.cwd()

        meta = gen.StageMeta(3.5, 3)

        ideal = gen.PipeParameters([meta.generate_ideal()], gen.compute_thres(5, *meta.fsr))
        real_stage = meta.generate_gaussian(s_eff=0.1, s_cap=0.1, s_refs=0.1, s_thres=0, s_cm=0.1)
        real = gen.PipeParameters([real_stage], gen.compute_thres(5, *meta.fsr))

        self.meta = meta
        self.ideal = ideal
        self.real = real

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_clear(self):
        args = misc.Namespace()
        args.samples = [64, 64]
        args.n_test = 0
        args.loop = 1
        args.full = False
        args.period = 16
        args.ic = "clear"
        args.seed = SEED
        args.inputs = INPUTS

        conf_seq = mconf.calib(self.meta, args, True)

        tb = sims.StageTestbench.Scalar(self.real.as_delta_sigma()[0], [], conf_seq)
        self.system_compare(tb)

    def test_precharged(self):
        args = misc.Namespace()
        args.samples = [64, 64]
        args.n_test = 0
        args.loop = 1
        args.full = False
        args.period = 16
        args.ic = "precharge"
        args.seed = SEED
        args.inputs = INPUTS

        conf_seq = mconf.calib(self.meta, args, True)

        tb = sims.StageTestbench.Scalar(self.real.as_delta_sigma()[0], [], conf_seq)
        self.system_compare(tb)

    def test_simplest_clear(self):
        args = misc.Namespace()
        args.samples = [12]
        args.n_test = 0
        args.loop = 1
        args.full = False
        args.period = 16
        args.ic = "clear"
        args.seed = SEED
        args.inputs = INPUTS

        conf_seq = mconf.calib(self.meta, args, True)

        tb = sims.StageTestbench.Scalar(self.real.as_delta_sigma()[0], [], conf_seq)
        self.system_compare(tb)

    def test_simplest_precharged(self):
        args = misc.Namespace()
        args.samples = [64]
        args.n_test = 0
        args.loop = 1
        args.full = False
        args.period = 16
        args.ic = "precharge"
        args.seed = SEED
        args.inputs = INPUTS

        conf_seq = mconf.calib(self.meta, args, True)

        tb = sims.StageTestbench.Scalar(self.real.as_delta_sigma()[0], [], conf_seq)
        self.system_compare(tb)

    def system_compare(self, tb):
        for idx in tb.iter_conf_idx():
            tb.configuration_sequence[idx].clear_cache()
            tb.configuration_sequence[idx].build_cache()

        simulator = sims.Simulator(None)
        c, u_real = tb.simulate(simulator)

        stage = tb.stages[tuple()]
        conf_seq = tb.configuration_sequence[tuple()]

        system = cal.CalibrationSystem(stage, conf_seq, c)
        x = system.map_in(stage, np.zeros((0, 1,)))

        error = system.system(x, scalar=True, use_bands=True)
        indexes = np.nonzero(error)[0]

        if len(indexes) > 0:
            print("BAD X")
            print(x)

        self.assertTrue(np.allclose(error, 0))

    def vec_system_compare(self, tb, n_vec=32):
        for idx in tb.iter_conf_idx():
            tb.configuration_sequence[idx].clear_cache()
            tb.configuration_sequence[idx].build_cache()

        simulator = sims.Simulator(None)
        c, u_real = tb.simulate(simulator)

        stage = tb.stages[tuple()]
        conf_seq = tb.configuration_sequence[tuple()]
        meta = stage.meta

        system = cal.CalibrationSystem(stage, conf_seq, c)

        real_stages = [meta.generate_gaussian(s_eff=0.1, s_cap=0.1, s_refs=0.1, s_thres=0, s_cm=0.1)
            for _ in range(n_vec)]
        for _stage in real_stages:
            _stage._thres = stage._thres

        xs = np.array([system.map_in(_stage, np.zeros((0, meta.n_diff,))) for _stage in real_stages])
        m_errors = system.system(xs, scalar=False, use_bands=True)
        system.system(xs[0, :], scalar=True, use_bands=True)
        s_errors = np.array([system.system(xs[ii, :], scalar=True, use_bands=True)
            for ii in range(np.size(xs, 0))])

        self.assertTrue(np.allclose(m_errors, s_errors))

    def test_vec_system_simplest(self):
        args = misc.Namespace()
        args.samples = [64]
        args.n_test = 0
        args.loop = 1
        args.full = False
        args.period = 16
        args.ic = "clear"
        args.seed = SEED
        args.inputs = INPUTS

        conf_seq = mconf.calib(self.meta, args, True)
        tb = sims.StageTestbench.Scalar(self.real.as_delta_sigma()[0], [], conf_seq)

        self.vec_system_compare(tb)

    def test_vec_system_loop(self):
        args = misc.Namespace()
        args.samples = [64, 64]
        args.n_test = 0
        args.loop = 2
        args.full = False
        args.period = 16
        args.ic = "clear"
        args.seed = SEED
        args.inputs = INPUTS

        conf_seq = mconf.calib(self.meta, args, True)
        tb = sims.StageTestbench.Scalar(self.real.as_delta_sigma()[0], [], conf_seq)

        self.vec_system_compare(tb)

    def test_vec_system_precharged(self):
        args = misc.Namespace()
        args.samples = [64]
        args.n_test = 0
        args.loop = 1
        args.full = False
        args.period = 16
        args.ic = "precharge"
        args.seed = SEED
        args.inputs = INPUTS

        conf_seq = mconf.calib(self.meta, args, True)
        tb = sims.StageTestbench.Scalar(self.real.as_delta_sigma()[0], [], conf_seq)

        self.vec_system_compare(tb)

    def test_vec_system_precharged_loop(self):
        args = misc.Namespace()
        args.samples = [64, 64]
        args.n_test = 0
        args.loop = 2
        args.full = False
        args.period = 16
        args.ic = "precharge"
        args.seed = SEED
        args.inputs = INPUTS

        conf_seq = mconf.calib(self.meta, args, True)
        tb = sims.StageTestbench.Scalar(self.real.as_delta_sigma()[0], [], conf_seq)

        self.vec_system_compare(tb)

    def test_vec_system_clear_sloop(self):
        args = misc.Namespace()
        args.samples = [64]
        args.n_test = 0
        args.loop = 2
        args.full = False
        args.period = 16
        args.ic = "clear"
        args.seed = SEED
        args.inputs = INPUTS

        conf_seq = mconf.calib(self.meta, args, True)
        tb = sims.StageTestbench.Scalar(self.real.as_delta_sigma()[0], [], conf_seq)

        self.vec_system_compare(tb)

    def test_vec_system_precharged_sloop(self):
        args = misc.Namespace()
        args.samples = [64]
        args.n_test = 0
        args.loop = 2
        args.full = False
        args.period = 16
        args.ic = "precharge"
        args.seed = SEED
        args.inputs = INPUTS

        conf_seq = mconf.calib(self.meta, args, True)
        tb = sims.StageTestbench.Scalar(self.real.as_delta_sigma()[0], [], conf_seq)

        self.vec_system_compare(tb)
