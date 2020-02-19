#! /usr/bin/env python

import unittest
import tempfile
import shutil

import numpy as np

import calib.gen as gen
import calib.data as data
import calib.simulation as sims
import calib.calibration as cal


class TestSimulation(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path_context = data.PathContext.cwd()

        meta = gen.StageMeta(4.5, 3)
        self.meta = meta

        self.ideal = gen.PipeParameters([meta.generate_ideal()], gen.compute_thres(5, *meta.fsr))

        conf = gen.Configuration(meta, [[0],[1],[2],[3],[4],[5],[6]])
        in_ = gen.ACCombinator(meta, gen.InternalDC(meta, [[2],[2],[2],[2],[2],[2],[1]]), gen.InternalDC(meta, [[0],[0],[0],[0],[0],[0],[1]]), 16)

        conf_set_0 = gen.ConfigurationSet(512, [in_], [conf])

        conf = gen.Configuration(meta, [[0],[1],[2]])
        in_ = gen.ACCombinator(meta, gen.InternalDC(meta, [[2], [2], [1]]), gen.InternalDC(meta, [[1],[1],[0]]), 16)

        conf_set_1 = gen.ConfigurationSet(512, [in_], [conf])

        self.conf_seq = gen.ConfigurationSequence(
            [gen.InitialCondition.Discharged(meta, 9)], [conf_set_0, conf_set_1,]*4)

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_recreate(self):
        simulator = sims.Simulator(None)
        tb = sims.StageTestbench(self.ideal.as_delta_sigma()[0], [], self.conf_seq)
        c, u_real = simulator.simulate(*tb.simulation_args(tuple()))
        fns = cal.CalibrationSystem
        cache = fns.recreate_cache(*tb.simulation_args(tuple()), c)
        u_est = fns.recreate(*tb.simulation_args(tuple()), cache)

        self.assertTrue(np.allclose(u_real, u_est))
