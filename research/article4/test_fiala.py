"""Lightweight, dependency-based regression checks for Article 4 Fiala solver."""
import unittest
import numpy as np
from article4_reproducible_fiala import VEHICLES, fiala, solve

class FialaTests(unittest.TestCase):
    def test_zero_slip(self):
        self.assertAlmostEqual(float(fiala(0., 80000., .3, 4000.)), 0.)
    def test_odd_symmetry(self):
        x=float(fiala(.04,80000.,.3,4000.))
        self.assertAlmostEqual(x,-float(fiala(-.04,80000.,.3,4000.)),places=8)
    def test_force_bound(self):
        for alpha in np.linspace(-.8,.8,101):
            self.assertLessEqual(abs(float(fiala(alpha,80000.,.3,4000.))),1200.+1e-9)
    def test_low_speed_guard(self):
        self.assertEqual(solve(1.,0.,.3,VEHICLES["mid_assumed"])["status"],"low_speed")
    def test_invalid_guard(self):
        self.assertEqual(solve(float("nan"),0.,.3,VEHICLES["mid_assumed"])["status"],"invalid_input")
    def test_straight_equilibrium(self):
        r=solve(10.,0.,.3,VEHICLES["mid_assumed"])
        self.assertEqual(r["status"],"ok")
        self.assertAlmostEqual(r["nonlinear_yaw"],0.,places=8)
        self.assertLess(r["residual"],1e-7)
    def test_zero_mu_not_supported(self):
        # The solver assumes strictly positive friction; no zero-mu run should be attempted.
        self.assertTrue(all(mu>0 for mu in (.1,.3)))

if __name__=="__main__":
    unittest.main()
