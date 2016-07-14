import unittest
import numpy as np
import latenttrees.lt_helper as lth
from scipy.stats import norm

class TestLtHelper(unittest.TestCase):
    pass

def test_norm_logpdf_generator(x, mu, std):
    def test(self):
        scipy_d = norm(mu, std)  # scipy normal distribution
        logpdf_scipy = scipy_d.logpdf(x)
        logpdf = lth.norm_logpdf(x, mu, std)
        # self.assertEqual(True, False)
        np.testing.assert_allclose(logpdf, logpdf_scipy)
    return test

if __name__ == '__main__':
    for i in range(10):
        test_name = 'test_norm_logpdf_{}'.format(i)
        d1 = 100
        d2 = 1
        mu = np.random.randn(d1, d2)
        std = np.random.rand(d1, d2)
        x = (np.random.rand(d1, d2) * 20) - 10
        test = test_norm_logpdf_generator(x, mu, std)
        setattr(TestLtHelper, test_name, test)
    unittest.main()
