"""

Tests functions in data_generator.py

"""
import unittest
import numpy as np
from scipy.integrate import quad
import net_est.data.data_generator as data_gen


class TestDataGenerator(unittest.TestCase):
    def testGenerateInput(self):
        with self.assertRaises(TypeError):
            data_gen.generate_training_data(n_samples=1.0)

        with self.assertRaises(TypeError):
            data_gen.generate_training_data(n_samples="a")

    def testGenerateOutput(self):
        result = data_gen.generate_training_data()
        self.assertEqual(len(result), 2)

    def testGenerateOutputShape(self):
        x_sampled, y_e = data_gen.generate_training_data(n_samples=10)
        self.assertEqual(x_sampled.shape[0], y_e.shape[0], 10)


class TestNoiseFunction(unittest.TestCase):
    def test_noise_function_input(self):
        with self.assertRaises(TypeError):
            data_gen.noise_function('bad_input')
        with self.assertRaises(TypeError):
            data_gen.noise_function(None)

    def test_noise_output_shape(self):
        x = np.arange(0, 1, 0.1)
        result = data_gen.noise_function(x)
        self.assertEqual(len(result), 2)

    def test_noise_shape(self):
        x = np.arange(0, 1, 0.1)
        y, sigma = data_gen.noise_function(x)
        self.assertEqual(x.shape, y.shape)

        x = np.array([[0, 10], [14, 12]])
        y, sigma = data_gen.noise_function(x)
        self.assertEqual(x.shape, y.shape)

    def test_noise_iterable_input(self):
        x = [8, 10, 12, 14]
        y, sigma = data_gen.noise_function(x)
        self.assertEqual(len(x), y.shape[0])


class TestTargetFunction(unittest.TestCase):
    def testTargetInput(self):
        x = 'badInput'
        with self.assertRaises(TypeError):
            data_gen.target_function(x)

    def testTargetOutput(self):
        y = data_gen.target_function(0.5)
        self.assertAlmostEqual(y, -0.3826, places=2)

    def testTargetOutputShape(self):
        x_in = [10, 23, 45]
        y = data_gen.target_function(x_in)
        self.assertEqual(len(x_in), y.shape[0])


class TestInputDistribution(unittest.TestCase):
    def setUp(self) -> None:
        self.x_dist = data_gen.abs_value_dist(name='x_abs')

    def tearDown(self) -> None:
        self.x_dist = None  # Not really necessary

    def test_dist_name(self):
        with self.assertRaises(TypeError):
            data_gen.abs_value_dist(name=10.0)
        with self.assertRaises(TypeError):
            data_gen.abs_value_dist(name=[None])
        with self.assertRaises(TypeError):
            data_gen.abs_value_dist(name=['x_int'])

    def test_valid_pdf(self):
        pdf_area = quad(self.x_dist.pdf, -1.0, 1.0)[0]
        self.assertAlmostEquals(pdf_area, 1.0, delta=1e-2)


if __name__ == '__main__':
    unittest.main()

