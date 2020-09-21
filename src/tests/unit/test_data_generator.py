"""

Tests functions in data_generator.py

"""
import unittest
import net_est.data.data_generator as data_gen


class TestDataGenerator(unittest.TestCase):
    def test_noise_function(self):
        with self.assertRaises(TypeError):
            data_gen.noise_function('bad_input')

        with self.assertRaises(TypeError):
            data_gen.noise_function(None)


