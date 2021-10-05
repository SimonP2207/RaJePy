import os
import unittest
from classes import JetModel

TEST_PARAM_DCY = os.sep.join([os.path.dirname(__file__), 'test_cases'])

class TestJetModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        files = sorted([os.sep.join([TEST_PARAM_DCY, _]) for _ in os.listdir(TEST_PARAM_DCY)])
        files = list(filter(lambda x: x[-9:] == 'params.py', files))
        cls.param_files = list(zip(files[::2], files[1::2]))
        cls.model_params = {os.path.basename(_[0].split('-')[0]) : JetModel.py_to_dict(_[0]) for _ in cls.param_files}

    def test_lz_to_grid_dims(self):
        correct_dims = {'test1': (80, 40, 20), 'test2': (80, 40, 20)}
        for test_case in self.model_params:
            test_case_file = list(filter(lambda x: x[:len(test_case)] == test_case,
                                         self.model_params))[0]
            dims = JetModel.lz_to_grid_dims(self.model_params[test_case])
            self.assertEqual(dims, correct_dims[test_case],
                             f"Model param file is {test_case_file}")


if __name__ == '__main__':
    unittest.main()
