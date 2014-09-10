

import unittest
from soundex import letters_to_numbers, truncate_to_three_digits, add_zero_padding

class TestSequenceFunctions(unittest.TestCase):
    def setUp(self):
        self.s1 = letters_to_numbers()
        self.s2 = truncate_to_three_digits()
        self.s3 = add_zero_padding()

    def test_letters(self):
        self.assertEqual("".join(self.s1.transduce(x for x in "washington")), "w25235")
        self.assertEqual("".join(self.s1.transduce(x for x in "jefferson")), "j1625")
        self.assertEqual("".join(self.s1.transduce(x for x in "adams")), "a352")
        self.assertEqual("".join(self.s1.transduce(x for x in "bush")), "b2")

    def test_truncation(self):
        self.assertEqual("".join(self.s2.transduce(x for x in "a33333")), "a333")
        self.assertEqual("".join(self.s2.transduce(x for x in "123456")), "123")
        self.assertEqual("".join(self.s2.transduce(x for x in "11")), "11")
        self.assertEqual("".join(self.s2.transduce(x for x in "5")), "5")

    def test_padding(self):
        self.assertEqual("".join(self.s3.transduce(x for x in "3")), "300")
        self.assertEqual("".join(self.s3.transduce(x for x in "b56")), "b560")
        self.assertEqual("".join(self.s3.transduce(x for x in "c111")), "c111")

    def test_hindi(self):
        self.assertEqual("".join(self.h1.transduce(x for x in "3")), "300")
        
if __name__ == '__main__':
    unittest.main()
