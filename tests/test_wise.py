import unittest


class PackageSmokeTests(unittest.TestCase):
    def test_package_imports(self):
        import wise

        self.assertEqual(wise.__all__, [])


if __name__ == "__main__":
    unittest.main()
