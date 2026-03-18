import unittest

from mahjong_ai.rules.schema import RulesConfig


class TestRulesSchema(unittest.TestCase):
    def test_validate_rejects_negative_pay_and_penalty_knobs(self):
        invalid_cases = (
            {"gang_ming_pay": -1},
            {"gang_an_pay": -1},
            {"gang_bu_pay": -1},
            {"hua_zhu_penalty": -1},
            {"cha_jiao_penalty": -1},
        )

        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    RulesConfig(**kwargs).validate()

    def test_validate_rejects_invalid_fan_pattern_definitions(self):
        invalid_cases = (
            {1: 2},
            {"qidui": -1},
            {"qidui": 1.5},
        )

        for fan_patterns in invalid_cases:
            with self.subTest(fan_patterns=fan_patterns):
                with self.assertRaises(ValueError):
                    RulesConfig(fan_patterns=fan_patterns).validate()


if __name__ == "__main__":
    unittest.main()
