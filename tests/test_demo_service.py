import importlib.util
import unittest
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "backend" / "services" / "demo.py"
SPEC = importlib.util.spec_from_file_location("demo_service_test", PATH)
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


class DemoServiceTests(unittest.TestCase):
    def test_overview_uses_final_metrics(self):
        overview = demo.get_overview()
        h1 = next(item for item in overview["metrics"] if item["key"] == "2026_h1")
        self.assertAlmostEqual(h1["total_return"], .1134009, places=6)
        self.assertAlmostEqual(h1["max_drawdown"], .0763436, places=6)
        self.assertEqual(h1["days"], 115)

    def test_fixed_sample_matches_official_advice(self):
        result = demo.chat("advice", "2026-06-30")["result"]
        actual = [(p["name"], p["symbol"], p["shares"]) for p in result["positions"]]
        self.assertEqual(actual, [("重庆银行", "601963", 16900), ("中国神华", "601088", 4200), ("海峡股份", "002320", 26000)])
        self.assertEqual(result["signal_date"], "2026-06-29")

    def test_non_trading_day_falls_back(self):
        result = demo.chat("advice", "2026-06-28")["result"]
        self.assertTrue(result["adjusted"])
        self.assertEqual(result["date"], "2026-06-26")

    def test_recent_official_advice_is_cached(self):
        response = demo.chat("advice", "2026-07-13")
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["result"]["date"], "2026-07-13")
        self.assertEqual(len(response["result"]["positions"]), 3)

    def test_chinese_and_dotted_dates_are_parsed(self):
        self.assertEqual(demo._parse_requested_date("7月13日交易建议").isoformat(), "2026-07-13")
        self.assertEqual(demo._parse_requested_date("2026.7.14的交易建议").isoformat(), "2026-07-14")

    def test_live_result_never_exceeds_available_capital(self):
        response = demo.chat("advice", "2026-07-14")
        result = response["result"]
        self.assertLessEqual(result["invested_amount"], result["available_capital"])
        self.assertEqual(result["signal_date"], "2026-07-13")
        self.assertEqual(len(result["positions"]), 3)

    def test_future_date_rejected(self):
        with self.assertRaises(ValueError):
            demo.chat("advice", "2099-01-01")


if __name__ == "__main__":
    unittest.main()
