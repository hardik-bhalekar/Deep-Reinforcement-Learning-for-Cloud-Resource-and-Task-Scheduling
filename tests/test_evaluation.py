"""Unit tests for BenchmarkEvaluator."""

import unittest
import tempfile
import os
from src.simulator.environment import CloudSchedulingEnv
from src.evaluation.evaluator import BenchmarkEvaluator


class TestBenchmarkEvaluator(unittest.TestCase):
    def test_benchmark_runs_and_saves_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = CloudSchedulingEnv()
            evaluator = BenchmarkEvaluator(
                env=env,
                dqn_agent=None,  # Tests baseline heuristics
                test_seeds=[42, 43],
                results_dir=tmpdir,
            )

            results = evaluator.run_benchmark(num_tasks=10, workload_type="uniform")
            self.assertIn("summary", results)
            self.assertIn("Random", results["summary"])
            self.assertIn("Min-Min", results["summary"])

            json_file = os.path.join(tmpdir, "benchmark_results.json")
            csv_file = os.path.join(tmpdir, "benchmark_summary.csv")
            self.assertTrue(os.path.exists(json_file))
            self.assertTrue(os.path.exists(csv_file))

            plots = evaluator.generate_comparison_plots(results)
            self.assertGreater(len(plots), 0)
            for p in plots:
                self.assertTrue(os.path.exists(p))


if __name__ == "__main__":
    unittest.main()
