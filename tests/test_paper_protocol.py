from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from src.paper_protocol import ObservableElectricPrior, ObservableWaterPrior, evaluate_reference, load_observations, train_run, resolve_device, PaperDGMForecaster
from scripts.benchmarks.generate_smoke_benchmark import generate
from scripts.benchmarks.run_paper_campaign import DEFAULTS


class PaperProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        generate(self.root / "data")
        self.dataset = self.root / "data"

    def test_observation_artifact_is_reference_free_and_windows_do_not_cross(self) -> None:
        observation_dir = self.dataset / "observations"
        manifest = json.loads((observation_dir / "manifest.json").read_text(encoding="utf-8"))
        data = np.load(observation_dir / manifest["observation_file"])
        self.assertFalse(any("edge" in key.lower() for key in data.files))
        self.assertNotIn("reference_file", manifest)
        bundle = load_observations(observation_dir)
        lags = manifest["window"]["lags"]
        horizon = manifest["window"]["forecast_horizon"]
        for split, dataset in bundle.datasets.items():
            start, stop = manifest["splits"][split]
            self.assertGreaterEqual(int(dataset.starts.min()), start)
            self.assertLessEqual(int(dataset.starts.max()) + lags + horizon, stop)

    def test_normalization_and_prior_are_train_fitted(self) -> None:
        bundle = load_observations(self.dataset / "observations")
        train_start, train_stop = bundle.manifest["splits"]["train"]
        train = bundle.values[train_start:train_stop]
        np.testing.assert_allclose(train.mean(axis=(0, 1)), 0.0, atol=1e-4)
        np.testing.assert_allclose(train.std(axis=(0, 1)), 1.0, atol=1e-4)
        prior = ObservableWaterPrior(train, bundle.coordinates, candidate_size=5)
        self.assertTrue(torch.all(prior.candidate_mask.sum(dim=1) == 5))
        windows = torch.as_tensor(train[:12][None], dtype=torch.float32)
        q = prior.distribution(windows)
        torch.testing.assert_close(q.sum(dim=-1), torch.ones((1, q.shape[1])))
        self.assertEqual(int(torch.count_nonzero(q.masked_select(~prior.candidate_mask[None]))), 0)

    def test_electric_prior_uses_fixed_candidate_support(self) -> None:
        generator = np.random.default_rng(7)
        values = generator.normal(size=(40, 8, 2)).astype(np.float32)
        coordinates = generator.uniform(size=(8, 2)).astype(np.float32)
        prior = ObservableElectricPrior(values, coordinates, candidate_size=3, ridge=0.5)
        self.assertTrue(torch.all(prior.candidate_mask.sum(dim=1) == 3))
        windows = torch.as_tensor(values[:6][None], dtype=torch.float32)
        q = prior.distribution(windows)
        self.assertTrue(torch.isfinite(q).all())
        torch.testing.assert_close(q.sum(dim=-1), torch.ones((1, 8)))
        self.assertEqual(int(torch.count_nonzero(q.masked_select(~prior.candidate_mask[None]))), 0)

    def test_invalid_splits_and_nonfinite_values_are_rejected(self) -> None:
        observations = self.dataset / "observations"
        path = observations / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["splits"]["val"] = [179, 204]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Invalid val split"):
            load_observations(observations)
        generate(self.dataset)
        with np.load(observations / "data.npz") as data:
            arrays = dict(data)
        arrays["pressure"][0, 0] = np.nan
        np.savez_compressed(observations / "data.npz", **arrays)
        with self.assertRaisesRegex(ValueError, "finite"):
            load_observations(observations)

    def test_heldout_values_do_not_change_training_statistics_or_candidates(self) -> None:
        observations = self.dataset / "observations"
        original = load_observations(observations)
        with np.load(observations / "data.npz") as data:
            arrays = dict(data)
        arrays["pressure"][180:] += 10000
        arrays["demand"][180:] *= 100
        np.savez_compressed(observations / "data.npz", **arrays)
        modified = load_observations(observations)
        np.testing.assert_array_equal(original.mean, modified.mean)
        np.testing.assert_array_equal(original.std, modified.std)
        for prior_type in (ObservableWaterPrior, ObservableElectricPrior):
            before = prior_type(original.values[:180], original.coordinates, 3)
            after = prior_type(modified.values[:180], modified.coordinates, 3)
            torch.testing.assert_close(before.training_support, after.training_support)

    def test_cpu_training_resume_and_input_identity(self) -> None:
        config = {**DEFAULTS, "device": "cpu", "epochs": 1, "hidden_dim": 8,
                  "embedding_dim": 4, "message_layers": 1, "candidate_size": 3, "k": 2}
        observations = self.dataset / "observations"
        output = self.root / "run"
        result = train_run(observations, output, config, 3)
        self.assertTrue(np.isfinite(result["forecast"]["mse"]))
        self.assertTrue((output / "COMPLETE").exists())
        self.assertFalse((output / "metrics_posthoc.json").exists())
        # Training never needs the evaluation directory, even for finalization.
        reference = self.dataset / "evaluation_only/reference_topology.npz"
        reference.unlink()
        checkpoint_time = (output / "last.pt").stat().st_mtime_ns
        self.assertEqual(result, train_run(observations, output, config, 3))
        self.assertEqual(checkpoint_time, (output / "last.pt").stat().st_mtime_ns)
        manifest_path = observations / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["dataset"] = "different_input"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "changed"):
            train_run(observations, output, config, 3)

    def test_interrupted_training_restores_optimizer_and_rng(self) -> None:
        config = {**DEFAULTS, "device": "cpu", "epochs": 2, "hidden_dim": 8,
                  "embedding_dim": 4, "message_layers": 1, "candidate_size": 3, "k": 2}
        observations = self.dataset / "observations"
        uninterrupted, interrupted = self.root / "full", self.root / "interrupted"
        train_run(observations, uninterrupted, config, 7, finalize=False)
        original_save = torch.save

        def interrupt_after_checkpoint(payload, path):
            original_save(payload, path)
            if Path(path).name == "last.pt" and payload["epoch"] == 0:
                raise RuntimeError("simulated interruption")

        with patch("src.paper_protocol.torch.save", side_effect=interrupt_after_checkpoint):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                train_run(observations, interrupted, config, 7, finalize=False)
        train_run(observations, interrupted, config, 7, finalize=False)
        full = torch.load(uninterrupted / "last.pt", weights_only=False)
        resumed = torch.load(interrupted / "last.pt", weights_only=False)
        for name, value in full["model"].items():
            torch.testing.assert_close(value, resumed["model"][name], rtol=0, atol=0)
        self.assertEqual(full["best_value"], resumed["best_value"])

    def test_sampler_never_selects_outside_candidate_support(self) -> None:
        mask = torch.zeros((6, 6), dtype=torch.bool)
        for row in range(6):
            mask[row, (row + 1) % 6] = True
            mask[row, (row + 2) % 6] = True
        config = {**DEFAULTS, "k": 2}
        model = PaperDGMForecaster(6, 6, 12, 2, mask, config)
        logits = torch.where(mask[None], -1000.0, 1000.0)
        for stochastic in (False, True):
            adjacency, log_probability = model._sample(logits, stochastic)
            self.assertEqual(int(adjacency.masked_select(~mask[None]).sum()), 0)
            self.assertTrue(torch.isfinite(log_probability).all())

    def test_invalid_device_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "device must"):
            resolve_device("typo")

    def test_posthoc_all_pairs_confusion_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, evaluation = root / "run", root / "evaluation"
            run.mkdir()
            evaluation.mkdir()
            (run / "node_ids.json").write_text(json.dumps(["a", "b", "c", "d"]))
            (run / "learned_edges.csv").write_text("source,target\n0,1\n0,2\n", encoding="utf-8")
            (run / "metrics_gt_free.json").write_text(json.dumps({"forecast": {"mse": 1.0}}), encoding="utf-8")
            np.savez_compressed(
                evaluation / "reference_topology.npz",
                edge_index=np.asarray([[0, 1], [1, 2]], dtype=np.int64).T,
                node_ids=np.asarray(["a", "b", "c", "d"]),
            )
            result = evaluate_reference(run, evaluation)
            self.assertEqual(result["topology"], {
                "tp": 1, "tn": 3, "fp": 1, "fn": 1,
                "precision": 0.5, "accuracy": 4 / 6, "recall": 0.5, "f1": 0.5, "jaccard": 1 / 3,
            })
            self.assertEqual(result["joint"]["S_task"], 0.5)
            self.assertEqual(result["joint"]["S_0.5"], 0.5)
            np.savez_compressed(evaluation / "reference_topology.npz",
                                node_ids=np.asarray(["b", "a", "c", "d"]),
                                edge_index=np.asarray([[0], [1]]))
            with self.assertRaisesRegex(ValueError, "node IDs/order"):
                evaluate_reference(run, evaluation)


if __name__ == "__main__":
    unittest.main()
