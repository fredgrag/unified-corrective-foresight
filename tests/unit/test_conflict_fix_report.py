from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from corrective_foresight.config.loader import load_action_spec
from corrective_foresight.evaluation.conflict_fix_report import (
    EpisodeSummary,
    build_paired_report,
    load_episode_summaries,
    validate_policy_label,
)
from corrective_foresight.evaluation.records import EvaluationRecord


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTION_SPEC_PATH = (
    PROJECT_ROOT
    / "configs/actions/maniskill_panda_pd_ee_delta_pose_physical_v1.yaml"
)


def summaries(
    *,
    successes: set[int],
    rewards: list[float],
    protocol_hash: str = "a" * 64,
) -> tuple[EpisodeSummary, ...]:
    return tuple(
        EpisodeSummary(
            seed=seed,
            success=seed in successes,
            total_reward=rewards[seed],
            episode_length=200,
            action_bounds_ok=True,
            protocol_hash=protocol_hash,
        )
        for seed in range(10)
    )


class ConflictFixReportTest(unittest.TestCase):
    def test_gate_requires_two_more_successes_and_positive_median_reward(self) -> None:
        baseline = summaries(successes={1}, rewards=[1.0] * 10)
        candidate = summaries(successes={1, 2, 3}, rewards=[1.2] * 10)

        report = build_paired_report(
            baseline,
            candidate,
            required_success_gain=2,
        )

        self.assertEqual(report.success_gain, 2)
        self.assertGreater(report.median_reward_change, 0.0)
        self.assertTrue(report.effect_gate_passed)

    def test_reward_gain_without_success_gain_is_insufficient(self) -> None:
        baseline = summaries(successes={1}, rewards=[1.0] * 10)
        candidate = summaries(successes={1, 2}, rewards=[2.0] * 10)

        report = build_paired_report(
            baseline,
            candidate,
            required_success_gain=2,
        )

        self.assertFalse(report.effect_gate_passed)
        self.assertIn("success_gain_below_required", report.reasons)

    def test_pairing_rejects_missing_seed_or_protocol_drift(self) -> None:
        baseline = summaries(successes=set(), rewards=[0.0] * 10)
        candidate = summaries(successes=set(), rewards=[0.1] * 10)

        with self.assertRaisesRegex(ValueError, "seeds"):
            build_paired_report(baseline[:-1], candidate)
        drifted = list(candidate)
        drifted[4] = EpisodeSummary(
            seed=4,
            success=False,
            total_reward=0.1,
            episode_length=200,
            action_bounds_ok=True,
            protocol_hash="b" * 64,
        )
        with self.assertRaisesRegex(ValueError, "protocol"):
            build_paired_report(baseline, tuple(drifted))

    def test_world_checkpoint_is_rejected_as_policy_result(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "world-pretrain is not a policy result",
        ):
            validate_policy_label("world_pretrain_5000")
        for label in (
            "untrained_action_from_world_5000",
            "unified_5000",
            "unified_20000",
        ):
            self.assertEqual(validate_policy_label(label), label)

    def test_strict_loader_verifies_records_actions_flow_and_video_hashes(self) -> None:
        action_spec = load_action_spec(ACTION_SPEC_PATH)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, reward in (
                ("untrained_action_from_world_5000", 1.0),
                ("unified_5000", 1.2),
            ):
                checkpoint = root / "checkpoints" / label
                checkpoint.mkdir(parents=True)
                (checkpoint / "manifest.json").write_text(
                    f"{label}\n",
                    encoding="utf-8",
                )
                self._write_evaluation_records(
                    root,
                    label,
                    checkpoint,
                    reward,
                    action_spec,
                )
            baseline = load_episode_summaries(
                root,
                "untrained_action_from_world_5000",
                action_spec=action_spec,
            )
            candidate = load_episode_summaries(
                root,
                "unified_5000",
                action_spec=action_spec,
            )

            report = build_paired_report(baseline, candidate)

            self.assertTrue(report.effect_gate_passed)
            video = root / "unified_5000/seed-4.mp4"
            original_video = video.read_bytes()
            video.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "video SHA-256"):
                load_episode_summaries(
                    root,
                    "unified_5000",
                    action_spec=action_spec,
                )
            video.write_bytes(original_video)
            manifest = root / "checkpoints/unified_5000/manifest.json"
            manifest.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checkpoint manifest SHA-256"):
                load_episode_summaries(
                    root,
                    "unified_5000",
                    action_spec=action_spec,
                )

    @staticmethod
    def _write_evaluation_records(
        root: Path,
        label: str,
        checkpoint: Path,
        reward: float,
        action_spec,
    ) -> None:
        tag_root = root / label
        tag_root.mkdir()
        manifest = hashlib.sha256(
            (checkpoint / "manifest.json").read_bytes()
        ).hexdigest()
        for seed in range(10):
            video = tag_root / f"seed-{seed}.mp4"
            video.write_bytes(f"video-{label}-{seed}".encode("ascii"))
            video_sha = hashlib.sha256(video.read_bytes()).hexdigest()
            physical_action = [0.0] * action_spec.dimension
            record = EvaluationRecord(
                {
                    "format_version": 1,
                    "tag": label,
                    "checkpoint": {
                        "label": label,
                        "directory": str(checkpoint),
                        "manifest_sha256": manifest,
                    },
                    "dataset": {
                        "dataset_id": "maniskill.pick_cube.test",
                        "revision": "test-revision",
                        "spec_hash": "c" * 64,
                    },
                    "action": {
                        "spec_id": action_spec.spec_id,
                        "spec_hash": action_spec.content_hash,
                    },
                    "environment": {
                        "env_id": "PickCube-v1",
                        "robot_uid": "panda_wristcam",
                        "obs_mode": "rgb",
                        "control_mode": "pd_ee_delta_pose",
                        "sim_backend": "physx_cpu",
                        "seed": seed,
                    },
                    "protocol": {
                        "action_horizon": 8,
                        "execution_horizon": 1,
                        "temporal_ensemble": False,
                        "context_steps": 2,
                    },
                    "flow": {
                        "solver": "midpoint",
                        "time_grid": [index / 10 for index in range(11)],
                        "intervals": 10,
                        "nfe_per_step": 20,
                        "total_nfe": 20,
                        "seeds": [1000 + seed],
                    },
                    "result": {
                        "success": seed < (3 if label == "unified_5000" else 1),
                        "total_reward": reward,
                        "length": 1,
                        "termination_reason": "truncated",
                    },
                    "diagnostics": {
                        "consistency_mean": 0.5,
                        "inverse_variance_mean": 0.25,
                    },
                    "steps": [
                        {
                            "index": 0,
                            "flow_seed": 1000 + seed,
                            "executed_chunk_index": 0,
                            "normalized_action": physical_action,
                            "physical_action": physical_action,
                            "environment_action": physical_action,
                            "reward": reward,
                            "terminated": False,
                            "truncated": True,
                            "success": seed < (
                                3 if label == "unified_5000" else 1
                            ),
                            "consistency": 0.5,
                            "inverse_variance_mean": 0.25,
                            "solver": "midpoint",
                            "time_grid": [index / 10 for index in range(11)],
                            "intervals": 10,
                            "nfe": 20,
                        }
                    ],
                    "video": {
                        "path": str(video),
                        "sha256": video_sha,
                        "frames": 2,
                        "fps": 20.0,
                    },
                }
            )
            record.write_atomic(tag_root / f"seed-{seed}.json")


if __name__ == "__main__":
    unittest.main()
