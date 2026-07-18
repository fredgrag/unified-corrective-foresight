from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterator

import torch

from corrective_foresight.evaluation.maniskill_runner import (
    CheckpointProvenance,
    EvaluationProtocol,
    ManiSkillClosedLoopRunner,
    ManiSkillEnvironmentSpec,
    ManiSkillOnlineObservationAdapter,
    physical_action_to_maniskill_controller,
)
from corrective_foresight.runtime import (
    assemble_runtime_policy,
    encode_condition_ids,
    expected_policy_checkpoint_contract,
    load_production_runtime,
)
from corrective_foresight.training.checkpoint import load_policy_checkpoint_strict
from corrective_foresight.training.distributed_checkpoint import (
    load_distributed_policy_checkpoint_for_evaluation,
)
from corrective_foresight.training.run_manifest import RunManifest


def flow_seed_stream(base_seed: int, environment_seed: int) -> Iterator[int]:
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("base seed must be a nonnegative integer")
    if type(environment_seed) is not int or environment_seed < 0:
        raise ValueError("environment seed must be a nonnegative integer")
    material = f"ucf-flow-v1:{base_seed}:{environment_seed}".encode("ascii")
    generator_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    generator_seed %= 2**63 - 1
    generator = torch.Generator(device="cpu").manual_seed(generator_seed)
    while True:
        yield int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("production evaluation requires CUDA")
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("evaluation seeds must be unique nonnegative integers")
    runtime = load_production_runtime(args.config)
    if len(runtime.dataset_specs) != 1 or len(runtime.action_specs) != 1:
        raise ValueError("ManiSkill evaluator requires one dataset and one ActionSpec")
    dataset_spec = runtime.dataset_specs[0]
    action_spec = runtime.action_specs[0]
    tasks = runtime.config.task_texts[dataset_spec.dataset_id]
    if len(tasks) != 1:
        raise ValueError("ManiSkill evaluator requires exactly one task text")
    task_text = tasks[0]
    evaluation = runtime.config.evaluation
    if (
        evaluation.env_id != "PickCube-v1"
        or evaluation.robot_uid != "panda_wristcam"
        or evaluation.obs_mode != "rgb"
        or evaluation.control_mode != "pd_ee_delta_pose"
        or evaluation.sim_backend != "physx_cpu"
    ):
        raise ValueError("unsupported ManiSkill environment contract")
    max_steps = evaluation.max_steps if args.max_steps is None else args.max_steps
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")

    device = torch.device("cuda:0")
    torch.manual_seed(runtime.config.seed)
    torch.cuda.manual_seed_all(runtime.config.seed)
    policy = assemble_runtime_policy(runtime, device=device)
    checkpoint = args.checkpoint.resolve(strict=True)
    manifest = RunManifest.from_json(
        (checkpoint / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.value["format_version"] == 2:
        load_distributed_policy_checkpoint_for_evaluation(
            checkpoint,
            expected_policy_checkpoint_contract(runtime, policy),
        )
    else:
        load_policy_checkpoint_strict(
            checkpoint,
            expected_policy_checkpoint_contract(runtime, policy),
        )
    manifest_hash = hashlib.sha256((checkpoint / "manifest.json").read_bytes()).hexdigest()
    condition_ids = encode_condition_ids(
        runtime.vocabulary,
        dataset_spec=dataset_spec,
        action_spec=action_spec,
        task_text=task_text,
        device=device,
    )

    from corrective_foresight.data.maniskill_conversion import make_pick_cube_rgb_env

    runner = ManiSkillClosedLoopRunner(
        env_factory=make_pick_cube_rgb_env,
        policy=policy,
        observation_adapter=ManiSkillOnlineObservationAdapter(
            dataset_spec=dataset_spec,
            condition_ids=condition_ids,
            task_text=task_text,
            device=device,
            context_steps=evaluation.context_steps,
        ),
        action_spec=action_spec,
        dataset_spec=dataset_spec,
        checkpoint=CheckpointProvenance(
            label=args.tag,
            directory=str(checkpoint),
            manifest_sha256=manifest_hash,
        ),
        environment_spec=ManiSkillEnvironmentSpec(
            env_id=evaluation.env_id,
            robot_uid=evaluation.robot_uid,
            obs_mode=evaluation.obs_mode,
            control_mode=evaluation.control_mode,
            sim_backend=evaluation.sim_backend,
        ),
        protocol=EvaluationProtocol(
            action_horizon=evaluation.action_horizon,
            execution_horizon=evaluation.execution_horizon,
            temporal_ensemble=evaluation.temporal_ensemble,
            solver=evaluation.solver,
            solver_intervals=evaluation.solver_intervals,
            context_steps=evaluation.context_steps,
        ),
        output_root=args.output_root or evaluation.output_root,
        max_steps=max_steps,
        action_to_environment=physical_action_to_maniskill_controller,
    )
    for seed in args.seeds:
        record = runner.run_episode(
            seed,
            flow_seed_stream(runtime.config.seed, seed),
            tag=args.tag,
        )
        result = record.value["result"]
        print(
            f"seed={seed} success={result['success']} length={result['length']} "
            f"video_sha256={record.value['video']['sha256']}"
        )


if __name__ == "__main__":
    main()
