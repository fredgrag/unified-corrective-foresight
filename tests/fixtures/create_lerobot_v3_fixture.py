from __future__ import annotations

from pathlib import Path

import numpy as np

from corrective_foresight.config.schema import ActionSpec, DatasetSpec


FPS = 10
HEIGHT = 32
WIDTH = 32
EPISODE_LENGTH = 12
REPO_ID = "fixtures/tiny_lerobot_v3"


def _video_feature() -> dict:
    return {
        "dtype": "video",
        "shape": (HEIGHT, WIDTH, 3),
        "names": ["height", "width", "channels"],
        "video.fps": FPS,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False,
    }


def _camera_frame(episode: int, frame: int, wrist: bool) -> np.ndarray:
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[..., 0] = 30 + episode * 80 + frame * 5
    image[..., 1] = 180 if wrist else 40
    image[:, frame % WIDTH, 2] = 255
    return image


def create_lerobot_v3_fixture(root: Path) -> tuple[DatasetSpec, ActionSpec]:
    import av
    from lerobot.datasets.io_utils import write_info
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    av.logging.set_level(av.logging.ERROR)
    features = {
        "observation.images.base": _video_feature(),
        "observation.images.wrist": _video_feature(),
        "observation.state": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["joint_a", "joint_b", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["x", "gripper"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=features,
        root=root,
        robot_type="fixture_robot",
        use_videos=True,
        image_writer_threads=2,
        vcodec="h264",
    )
    for episode in range(2):
        for frame in range(EPISODE_LENGTH):
            dataset.add_frame(
                {
                    "task": f"fixture task {episode}",
                    "observation.images.base": _camera_frame(episode, frame, False),
                    "observation.images.wrist": _camera_frame(episode, frame, True),
                    "observation.state": np.asarray(
                        [episode, frame / 10.0, (frame % 3) / 2.0], dtype=np.float32
                    ),
                    "action": np.asarray(
                        [frame / 20.0 + episode * 0.1, -frame / 30.0],
                        dtype=np.float32,
                    ),
                }
            )
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()

    training_actions = np.stack(
        [
            np.asarray([frame / 20.0, -frame / 30.0], dtype=np.float32)
            for frame in range(EPISODE_LENGTH)
        ]
    )
    training_mean = training_actions.mean(axis=0)
    training_std = training_actions.std(axis=0)
    dataset.meta.info["ucf"] = {
        "schema_version": 1,
        "train_action_stats": {
            "feature": "action",
            "mean": training_mean.tolist(),
            "std": training_std.tolist(),
            "count": int(training_actions.shape[0]),
        },
    }
    write_info(dataset.meta.info, dataset.meta.root)

    reopened = LeRobotDataset(repo_id=REPO_ID, root=root, video_backend="torchcodec")
    action_spec = ActionSpec(
        schema_version=1,
        spec_id="fixture.ee_delta.v1",
        dimension=2,
        names=("x", "gripper"),
        units=("m", "unitless"),
        action_space="end_effector",
        mode="delta",
        rotation_representation="none",
        arm_count=1,
        gripper_indices=(1,),
        control_mode="fixture_delta",
        frequency_hz=float(FPS),
        normalization_mean=tuple(float(value) for value in training_mean),
        normalization_std=tuple(float(value) for value in training_std),
        minimum=(-1.0, -1.0),
        maximum=(1.0, 1.0),
    )
    dataset_spec = DatasetSpec(
        schema_version=1,
        dataset_id=REPO_ID,
        repo_id=None,
        local_root=str(root.resolve()),
        revision="fixture-v1",
        fps=float(FPS),
        camera_features={
            "base": "observation.images.base",
            "wrist": "observation.images.wrist",
        },
        optional_camera_roles=(),
        proprio_features=("observation.state",),
        proprio_required=True,
        task_feature="task",
        language_feature=None,
        embodiment_id="fixture.robot.v1",
        action_feature="action",
        action_spec_id=action_spec.spec_id,
        sample_weight=1.0,
        split_episode_ids={"train": (0,), "validation": (1,), "evaluation": ()},
    )
    return dataset_spec, action_spec
