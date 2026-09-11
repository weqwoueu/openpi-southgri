"""Compute SouthGrid normalization statistics without decoding camera videos.

This is a fast path for LeRobot v2.1 datasets whose robot-specific transform
passes ``observation.state`` and ``action`` through unchanged. It reads only
those Parquet columns and reconstructs the future action chunks used by the
standard OpenPI data loader, including end-of-episode clamping.
"""

from pathlib import Path

import numpy as np
import polars as pl
import tqdm
import tyro

from openpi.shared import normalize
from openpi.training import config as _config


class _BatchedStats:
    """Feed RunningStats in the same outer batch sizes as the standard script."""

    def __init__(self, batch_size: int):
        self.stats = normalize.RunningStats()
        self.batch_size = batch_size
        self._pending: list[np.ndarray] = []
        self._pending_count = 0
        self.count = 0

    def add(self, values: np.ndarray) -> None:
        if not len(values):
            return
        self._pending.append(values)
        self._pending_count += len(values)
        if self._pending_count < self.batch_size:
            return

        combined = np.concatenate(self._pending, axis=0)
        full_count = len(combined) // self.batch_size * self.batch_size
        for start in range(0, full_count, self.batch_size):
            self.stats.update(combined[start : start + self.batch_size])
        self.count += full_count
        self._pending = [combined[full_count:]] if full_count < len(combined) else []
        self._pending_count = len(combined) - full_count

    def finish(self) -> normalize.NormStats:
        # The standard Torch DataLoader uses drop_last=True, so intentionally
        # leave an incomplete final outer batch out of the statistics.
        if self.count == 0:
            raise ValueError("Dataset does not contain one complete statistics batch")
        return self.stats.get_statistics()


def _as_matrix(frame: pl.DataFrame, key: str) -> np.ndarray:
    values = np.asarray(frame.get_column(key).to_list(), dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"Expected finite vector column {key!r}, got shape {values.shape}")
    return values


def _resolve_dataset_dir(repo_id: str, dataset_dir: str | None) -> Path:
    if dataset_dir is not None:
        return Path(dataset_dir).expanduser().resolve()

    from lerobot.common.constants import HF_LEROBOT_HOME

    return (Path(HF_LEROBOT_HOME) / repo_id).expanduser().resolve()


def main(
    config_name: str,
    dataset_dir: str | None = None,
    max_frames: int | None = None,
) -> None:
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("The selected config does not define a LeRobot repo_id")
    if data_config.action_sequence_keys != ("action",):
        raise ValueError(
            "Fast statistics currently require action_sequence_keys=('action',); "
            f"got {data_config.action_sequence_keys!r}"
        )
    if config.policy_metadata is None or config.policy_metadata.get("robot") != "g1_omnipicker":
        raise ValueError("This fast path currently supports the SouthGrid g1_omnipicker transform only")

    dataset_path = _resolve_dataset_dir(data_config.repo_id, dataset_dir)
    parquet_files = sorted((dataset_path / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found under {dataset_path / 'data'}")

    horizon = config.model.action_horizon
    stats_batch_size = config.batch_size
    expected_dim = int(config.policy_metadata["robot_action_dim"])
    state_stats = _BatchedStats(stats_batch_size)
    action_stats = _BatchedStats(stats_batch_size)
    processed_frames = 0

    print(f"Dataset: {dataset_path}")
    print(f"Parquet files: {len(parquet_files)}")
    print(f"Robot action dimension: {expected_dim}")
    print(f"Action horizon: {horizon}")
    print("Camera videos: skipped")

    columns = ["episode_index", "frame_index", "observation.state", "action"]
    for parquet_path in tqdm.tqdm(parquet_files, desc="Reading Parquet"):
        frame = pl.read_parquet(parquet_path, columns=columns).sort(["episode_index", "frame_index"])
        for episode in frame.partition_by("episode_index", maintain_order=True):
            states = _as_matrix(episode, "observation.state")
            actions = _as_matrix(episode, "action")
            if states.shape[-1] != expected_dim or actions.shape[-1] != expected_dim:
                raise ValueError(
                    f"Expected {expected_dim}-D state/action in {parquet_path}, "
                    f"got {states.shape[-1]} and {actions.shape[-1]}"
                )
            if len(states) != len(actions):
                raise ValueError(f"State/action length mismatch in {parquet_path}")

            take = len(states)
            if max_frames is not None:
                take = min(take, max_frames - processed_frames)
            if take <= 0:
                break

            state_stats.add(states[:take])
            offsets = np.arange(horizon)[None, :]
            for start in range(0, take, stats_batch_size):
                stop = min(start + stats_batch_size, take)
                indices = np.arange(start, stop)[:, None] + offsets
                indices = np.minimum(indices, len(actions) - 1)
                action_stats.add(actions[indices])
            processed_frames += take

            if max_frames is not None and processed_frames >= max_frames:
                break
        if max_frames is not None and processed_frames >= max_frames:
            break

    norm_stats = {
        "state": state_stats.finish(),
        "actions": action_stats.finish(),
    }
    output_path = config.assets_dirs / data_config.repo_id
    normalize.save(output_path, norm_stats)

    print(f"Processed frames: {processed_frames}")
    print(f"Statistics samples: state={state_stats.count}, action chunks={action_stats.count}")
    print(f"Wrote normalization statistics to: {output_path / 'norm_stats.json'}")


if __name__ == "__main__":
    tyro.cli(main)
