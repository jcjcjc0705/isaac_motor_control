"""Episode dataset and the train/validation split."""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import ExperimentConfig
from .features import Scaler, build_features

SIGNAL_TYPE_COL = "signal_type"


class EpisodeDataset(Dataset):
    """One sample per episode: a full command sequence and its response.

    Episodes whose length differs from ``cfg.seq_len`` are dropped, since the
    model is rolled out over a fixed horizon. ``episode_ids`` records which
    source episode each sample came from, which is what lets a split be built
    from the CSV's own metadata rather than from positional arithmetic.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: ExperimentConfig,
        scaler: Optional[Scaler] = None,
    ):
        self.cfg = cfg
        self.episode_ids: List[int] = []

        commands: List[np.ndarray] = []
        responses: List[np.ndarray] = []

        missing = [name for name in cfg.input_cols + cfg.target_cols
                   if name not in df.columns]
        if missing:
            raise ValueError(
                f"The dataset has no {', '.join(missing)} column(s). A model is "
                f"fitted to the columns named by input_cols and target_cols, so "
                f"a dataset recorded before they were chosen has to be collected "
                f"again."
            )

        groups = df.groupby("episode_id")
        for episode_id, group in sorted(groups, key=lambda item: item[0]):
            if len(group) != cfg.seq_len:
                continue
            commands.append(build_features(
                group[list(cfg.input_cols)].values, cfg.history_window
            ))
            responses.append(group[list(cfg.target_cols)].values)
            self.episode_ids.append(int(episode_id))

        if not commands:
            raise ValueError(
                f"No episode has exactly {cfg.seq_len} rows; check episode_len "
                f"and reset_len against the CSV."
            )

        self.u_data = torch.tensor(np.array(commands), dtype=torch.float32)
        self.y_data = torch.tensor(np.array(responses), dtype=torch.float32)

        dropped = groups.ngroups - len(self.episode_ids)
        if dropped:
            print(f"Warning: dropped {dropped} episode(s) with unexpected length")

        self.scaler = scaler if scaler is not None else Scaler.fit(self.u_data, self.y_data)
        self.u_data = self.scaler.normalize_u(self.u_data)
        self.y_data = self.scaler.normalize_y(self.y_data)

    def __len__(self) -> int:
        return self.u_data.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.u_data[index], self.y_data[index]


def episode_signal_types(
    df: pd.DataFrame, dataset: EpisodeDataset, cfg: ExperimentConfig
) -> Dict[int, str]:
    """Map each sample index to the signal type that excited it.

    Read from the ``signal_type`` column the collector writes. A CSV without
    that column falls back to the order implied by ``cfg.signal_mix``, which
    holds only for data recorded by the interleaved schedule.
    """
    if SIGNAL_TYPE_COL in df.columns:
        by_episode = df.groupby("episode_id")[SIGNAL_TYPE_COL].first().to_dict()
        return {i: by_episode[ep] for i, ep in enumerate(dataset.episode_ids)}

    print(
        f"Warning: '{SIGNAL_TYPE_COL}' column missing, inferring signal type from "
        f"episode order. Re-collect the dataset to remove this guess."
    )
    names = [name for name, _ in cfg.signal_mix]
    return {i: names[ep % len(names)] for i, ep in enumerate(dataset.episode_ids)}


def stratified_split(
    types: Dict[int, str], cfg: ExperimentConfig
) -> Tuple[List[int], List[int]]:
    """Split sample indices so every signal type is represented in validation.

    Each signal type is bucketed separately and sampled at a fixed stride, so
    validation covers every type whatever ``cfg.val_ratio`` is set to.
    """
    buckets: Dict[str, List[int]] = {}
    for index, signal_type in types.items():
        buckets.setdefault(signal_type, []).append(index)

    stride = max(2, round(1.0 / cfg.val_ratio))
    train_indices: List[int] = []
    val_indices: List[int] = []
    for indices in buckets.values():
        held_out = set(indices[::stride])
        val_indices.extend(sorted(held_out))
        train_indices.extend(i for i in indices if i not in held_out)

    return sorted(train_indices), sorted(val_indices)


def describe_split(
    types: Dict[int, str],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
) -> str:
    """Per-type breakdown of the split, for the training log."""
    lines = []
    for name in sorted({t for t in types.values()}):
        n_train = sum(1 for i in train_indices if types[i] == name)
        n_val = sum(1 for i in val_indices if types[i] == name)
        lines.append(f"  {name:<14} train {n_train:>4}  val {n_val:>4}")
    return "\n".join(lines)
