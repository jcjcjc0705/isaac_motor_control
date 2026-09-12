"""Inspect a collected dataset: limit violations and per-episode waveforms.

All settings come from src/speed_control/speed_control/config.py. Edit that file
and re-run; this script takes no command-line arguments.
"""

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "speed_control"))

from speed_control.config import DEFAULT_CONFIG as cfg


def resolve_csv(config) -> str:
    """Use the configured file, falling back to the newest CSV in data_dir."""
    if os.path.exists(config.inspect_file):
        print(f"Reading {config.inspect_file}")
        return config.inspect_file

    print(f"Warning: {config.inspect_file} not found, falling back to the newest CSV")
    candidates = glob.glob(os.path.join(config.data_dir, "*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV files under {config.data_dir}")

    newest = max(candidates, key=os.path.getctime)
    print(f"Reading {newest}")
    return newest


def report_limits(df: pd.DataFrame, config):
    """Print how many episodes exceeded the hard limit, and which ones."""
    has_joint2 = "pos_joint2" in df.columns

    exceeded = (df["pos_motor"].abs() > config.hard_limit) | \
               (df["pos_joint1"].abs() > config.hard_limit)
    if has_joint2:
        exceeded |= df["pos_joint2"].abs() > config.hard_limit

    failed = df[exceeded]["episode_id"].unique().astype(int)

    print("")
    print("Dataset limit report")
    print(f"  Layout         : {'motor + joint1 + joint2' if has_joint2 else 'motor + joint1'}")
    print(f"  Episodes       : {df['episode_id'].nunique()}")
    print(f"  Hard limit     : {config.hard_limit} rad")
    print(f"  Rows exceeding : {int(exceeded.sum())} ({exceeded.mean() * 100:.2f}%)")
    print(f"  Episodes hit   : {len(failed)}")
    if len(failed):
        print(f"  Episode ids    : {failed.tolist()}")
    print("")

    return failed, has_joint2


def plot_dataset(df: pd.DataFrame, config, failed_episodes, has_joint2, csv_path: str) -> str:
    time_col = "time_actual"

    episode_id = config.inspect_episode_id
    if episode_id not in df["episode_id"].values:
        episode_id = df["episode_id"].iloc[0]
        print(f"Warning: episode {config.inspect_episode_id} not present, "
              f"plotting episode {int(episode_id)} instead")
    episode = df[df["episode_id"] == episode_id]

    _, (ax_velocity, ax_position, ax_overview) = plt.subplots(3, 1, figsize=(14, 15))

    # matplotlib cannot index a pandas Series directly; hand it plain arrays.
    episode_time = episode[time_col].to_numpy()
    session_time = df[time_col].to_numpy()

    ax_velocity.plot(episode_time, episode["input_u"].to_numpy(), "b--", alpha=0.6, label="Input (u)")
    ax_velocity.plot(episode_time, episode["vel_motor"].to_numpy(), "r-", alpha=0.4, label="Motor vel")
    ax_velocity.plot(episode_time, episode["vel_joint1"].to_numpy(), "g-", label="Joint1 vel")
    if has_joint2:
        ax_velocity.plot(episode_time, episode["vel_joint2"].to_numpy(), "m-", alpha=0.8, label="Joint2 vel")

    reset_index = min(config.episode_len - 1, len(episode) - 1)
    if reset_index > 0:
        ax_velocity.axvline(x=episode_time[reset_index], color="orange",
                            linestyle=":", label="Reset phase start")
    ax_velocity.set_title(f"Velocity detail (episode {int(episode_id)})")
    ax_velocity.legend(loc="upper right")
    ax_velocity.grid(True, alpha=0.3)

    ax_position.plot(episode_time, episode["pos_motor"].to_numpy(), "r-", alpha=0.4, label="Motor pos")
    ax_position.plot(episode_time, episode["pos_joint1"].to_numpy(), "g-", label="Joint1 pos")
    if has_joint2:
        ax_position.plot(episode_time, episode["pos_joint2"].to_numpy(), "m-", alpha=0.8, label="Joint2 pos")
    for sign in (1, -1):
        ax_position.axhline(y=sign * config.hard_limit, color="r", linestyle="-.", alpha=0.8)
    ax_position.set_title(f"Position detail (episode {int(episode_id)})")
    ax_position.legend(loc="upper right")
    ax_position.grid(True, alpha=0.3)

    ax_overview.plot(session_time, df["pos_joint1"].to_numpy(), color="green", alpha=0.5, label="Joint1")
    if has_joint2:
        ax_overview.plot(session_time, df["pos_joint2"].to_numpy(), color="purple", alpha=0.5, label="Joint2")
    for sign in (1, -1):
        ax_overview.axhline(y=sign * config.hard_limit, color="red", linestyle="-.", alpha=0.8)

    labelled = False
    for episode_key, group in df.groupby("episode_id"):
        if episode_key in failed_episodes:
            ax_overview.axvspan(group[time_col].min(), group[time_col].max(),
                                color="red", alpha=0.3,
                                label="" if labelled else "Exceeds limit")
            labelled = True
    ax_overview.set_title("Full session overview (red spans exceed the hard limit)")
    ax_overview.set_xlabel("Time [s]")
    ax_overview.legend(loc="upper right")
    ax_overview.grid(True, alpha=0.3)

    plt.tight_layout()
    output = os.path.join(
        config.data_dir, os.path.basename(csv_path).replace(".csv", "_analysis.png")
    )
    plt.savefig(output)
    plt.close()
    return output


def main() -> None:
    csv_path = resolve_csv(cfg)
    df = pd.read_csv(csv_path)
    failed_episodes, has_joint2 = report_limits(df, cfg)
    print(f"Saved {plot_dataset(df, cfg, failed_episodes, has_joint2, csv_path)}")


if __name__ == "__main__":
    main()
