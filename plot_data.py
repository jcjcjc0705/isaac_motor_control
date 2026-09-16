"""Inspect collected datasets: limit violations and per-episode waveforms.

Every CSV under ``cfg.data_dir`` is read in turn. Each one gets a limit report
on the terminal and a three-panel figure written alongside it: velocity and
position for ``cfg.inspect_episode_id``, plus an overview of the whole session
with the over-limit episodes shaded. A file that is unreadable, empty, or
missing a required column is reported and skipped, so one bad file does not
stop the rest.

Usage:

    python plot_data.py

All settings come from src/speed_control/speed_control/config.py. Edit that
file and re-run; this script takes no command-line arguments.
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

# Columns every plot and every line of the report needs.
REQUIRED_COLUMNS = (
    "time_actual", "episode_id", "input_u",
    "pos_motor", "vel_motor", "pos_joint1", "vel_joint1",
)


def dataset_paths(config) -> list:
    """Every CSV under data_dir, in name order."""
    paths = sorted(glob.glob(os.path.join(config.data_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files under {config.data_dir}")
    return paths


def load_dataset(path: str):
    """Read one CSV, or return None with a reason printed."""
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        print(f"  Skipped: {error}")
        return None
    if df.empty:
        print("  Skipped: no rows")
        return None
    missing = [name for name in REQUIRED_COLUMNS if name not in df.columns]
    if missing:
        print(f"  Skipped: missing column(s) {', '.join(missing)}")
        return None
    return df


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


def excitation_end(input_u) -> int:
    """Index where the excitation stops and the reset phase begins.

    Reads the boundary off the command column, so it holds for an episode of
    any length. Returns 0 when the episode carries no excitation.
    """
    nonzero = [i for i, value in enumerate(input_u) if value != 0.0]
    return nonzero[-1] + 1 if nonzero else 0


def plot_dataset(df: pd.DataFrame, config, failed_episodes, has_joint2, csv_path: str) -> str:
    time_col = "time_actual"
    name = os.path.basename(csv_path)

    episode_id = config.inspect_episode_id
    if episode_id not in df["episode_id"].values:
        episode_id = df["episode_id"].iloc[0]
        print(f"Warning: episode {config.inspect_episode_id} not present, "
              f"plotting episode {int(episode_id)} instead")
    episode = df[df["episode_id"] == episode_id]

    figure, (ax_velocity, ax_position, ax_overview) = plt.subplots(3, 1, figsize=(14, 15))
    figure.suptitle(name, fontsize=16)

    # Plain arrays, since matplotlib cannot index a pandas Series directly.
    episode_time = episode[time_col].to_numpy()
    session_time = df[time_col].to_numpy()

    ax_velocity.plot(episode_time, episode["input_u"].to_numpy(), "b--", alpha=0.6, label="Input (u)")
    ax_velocity.plot(episode_time, episode["vel_motor"].to_numpy(), "r-", alpha=0.4, label="Motor vel")
    ax_velocity.plot(episode_time, episode["vel_joint1"].to_numpy(), "g-", label="Joint1 vel")
    if has_joint2:
        ax_velocity.plot(episode_time, episode["vel_joint2"].to_numpy(), "m-", alpha=0.8, label="Joint2 vel")

    reset_index = min(excitation_end(episode["input_u"].to_numpy()),
                      len(episode) - 1)
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
    plt.subplots_adjust(top=0.95)
    output = os.path.join(
        config.data_dir, os.path.basename(csv_path).replace(".csv", "_analysis.png")
    )
    plt.savefig(output)
    plt.close()
    return output


def main() -> None:
    paths = dataset_paths(cfg)
    print(f"{len(paths)} CSV file(s) under {os.path.abspath(cfg.data_dir)}")

    plotted = 0
    for path in paths:
        print("")
        print(f"--- {path}")
        df = load_dataset(path)
        if df is None:
            continue
        failed_episodes, has_joint2 = report_limits(df, cfg)
        print(f"Saved {plot_dataset(df, cfg, failed_episodes, has_joint2, path)}")
        plotted += 1

    print("")
    print(f"Plotted {plotted} of {len(paths)} file(s)")


if __name__ == "__main__":
    main()
