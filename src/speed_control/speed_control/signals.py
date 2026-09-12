"""Excitation signal generation and episode scheduling.

Pure NumPy with no ROS dependency, so waveforms can be generated and plotted
without launching the simulator.

Every generator shapes its output so the plant is predicted to stay inside
``cfg.planner_safe_limit``. The prediction is a crude one, and the collector
enforces a real-time abort on top of it.
"""

import math
import random
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import chirp

from .config import ExperimentConfig


class EpisodePlan(NamedTuple):
    """One episode's excitation recipe."""

    signal_type: str
    amplitude: float
    param_range: Optional[Tuple[float, float]]
    sign: int


def seed_everything(seed: int) -> None:
    """Seed both RNGs used by the generators so a run can be reproduced."""
    np.random.seed(seed)
    random.seed(seed)


# ----------------------------------------------------------------------
# Safety shaping
# ----------------------------------------------------------------------

def predict_peak(u: np.ndarray, gain: float) -> float:
    """Predict the largest joint excursion a command sequence will produce.

    A held effort settles at the angle where gravity balances it, so the
    excursion follows the command's peak rather than its integral and the
    prediction is a single multiplication by ``gain``, the radians per unit of
    effort held in ``cfg.effort_to_pos_gain``.

    The gain belongs to whichever joint swings furthest, which on a linkage is
    not necessarily the driven one.
    """
    return gain * float(np.max(np.abs(u)))


def scale_to_safe_peak(u: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    """Uniformly shrink a waveform until its predicted peak fits the safe band."""
    limit = cfg.planner_safe_limit - cfg.planner_margin
    peak = predict_peak(u, cfg.effort_to_pos_gain)
    if peak > limit:
        u = u * (limit / peak)
    return u


def limit_slew_rate(signal: np.ndarray, max_step: float) -> np.ndarray:
    """Clamp the step-to-step change so the drive never sees a jump discontinuity."""
    smoothed = np.zeros_like(signal)
    smoothed[0] = signal[0]
    for i in range(1, len(signal)):
        delta = np.clip(signal[i] - smoothed[i - 1], -max_step, max_step)
        smoothed[i] = smoothed[i - 1] + delta
    return smoothed


# ----------------------------------------------------------------------
# Generators, all sharing the (cfg, plan, length) signature
# ----------------------------------------------------------------------

def _make_prbs(cfg: ExperimentConfig, plan: EpisodePlan, length: int) -> np.ndarray:
    """Alternating-sign square pulses with randomised hold times.

    ``plan.param_range`` is the hold time in steps. Hold time does not bound
    the excursion -- a joint stops at the angle that balances the torque and
    stays there -- so the level is what the peak scaling shapes.
    """
    u = np.zeros(length)
    min_hold, max_hold = int(plan.param_range[0]), int(plan.param_range[1])

    index = 0
    sign = plan.sign
    while index < length:
        level = sign * np.random.uniform(0.3 * plan.amplitude, plan.amplitude)
        end = min(index + np.random.randint(min_hold, max_hold + 1), length)
        u[index:end] = level
        sign *= -1
        index = end
    return scale_to_safe_peak(u, cfg)


def _make_ramps(cfg: ExperimentConfig, plan: EpisodePlan, length: int) -> np.ndarray:
    """Triangle-like sweeps between alternating-sign targets."""
    u = np.zeros(length)
    low, high = plan.param_range

    index = 0
    sign = plan.sign
    last_value = 0.0
    while index < length:
        frequency = np.random.uniform(low, high)
        half_period_steps = max(1, int((1.0 / (2.0 * frequency)) / cfg.dt))
        end = min(index + half_period_steps, length)

        target = sign * np.random.uniform(0.3 * plan.amplitude, plan.amplitude)
        u[index:end] = np.linspace(last_value, target, end - index)

        last_value = target
        sign *= -1
        index = end
    return scale_to_safe_peak(u, cfg)


def _make_chirp(cfg: ExperimentConfig, plan: EpisodePlan, length: int) -> np.ndarray:
    """Linear frequency sweep across the plan's range."""
    t = np.arange(length) * cfg.dt
    low, high = plan.param_range
    u = plan.sign * plan.amplitude * chirp(t, f0=low, t1=t[-1], f1=high, method="linear")
    return scale_to_safe_peak(u, cfg)


def _make_multisine(cfg: ExperimentConfig, plan: EpisodePlan, length: int) -> np.ndarray:
    """Sum of three randomised tones, one per fixed frequency band.

    The bands are independent of ``plan.param_range``, so every episode of this
    type covers the whole spectrum.
    """
    bands = [(0.5, 1.5), (1.5, 3.0), (3.0, 6.0)]
    t = np.arange(length) * cfg.dt

    u = np.zeros(length)
    normaliser = 0.0
    for low, high in bands:
        wave = np.sin if random.choice([True, False]) else np.cos
        weight = random.uniform(0.5, 1.5)
        u += weight * wave(random.uniform(low, high) * t + random.uniform(0, 2 * np.pi))
        normaliser += 1.5

    if normaliser > 0:
        u = (u / normaliser) * plan.amplitude
    return scale_to_safe_peak(u, cfg)


def _make_smooth_noise(cfg: ExperimentConfig, plan: EpisodePlan, length: int) -> np.ndarray:
    """Cubic spline through alternating-sign random key points."""
    spacing = max(1, int(0.5 / cfg.dt))
    num_points = length // spacing + 2

    x_key = np.linspace(0, length, num_points)
    magnitudes = np.random.uniform(0.1 * plan.amplitude, plan.amplitude, num_points)
    signs = np.ones(num_points)
    signs[1::2] = -1

    spline = interp1d(x_key, magnitudes * signs, kind="cubic", fill_value="extrapolate")
    u = np.clip(spline(np.arange(length)), -plan.amplitude, plan.amplitude)
    return scale_to_safe_peak(u, cfg)


GENERATORS = {
    "PRBS": _make_prbs,
    "RAMPS": _make_ramps,
    "CHIRP": _make_chirp,
    "MULTISINE": _make_multisine,
    "SMOOTH_NOISE": _make_smooth_noise,
}


def render_signal(cfg: ExperimentConfig, plan: EpisodePlan, length: int) -> np.ndarray:
    """Build the command sequence for one episode."""
    generator = GENERATORS.get(plan.signal_type)
    if generator is None:
        raise ValueError(
            f"Unknown signal type {plan.signal_type!r}, expected one of {sorted(GENERATORS)}"
        )
    return limit_slew_rate(generator(cfg, plan, length), cfg.max_slew_rate)


# ----------------------------------------------------------------------
# Scheduling
# ----------------------------------------------------------------------

def frequency_split_levels(num_episodes: int) -> List[int]:
    """Powers of two up to sqrt(num_episodes), coarsest first.

    Sets how finely a signal type's parameter range is subdivided, so the
    schedule sweeps from wide ranges down to narrow ones.
    """
    if num_episodes < 2:
        return [1]
    levels = [1]
    current = 2
    while current <= int(math.sqrt(num_episodes)):
        levels.append(current)
        current *= 2
    return sorted(levels, reverse=True)


def build_type_schedule(
    signal_type: str,
    num_episodes: int,
    ranges: Dict[str, Tuple[float, float]],
    base_amplitude: float,
) -> List[EpisodePlan]:
    """Grid-sweep amplitude x sub-range x sign for one signal type."""
    amplitude_scales = [0.4, 0.6, 0.8, 1.0]
    signs = [1, -1]
    full_range = ranges.get(signal_type, (0.1, 1.0))

    schedule: List[EpisodePlan] = []
    while len(schedule) < num_episodes:
        for scale in amplitude_scales:
            for splits in frequency_split_levels(num_episodes):
                edges = np.linspace(full_range[0], full_range[1], splits + 1)
                # PRBS ranges are integer hold steps; collapse degenerate splits.
                if signal_type == "PRBS" and len(np.unique(edges.astype(int))) < 2:
                    edges = np.array([full_range[0], full_range[1]])
                for i in range(len(edges) - 1):
                    for sign in signs:
                        schedule.append(EpisodePlan(
                            signal_type=signal_type,
                            amplitude=scale * base_amplitude,
                            param_range=(edges[i], edges[i + 1]),
                            sign=sign,
                        ))
    return schedule[:num_episodes]


def split_episode_counts(
    total_episodes: int, signal_mix: Tuple[Tuple[str, float], ...]
) -> Dict[str, int]:
    """Divide episodes by ratio, giving the remainder to the last signal type."""
    counts: Dict[str, int] = {}
    assigned = 0
    for name, ratio in signal_mix[:-1]:
        counts[name] = int(total_episodes * ratio)
        assigned += counts[name]
    counts[signal_mix[-1][0]] = total_episodes - assigned
    return counts


def build_training_schedule(cfg: ExperimentConfig) -> List[EpisodePlan]:
    """Interleave the per-type schedules into the episode order to be recorded.

    Interleaving spreads every signal type across the whole session, so a drift
    part-way through does not land entirely on one type.
    """
    counts = split_episode_counts(cfg.total_episodes, cfg.signal_mix)
    per_type = {
        name: build_type_schedule(name, count, cfg.signal_ranges, cfg.amplitude)
        for name, count in counts.items()
    }

    schedule: List[EpisodePlan] = []
    longest = max((len(plans) for plans in per_type.values()), default=0)
    for i in range(longest):
        for name, _ in cfg.signal_mix:
            if i < len(per_type[name]):
                schedule.append(per_type[name][i])
    return schedule


def build_test_schedule(cfg: ExperimentConfig) -> List[EpisodePlan]:
    """Uniform schedule for the held-out test set."""
    return [
        EpisodePlan(cfg.test_signal_type, cfg.amplitude, cfg.signal_ranges.get(cfg.test_signal_type), 1)
        for _ in range(cfg.test_episodes)
    ]
