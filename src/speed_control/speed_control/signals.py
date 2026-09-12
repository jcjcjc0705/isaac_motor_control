"""Excitation signal generation and episode scheduling.

Pure NumPy with no ROS dependency, so waveforms can be generated and plotted
without launching the simulator.

Every generator produces a waveform shape at unit amplitude, which
:func:`render_signal` then scales so the excursion it is predicted to produce
is the share of ``cfg.planner_safe_limit`` that the episode plan asks for. The
collector enforces a real-time abort on top of that prediction.
"""

import math
import random
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import chirp

from .config import ExperimentConfig


class EpisodePlan(NamedTuple):
    """One episode's excitation recipe.

    ``travel_fraction`` is the share of the safe band the episode should fill,
    so the effort it takes to get there is worked out per waveform rather than
    being carried here.
    """

    signal_type: str
    travel_fraction: float
    param_range: Optional[Tuple[float, float]]
    sign: int


def seed_everything(seed: int) -> None:
    """Seed both RNGs used by the generators so a run can be reproduced."""
    np.random.seed(seed)
    random.seed(seed)


# ----------------------------------------------------------------------
# Safety shaping
# ----------------------------------------------------------------------

def low_pass(u: np.ndarray, tau: float, dt: float) -> np.ndarray:
    """One-pole low-pass filter, the shape of the plant's approach to an angle."""
    u = np.asarray(u, dtype=float)
    if tau <= 0.0:
        return u.copy()
    alpha = dt / (tau + dt)
    filtered = np.zeros_like(u)
    value = 0.0
    for index, sample in enumerate(u):
        value += alpha * (sample - value)
        filtered[index] = value
    return filtered


def predict_peak(u: np.ndarray, cfg: ExperimentConfig) -> float:
    """Predict the largest joint excursion a command sequence will produce.

    A held effort settles at the angle where gravity balances it, so a command
    that holds still long enough reaches ``effort_to_pos_gain`` radians per
    unit, while one that reverses first does not get that far. Low-passing the
    command with ``cfg.plant_time_constant`` before taking its peak reproduces
    both ends: the filter passes a held level unchanged and attenuates a fast
    alternating one in proportion to how little time it spends on one side.

    The gain belongs to whichever joint swings furthest, which on a linkage is
    not necessarily the driven one.
    """
    filtered = low_pass(u, cfg.plant_time_constant, cfg.dt)
    return cfg.effort_to_pos_gain * float(np.max(np.abs(filtered)))


def predict_abort_measure(u: np.ndarray, cfg: ExperimentConfig) -> float:
    """Predict the largest value the collector's abort test will see.

    That test watches position plus ``cfg.lookahead`` seconds of velocity, so a
    waveform can pass :func:`predict_peak` on position and still trip it by
    being fast. Differentiating the same filtered command gives the velocity
    term, which makes this the command-side form of the abort test.
    """
    filtered = low_pass(u, cfg.plant_time_constant, cfg.dt)
    rate = np.gradient(filtered, cfg.dt)
    return cfg.effort_to_pos_gain * float(
        np.max(np.abs(filtered + cfg.lookahead * rate))
    )


def fit_to_travel(u: np.ndarray, cfg: ExperimentConfig, travel_fraction: float) -> np.ndarray:
    """Scale a waveform to the travel it is asked for, without tripping the abort.

    The position scaling runs in both directions: a waveform that would
    overshoot the band is shrunk, and one that would barely move the rig is
    grown, which is what lets every signal type use the same share of the
    travel. A waveform whose speed would then trip the abort test is shrunk
    further, so only the fast ones pay for it. The result is clipped to
    ``cfg.max_effort``, the ceiling on anything published.
    """
    peak = predict_peak(u, cfg)
    if peak < 1e-9:
        return np.zeros_like(u, dtype=float)
    scale = travel_fraction * cfg.safe_travel / peak

    measure = scale * predict_abort_measure(u, cfg)
    if measure > cfg.planner_lookahead_limit:
        scale *= cfg.planner_lookahead_limit / measure
    return np.clip(u * scale, -cfg.max_effort, cfg.max_effort)


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

    ``plan.param_range`` is the hold time in steps. It is what decides how far
    a given level carries the joint, so it also decides the effort the peak
    prediction ends up asking for.
    """
    u = np.zeros(length)
    min_hold, max_hold = int(plan.param_range[0]), int(plan.param_range[1])

    index = 0
    sign = plan.sign
    while index < length:
        level = sign * np.random.uniform(0.3, 1.0)
        end = min(index + np.random.randint(min_hold, max_hold + 1), length)
        u[index:end] = level
        sign *= -1
        index = end
    return u


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

        target = sign * np.random.uniform(0.3, 1.0)
        u[index:end] = np.linspace(last_value, target, end - index)

        last_value = target
        sign *= -1
        index = end
    return u


def _make_chirp(cfg: ExperimentConfig, plan: EpisodePlan, length: int) -> np.ndarray:
    """Linear frequency sweep across the plan's range."""
    t = np.arange(length) * cfg.dt
    low, high = plan.param_range
    return plan.sign * chirp(t, f0=low, t1=t[-1], f1=high, method="linear")


# Hz, one tone drawn from each. They straddle the linkage's modes rather than
# spreading evenly, and stop at 4 Hz, which the 20 Hz command rate still
# resolves with five samples per cycle.
MULTISINE_BANDS = ((0.2, 0.8), (0.8, 2.0), (2.0, 4.0))


def _make_multisine(cfg: ExperimentConfig, plan: EpisodePlan, length: int) -> np.ndarray:
    """Sum of three randomised tones, one per band of MULTISINE_BANDS.

    The bands are independent of ``plan.param_range``, so every episode of this
    type covers the whole spectrum.
    """
    t = np.arange(length) * cfg.dt

    u = np.zeros(length)
    normaliser = 0.0
    for low, high in MULTISINE_BANDS:
        wave = np.sin if random.choice([True, False]) else np.cos
        weight = random.uniform(0.5, 1.5)
        frequency = random.uniform(low, high)
        u += weight * wave(2.0 * np.pi * frequency * t + random.uniform(0, 2 * np.pi))
        normaliser += 1.5

    if normaliser > 0:
        u = u / normaliser
    return u


def _make_smooth_noise(cfg: ExperimentConfig, plan: EpisodePlan, length: int) -> np.ndarray:
    """Cubic spline through alternating-sign random key points.

    ``plan.param_range`` is the key-point rate in Hz. Successive points
    alternate in sign, so placing them half a period apart puts the waveform's
    fundamental at the drawn frequency.
    """
    low, high = plan.param_range
    frequency = np.random.uniform(low, high)
    spacing = max(1, int((1.0 / (2.0 * frequency)) / cfg.dt))
    num_points = length // spacing + 2

    x_key = np.linspace(0, length, num_points)
    magnitudes = np.random.uniform(0.1, 1.0, num_points)
    signs = np.ones(num_points)
    signs[1::2] = -1

    spline = interp1d(x_key, magnitudes * signs, kind="cubic", fill_value="extrapolate")
    return np.clip(spline(np.arange(length)), -1.0, 1.0)


GENERATORS = {
    "PRBS": _make_prbs,
    "RAMPS": _make_ramps,
    "CHIRP": _make_chirp,
    "MULTISINE": _make_multisine,
    "SMOOTH_NOISE": _make_smooth_noise,
}


def render_signal(cfg: ExperimentConfig, plan: EpisodePlan, length: int) -> np.ndarray:
    """Build the command sequence for one episode, scaled to its travel share."""
    generator = GENERATORS.get(plan.signal_type)
    if generator is None:
        raise ValueError(
            f"Unknown signal type {plan.signal_type!r}, expected one of {sorted(GENERATORS)}"
        )
    shape = generator(cfg, plan, length)
    return limit_slew_rate(fit_to_travel(shape, cfg, plan.travel_fraction),
                           cfg.max_slew_rate)


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
) -> List[EpisodePlan]:
    """Grid-sweep travel share x sub-range x sign for one signal type."""
    travel_fractions = [0.4, 0.6, 0.8, 1.0]
    signs = [1, -1]
    full_range = ranges.get(signal_type, (0.1, 1.0))

    schedule: List[EpisodePlan] = []
    while len(schedule) < num_episodes:
        for fraction in travel_fractions:
            for splits in frequency_split_levels(num_episodes):
                edges = np.linspace(full_range[0], full_range[1], splits + 1)
                # PRBS ranges are integer hold steps; collapse degenerate splits.
                if signal_type == "PRBS" and len(np.unique(edges.astype(int))) < 2:
                    edges = np.array([full_range[0], full_range[1]])
                for i in range(len(edges) - 1):
                    for sign in signs:
                        schedule.append(EpisodePlan(
                            signal_type=signal_type,
                            travel_fraction=fraction,
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
        name: build_type_schedule(name, count, cfg.signal_ranges)
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
        EpisodePlan(cfg.test_signal_type, 1.0,
                    cfg.signal_ranges.get(cfg.test_signal_type), 1)
        for _ in range(cfg.test_episodes)
    ]
