"""Neural state-space model of the motor and its driven link.

Built through :func:`build_model` by training and inference alike, so the two
always instantiate the same architecture. :func:`load_model` adds the
checkpoint named by the config.
"""

import torch
import torch.nn as nn

from .config import ExperimentConfig
from .features import feature_dim


class NSSM(nn.Module):
    """Discrete-time non-linear state-space model.

        y_t     = g(x_t) + D u_t
        x_{t+1} = (1 - a) x_t + a f(x_t, u_t)

    The state update is a residual (forward Euler) step whose leak rate ``a`` is
    learned through a sigmoid, initialised small so the state starts out nearly
    constant and backpropagation through several hundred steps stays stable.

    ``initial_from_output`` adds an encoder that maps the first observed output
    of a sequence to the state the rollout starts from, for a plant whose
    episodes do not begin at rest.
    """

    def __init__(self, state_dim: int, input_dim: int, output_dim: int,
                 initial_from_output: bool = False):
        super().__init__()
        self.state_dim = state_dim

        self.f_net = nn.Sequential(
            nn.Linear(state_dim + input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, state_dim),
        )
        self.g_net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim),
        )
        # Direct feedthrough from command to output.
        self.d_net = nn.Linear(input_dim, output_dim, bias=False)
        # sigmoid(-2.0) ~ 0.12
        self.alpha_raw = nn.Parameter(torch.tensor([-2.0]))
        self.x0_net = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.Tanh(),
            nn.Linear(32, state_dim),
        ) if initial_from_output else None

    def initial_state(self, y_initial: torch.Tensor) -> torch.Tensor:
        """State to roll from, encoded from the sequence's first output.

        ``y_initial`` is [batch, output_dim] in the same normalised units the
        forward pass predicts. Without the encoder the state starts at zero.
        """
        if self.x0_net is None:
            return torch.zeros(y_initial.shape[0], self.state_dim,
                               device=y_initial.device)
        return self.x0_net(y_initial)

    def forward(self, u_sequence: torch.Tensor, x_initial: torch.Tensor) -> torch.Tensor:
        """Roll the model open-loop over a command sequence.

        Args:
            u_sequence: [batch, seq_len, input_dim]
            x_initial: [batch, state_dim]

        Returns:
            [batch, seq_len, output_dim]
        """
        alpha = torch.sigmoid(self.alpha_raw)

        state = x_initial
        outputs = []
        for t in range(u_sequence.shape[1]):
            u_t = u_sequence[:, t, :]
            outputs.append((self.g_net(state) + self.d_net(u_t)).unsqueeze(1))
            state = (1.0 - alpha) * state + alpha * self.f_net(torch.cat((state, u_t), dim=1))

        return torch.cat(outputs, dim=1)


class CascadedSystem(nn.Module):
    """One NSSM stage per joint, chained along the mechanism.

    Each stage predicts its joint's position and velocity from the command
    features; every stage after the first also takes the previous stage's
    prediction, detached, so a stage's error does not back-propagate into the
    one before it and the stages train independently.

    Output channel order matches ``cfg.target_cols``: two channels per stage,
    in the order the stages are chained.

    ``initial_state_from_data`` gives every stage an encoder from its first
    observed output to the state it rolls from, so a sequence that does not
    begin at rest is not asked to.
    """

    def __init__(self, state_dim: int, input_dim: int, history_window: int,
                 output_dim: int, initial_state_from_data: bool = False):
        super().__init__()
        if output_dim < 2 or output_dim % 2:
            raise ValueError(
                "CascadedSystem predicts a position and a velocity per stage, so "
                f"output_dim must be an even number of at least 2, got {output_dim}"
            )
        self.output_dim = output_dim

        command_dim = feature_dim(input_dim, history_window)
        self.stages = nn.ModuleList(
            NSSM(state_dim, command_dim + (0 if index == 0 else 2), output_dim=2,
                 initial_from_output=initial_state_from_data)
            for index in range(output_dim // 2)
        )

    def forward(self, u_sequence: torch.Tensor, initial_states) -> torch.Tensor:
        predictions = []
        stage_input = u_sequence
        for index, stage in enumerate(self.stages):
            prediction = stage(stage_input, initial_states[index])
            predictions.append(prediction)
            stage_input = torch.cat([u_sequence, prediction.detach()], dim=2)
        return torch.cat(predictions, dim=2)

    def initial_states(self, y_initial: torch.Tensor, device) -> list:
        """State each stage rolls from, one per stage.

        ``y_initial`` is the sequence's first output, normalised and shaped
        [batch, output_dim]. Each stage reads the two channels it predicts.
        """
        y_initial = y_initial.to(device)
        return [
            stage.initial_state(y_initial[:, 2 * index:2 * index + 2])
            for index, stage in enumerate(self.stages)
        ]


def build_model(cfg: ExperimentConfig) -> CascadedSystem:
    """Construct the model described by the config."""
    return CascadedSystem(cfg.state_dim, cfg.input_dim, cfg.history_window,
                          cfg.output_dim, cfg.initial_state_from_data)


def load_model(cfg: ExperimentConfig, device) -> CascadedSystem:
    """Build the model and restore the checkpoint named by the config."""
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(cfg.model_save_path, map_location=device))
    model.eval()
    return model
