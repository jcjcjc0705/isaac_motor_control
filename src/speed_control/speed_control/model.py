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
    """

    def __init__(self, state_dim: int, input_dim: int, output_dim: int):
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
    """Two NSSM stages: motor first, then the link it drives.

    The joint stage takes the motor stage's prediction as an extra input,
    detached, so joint-side error does not back-propagate into the motor model
    and the two stages train independently.

    Output channel order matches ``cfg.target_cols``: the first two channels
    come from the motor stage, the last two from the joint stage.
    """

    def __init__(self, state_dim: int, input_dim: int, history_window: int, output_dim: int):
        super().__init__()
        if output_dim != 4:
            raise ValueError(
                f"CascadedSystem is a fixed 2 + 2 cascade, got output_dim={output_dim}"
            )
        self.output_dim = output_dim

        command_dim = feature_dim(input_dim, history_window)
        self.stage_motor = NSSM(state_dim, command_dim, output_dim=2)
        self.stage_joint = NSSM(state_dim, command_dim + 2, output_dim=2)

    def forward(self, u_sequence: torch.Tensor, initial_states) -> torch.Tensor:
        pred_motor = self.stage_motor(u_sequence, initial_states[0])
        joint_input = torch.cat([u_sequence, pred_motor.detach()], dim=2)
        pred_joint = self.stage_joint(joint_input, initial_states[1])
        return torch.cat([pred_motor, pred_joint], dim=2)

    def initial_states(self, batch_size: int, device) -> list:
        """Zero initial state for each stage."""
        return [
            torch.zeros(batch_size, stage.state_dim, device=device)
            for stage in (self.stage_motor, self.stage_joint)
        ]


def build_model(cfg: ExperimentConfig) -> CascadedSystem:
    """Construct the model described by the config."""
    return CascadedSystem(cfg.state_dim, cfg.input_dim, cfg.history_window, cfg.output_dim)


def load_model(cfg: ExperimentConfig, device) -> CascadedSystem:
    """Build the model and restore the checkpoint named by the config."""
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(cfg.model_save_path, map_location=device))
    model.eval()
    return model
