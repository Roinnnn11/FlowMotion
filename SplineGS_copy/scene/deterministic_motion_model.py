import torch
import torch.nn as nn

from scene.motion_model_base import MotionModelBaseCZ, expand_tau


class _MotionMLP(nn.Module):
    """Shared MLP: (x0, tau) -> delta_x.

    Input:  x0   (N, 3) canonical Gaussian positions
            tau  (N, 1) physical time in [0, 1]
    Output: delta_x (N, 3) displacement
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, 3),
        )
        # Zero-init last layer so the model starts as identity (no motion)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x0: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x0, tau], dim=-1)  # (N, 4)
        return self.net(inp)               # (N, 3)


class DeterministicMotionModelCZ(MotionModelBaseCZ):
    """Deterministic MLP motion model: x_tau = x0 + f_theta(x0, tau).

    Replaces the spline trajectory with a shared MLP that maps canonical
    position + physical time to a displacement.  This serves as the
    'Ours w/o FM' baseline in the paper.

    The per-Gaussian `control_xyz` tensor is kept as a dummy parameter
    (requires_grad=False) solely to satisfy the densification interface in
    GaussianModel.  The actual learnable parameters live in `self.mlp`.
    """

    motion_model_type = "deterministic"

    def __init__(self, control_num: int):
        super().__init__(control_num)
        self.mlp = _MotionMLP(hidden=64)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def initialize_random(self, fused_point_cloud: torch.Tensor):
        N = fused_point_cloud.shape[0]
        device, dtype = fused_point_cloud.device, fused_point_cloud.dtype
        # Dummy control_xyz: shape (N, 2, 3), not optimised
        control_xyz = torch.zeros((N, 2, 3), dtype=dtype, device=device)
        current_control_num = torch.full((N, 1), 2, dtype=torch.int64, device=device)
        self.control_xyz = nn.Parameter(control_xyz, requires_grad=False)
        self.current_control_num = nn.Parameter(current_control_num, requires_grad=False)
        self.mlp = self.mlp.to(device=device, dtype=dtype)

    def initialize_from_trajectory(self, canonical_xyz, dyn_trajectory, deform_spatial_scale):
        N = canonical_xyz.shape[0]
        device, dtype = canonical_xyz.device, canonical_xyz.dtype
        control_xyz = torch.zeros((N, 2, 3), dtype=dtype, device=device)
        current_control_num = torch.full((N, 1), 2, dtype=torch.int64, device=device)
        self.control_xyz = nn.Parameter(control_xyz, requires_grad=False)
        self.current_control_num = nn.Parameter(current_control_num, requires_grad=False)
        self.mlp = self.mlp.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x0: torch.Tensor, tau, deform_spatial_scale: float = 1.0):
        tau_t = expand_tau(tau, x0.shape[0], x0.device, x0.dtype)  # (N, 1)
        delta = self.mlp(x0, tau_t)                                  # (N, 3)
        return x0 + delta

    def forward_infer(self, x0: torch.Tensor, tau, deform_spatial_scale: float = 1.0):
        return self.forward(x0, tau, deform_spatial_scale)

    # ------------------------------------------------------------------
    # Optimizer helpers
    # ------------------------------------------------------------------

    def get_mlp_parameters(self):
        """Return MLP parameters for the external optimizer."""
        return self.mlp.parameters()

    # ------------------------------------------------------------------
    # Regularisation
    # ------------------------------------------------------------------

    def smoothness_loss(self, x0: torch.Tensor, tau, eps: float = 0.02) -> torch.Tensor:
        """Temporal smoothness via second-order finite differences.

        Penalises large acceleration in the predicted displacement:
            L_smooth = || delta(tau+eps) - 2*delta(tau) + delta(tau-eps) ||^2
        """
        tau_t = expand_tau(tau, x0.shape[0], x0.device, x0.dtype)  # (N, 1)
        tau_prev = (tau_t - eps).clamp(0.0, 1.0)
        tau_next = (tau_t + eps).clamp(0.0, 1.0)

        x0_d = x0.detach()  # don't propagate smoothness grad through positions
        d_curr = self.mlp(x0_d, tau_t)
        d_prev = self.mlp(x0_d, tau_prev)
        d_next = self.mlp(x0_d, tau_next)

        accel = d_next - 2.0 * d_curr + d_prev  # (N, 3)
        return accel.pow(2).mean()
