import torch
import torch.nn as nn

from scene.motion_model_base import MotionModelBaseCZ, expand_tau


class _FMVelocityMLP(nn.Module):
    """Neural velocity field: (x_t, t, tau) -> velocity.

    Input:  x_t  (N, 3) interpolated position at flow time t
            t    (N, 1) flow-matching internal time in [0, 1]
            tau  (N, 1) physical scene time in [0, 1]
    Output: v    (N, 3) velocity
    """

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, 3),
        )
        # Zero-init last layer so the model starts as identity (no motion)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x_t, t, tau], dim=-1)  # (N, 5)
        return self.net(inp)                      # (N, 3)


class FMMotionModelCZ(MotionModelBaseCZ):
    """Flow Matching motion model: x_tau = ODE_integrate(v_theta, x0, tau).

    Learns a neural velocity field v_theta(x_t, t, tau) that transports
    canonical Gaussian positions x0 to deformed positions x_tau via a
    continuous-time ODE.  Training uses Conditional Flow Matching (CFM):
    the velocity field is trained to follow straight-line optimal transport
    paths between x0 and x_tau.

    This is the 'Ours full' model in the paper.
    """

    motion_model_type = "fm"

    def __init__(self, control_num: int):
        super().__init__(control_num)
        self.velocity_mlp = _FMVelocityMLP(hidden=128)
        self.n_euler_steps = 10

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def initialize_random(self, fused_point_cloud: torch.Tensor):
        N = fused_point_cloud.shape[0]
        device, dtype = fused_point_cloud.device, fused_point_cloud.dtype
        control_xyz = torch.zeros((N, 2, 3), dtype=dtype, device=device)
        current_control_num = torch.full((N, 1), 2, dtype=torch.int64, device=device)
        self.control_xyz = nn.Parameter(control_xyz, requires_grad=False)
        self.current_control_num = nn.Parameter(current_control_num, requires_grad=False)
        self.velocity_mlp = self.velocity_mlp.to(device=device, dtype=dtype)

    def initialize_from_trajectory(self, canonical_xyz, dyn_trajectory, deform_spatial_scale):
        N = canonical_xyz.shape[0]
        device, dtype = canonical_xyz.device, canonical_xyz.dtype
        control_xyz = torch.zeros((N, 2, 3), dtype=dtype, device=device)
        current_control_num = torch.full((N, 1), 2, dtype=torch.int64, device=device)
        self.control_xyz = nn.Parameter(control_xyz, requires_grad=False)
        self.current_control_num = nn.Parameter(current_control_num, requires_grad=False)
        self.velocity_mlp = self.velocity_mlp.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Forward: Euler ODE integration from t=0 to t=1
    # ------------------------------------------------------------------

    def forward(self, x0: torch.Tensor, tau, deform_spatial_scale: float = 1.0):
        tau_t = expand_tau(tau, x0.shape[0], x0.device, x0.dtype)  # (N, 1)
        dt = 1.0 / self.n_euler_steps
        x = x0
        for i in range(self.n_euler_steps):
            t_val = torch.full((x.shape[0], 1), i * dt, device=x.device, dtype=x.dtype)
            v = self.velocity_mlp(x, t_val, tau_t)
            x = x + dt * v
        return x

    def forward_infer(self, x0: torch.Tensor, tau, deform_spatial_scale: float = 1.0):
        return self.forward(x0, tau, deform_spatial_scale)

    # ------------------------------------------------------------------
    # Velocity interface (satisfies base class)
    # ------------------------------------------------------------------

    def velocity(self, x_t: torch.Tensor, t, tau):
        t_t = expand_tau(t, x_t.shape[0], x_t.device, x_t.dtype)
        tau_t = expand_tau(tau, x_t.shape[0], x_t.device, x_t.dtype)
        return self.velocity_mlp(x_t, t_t, tau_t)

    # ------------------------------------------------------------------
    # Optimizer helpers
    # ------------------------------------------------------------------

    def get_mlp_parameters(self):
        """Return velocity MLP parameters for the external optimizer."""
        return self.velocity_mlp.parameters()

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def fm_loss(self, x0: torch.Tensor, tau) -> torch.Tensor:
        """Conditional Flow Matching loss.

        Trains the velocity field to follow straight-line optimal transport
        paths from x0 (canonical) to x1 (deformed at physical time tau).

        L_FM = || v_theta(x_t, t, tau) - v* ||^2
        where:
            x1    = forward(x0, tau).detach()   (current ODE target)
            t     ~ U(0, 1)
            x_t   = (1-t)*x0 + t*x1             (linear interpolation)
            v*    = x1 - x0                      (straight-line target velocity)
        """
        tau_t = expand_tau(tau, x0.shape[0], x0.device, x0.dtype)  # (N, 1)
        x0_d = x0.detach()

        # Get ODE-integrated target positions (detached to avoid 2nd-order grads)
        with torch.no_grad():
            x1 = self.forward(x0_d, tau_t)

        # Sample random flow time t ~ U(0, 1)
        t = torch.rand(x0_d.shape[0], 1, device=x0_d.device, dtype=x0_d.dtype)

        # Linear interpolation and target velocity
        x_t = (1.0 - t) * x0_d + t * x1   # (N, 3)
        v_star = x1 - x0_d                  # (N, 3)

        # Velocity prediction
        v_pred = self.velocity_mlp(x_t, t, tau_t)

        return (v_pred - v_star).pow(2).mean()

    def smoothness_loss(self, x0: torch.Tensor, tau, eps: float = 0.02) -> torch.Tensor:
        """Temporal smoothness via second-order finite differences on ODE output.

        Penalises large acceleration in the deformed positions:
            L_smooth = || delta(tau+eps) - 2*delta(tau) + delta(tau-eps) ||^2
        where delta(tau) = forward(x0, tau) - x0.
        """
        tau_t = expand_tau(tau, x0.shape[0], x0.device, x0.dtype)  # (N, 1)
        tau_prev = (tau_t - eps).clamp(0.0, 1.0)
        tau_next = (tau_t + eps).clamp(0.0, 1.0)

        x0_d = x0.detach()
        d_curr = self.forward(x0_d, tau_t) - x0_d
        d_prev = self.forward(x0_d, tau_prev) - x0_d
        d_next = self.forward(x0_d, tau_next) - x0_d

        accel = d_next - 2.0 * d_curr + d_prev  # (N, 3)
        return accel.pow(2).mean()
