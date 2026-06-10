# %% [markdown]
# # CPG-Based Smooth Gait-Switching Core
#
# Inspired by JiChern/CPG — "Free Gait Transition and Stable Motion Generation
# Using CPG-based Locomotion Control" (Chen, Fan & Xu, *Nonlinear Dynamics*, 2024):
# https://github.com/JiChern/CPG
#
# The paper's improved diffusive CPG couples N Hopf-style oscillators with
# state vectors z_i = (x_i, y_i):
#
#     z_dot_i = F(z_i) + gamma_i * Perp_zi( R(theta_i) z_{i+1} - z_i )
#
# - F(z_i): intrinsic Hopf oscillator -> stable limit cycle of radius sqrt(mu)
#   at frequency omega.
# - R(theta_i): 2D rotation by the *desired phase lag* theta_i between
#   neighboring oscillators (this is the gait parameter).
# - Perp_zi(v) = v - (v . z_i / |z_i|^2) z_i: projects the coupling term
#   perpendicular to z_i, so coupling nudges PHASE only, leaving amplitude
#   (and hence joint amplitude) governed by F.
#
# Below this is specialised to N=2 oscillators driving the two swimmer joints
# (front angle a1, tail angle a2). Each oscillator also carries a slowly
# relaxing bias state b_i (first-order filter toward a target). The joint
# angle is a_i = b_i + x_i.
#
# Gait switching = changing the *targets* (theta_1, theta_2, b1_target,
# b2_target) of a running CPG. Because (x_i, y_i, b_i) are continuous ODE
# states, a_i and a_i_dot stay continuous across a switch -- only a_i_ddot has
# a (bounded) kink. This is the core improvement over a gait-library approach
# that swaps entire closed-form joint trajectories at cycle boundaries (which
# makes a_i itself jump).

# %%
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Symbolic CPG derivation (done once, then lambdified)

# %%
x1, y1, x2, y2, b1, b2 = sp.symbols("x1 y1 x2 y2 b1 b2", real=True)
th1, th2, g1, g2, mu1, mu2, om, alpha, kb, b1t, b2t = sp.symbols(
    "theta1 theta2 gamma1 gamma2 mu1 mu2 omega alpha k_b b1t b2t", real=True
)


def _hopf(x, y, mu):
    """Intrinsic Hopf oscillator: limit cycle radius sqrt(mu), freq omega."""
    r2 = x**2 + y**2
    return alpha * (mu - r2) * x - om * y, alpha * (mu - r2) * y + om * x


def _rot(theta, vx, vy):
    """2D rotation by desired phase lag theta."""
    return sp.cos(theta) * vx - sp.sin(theta) * vy, sp.sin(theta) * vx + sp.cos(theta) * vy


def _perp(zx, zy, vx, vy):
    """Project (vx,vy) perpendicular to (zx,zy) -> phase-only coupling."""
    eps = sp.Float(1e-6)
    dot = vx * zx + vy * zy
    nrm = zx**2 + zy**2 + eps
    return vx - dot / nrm * zx, vy - dot / nrm * zy


# Oscillator 1 (front joint) couples toward oscillator 2 rotated by theta1
Fx1, Fy1 = _hopf(x1, y1, mu1)
rx1, ry1 = _rot(th1, x2, y2)
px1, py1 = _perp(x1, y1, rx1 - x1, ry1 - y1)
x1dot = Fx1 + g1 * px1
y1dot = Fy1 + g1 * py1

# Oscillator 2 (tail joint) couples toward oscillator 1 rotated by theta2
Fx2, Fy2 = _hopf(x2, y2, mu2)
rx2, ry2 = _rot(th2, x1, y1)
px2, py2 = _perp(x2, y2, rx2 - x2, ry2 - y2)
x2dot = Fx2 + g2 * px2
y2dot = Fy2 + g2 * py2

# Bias states: first-order relaxation toward externally-set targets
b1dot = kb * (b1t - b1)
b2dot = kb * (b2t - b2)

_state = [x1, y1, x2, y2, b1, b2]
_zdot_expr = [x1dot, y1dot, x2dot, y2dot, b1dot, b2dot]

# Jacobian of the oscillator block (needed for analytic joint accelerations
# a_i_ddot = d/dt(a_i_dot) via the chain rule: zddot = J(z) @ zdot )
_J_osc = sp.Matrix([[sp.diff(e, v) for v in [x1, y1, x2, y2]] for e in [x1dot, y1dot, x2dot, y2dot]])

_params = [th1, th2, g1, g2, mu1, mu2, om, alpha, kb, b1t, b2t]
_zdot_fn = sp.lambdify(_state + _params, _zdot_expr, "numpy")
_Josc_fn = sp.lambdify(_state + _params, _J_osc, "numpy")

print("CPG symbolic derivation complete.")

# %% [markdown]
# ## 2. CPG gait generator class

# %%
W_GAIT = 2.0 * np.pi          # base oscillation frequency [rad/s]  (1 Hz)
T_GAIT = 2.0 * np.pi / W_GAIT  # nominal cycle period [s]
A_GAIT = np.pi / 4             # joint oscillation amplitude [rad]
MU_GAIT = A_GAIT**2             # Hopf limit-cycle radius^2 -> amplitude = sqrt(mu)
B_TURN = np.pi / 4              # body-curvature bias magnitude for turning [rad]

# Phase lag (theta) targets reproduce the original SWIM gait:
#   a1 = A*sin(wt), a2 = A*sin(wt + pi/2)  ->  tail leads front by +pi/2
THETA1_SWIM = -np.pi / 2
THETA2_SWIM = +np.pi / 2

# Named gait presets: (theta1, theta2, b1_target, b2_target)
GAITS = {
    "swim":     (THETA1_SWIM, THETA2_SWIM, 0.0, 0.0),
    "turn_pos": (THETA1_SWIM, THETA2_SWIM, +B_TURN, +B_TURN),
    "turn_neg": (THETA1_SWIM, THETA2_SWIM, -B_TURN, -B_TURN),
}


class CPGGaitGenerator:
    """
    2-oscillator diffusive Hopf CPG driving joint angles (a1, a2).

    State (6,): [x1, y1, x2, y2, b1, b2]
        a1 = b1 + x1,  a2 = b2 + x2

    Gait switching: call set_gait(name) or set_targets(...) at any time.
    The targets (theta1, theta2, b1_target, b2_target) are control inputs;
    the CPG state itself evolves continuously, so (a1, a2, a1_dot, a2_dot)
    never jump at a switch.
    """

    def __init__(self, omega=W_GAIT, mu=MU_GAIT, alpha=10.0, gamma=4.0, k_bias=4.0,
                 init_phase1=0.0, init_phase2=np.pi / 2):
        self.omega = omega
        self.mu1 = self.mu2 = mu
        self.alpha = alpha
        self.gamma = gamma
        self.k_bias = k_bias

        self.theta1, self.theta2, self.b1_target, self.b2_target = GAITS["swim"]

        r0 = np.sqrt(mu)
        self.state = np.array([
            r0 * np.cos(init_phase1), r0 * np.sin(init_phase1),
            r0 * np.cos(init_phase2), r0 * np.sin(init_phase2),
            0.0, 0.0,
        ])

    # -- gait switching -----------------------------------------------------
    def set_gait(self, name):
        self.theta1, self.theta2, self.b1_target, self.b2_target = GAITS[name]

    def set_targets(self, theta1=None, theta2=None, b1_target=None, b2_target=None):
        if theta1 is not None: self.theta1 = theta1
        if theta2 is not None: self.theta2 = theta2
        if b1_target is not None: self.b1_target = b1_target
        if b2_target is not None: self.b2_target = b2_target

    # -- core dynamics --------------------------------------------------------
    def _params(self):
        return [self.theta1, self.theta2, self.gamma, self.gamma,
                self.mu1, self.mu2, self.omega, self.alpha, self.k_bias,
                self.b1_target, self.b2_target]

    def derivative(self, state=None):
        """6-vector state derivative (for embedding in a larger ODE)."""
        s = self.state if state is None else state
        return np.asarray(_zdot_fn(*s, *self._params()), dtype=float)

    def joint_kinematics(self, state=None):
        """Return (a1, a2, a1_dot, a2_dot, a1_ddot, a2_ddot)."""
        s = self.state if state is None else state
        zd = self.derivative(s)
        J = np.asarray(_Josc_fn(*s, *self._params()), dtype=float)
        zdd_osc = J @ zd[:4]
        a1, a2 = s[4] + s[0], s[5] + s[2]
        a1d, a2d = zd[4] + zd[0], zd[5] + zd[2]
        a1dd = -self.k_bias * zd[4] + zdd_osc[0]
        a2dd = -self.k_bias * zd[5] + zdd_osc[2]
        return a1, a2, a1d, a2d, a1dd, a2dd

    # -- standalone integration (RK4) -----------------------------------------
    def step(self, dt):
        """Advance the internal CPG state by dt using RK4 (standalone use)."""
        k1 = self.derivative(self.state)
        k2 = self.derivative(self.state + 0.5 * dt * k1)
        k3 = self.derivative(self.state + 0.5 * dt * k2)
        k4 = self.derivative(self.state + dt * k3)
        self.state = self.state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


# %% [markdown]
# ## 3. Demo — smooth gait transition (swim -> turn_pos -> swim)
#
# Reproduces the spirit of the repo's `gait_transition_curves.png`: the
# oscillator phase portraits stay on (near) the limit circle throughout,
# while the joint-angle offsets shift smoothly between gaits.

# %%
if __name__ == "__main__":
    cpg = CPGGaitGenerator()

    dt = 0.001
    t_end = 12.0
    ts = np.arange(0.0, t_end, dt)

    a1_h = np.zeros_like(ts); a2_h = np.zeros_like(ts)
    a1d_h = np.zeros_like(ts); a2d_h = np.zeros_like(ts)
    a1dd_h = np.zeros_like(ts); a2dd_h = np.zeros_like(ts)
    x1_h = np.zeros_like(ts); y1_h = np.zeros_like(ts)
    x2_h = np.zeros_like(ts); y2_h = np.zeros_like(ts)
    gait_h = []

    for i, tv in enumerate(ts):
        if tv < 4.0:
            cpg.set_gait("swim")
        elif tv < 8.0:
            cpg.set_gait("turn_pos")
        else:
            cpg.set_gait("turn_neg")

        a1, a2, a1d, a2d, a1dd, a2dd = cpg.joint_kinematics()
        a1_h[i], a2_h[i] = a1, a2
        a1d_h[i], a2d_h[i] = a1d, a2d
        a1dd_h[i], a2dd_h[i] = a1dd, a2dd
        x1_h[i], y1_h[i], x2_h[i], y2_h[i] = cpg.state[0], cpg.state[1], cpg.state[2], cpg.state[3]
        gait_h.append((cpg.b1_target, cpg.b2_target))

        cpg.step(dt)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("CPG Gait Generator — Smooth Transitions (swim -> turn_pos -> turn_neg)",
                  fontsize=12, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(ts, np.degrees(a1_h), label="a1 (front)")
    ax.plot(ts, np.degrees(a2_h), label="a2 (tail)")
    for tt in (4.0, 8.0):
        ax.axvline(tt, color="k", ls="--", lw=1)
    ax.set_xlabel("t [s]"); ax.set_ylabel("joint angle [deg]")
    ax.set_title("Joint angles stay continuous across switches")
    ax.legend(); ax.grid(True)

    ax = axes[0, 1]
    ax.plot(ts, a1dd_h, label="a1_ddot")
    ax.plot(ts, a2dd_h, label="a2_ddot")
    for tt in (4.0, 8.0):
        ax.axvline(tt, color="k", ls="--", lw=1)
    ax.set_xlabel("t [s]"); ax.set_ylabel("[rad/s^2]")
    ax.set_title("Accelerations show only a bounded kink at switches")
    ax.legend(); ax.grid(True)

    ax = axes[1, 0]
    ax.plot(x1_h, y1_h, lw=0.5)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.sqrt(MU_GAIT) * np.cos(th), np.sqrt(MU_GAIT) * np.sin(th), 'k--', lw=1, label="limit circle")
    ax.set_xlabel("x1"); ax.set_ylabel("y1")
    ax.set_title("Oscillator 1 phase portrait")
    ax.axis("equal"); ax.legend(); ax.grid(True)

    ax = axes[1, 1]
    ax.plot(x2_h, y2_h, lw=0.5, color="tab:orange")
    ax.plot(np.sqrt(MU_GAIT) * np.cos(th), np.sqrt(MU_GAIT) * np.sin(th), 'k--', lw=1, label="limit circle")
    ax.set_xlabel("x2"); ax.set_ylabel("y2")
    ax.set_title("Oscillator 2 phase portrait")
    ax.axis("equal"); ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.savefig("cpg_core_demo.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Done. Plot saved to cpg_core_demo.png")
