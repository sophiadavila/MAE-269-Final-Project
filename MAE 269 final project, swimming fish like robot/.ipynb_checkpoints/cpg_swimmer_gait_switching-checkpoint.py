# %% [markdown]
# # CPG-Based Gait-Switching Controller for the 3-Link Swimmer
#
# This builds on `cpg_core.py` (a 2-oscillator diffusive Hopf CPG, inspired by
# JiChern/CPG: https://github.com/JiChern/CPG) and plugs it into the 3-link
# swimmer body dynamics from `path_tracking_gait_switching.py`.
#
# **Old controller** (path_tracking_gait_switching.py): the cascade controller
# picks a gait *name* once per cycle and the joint trajectories a1(t), a2(t)
# are recomputed from a fresh closed-form sin() with a different bias/phase --
# a1(t) JUMPS at the cycle boundary when the gait changes.
#
# **New controller** (this file): the cascade controller instead sets CPG
# *targets* (bias_1, bias_2) once per cycle. The CPG's internal state (which
# includes first-order bias filters) evolves continuously, so a1(t), a2(t),
# a1_dot(t), a2_dot(t) never jump -- only the accelerations show a bounded
# kink. Run this notebook to compare both controllers side by side.
#
# Run `cpg_core.py` first (or make sure it's in the same folder) -- this file
# imports the CPG class from it.

# %%
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from cpg_core import CPGGaitGenerator, GAITS, W_GAIT, T_GAIT, A_GAIT, B_TURN

PHI_G = np.pi / 2  # tail-front phase offset (matches CPG swim phase lag)

# %% [markdown]
# ## 1. Body dynamics (3-link swimmer Lagrangian)
#
# Identical derivation to `path_tracking_gait_switching.py` -- repeated here
# so this file is self-contained.

# %%
t = sp.symbols("t", real=True)
l, m, I_link, c = sp.symbols("l m I c", positive=True, real=True)
U_flow, U_flow_y = sp.symbols("U_flow U_flow_y", real=True)

x = sp.Function("x")(t)
y = sp.Function("y")(t)
theta = sp.Function("theta")(t)
a1 = sp.Function("a1")(t)
a2 = sp.Function("a2")(t)

q = [x, y, theta, a1, a2]
q_dt = [sp.diff(v, t) for v in q]
n = 5


def R_z(ang):
    return sp.Matrix([[sp.cos(ang), -sp.sin(ang), 0],
                       [sp.sin(ang),  sp.cos(ang), 0],
                       [0,            0,           1]])


def make_T(R, p):
    return sp.Matrix([
        [R[0, 0], R[0, 1], R[0, 2], p[0, 0]],
        [R[1, 0], R[1, 1], R[1, 2], p[1, 0]],
        [R[2, 0], R[2, 1], R[2, 2], p[2, 0]],
        [0,       0,       0,       1],
    ])


T_base = make_T(R_z(theta), sp.Matrix([[x], [y], [0]]))
T_front = (T_base
           @ make_T(sp.eye(3), sp.Matrix([[-l / 2], [0], [0]]))
           @ make_T(R_z(a1), sp.Matrix([[0], [0], [0]]))
           @ make_T(sp.eye(3), sp.Matrix([[-l / 2], [0], [0]])))
T_tail = (T_base
          @ make_T(sp.eye(3), sp.Matrix([[l / 2], [0], [0]]))
          @ make_T(R_z(a2), sp.Matrix([[0], [0], [0]]))
          @ make_T(sp.eye(3), sp.Matrix([[l / 2], [0], [0]])))

r_base = sp.Matrix([T_base[0, 3], T_base[1, 3]])
r_front = sp.simplify(sp.Matrix([T_front[0, 3], T_front[1, 3]]))
r_tail = sp.simplify(sp.Matrix([T_tail[0, 3], T_tail[1, 3]]))

J_base = r_base.jacobian(q)
J_front = sp.simplify(r_front.jacobian(q))
J_tail = sp.simplify(r_tail.jacobian(q))

qd = sp.Matrix(q_dt)


def link_KE(J_v, omega_sym):
    v = J_v @ qd
    return sp.Rational(1, 2) * m * (v.T @ v)[0, 0] + sp.Rational(1, 2) * I_link * omega_sym**2


T_kin = sp.expand(
    link_KE(J_base, q_dt[2])
    + link_KE(J_front, q_dt[2] + q_dt[3])
    + link_KE(J_tail, q_dt[2] + q_dt[4])
)

M_mat = sp.simplify(sp.Matrix([
    [sp.diff(sp.diff(T_kin, q_dt[i]), q_dt[j]) for j in range(n)]
    for i in range(n)
]))

q_ddt_syms = sp.symbols("xdd ydd thdd a1dd a2dd")
h_vec = sp.Matrix.zeros(n, 1)
for i in range(n):
    dT_i = sp.diff(T_kin, q_dt[i])
    ddt_i = sum(
        sp.diff(dT_i, q[j]) * q_dt[j] + sp.diff(dT_i, q_dt[j]) * q_ddt_syms[j]
        for j in range(n)
    )
    h_vec[i] = sp.simplify(ddt_i.subs(dict(zip(q_ddt_syms, [0] * n))) - sp.diff(T_kin, q[i]))


def flow_field(px, py):
    return sp.Matrix([U_flow, U_flow_y])


def generalized_drag_force(J_v, r_i):
    vf_i = flow_field(r_i[0], r_i[1])
    vrel = J_v @ qd - vf_i
    return J_v.T @ (-c * vrel)


Q_flow = sp.simplify(
    generalized_drag_force(J_base, r_base)
    + generalized_drag_force(J_front, r_front)
    + generalized_drag_force(J_tail, r_tail)
)

params_sym = [l, m, I_link, c, U_flow, U_flow_y]
M_fn = sp.lambdify([q, params_sym], M_mat, "numpy")
h_fn = sp.lambdify([q, q_dt, params_sym], h_vec, "numpy")
Qf_fn = sp.lambdify([q, q_dt, params_sym], Q_flow, "numpy")

print("Body dynamics symbolic derivation complete.")

# %% [markdown]
# ## 2. Physical parameters & disturbance helper

# %%
l_val = 1.0
m_val = 0.1
I_val = m_val * l_val**2 / 12.0
c_val = 1.0


def make_params(Ux=0.0, Uy=0.0):
    return [l_val, m_val, I_val, c_val, Ux, Uy]


def get_disturbance(tv, disturbances):
    for t0, t1, Ux, Uy in disturbances:
        if t0 <= tv < t1:
            return Ux, Uy
    return 0.0, 0.0


# %% [markdown]
# ## 3. NEW controller: CPG-driven dynamics (12-state system)
#
# State = [x, y, theta, xd, yd, thd,  x1, y1, x2, y2, b1, b2]
#                body (6)                   CPG (6)

# %%
def dynamics_cpg(tv, state, cpg, Ux, Uy):
    body = state[0:6]
    cpg_state = state[6:12]
    xv, yv, thv, xd, yd, thd = body

    a1v, a2v, a1d, a2d, a1dd, a2dd = cpg.joint_kinematics(cpg_state)

    p_vals = make_params(Ux=Ux, Uy=Uy)
    qc = [xv, yv, thv, a1v, a2v]
    qdc = [xd, yd, thd, a1d, a2d]
    M_ = np.array(M_fn(qc, p_vals), dtype=float)
    h_ = np.array(h_fn(qc, qdc, p_vals), dtype=float).flatten()
    Qf_ = np.array(Qf_fn(qc, qdc, p_vals), dtype=float).flatten()

    rhs = Qf_[0:3] - h_[0:3] - M_[0:3, 3:5] @ np.array([a1dd, a2dd])
    body_acc = np.linalg.solve(M_[0:3, 0:3], rhs)

    cpg_dot = cpg.derivative(cpg_state)
    return [xd, yd, thd, *body_acc, *cpg_dot]


# %% [markdown]
# ## 4. OLD controller: gait-library dynamics (for comparison)
#
# Same gait library / dynamics as `path_tracking_gait_switching.py`.

# %%
GAIT_PARAMS = {
    'swim':     (0.0, 0.0),
    'turn_pos': (+B_TURN, +B_TURN),
    'turn_neg': (-B_TURN, -B_TURN),
}


def gait_kinematics_old(tv, gait_name):
    d1, d2 = GAIT_PARAMS[gait_name]
    a1v = d1 + A_GAIT * np.sin(W_GAIT * tv)
    a2v = d2 + A_GAIT * np.sin(W_GAIT * tv + PHI_G)
    a1d = A_GAIT * W_GAIT * np.cos(W_GAIT * tv)
    a2d = A_GAIT * W_GAIT * np.cos(W_GAIT * tv + PHI_G)
    a1dd = -A_GAIT * W_GAIT**2 * np.sin(W_GAIT * tv)
    a2dd = -A_GAIT * W_GAIT**2 * np.sin(W_GAIT * tv + PHI_G)
    return a1v, a2v, a1d, a2d, a1dd, a2dd


def dynamics_old(tv, state, gait_name, Ux, Uy):
    xv, yv, thv, xd, yd, thd = state
    p_vals = make_params(Ux=Ux, Uy=Uy)
    a1v, a2v, a1d, a2d, a1dd, a2dd = gait_kinematics_old(tv, gait_name)
    qc = [xv, yv, thv, a1v, a2v]
    qdc = [xd, yd, thd, a1d, a2d]
    M_ = np.array(M_fn(qc, p_vals), dtype=float)
    h_ = np.array(h_fn(qc, qdc, p_vals), dtype=float).flatten()
    Qf_ = np.array(Qf_fn(qc, qdc, p_vals), dtype=float).flatten()
    rhs = Qf_[0:3] - h_[0:3] - M_[0:3, 3:5] @ np.array([a1dd, a2dd])
    q_ddot_pas = np.linalg.solve(M_[0:3, 0:3], rhs)
    return [xd, yd, thd, *q_ddot_pas]


# %% [markdown]
# ## 5. Shared cascade controller (heading -> gait label)
#
# Same control law as the original file. The only thing that differs between
# old/new is how the resulting label is *realized* in the joint trajectories.

# %%
K_POS = 0.20
PSI_MAX = np.radians(15)
TURN_THRESH = np.radians(5)


def desired_heading(y_end):
    return float(np.clip(K_POS * y_end, -PSI_MAX, PSI_MAX))


def cascade_controller(y_end, th_avg):
    psi_d = desired_heading(y_end)
    e_psi = (th_avg - psi_d + np.pi) % (2.0 * np.pi) - np.pi
    if e_psi > +TURN_THRESH:
        return 'turn_cw', psi_d, e_psi
    elif e_psi < -TURN_THRESH:
        return 'turn_ccw', psi_d, e_psi
    else:
        return 'swim', psi_d, e_psi


# %% [markdown]
# ## 6. Gait-turning diagnostic -> assign CW/CCW bias sign
#
# Runs 3 cycles of `turn_pos`/`turn_neg` (old dynamics, from rest) and measures
# net Delta-theta. This is a pure body-dynamics property, so the same
# CW/CCW assignment applies whether the bias is realized via the old gait
# library or the new CPG.

# %%
print("Gait turning diagnostic (3 cycles from rest, no flow)")
_dtheta = {}
for _gname in ['swim', 'turn_pos', 'turn_neg']:
    _sv = solve_ivp(
        lambda tv, s, g=_gname: dynamics_old(tv, s, g, 0.0, 0.0),
        [0.0, 3 * T_GAIT], [0.0] * 6,
        t_eval=np.linspace(0, 3 * T_GAIT, 300), rtol=1e-8, atol=1e-8
    )
    _dtheta[_gname] = _sv.y[2, -1]
    print(f"  {_gname:10s}: Delta theta = {np.degrees(_dtheta[_gname]):+.2f} deg (per 3 cycles)")

if _dtheta['turn_pos'] < _dtheta['turn_neg']:
    GAIT_CW, GAIT_CCW = 'turn_pos', 'turn_neg'
    BIAS_CW, BIAS_CCW = (+B_TURN, +B_TURN), (-B_TURN, -B_TURN)
else:
    GAIT_CW, GAIT_CCW = 'turn_neg', 'turn_pos'
    BIAS_CW, BIAS_CCW = (-B_TURN, -B_TURN), (+B_TURN, +B_TURN)

print(f"\n  -> GAIT_CW  = '{GAIT_CW}'   bias = {BIAS_CW}")
print(f"  -> GAIT_CCW = '{GAIT_CCW}'   bias = {BIAS_CCW}")


def resolve_gait_old(label):
    if label == 'turn_cw':  return GAIT_CW
    if label == 'turn_ccw': return GAIT_CCW
    return 'swim'


def resolve_bias_new(label):
    if label == 'turn_cw':  return BIAS_CW
    if label == 'turn_ccw': return BIAS_CCW
    return (0.0, 0.0)


# %% [markdown]
# ## 7. Cycle-based simulators
#
# Both simulators run one gait cycle at a time, read end-of-cycle (y, theta),
# run the shared cascade controller, and update the gait for the next cycle.
# - OLD: switches `gait_now` (a discrete label -> new closed-form a1(t),a2(t)).
# - NEW: calls `cpg.set_targets(b1_target=, b2_target=)` -- the CPG state
#   (already mid-trajectory) relaxes toward the new bias smoothly.

# %%
def simulate_old(t_end, s0, disturbances, pts_per_cycle=200):
    state = np.array(s0, dtype=float)
    t_now, gait_now = 0.0, 'swim'
    t_segs, y_segs, a1_segs, log = [], [], [], []

    while t_now < t_end - 1e-10:
        t_next = min(t_now + T_GAIT, t_end)
        t_seg = np.linspace(t_now, t_next, pts_per_cycle)
        g_hold = gait_now

        sol = solve_ivp(
            lambda tv, s: dynamics_old(tv, s, g_hold, *get_disturbance(tv, disturbances)),
            [t_now, t_next], state, t_eval=t_seg, rtol=1e-8, atol=1e-8, method='RK45'
        )

        t_segs.append(sol.t[:-1])
        y_segs.append(sol.y[:, :-1])
        a1_segs.append(np.array([gait_kinematics_old(tv, g_hold)[0] for tv in sol.t[:-1]]))

        state = sol.y[:, -1]
        t_now = t_next

        y_end = state[1]
        th_avg = float(np.angle(np.mean(np.exp(1j * sol.y[2]))))
        label, psi_d, e_psi = cascade_controller(y_end, th_avg)
        gait_now = resolve_gait_old(label)

        log.append({'t': t_now, 'y_end': y_end, 'th_avg': th_avg,
                     'psi_d': psi_d, 'e_psi': e_psi, 'label': label, 'gait': gait_now})

    t_segs.append(np.array([t_now]))
    y_segs.append(state.reshape(6, 1))
    a1_segs.append(np.array([gait_kinematics_old(t_now, gait_now)[0]]))

    return (np.concatenate(t_segs), np.concatenate(y_segs, axis=1),
            np.concatenate(a1_segs), log)


def simulate_cpg(t_end, s0, disturbances, pts_per_cycle=200):
    cpg = CPGGaitGenerator()
    state = np.concatenate([np.asarray(s0, dtype=float), cpg.state])
    t_now = 0.0
    t_segs, y_segs, log = [], [], []

    while t_now < t_end - 1e-10:
        t_next = min(t_now + T_GAIT, t_end)
        t_seg = np.linspace(t_now, t_next, pts_per_cycle)

        sol = solve_ivp(
            lambda tv, s: dynamics_cpg(tv, s, cpg, *get_disturbance(tv, disturbances)),
            [t_now, t_next], state, t_eval=t_seg, rtol=1e-8, atol=1e-8, method='RK45'
        )

        t_segs.append(sol.t[:-1])
        y_segs.append(sol.y[:, :-1])

        state = sol.y[:, -1]
        cpg.state = state[6:12]
        t_now = t_next

        y_end = state[1]
        th_avg = float(np.angle(np.mean(np.exp(1j * sol.y[2]))))
        label, psi_d, e_psi = cascade_controller(y_end, th_avg)
        b1_t, b2_t = resolve_bias_new(label)
        cpg.set_targets(b1_target=b1_t, b2_target=b2_t)

        log.append({'t': t_now, 'y_end': y_end, 'th_avg': th_avg,
                     'psi_d': psi_d, 'e_psi': e_psi, 'label': label,
                     'b1_target': b1_t, 'b2_target': b2_t})

    t_segs.append(np.array([t_now]))
    y_segs.append(state.reshape(12, 1))

    t_arr = np.concatenate(t_segs)
    y_arr = np.concatenate(y_segs, axis=1)
    a1_arr = y_arr[10, :] + y_arr[6, :]   # a1 = b1 + x1
    return t_arr, y_arr, a1_arr, log


# %% [markdown]
# ## 8. Run scenarios
#
# Same scenarios as the original file: A) no disturbance, B) one cross-flow
# pulse, C) two opposing pulses. Run with both controllers.

# %%
T_END = 40.0
s0 = [0.0] * 6
dist_B = [(3.0, 8.0, 0.0, 0.4)]
dist_C = [(3.0, 8.0, 0.0, 0.4), (22.0, 27.0, 0.0, -0.4)]

print("\n--- OLD (discrete gait-library) controller ---")
tA_o, yA_o, a1A_o, logA_o = simulate_old(T_END, s0, [])
tB_o, yB_o, a1B_o, logB_o = simulate_old(T_END, s0, dist_B)
tC_o, yC_o, a1C_o, logC_o = simulate_old(T_END, s0, dist_C)
print(f"Final y: A={yA_o[1,-1]:.4f}  B={yB_o[1,-1]:.4f}  C={yC_o[1,-1]:.4f}")

print("\n--- NEW (CPG smooth) controller ---")
tA_n, yA_n, a1A_n, logA_n = simulate_cpg(T_END, s0, [])
tB_n, yB_n, a1B_n, logB_n = simulate_cpg(T_END, s0, dist_B)
tC_n, yC_n, a1C_n, logC_n = simulate_cpg(T_END, s0, dist_C)
print(f"Final y: A={yA_n[1,-1]:.4f}  B={yB_n[1,-1]:.4f}  C={yC_n[1,-1]:.4f}")

# %% [markdown]
# ## 9. Compare: trajectories + gait switching + joint-angle continuity

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("OLD (discrete gait library) vs NEW (CPG smooth) gait-switching controller\n"
              f"Scenario B: Uy=0.4 m/s, t in [3,8] s   |   GAIT_CW={GAIT_CW}, GAIT_CCW={GAIT_CCW}",
              fontsize=12, fontweight="bold")

ax = axes[0, 0]
ax.axhline(0, color='k', ls='--', lw=1.5, label='Desired y=0')
ax.plot(tB_o, yB_o[1], 'r-', lw=1.3, label='OLD: y(t)')
ax.plot(tB_n, yB_n[1], 'b-', lw=1.3, label='NEW (CPG): y(t)')
for t0, t1, _, _ in dist_B:
    ax.axvspan(t0, t1, alpha=0.12, color='orange', label='Disturbance')
ax.set_xlabel('t [s]'); ax.set_ylabel('y [m]')
ax.set_title('Lateral position vs time')
ax.legend(fontsize=8); ax.grid(True)

ax = axes[0, 1]
ax.plot(tB_o, np.degrees(a1B_o), 'r-', lw=1.0, label='OLD: a1(t)')
ax.plot(tB_n, np.degrees(a1B_n), 'b-', lw=1.0, label='NEW (CPG): a1(t)')
for t0, t1, _, _ in dist_B:
    ax.axvspan(t0, t1, alpha=0.12, color='orange')
ax.set_xlabel('t [s]'); ax.set_ylabel('a1 [deg]')
ax.set_title('Front joint angle a1(t) -- whole run')
ax.legend(fontsize=8); ax.grid(True)

# zoom near the first switch out of 'swim'
switch_t = next((d['t'] for d in logB_n if d['label'] != 'swim'), T_GAIT)
zoom_lo, zoom_hi = max(0, switch_t - 2 * T_GAIT), switch_t + 2 * T_GAIT

ax = axes[1, 0]
mo = (tB_o >= zoom_lo) & (tB_o <= zoom_hi)
mn = (tB_n >= zoom_lo) & (tB_n <= zoom_hi)
ax.plot(tB_o[mo], np.degrees(a1B_o[mo]), 'r.-', lw=1.2, ms=3, label='OLD: a1(t)')
ax.plot(tB_n[mn], np.degrees(a1B_n[mn]), 'b.-', lw=1.2, ms=3, label='NEW (CPG): a1(t)')
ax.axvline(switch_t, color='k', ls='--', lw=1, label='gait switch instant')
ax.set_xlabel('t [s]'); ax.set_ylabel('a1 [deg]')
ax.set_title(f'Zoom near first gait switch (t~{switch_t:.1f}s)\nOLD jumps, NEW stays continuous')
ax.legend(fontsize=8); ax.grid(True)

ax = axes[1, 1]
labels_o = [d['label'] for d in logB_o]
labels_n = [d['label'] for d in logB_n]
times_o = [d['t'] - T_GAIT for d in logB_o]
times_n = [d['t'] - T_GAIT for d in logB_n]
code = {'swim': 0, 'turn_cw': -1, 'turn_ccw': 1}
ax.step(times_o, [code[l] for l in labels_o], 'r-', where='post', lw=1.5, label='OLD label')
ax.step(times_n, [code[l] for l in labels_n], 'b--', where='post', lw=1.5, label='NEW label')
for t0, t1, _, _ in dist_B:
    ax.axvspan(t0, t1, alpha=0.12, color='orange')
ax.set_yticks([-1, 0, 1]); ax.set_yticklabels(['turn_cw', 'swim', 'turn_ccw'])
ax.set_xlabel('t [s]'); ax.set_title('Per-cycle gait label (controller output)')
ax.legend(fontsize=8); ax.grid(True)

plt.tight_layout()
plt.savefig("cpg_vs_discrete_gait_switching.png", dpi=120, bbox_inches="tight")
plt.show()
print("\nDone. Plot saved to cpg_vs_discrete_gait_switching.png")
