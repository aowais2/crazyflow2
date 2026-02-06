import numpy as np
import matplotlib.pyplot as plt

from crazyflow.control import Control
from crazyflow.sim import Sim


def virtual_leader(t, p0=np.array([0.0, 0.0, 1.0]), v_ref=np.array([0.2, 0.0, 0.0])):
    p_v = p0 + v_ref * t
    return p_v, v_ref


def cube_offsets(edge=1.0):
    hs = edge / 2.0
    corners = np.array(
        [
            [-hs, -hs, -hs],
            [-hs, -hs,  hs],
            [-hs,  hs, -hs],
            [-hs,  hs,  hs],
            [ hs, -hs, -hs],
            [ hs, -hs,  hs],
            [ hs,  hs, -hs],
            [ hs,  hs,  hs],
        ]
    )
    return corners


# -------- adjacency (approximate Fig. 3) --------
def build_adjacency():
    """
    Followers: 0..7, Leaders: 8..15
    Simple chain among leaders + each follower connected to two leaders.
    """
    N = 16
    A = np.zeros((N, N))

    # Leaders: chain + reverse chain
    for i in range(8, 15):
        A[i, i + 1] = 1.0
    for i in range(9, 16):
        A[i, i - 1] = 1.0

    # Followers: each connected to two leaders
    for k in range(0, 8):
        A[k, 8 + (k % 4)] = 1.0
        A[k, 8 + ((k + 1) % 4)] = 1.0

    return A


# -------- ξ, ζ as in the paper --------
def compute_xi_zeta(p, v, leader_idx, follower_idx, h_leaders, A):
    """
    p, v: (N,3)
    h_leaders: (8,3) formation offsets for leaders
    A: (N,N) adjacency
    Returns:
        xi, zeta: (N,3)
    """
    N = p.shape[0]
    h = np.zeros((N, 3))
    h[leader_idx] = h_leaders

    xi = np.zeros_like(p)
    zeta = np.zeros_like(p)

    # Leaders: xi_i = Σ a_ij (p_i - p_j - d_ij), d_ij = h_i - h_j
    for i in leader_idx:
        for j in range(N):
            if A[i, j] != 0.0:
                d_ij = h[i] - h[j]
                xi[i] += A[i, j] * (p[i] - p[j] - d_ij)
                zeta[i] += A[i, j] * (v[i] - v[j])

    # Followers: xi_k = Σ a_kj (p_k - p_j)
    for k in follower_idx:
        for j in range(N):
            if A[k, j] != 0.0:
                xi[k] += A[k, j] * (p[k] - p[j])
                zeta[k] += A[k, j] * (v[k] - v[j])

    return xi, zeta


# -------- global, minimal collision avoidance for all robots --------
def all_pair_repulsion(p, d_safe=0.2, k_rep=0.05):
    """
    p: (N,3) positions
    d_safe: safety distance
    k_rep: repulsion gain (small so deviation is minimal)
    Returns: (N,3) repulsive accelerations for all robots
    """
    N = p.shape[0]
    u_rep = np.zeros_like(p)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            diff = p[i] - p[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                continue
            if dist < d_safe:
                dir_ij = diff / dist
                mag = k_rep * (1.0 / dist - 1.0 / d_safe)
                u_rep[i] += mag * dir_ij
    return u_rep


def benchmark_control(p, v, p_v, v_v, leader_idx, follower_idx, h_leaders,
                      k_p=0.8, k_v=0.8, d_safe=0.2, k_rep=0.05):
    u = np.zeros_like(p)

    # Leaders: track p_v + h_i
    for i_local, i in enumerate(leader_idx):
        p_des = p_v + h_leaders[i_local]
        v_des = v_v
        e = p[i] - p_des
        de = v[i] - v_des
        u[i] = -k_p * e - k_v * de

    # Followers: converge to convex hull (average of leaders)
    p_leaders = p[leader_idx]
    v_leaders = v[leader_idx]
    p_ref = np.mean(p_leaders, axis=0)
    v_ref = np.mean(v_leaders, axis=0)

    for k in follower_idx:
        e = p[k] - p_ref
        de = v[k] - v_ref
        u[k] = -k_p * e - k_v * de

    # Small repulsion for all robots
    u += all_pair_repulsion(p, d_safe=d_safe, k_rep=k_rep)
    return u


def fixed_time_control(p, v, p_v, v_v, leader_idx, follower_idx, h_leaders,
                       a=1.0, b=0.8, c=0.5, h=5.0, q=0.5,
                       d_safe=0.2, k_rep=0.05):
    u = np.zeros_like(p)

    # Leaders: fixed-time formation tracking
    for i_local, i in enumerate(leader_idx):
        p_des = p_v + h_leaders[i_local]
        v_des = v_v
        e = p[i] - p_des
        de = v[i] - v_des
        s = e + (1.0 / h) * de
        u[i] = -a * s - b * np.power(np.abs(s), q) * np.sign(s) - c * de

    # Followers: fixed-time containment (to average of leaders)
    p_leaders = p[leader_idx]
    v_leaders = v[leader_idx]
    p_ref = np.mean(p_leaders, axis=0)
    v_ref = np.mean(v_leaders, axis=0)

    for k in follower_idx:
        e = p[k] - p_ref
        de = v[k] - v_ref
        s = e + (1.0 / h) * de
        u[k] = -a * s - b * np.power(np.abs(s), q) * np.sign(s) - c * de

    # Small repulsion for all robots
    u += all_pair_repulsion(p, d_safe=d_safe, k_rep=k_rep)
    return u


# -------- NEW: paper-style error using ξ --------
def compute_errors(p, v, leader_idx, follower_idx, h_leaders, A):
    """
    Returns per-agent ||ξ_i|| for leaders and followers (no averaging).
    """
    xi, zeta = compute_xi_zeta(p, v, leader_idx, follower_idx, h_leaders, A)

    xi_leaders = np.linalg.norm(xi[leader_idx], axis=1)     # shape (8,)
    xi_followers = np.linalg.norm(xi[follower_idx], axis=1) # shape (8,)

    return xi_leaders, xi_followers


def run_sim(use_fixed_time=True, T=30.0, v_ref=np.array([0.2, 0.0, 0.0])):
    N = 16
    n_leaders = 8
    n_followers = 8
    follower_idx = np.arange(0, n_followers)
    leader_idx = np.arange(n_followers, n_followers + n_leaders)

    p = np.zeros((N, 3))
    v = np.zeros((N, 3))

    rng = np.random.default_rng(0)
    p[leader_idx] = rng.uniform(low=[-1.0, -1.0, 0.5], high=[1.0, 1.0, 1.5], size=(n_leaders, 3))
    p[follower_idx] = rng.uniform(low=[-2.0, -2.0, 0.0], high=[2.0, 2.0, 0.5], size=(n_followers, 3))

    h_leaders = cube_offsets(edge=1.0)
    A = build_adjacency()  # NEW

    control_freq = 50.0
    dt = 1.0 / control_freq
    steps = int(T * control_freq)

    leader_err_hist = []
    follower_err_hist = []
    t_hist = []

    sim = Sim(n_drones=N, control=Control.state)
    sim.reset()
    fps = 30

    for k in range(steps):
        t = k * dt
        p_v, v_v = virtual_leader(t, p0=np.array([0.0, 0.0, 1.0]), v_ref=v_ref)

        if use_fixed_time:
            u = fixed_time_control(p, v, p_v, v_v, leader_idx, follower_idx, h_leaders)
        else:
            u = benchmark_control(p, v, p_v, v_v, leader_idx, follower_idx, h_leaders)

        v = v + u * dt
        p = p + v * dt

        # NEW: paper-style ξ-based errors
        eL, eF = compute_errors(p, v, leader_idx, follower_idx, h_leaders, A)
        leader_err_hist.append(eL)
        follower_err_hist.append(eF)
        t_hist.append(t)

        states = sim.data.states.replace(
            pos=sim.data.states.pos.at[0].set(p),
            vel=sim.data.states.vel.at[0].set(v),
        )
        sim.data = sim.data.replace(states=states)

        sim.step(1)
        if (k % int(control_freq // fps)) == 0:
            sim.render()

    sim.close()
    return np.array(t_hist), np.array(leader_err_hist), np.array(follower_err_hist)


def main():
    t_ft, eL_ft, eF_ft = run_sim(use_fixed_time=True, T=20.0)
    t_bm, eL_bm, eF_bm = run_sim(use_fixed_time=False, T=30.0)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t_ft, eL_ft, label="Leaders (||ξ_i||)")
    plt.plot(t_ft, eF_ft, label="Followers (||ξ_k||)")
    plt.axvline(20.0, color="k", linestyle="--", label="T = 20 s")
    plt.title("Fixed-time controller (Theorem 2 approx) + mild repulsion")
    plt.xlabel("Time [s]")
    plt.ylabel("Mean error norm")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(t_bm, eL_bm, label="Leaders (||ξ_i||)")
    plt.plot(t_bm, eF_bm, label="Followers (||ξ_k||)")
    plt.axvline(20.0, color="k", linestyle="--", label="T = 20 s")
    plt.title("Benchmark controller (Theorem 1 approx) + mild repulsion")
    plt.xlabel("Time [s]")
    plt.ylabel("Mean error norm")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()





"""
import numpy as np
import matplotlib.pyplot as plt

from crazyflow.control import Control
from crazyflow.sim import Sim


def virtual_leader(t, p0=np.array([0.0, 0.0, 1.0]), v_ref=np.array([0.2, 0.0, 0.0])):
    p_v = p0 + v_ref * t
    return p_v, v_ref


def cube_offsets(edge=1.0):
    hs = edge / 2.0
    corners = np.array(
        [
            [-hs, -hs, -hs],
            [-hs, -hs,  hs],
            [-hs,  hs, -hs],
            [-hs,  hs,  hs],
            [ hs, -hs, -hs],
            [ hs, -hs,  hs],
            [ hs,  hs, -hs],
            [ hs,  hs,  hs],
        ]
    )
    return corners


# -------- global, minimal collision avoidance for all robots --------
def all_pair_repulsion(p, d_safe=0.2, k_rep=0.05):
"""
"""
    p: (N,3) positions
    d_safe: safety distance
    k_rep: repulsion gain (small so deviation is minimal)
    Returns: (N,3) repulsive accelerations for all robots
"""
"""
    N = p.shape[0]
    u_rep = np.zeros_like(p)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            diff = p[i] - p[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                continue
            if dist < d_safe:
                dir_ij = diff / dist
                mag = k_rep * (1.0 / dist - 1.0 / d_safe)
                u_rep[i] += mag * dir_ij
    return u_rep


def benchmark_control(p, v, p_v, v_v, leader_idx, follower_idx, h_leaders,
                      k_p=0.8, k_v=0.8, d_safe=0.2, k_rep=0.05):
    u = np.zeros_like(p)

    # Leaders: track p_v + h_i
    for i_local, i in enumerate(leader_idx):
        p_des = p_v + h_leaders[i_local]
        v_des = v_v
        e = p[i] - p_des
        de = v[i] - v_des
        u[i] = -k_p * e - k_v * de

    # Followers: converge to convex hull (average of leaders)
    p_leaders = p[leader_idx]
    v_leaders = v[leader_idx]
    p_ref = np.mean(p_leaders, axis=0)
    v_ref = np.mean(v_leaders, axis=0)

    for k in follower_idx:
        e = p[k] - p_ref
        de = v[k] - v_ref
        u[k] = -k_p * e - k_v * de

    # Small repulsion for all robots
    u += all_pair_repulsion(p, d_safe=d_safe, k_rep=k_rep)
    return u


def fixed_time_control(p, v, p_v, v_v, leader_idx, follower_idx, h_leaders,
                       a=1.0, b=0.8, c=0.5, h=5.0, q=0.5,
                       d_safe=0.2, k_rep=0.05):
    u = np.zeros_like(p)

    # Leaders: fixed-time formation tracking
    for i_local, i in enumerate(leader_idx):
        p_des = p_v + h_leaders[i_local]
        v_des = v_v
        e = p[i] - p_des
        de = v[i] - v_des
        s = e + (1.0 / h) * de
        u[i] = -a * s - b * np.power(np.abs(s), q) * np.sign(s) - c * de

    # Followers: fixed-time containment (to average of leaders)
    p_leaders = p[leader_idx]
    v_leaders = v[leader_idx]
    p_ref = np.mean(p_leaders, axis=0)
    v_ref = np.mean(v_leaders, axis=0)

    for k in follower_idx:
        e = p[k] - p_ref
        de = v[k] - v_ref
        s = e + (1.0 / h) * de
        u[k] = -a * s - b * np.power(np.abs(s), q) * np.sign(s) - c * de

    # Small repulsion for all robots
    u += all_pair_repulsion(p, d_safe=d_safe, k_rep=k_rep)
    return u


def compute_errors(p, v, p_v, v_v, leader_idx, follower_idx, h_leaders):
    eL = []
    for i_local, i in enumerate(leader_idx):
        p_des = p_v + h_leaders[i_local]
        eL.append(p[i] - p_des)
    eL = np.vstack(eL)
    leader_err = np.linalg.norm(eL, axis=1).mean()

    p_leaders = p[leader_idx]
    p_ref = np.mean(p_leaders, axis=0)
    eF = []
    for k in follower_idx:
        eF.append(p[k] - p_ref)
    eF = np.vstack(eF)
    follower_err = np.linalg.norm(eF, axis=1).mean()

    return leader_err, follower_err


def run_sim(use_fixed_time=True, T=30.0, v_ref=np.array([0.2, 0.0, 0.0])):
    N = 16
    n_leaders = 8
    n_followers = 8
    follower_idx = np.arange(0, n_followers)
    leader_idx = np.arange(n_followers, n_followers + n_leaders)

    p = np.zeros((N, 3))
    v = np.zeros((N, 3))

    rng = np.random.default_rng(0)
    p[leader_idx] = rng.uniform(low=[-1.0, -1.0, 0.5], high=[1.0, 1.0, 1.5], size=(n_leaders, 3))
    p[follower_idx] = rng.uniform(low=[-2.0, -2.0, 0.0], high=[2.0, 2.0, 0.5], size=(n_followers, 3))

    h_leaders = cube_offsets(edge=1.0)

    control_freq = 50.0
    dt = 1.0 / control_freq
    steps = int(T * control_freq)

    leader_err_hist = []
    follower_err_hist = []
    t_hist = []

    sim = Sim(n_drones=N, control=Control.state)
    sim.reset()
    fps = 30

    for k in range(steps):
        t = k * dt
        p_v, v_v = virtual_leader(t, p0=np.array([0.0, 0.0, 1.0]), v_ref=v_ref)

        if use_fixed_time:
            u = fixed_time_control(p, v, p_v, v_v, leader_idx, follower_idx, h_leaders)
        else:
            u = benchmark_control(p, v, p_v, v_v, leader_idx, follower_idx, h_leaders)

        v = v + u * dt
        p = p + v * dt

        eL, eF = compute_errors(p, v, p_v, v_v, leader_idx, follower_idx, h_leaders)
        leader_err_hist.append(eL)
        follower_err_hist.append(eF)
        t_hist.append(t)

        states = sim.data.states.replace(
            pos=sim.data.states.pos.at[0].set(p),
            vel=sim.data.states.vel.at[0].set(v),
        )
        sim.data = sim.data.replace(states=states)

        sim.step(1)
        if (k % int(control_freq // fps)) == 0:
            sim.render()

    sim.close()
    return np.array(t_hist), np.array(leader_err_hist), np.array(follower_err_hist)


def main():
    t_ft, eL_ft, eF_ft = run_sim(use_fixed_time=True, T=20.0)
    t_bm, eL_bm, eF_bm = run_sim(use_fixed_time=False, T=30.0)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t_ft, eL_ft, label="Leaders (formation error)")
    plt.plot(t_ft, eF_ft, label="Followers (containment error)")
    plt.axvline(20.0, color="k", linestyle="--", label="T = 20 s")
    plt.title("Fixed-time controller (Theorem 2) + mild repulsion")
    plt.xlabel("Time [s]")
    plt.ylabel("Mean error norm")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(t_bm, eL_bm, label="Leaders (formation error)")
    plt.plot(t_bm, eF_bm, label="Followers (containment error)")
    plt.axvline(20.0, color="k", linestyle="--", label="T = 20 s")
    plt.title("Benchmark controller (Theorem 1) + mild repulsion")
    plt.xlabel("Time [s]")
    plt.ylabel("Mean error norm")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
"""
