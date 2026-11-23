from __future__ import annotations

import time
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp


# ============================================================
# 0. PHYSICAL SCALES FOR VOLTAGE OUTPUT
# ============================================================

PHI0 = 2.067833848e-15          # Wb
PHI0_REDUCED = PHI0 / (2.0 * np.pi)

# These physical values are used only to convert normalized voltage
# V_norm = dphi/dtau into SI voltage:
#     V = (Phi0 / 2pi) * omega_pL * V_norm.
# If you do not need mV, you can ignore them.
IC_L_PHYS = 100e-6              # A, critical current of one LJJ discrete element
C_L_PHYS = 60e-15               # F, capacitance of one LJJ discrete element
OMEGA_PL = np.sqrt(IC_L_PHYS / (PHI0_REDUCED * C_L_PHYS))
V_SCALE = PHI0_REDUCED * OMEGA_PL


# ============================================================
# 1. USER PARAMETERS
# ============================================================

P = {
    # -------------------------
    # LJJ parameters
    # -------------------------
    "M": 143,                  # number of LJJ nodes
    "alpha_L": 0.2,           # LJJ damping
    "gamma_L": 0.6,           # LJJ bias current normalized to IcL
    "l_L": 0.10,               # discrete LJJ inductive parameter; dx = sqrt(l_L)

    # -------------------------
    # Initial fluxon
    # -------------------------
    "n_fluxons": 1,
    "fluxon_cells": [20],      # cell-like positions, 0-based
    "fluxon_velocity": 0.90,   # 0 <= v < 1
    "fluxon_direction": "right",  # "right" or "left"

    # -------------------------
    # JTL parameters
    # -------------------------
    "N_JTL": 5,               # number of JTL junctions
    "m_J": 1.0,                # C_J / C_L
    "s_J": 1.0,                # IcJ / IcL
    "alpha_J": 1,           # JTL damping in LJJ normalization
    "gamma_J": 0.7,           # JTL bias normalized to IcL
    "l_J": 9.00,               # JTL inductive parameter
           # extra damping on last JTL junction

    # -------------------------
    # Capacitive link: Cc / CL
    # -------------------------
    "kappa": 2,

    # -------------------------
    # Integration
    # -------------------------
    "tau_end": 400.0,
    "n_eval": 2500,
    "rtol": 1e-6,
    "atol": 1e-8,
    "max_step": 0.05,
    "method": "DOP853",

    # -------------------------
    # Output
    # -------------------------
    "results_dir": "ljj_capacitive_jtl_results",
    "save_animation": True,
    "show_plots": True,
    "animation_frames": 400,
    "animation_interval_ms": 25,
}



# ============================================================
# 1.1. OUTPUT FOLDER NAMED BY SYSTEM PARAMETERS
# ============================================================

def _safe_tag(value) -> str:
    """
    Convert a parameter value to a filesystem-safe short tag.
    Examples:
        0.05 -> 0p05
        [20, 35] -> 20-35
        "right" -> right
    """
    if isinstance(value, (list, tuple)):
        return "-".join(_safe_tag(v) for v in value)

    if isinstance(value, bool):
        return "1" if value else "0"

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        s = f"{float(value):.6g}"
    else:
        s = str(value)

    replacements = {
        ".": "p",
        "-": "m",
        "+": "p",
        " ": "",
        "/": "_",
        "\\": "_",
        ":": "_",
        ",": "-",
        "[": "",
        "]": "",
        "(": "",
        ")": "",
        "'": "",
        '"': "",
    }

    for old, new in replacements.items():
        s = s.replace(old, new)

    return s


def make_run_folder_name(p: dict) -> str:
    """
    Folder name contains the main physical/numerical parameters.
    Timestamp is appended so that repeated runs do not overwrite each other.
    Full parameters are also saved into parameters.json inside the folder.
    """
    keys_for_name = [
        "M",
        "alpha_L",
        "gamma_L",
        "l_L",
        "n_fluxons",
        "fluxon_cells",
        "fluxon_velocity",
        "fluxon_direction",
        "N_JTL",
        "m_J",
        "s_J",
        "alpha_J",
        "gamma_J",
        "l_J",
        "kappa",
        "tau_end",
    ]

    name_map = {
        "M": "M",
        "alpha_L": "aL",
        "gamma_L": "gL",
        "l_L": "lL",
        "n_fluxons": "Nfl",
        "fluxon_cells": "cells",
        "fluxon_velocity": "vfl",
        "fluxon_direction": "dir",
        "N_JTL": "NJTL",
        "m_J": "mJ",
        "s_J": "sJ",
        "alpha_J": "aJ",
        "gamma_J": "gJ",
        "l_J": "lJ",
        "kappa": "kap",
        "tau_end": "T",
    }

    parts = []
    for key in keys_for_name:
        if key in p:
            parts.append(f"{name_map.get(key, key)}{_safe_tag(p[key])}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return "__".join(parts) + "__" + timestamp


def make_output_dir(p: dict) -> Path:
    """
    Create a unique run directory:
        <script_dir>/<results_dir>/<folder_named_by_parameters_timestamp>/
    """
    script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    root_dir = script_dir / str(p["results_dir"])
    out_dir = root_dir / make_run_folder_name(p)
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def save_parameters(out_dir: Path, p: dict) -> None:
    """
    Save full parameter dictionary and physical scales into the run folder.
    """
    payload = {
        "parameters": dict(p),
        "physical_scales": {
            "PHI0_Wb": PHI0,
            "PHI0_REDUCED_Wb": PHI0_REDUCED,
            "IC_L_PHYS_A": IC_L_PHYS,
            "C_L_PHYS_F": C_L_PHYS,
            "OMEGA_PL_rad_s": OMEGA_PL,
            "V_SCALE_V": V_SCALE,
            "V_SCALE_mV": V_SCALE * 1e3,
        },
    }

    json_path = out_dir / "parameters.json"
    txt_path = out_dir / "parameters.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("Physical scales\n")
        f.write("================\n")
        for key, value in payload["physical_scales"].items():
            f.write(f"{key}: {value}\n")

        f.write("\nParameters\n")
        f.write("==========\n")
        for key, value in p.items():
            f.write(f"{key}: {value}\n")

    print("Saved parameters:", json_path.resolve())
    print("Saved parameters:", txt_path.resolve())



# ============================================================
# 2. STABLE FLUXON INITIAL CONDITION
# ============================================================

def stable_kink_phi(xi: np.ndarray) -> np.ndarray:
    """Stable version of 4 atan(exp(xi))."""
    xi = np.asarray(xi, dtype=float)
    phi = np.empty_like(xi, dtype=float)

    mask = xi < 0.0
    phi[mask] = 4.0 * np.arctan(np.exp(xi[mask]))
    phi[~mask] = 2.0 * np.pi - 4.0 * np.arctan(np.exp(-xi[~mask]))
    return phi


def stable_sech(xi: np.ndarray) -> np.ndarray:
    """Stable sech(xi) = 1 / cosh(xi)."""
    xi = np.asarray(xi, dtype=float)
    z = np.exp(-np.abs(xi))
    return 2.0 * z / (1.0 + z * z)


def make_one_fluxon_ic(
    M: int,
    l_L: float,
    fluxon_cell: int,
    velocity: float,
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Continuum sine-Gordon 2pi kink placed on a discrete LJJ lattice.

    dx = sqrt(l_L), x_j = j dx.

    Right-moving fluxon:
        phi(x,t) = 4 atan(exp((x - x0 - v t) / sqrt(1-v^2)))
        phi_tau(x,0) < 0 near the fluxon center.
    """
    if not (0.0 <= velocity < 1.0):
        raise ValueError("fluxon_velocity must satisfy 0 <= v < 1")

    if direction not in {"right", "left"}:
        raise ValueError("fluxon_direction must be 'right' or 'left'")

    dx = np.sqrt(l_L)
    x = np.arange(M) * dx
    x0 = float(fluxon_cell) * dx

    gamma_rel = np.sqrt(1.0 - velocity**2)
    xi = (x - x0) / gamma_rel

    phi = stable_kink_phi(xi)

    if direction == "right":
        sign = -1.0
    else:
        sign = +1.0

    psi = sign * (2.0 * velocity / gamma_rel) * stable_sech(xi)
    return phi, psi


def make_initial_state(p: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    State vector:
        y = [phi_0..phi_{M-1}, theta_0..theta_{N-1},
             psi_phi_0..psi_phi_{M-1}, psi_theta_0..psi_theta_{N-1}]
    """
    M = int(p["M"])
    N = int(p["N_JTL"])
    l_L = float(p["l_L"])

    if N < 2:
        raise ValueError("Use N_JTL >= 2, otherwise there is no propagating JTL output.")

    dx = np.sqrt(l_L)
    x_ljj = np.arange(M) * dx

    phi0 = np.zeros(M, dtype=float)
    psi0 = np.zeros(M, dtype=float)

    n_fluxons = int(p["n_fluxons"])
    cells = list(p["fluxon_cells"])
    if len(cells) != n_fluxons:
        raise ValueError("len(fluxon_cells) must be equal to n_fluxons")

    for cell in cells:
        phi_k, psi_k = make_one_fluxon_ic(
            M=M,
            l_L=l_L,
            fluxon_cell=int(cell),
            velocity=float(p["fluxon_velocity"]),
            direction=str(p["fluxon_direction"]),
        )
        phi0 += phi_k
        psi0 += psi_k

    s_J = float(p["s_J"])
    gamma_J = float(p["gamma_J"])
    if abs(gamma_J / s_J) >= 1.0:
        raise ValueError("Need abs(gamma_J / s_J) < 1 for static JTL initial state.")

    theta_eq = np.arcsin(gamma_J / s_J)
    theta0 = np.full(N, theta_eq, dtype=float)
    omega0 = np.zeros(N, dtype=float)

    y0 = np.concatenate([phi0, theta0, psi0, omega0])
    return x_ljj, y0


# ============================================================
# 3. COUPLED LJJ + CAPACITIVE LINK + JTL SYSTEM
# ============================================================

def system_ljj_cap_jtl(tau: float, y: np.ndarray, p: dict) -> np.ndarray:
    M = int(p["M"])
    N = int(p["N_JTL"])

    alpha_L = float(p["alpha_L"])
    gamma_L = float(p["gamma_L"])
    l_L = float(p["l_L"])

    m_J = float(p["m_J"])
    s_J = float(p["s_J"])
    alpha_J = float(p["alpha_J"])
    gamma_J = float(p["gamma_J"])
    l_J = float(p["l_J"])

    kappa = float(p["kappa"])

    phi = y[:M]
    theta = y[M:M + N]
    psi = y[M + N:M + N + M]
    omega = y[M + N + M:M + N + M + N]

    ddphi = np.zeros(M, dtype=float)
    ddtheta = np.zeros(N, dtype=float)

    # --------------------------------------------------------
    # LJJ left boundary: open end / Neumann boundary
    # equation consistent with user's open-chain code:
    # psi_dot[0] = i - sin(phi0) - alpha psi0 - (phi0 - phi1)/l
    # --------------------------------------------------------
    ddphi[0] = (
        gamma_L
        - np.sin(phi[0])
        - alpha_L * psi[0]
        - (phi[0] - phi[1]) / l_L
    )

    # --------------------------------------------------------
    # LJJ bulk
    # --------------------------------------------------------
    if M > 2:
        ddphi[1:-1] = (
            gamma_L
            - np.sin(phi[1:-1])
            - alpha_L * psi[1:-1]
            + (phi[2:] + phi[:-2] - 2.0 * phi[1:-1]) / l_L
        )

    # --------------------------------------------------------
    # Coupled equations for right LJJ edge and first JTL cell
    #
    # (1+kappa) phi_M'' - kappa theta_1'' = F
    # -kappa phi_M'' + (m_J+kappa) theta_1'' = G
    # --------------------------------------------------------
    F = (
        gamma_L
        - np.sin(phi[-1])
        - alpha_L * psi[-1]
        - (phi[-1] - phi[-2]) / l_L
    )

    G = (
        gamma_J
        - s_J * np.sin(theta[0])
        - alpha_J * omega[0]
        - (theta[0] - theta[1]) / l_J
    )

    det = (1.0 + kappa) * (m_J + kappa) - kappa**2
    if det <= 0.0:
        raise ValueError("Bad capacitance matrix: determinant <= 0. Check m_J and kappa.")

    ddphi[-1] = ((m_J + kappa) * F + kappa * G) / det
    ddtheta[0] = (kappa * F + (1.0 + kappa) * G) / det

    # --------------------------------------------------------
    # JTL bulk
    # --------------------------------------------------------
    if N > 2:
        ddtheta[1:-1] = (
            gamma_J
            - s_J * np.sin(theta[1:-1])
            - alpha_J * omega[1:-1]
            + (theta[2:] + theta[:-2] - 2.0 * theta[1:-1]) / l_J
        ) / m_J

    # --------------------------------------------------------
    # JTL right end + optional output load
    # --------------------------------------------------------
    ddtheta[-1] = (
        gamma_J
        - s_J * np.sin(theta[-1])
        - (alpha_J) * omega[-1]
        - (theta[-1] - theta[-2]) / l_J
    ) / m_J

    return np.concatenate([psi, omega, ddphi, ddtheta])


# ============================================================
# 4. SIMULATION
# ============================================================

def run_simulation(p: dict):
    x_ljj, y0 = make_initial_state(p)

    tau_eval = np.linspace(0.0, float(p["tau_end"]), int(p["n_eval"]))

    t0 = time.time()
    sol = solve_ivp(
        fun=lambda tau, y: system_ljj_cap_jtl(tau, y, p),
        t_span=(0.0, float(p["tau_end"])),
        y0=y0,
        t_eval=tau_eval,
        method=str(p["method"]),
        rtol=float(p["rtol"]),
        atol=float(p["atol"]),
        max_step=float(p["max_step"]),
    )

    elapsed = time.time() - t0
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    print(f"Integration done in {elapsed:.2f} s")
    return x_ljj, sol


def unpack_solution(sol, p: dict):
    M = int(p["M"])
    N = int(p["N_JTL"])

    tau = sol.t
    Y = sol.y.T

    phi = Y[:, :M]
    theta = Y[:, M:M + N]
    psi = Y[:, M + N:M + N + M]
    omega = Y[:, M + N + M:M + N + M + N]

    return tau, phi, theta, psi, omega


# ============================================================
# 5. PLOTS AND ANIMATION
# ============================================================

def save_csv(out_dir: Path, tau, psi_ljj, omega_jtl, theta_jtl):
    V_out_norm = omega_jtl[:, -1]
    flux_out = (theta_jtl[:, -1] - theta_jtl[0, -1]) / (2.0 * np.pi)

    data = np.column_stack([
        tau,
        V_out_norm,
        V_out_norm * V_SCALE * 1e3,
        flux_out,
    ])

    path = out_dir / "jtl_output.csv"
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="tau,V_out_normalized,V_out_mV,Delta_theta_N_over_2pi",
        comments="",
    )
    print("Saved:", path.resolve())


def plot_results(out_dir: Path, x_ljj, tau, psi_ljj, omega_jtl, theta_jtl, p: dict):
    N = int(p["N_JTL"])

    V_ljj_norm = psi_ljj
    V_jtl_norm = omega_jtl

    V_ljj_mV = V_ljj_norm * V_SCALE * 1e3
    V_jtl_mV = V_jtl_norm * V_SCALE * 1e3
    V_out_mV = V_jtl_mV[:, -1]
    flux_out = (theta_jtl[:, -1] - theta_jtl[0, -1]) / (2.0 * np.pi)

    print(f"Final output flux Delta theta_N / 2pi = {flux_out[-1]:.6g}")
    print("If this is close to 1, one SFQ pulse reached the JTL output.")

    # Output voltage
    plt.figure(figsize=(9, 4))
    plt.plot(tau, V_out_mV, lw=1.5)
    plt.xlabel(r"$\tau = \omega_{pL} t$")
    plt.ylabel(r"$V_{out}$, mV")
    plt.title("JTL output voltage")
    plt.grid(True)
    plt.tight_layout()
    path = out_dir / "jtl_output_voltage.png"
    plt.savefig(path, dpi=200)
    if p["show_plots"]:
        plt.show()
    else:
        plt.close()

    # Output flux
    plt.figure(figsize=(9, 4))
    plt.plot(tau, flux_out, lw=1.5)
    plt.xlabel(r"$\tau = \omega_{pL} t$")
    plt.ylabel(r"$\Delta\theta_N / 2\pi$")
    plt.title("Accumulated output flux in JTL")
    plt.grid(True)
    plt.tight_layout()
    path = out_dir / "jtl_output_flux.png"
    plt.savefig(path, dpi=200)
    if p["show_plots"]:
        plt.show()
    else:
        plt.close()

    # LJJ voltage heatmap
    plt.figure(figsize=(10, 5))
    plt.imshow(
        V_ljj_mV.T,
        aspect="auto",
        origin="lower",
        extent=[tau[0], tau[-1], x_ljj[0], x_ljj[-1]],
    )
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$x/\lambda_J$")
    plt.title("Voltage in LJJ, mV")
    plt.colorbar(label="mV")
    plt.tight_layout()
    path = out_dir / "ljj_voltage_heatmap.png"
    plt.savefig(path, dpi=200)
    if p["show_plots"]:
        plt.show()
    else:
        plt.close()

    # JTL voltage heatmap
    plt.figure(figsize=(10, 4.5))
    plt.imshow(
        V_jtl_mV.T,
        aspect="auto",
        origin="lower",
        extent=[tau[0], tau[-1], 1, N],
    )
    plt.xlabel(r"$\tau$")
    plt.ylabel("JTL junction number")
    plt.title("Voltage in JTL, mV")
    plt.colorbar(label="mV")
    plt.tight_layout()
    path = out_dir / "jtl_voltage_heatmap.png"
    plt.savefig(path, dpi=200)
    if p["show_plots"]:
        plt.show()
    else:
        plt.close()


def make_and_save_animation(out_dir: Path, x_ljj, tau, psi_ljj, omega_jtl, p: dict):
    if not p["save_animation"]:
        return

    N = int(p["N_JTL"])
    frames_count = int(p["animation_frames"])
    frames = np.linspace(0, len(tau) - 1, frames_count).astype(int)

    V_ljj_mV = psi_ljj * V_SCALE * 1e3
    V_jtl_mV = omega_jtl * V_SCALE * 1e3

    ljj_lim = np.percentile(np.abs(V_ljj_mV), 99.5)
    jtl_lim = np.percentile(np.abs(V_jtl_mV), 99.5)
    if ljj_lim <= 0:
        ljj_lim = 1.0
    if jtl_lim <= 0:
        jtl_lim = 1.0

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    ax_ljj, ax_jtl = axes

    line_ljj, = ax_ljj.plot(x_ljj, V_ljj_mV[0], lw=2)
    ax_ljj.set_xlim(x_ljj[0], x_ljj[-1])
    ax_ljj.set_ylim(-1.2 * ljj_lim, 1.2 * ljj_lim)
    ax_ljj.set_xlabel(r"$x/\lambda_J$")
    ax_ljj.set_ylabel("LJJ voltage, mV")
    ax_ljj.grid(True)

    jtl_cells = np.arange(1, N + 1)
    line_jtl, = ax_jtl.plot(jtl_cells, V_jtl_mV[0], "o-", lw=2, ms=4)
    ax_jtl.set_xlim(1, N)
    ax_jtl.set_ylim(-1.2 * jtl_lim, 1.2 * jtl_lim)
    ax_jtl.set_xlabel("JTL junction number")
    ax_jtl.set_ylabel("JTL voltage, mV")
    ax_jtl.grid(True)

    time_text = ax_ljj.text(0.02, 0.88, "", transform=ax_ljj.transAxes, fontsize=12)

    def update(k):
        idx = frames[k]
        line_ljj.set_ydata(V_ljj_mV[idx])
        line_jtl.set_ydata(V_jtl_mV[idx])
        time_text.set_text(rf"$\tau={tau[idx]:.2f}$")
        return line_ljj, line_jtl, time_text

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=int(p["animation_interval_ms"]),
        blit=True,
    )

    gif_path = out_dir / "ljj_jtl_voltage_animation.gif"
    print("Saving animation:", gif_path.resolve())
    ani.save(gif_path, writer="pillow", dpi=120)
    plt.close(fig)
    print("Saved:", gif_path.resolve())


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":
    print("omega_pL =", f"{OMEGA_PL:.6e}", "rad/s")
    print("V_scale =", f"{V_SCALE * 1e3:.6g}", "mV for normalized voltage = 1")

    out_dir = make_output_dir(P)

    print("Output directory:", out_dir.resolve())
    print("Parameters:")
    for key, value in P.items():
        print(f"  {key}: {value}")

    save_parameters(out_dir, P)

    x_ljj, sol = run_simulation(P)
    tau, phi, theta, psi, omega = unpack_solution(sol, P)

    save_csv(out_dir, tau, psi, omega, theta)
    plot_results(out_dir, x_ljj, tau, psi, omega, theta, P)
    make_and_save_animation(out_dir, x_ljj, tau, psi, omega, P)

    print("Done.")
