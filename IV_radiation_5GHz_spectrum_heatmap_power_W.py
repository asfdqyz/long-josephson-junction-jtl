from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time


# ============================================================
# RCSJ + inductive coupling + external flux per cell, open chain
# ============================================================
def system(t, y, N, alpha, i, l, a):
    """
    Open chain, nodes k=0..N-1, cells m=0..N-2 between k and k+1.

    Normalized equations:
        dphi_k/dt = psi_k
        dpsi_k/dt = i - sin(phi_k) - alpha*psi_k - coupling_k

    a[m] = 2*pi*Phi_e[m]/Phi0 is external flux in cell m.
    In the runs below, a is usually zero: fluxons are introduced through
    the initial phase profile, not through external flux.
    """
    phi = y[:N]
    psi = y[N:]

    # a can be None / scalar / array / function a(t, N-1)
    if callable(a):
        a = a(t, N - 1)
    if a is None:
        a = np.zeros(N - 1, dtype=float)
    else:
        a = np.asarray(a, dtype=float)
        if a.ndim == 0:
            a = np.full(N - 1, float(a))
    if a.shape != (N - 1,):
        raise ValueError(f"a must have shape ({N-1},), got {a.shape}")

    dphi_dt = psi
    dpsi_dt = np.zeros_like(psi)

    # left boundary k = 0
    dpsi_dt[0] = (
        i
        - np.sin(phi[0])
        - alpha * psi[0]
        - (phi[0] - phi[1] + a[0]) / l
    )

    # bulk k = 1..N-2
    if N > 2:
        dpsi_dt[1:-1] = (
            i
            - np.sin(phi[1:-1])
            - alpha * psi[1:-1]
            + (phi[2:] + phi[:-2] - 2.0 * phi[1:-1] + a[:-1] - a[1:]) / l
        )

    # right boundary k = N-1
    dpsi_dt[-1] = (
        i
        - np.sin(phi[-1])
        - alpha * psi[-1]
        - (phi[-1] - phi[-2] - a[-1]) / l
    )

    return np.concatenate([dphi_dt, dpsi_dt])


# ============================================================
# Fluxon initial conditions
# ============================================================
def _stable_kink_phi(xi):
    """Stable version of 4*atan(exp(xi))."""
    xi = np.asarray(xi, dtype=float)
    phi = np.empty_like(xi, dtype=float)

    mask = xi < 0.0
    phi[mask] = 4.0 * np.arctan(np.exp(xi[mask]))
    phi[~mask] = 2.0 * np.pi - 4.0 * np.arctan(np.exp(-xi[~mask]))

    return phi


def _stable_sech(xi):
    """Stable sech(xi) = 1/cosh(xi)."""
    xi = np.asarray(xi, dtype=float)
    z = np.exp(-np.abs(xi))
    return 2.0 * z / (1.0 + z * z)


def make_fluxon_ic(N, l, x0, v=0.9, direction="right"):
    """
    Continuum sine-Gordon 2*pi kink placed on the discrete lattice.

    N         -- number of nodes
    l         -- coupling parameter; lattice step dx = sqrt(l)
    x0        -- fluxon center in continuous coordinate x
    v         -- initial fluxon velocity, 0 <= v < 1
    direction -- "left" or "right"

    For a right-moving fluxon:
        phi(x,t) = 4 atan(exp((x - x0 - v*t)/sqrt(1-v^2)))
        psi(x,0) = dphi/dt < 0 near the center
    """
    if not (0.0 <= v < 1.0):
        raise ValueError("Need 0 <= v < 1")

    dx = np.sqrt(l) 
    x = np.arange(N) * dx

    gamma = np.sqrt(1.0 - v**2)
    xi = (x - x0) / gamma

    phi0 = _stable_kink_phi(xi)

    if direction == "left":
        sgn = +1.0
    elif direction == "right":
        sgn = -1.0
    else:
        raise ValueError("direction must be 'left' or 'right'")

    psi0 = sgn * (2.0 * v / gamma) * _stable_sech(xi)

    return phi0, psi0


def make_multi_fluxon_y0(
    N,
    l,
    n_fluxons,
    fluxon_cell=1,
    v0=0.9,
    direction="right",
    placement_mode="spread",       # "same", "spread", or "custom"
    spacing_cells=8,
    fluxon_cells=None,
):
    """
    Make initial conditions with a controllable number of fluxons.

    n_fluxons:
        Number of 2*pi kinks to put into the initial phase profile.

    placement_mode="same":
        All fluxons are placed at the same center. This imitates the
        Ustinov/Malomed/Sakai style for a 4*pi, 6*pi, ... initial kink:
        two 2*pi kinks at the same x0 produce an initial 4*pi kink.

    placement_mode="spread":
        Fluxons are placed in different cells:
            fluxon_cell, fluxon_cell + spacing_cells, ...

    placement_mode="custom":
        Use the explicit list fluxon_cells, for example [5, 20, 35].

    fluxon_cell and fluxon_cells are 1-based cell-like indices.
    """
    if n_fluxons < 0 or int(n_fluxons) != n_fluxons:
        raise ValueError("n_fluxons must be a non-negative integer")
    n_fluxons = int(n_fluxons)

    dx = np.sqrt(l)

    if n_fluxons == 0:
        phi0 = np.zeros(N)
        psi0 = np.zeros(N)
        return np.concatenate([phi0, psi0]), [], 0.0

    if placement_mode == "same":
        cells = [int(fluxon_cell)] * n_fluxons
    elif placement_mode == "spread":
        cells = [int(fluxon_cell + j * spacing_cells) for j in range(n_fluxons)]
    elif placement_mode == "custom":
        if fluxon_cells is None:
            raise ValueError("For placement_mode='custom', pass fluxon_cells")
        if len(fluxon_cells) != n_fluxons:
            raise ValueError("len(fluxon_cells) must be equal to n_fluxons")
        cells = [int(c) for c in fluxon_cells]
    else:
        raise ValueError("placement_mode must be 'same', 'spread', or 'custom'")

    if min(cells) < 0 or max(cells) > N - 1:
        raise ValueError(
            f"Fluxon centers are outside the chain: cells={cells}. "
            f"Use cells in the approximate range 0..{N-1}."
        )

    phi0 = np.zeros(N, dtype=float)
    psi0 = np.zeros(N, dtype=float)

    for cell in cells:
        x0 = float(cell) * dx
        phi_k, psi_k = make_fluxon_ic(
            N=N,
            l=l,
            x0=x0,
            v=v0,
            direction=direction,
        )
        phi0 += phi_k
        psi0 += psi_k

    y0 = np.concatenate([phi0, psi0])
    q_fluxon = (phi0[-1] - phi0[0]) / (2.0 * np.pi)

    return y0, cells, q_fluxon


# ============================================================
# Voltage extraction
# ============================================================
def avg_voltage_global(sol, N, transient_fraction=0.6):
    """
    Normalized voltage:
        V = <dphi/dt>, averaged over nodes and over the final time window.
    """
    cut = int(transient_fraction * sol.t.size)
    psi = sol.y[N:, cut:]
    return float(np.mean(psi))


def avg_voltage_single_node(sol, N, transient_fraction=0.6, node_index=None):
    if node_index is None:
        node_index = N // 2
    cut = int(transient_fraction * sol.t.size)
    psi = sol.y[N:, cut:]
    return float(np.mean(psi[node_index, :]))


# ============================================================
# Spectrum extraction
# ============================================================
def extract_voltage_signal_for_spectrum(
    sol,
    N,
    signal_mode="right_edge",
    node_index=None,
    transient_fraction=0.5,
):
    """
    Extract a voltage-like signal psi(tau)=dphi/dtau for spectral analysis.

    signal_mode:
        "right_edge" -- voltage at the right boundary node. This is usually
                        the best choice for checking the fluxon round-trip
                        radiation frequency u/(2L), because the same edge is
                        hit once per round trip.
        "left_edge"  -- voltage at the left boundary node.
        "node"       -- voltage at a selected node_index.
                        Be careful: a middle node can be crossed twice per
                        fluxon round trip and therefore can show a strong
                        component near 2*u/(2L).
        "global"     -- voltage averaged over all LJJ nodes.
    """
    cut = int(transient_fraction * sol.t.size)
    tau = sol.t[cut:]
    psi = sol.y[N:, cut:]

    if tau.size < 8:
        raise ValueError("Too few time samples for spectrum. Increase n_eval or decrease transient_fraction.")

    if signal_mode == "right_edge":
        signal = psi[-1, :]
    elif signal_mode == "left_edge":
        signal = psi[0, :]
    elif signal_mode == "node":
        if node_index is None:
            node_index = N // 2
        signal = psi[int(node_index), :]
    elif signal_mode == "global":
        signal = np.mean(psi, axis=0)
    else:
        raise ValueError("signal_mode must be 'right_edge', 'left_edge', 'node', or 'global'")

    return tau, np.asarray(signal, dtype=float)


def single_sided_spectrum_vs_tau(tau, signal, remove_dc=True, use_hann=True):
    """
    Compute one-sided spectrum of signal(tau).

    Returned frequency is in cycles per normalized time tau.
    Physical frequency is:
        f_Hz = f_tau * omega_p.

    Returned power is not dB. It is the normalized mean-square voltage
    contribution of each FFT bin:
        power_norm = V_norm,rms^2.

    To convert it to load power:
        P_W = power_norm * V_scale^2 / R_load,
    where
        V_scale = (Phi0 / 2pi) * omega_p.
    """
    tau = np.asarray(tau, dtype=float)
    signal = np.asarray(signal, dtype=float)

    if tau.size != signal.size:
        raise ValueError("tau and signal must have the same length")
    if tau.size < 8:
        raise ValueError("Too few time samples for FFT")

    dtau = float(np.mean(np.diff(tau)))
    if dtau <= 0.0:
        raise ValueError("Bad tau grid: dtau <= 0")

    x = signal.copy()
    if remove_dc:
        x -= np.mean(x)

    n = x.size

    if use_hann:
        window = np.hanning(n)
        coherent_gain = np.sum(window) / n
        xw = x * window
    else:
        coherent_gain = 1.0
        xw = x

    freq_tau = np.fft.rfftfreq(n, d=dtau)
    fft_values = np.fft.rfft(xw)

    # One-sided peak amplitude of normalized voltage for every frequency bin.
    amp_norm = 2.0 * np.abs(fft_values) / max(n * coherent_gain, 1e-300)

    # DC and Nyquist bins should not be doubled. DC is removed below anyway.
    if amp_norm.size > 0:
        amp_norm[0] *= 0.5
    if n % 2 == 0 and amp_norm.size > 1:
        amp_norm[-1] *= 0.5

    # Convert peak amplitude to RMS squared: V_rms^2 = V_peak^2 / 2.
    power_norm = 0.5 * amp_norm**2

    # Remove residual DC from the plotted spectrum.
    if power_norm.size > 0:
        power_norm[0] = 0.0

    return freq_tau, power_norm


def compute_solution_spectrum(
    sol,
    N,
    signal_mode="right_edge",
    node_index=None,
    transient_fraction=0.5,
):
    tau, signal = extract_voltage_signal_for_spectrum(
        sol=sol,
        N=N,
        signal_mode=signal_mode,
        node_index=node_index,
        transient_fraction=transient_fraction,
    )
    return single_sided_spectrum_vs_tau(tau, signal)


# ============================================================
# I-V scan for arbitrary initial condition
# ============================================================
def simulate_IV(
    N,
    alpha,
    l,
    y0,
    currents,
    a=None,
    t_end=1500.0,
    n_eval=5000,
    method="RK45",
    transient_fraction=0.8,
    voltage_mode="node",       # "global" or "node"
    node_index=None,
    reset_each_current=False,
    show_progress=True,
    label="",
    return_spectrum=False,
    spectrum_signal_mode="right_edge",
    spectrum_node_index=None,
    spectrum_transient_fraction=0.5,
    spectrum_max_freq_tau=None,
):
    """
    Return V(I) for a current sweep.

    If return_spectrum=True, also return a spectrum matrix:
        V, freq_tau, spectrum_power

    spectrum_power has shape:
        (number of current points, number of frequency points).

    freq_tau is measured in cycles per normalized time tau. Convert it by:
        f_Hz = freq_tau * omega_p.
    """
    t_span = (0.0, t_end)
    t_eval = np.linspace(t_span[0], t_span[1], n_eval)

    y_init = y0.copy()
    V = []

    spectrum_freq_tau = None
    spectrum_power_rows = []

    t0 = time.time()
    total = len(currents)

    def progress_line(k, i_val):
        if not show_progress:
            return
        done = k + 1
        frac = done / total
        bar_len = 28
        filled = int(bar_len * frac)
        bar = "#" * filled + "-" * (bar_len - filled)
        elapsed = time.time() - t0
        rate = elapsed / done
        eta = rate * (total - done)
        print(
            f"\r{label}[{bar}] {done:>4}/{total:<4}  i={i_val: .4f}  "
            f"elapsed={elapsed:7.1f}s  eta={eta:7.1f}s",
            end="",
            flush=True,
        )

    for k, i_val in enumerate(currents):
        if reset_each_current:
            y_init = y0.copy()

        sol = solve_ivp(
            system,
            t_span,
            y_init,
            args=(N, alpha, float(i_val), l, a),
            t_eval=t_eval,
            method=method,
            rtol=1e-6,
            atol=1e-9,
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed at i={i_val}: {sol.message}")

        if voltage_mode == "global":
            V.append(avg_voltage_global(sol, N, transient_fraction))
        elif voltage_mode == "node":
            V.append(avg_voltage_single_node(sol, N, transient_fraction, node_index))
        else:
            raise ValueError("voltage_mode must be 'global' or 'node'")

        if return_spectrum:
            freq_tau, power = compute_solution_spectrum(
                sol=sol,
                N=N,
                signal_mode=spectrum_signal_mode,
                node_index=spectrum_node_index,
                transient_fraction=spectrum_transient_fraction,
            )

            if spectrum_max_freq_tau is not None:
                mask = freq_tau <= float(spectrum_max_freq_tau)
                freq_tau = freq_tau[mask]
                power = power[mask]

            if spectrum_freq_tau is None:
                spectrum_freq_tau = freq_tau
            elif spectrum_freq_tau.shape != freq_tau.shape or not np.allclose(spectrum_freq_tau, freq_tau):
                raise RuntimeError("Frequency grid changed between current points")

            spectrum_power_rows.append(power)

        y_init = sol.y[:, -1].copy()
        progress_line(k, float(i_val))

    if show_progress:
        print()

    V = np.asarray(V)

    if return_spectrum:
        return V, spectrum_freq_tau, np.vstack(spectrum_power_rows)

    return V

# ============================================================
# Worker: one IV curve for one number of fluxons
# ============================================================
def run_one_fluxon_number(job):
    (
        n_fluxons,
        N,
        alpha,
        l,
        currents,
        a,
        fluxon_cell,
        v0,
        direction,
        placement_mode,
        spacing_cells,
        fluxon_cells_map,
        t_end,
        n_eval,
        method,
        transient_fraction,
        voltage_mode,
        node_index,
        reset_each_current,
        return_spectrum,
        spectrum_signal_mode,
        spectrum_node_index,
        spectrum_transient_fraction,
        spectrum_max_freq_tau,
    ) = job

    custom_cells = None
    if fluxon_cells_map is not None and int(n_fluxons) in fluxon_cells_map:
        custom_cells = fluxon_cells_map[int(n_fluxons)]

    actual_placement_mode = placement_mode
    if custom_cells is not None:
        actual_placement_mode = "custom"

    y0, cells, q_fluxon = make_multi_fluxon_y0(
        N=N,
        l=l,
        n_fluxons=int(n_fluxons),
        fluxon_cell=fluxon_cell,
        v0=v0,
        direction=direction,
        placement_mode=actual_placement_mode,
        spacing_cells=spacing_cells,
        fluxon_cells=custom_cells,
    )

    sim_result = simulate_IV(
        N=N,
        alpha=alpha,
        l=l,
        y0=y0,
        currents=currents,
        a=a,
        t_end=t_end,
        n_eval=n_eval,
        method=method,
        transient_fraction=transient_fraction,
        voltage_mode=voltage_mode,
        node_index=node_index,
        reset_each_current=reset_each_current,
        show_progress=False,
        return_spectrum=return_spectrum,
        spectrum_signal_mode=spectrum_signal_mode,
        spectrum_node_index=spectrum_node_index,
        spectrum_transient_fraction=spectrum_transient_fraction,
        spectrum_max_freq_tau=spectrum_max_freq_tau,
    )

    if return_spectrum:
        V, spectrum_freq_tau, spectrum_power = sim_result
    else:
        V = sim_result
        spectrum_freq_tau = None
        spectrum_power = None

    result = {
        "n_fluxons": int(n_fluxons),
        "cells": cells,
        "q_fluxon_initial": float(q_fluxon),
        "V": V,
    }

    if return_spectrum:
        result["spectrum_freq_tau"] = spectrum_freq_tau
        result["spectrum_power"] = spectrum_power
        result["spectrum_signal_mode"] = spectrum_signal_mode

    return result


# ============================================================
# Helpers
# ============================================================
def make_current_array(i_min, i_max, di, sweep_direction="up"):
    if di <= 0:
        raise ValueError("di must be positive")
    if sweep_direction == "up":
        return np.arange(i_min, i_max + 0.5 * di, di)
    if sweep_direction == "down":
        return np.arange(i_max, i_min - 0.5 * di, -di)
    raise ValueError("sweep_direction must be 'up' or 'down'")


def safe_float_tag(x):
    return f"{float(x):g}".replace(".", "p").replace("-", "m")


# ============================================================
# Physical frequency / voltage conversion for JOSEPHSON frequency
# ============================================================
PHI0_SI = 2.067833848e-15  # Wb = V*s
MU0_SI = 4.0 * np.pi * 1e-7  # H/m


def josephson_penetration_depth(Jc_A_per_m2, d_eff_m):
    """Josephson penetration depth lambda_J in meters."""
    if Jc_A_per_m2 <= 0 or d_eff_m <= 0:
        raise ValueError("Jc_A_per_m2 and d_eff_m must be positive")
    return float(np.sqrt(PHI0_SI / (2.0 * np.pi * MU0_SI * d_eff_m * Jc_A_per_m2)))


def normalized_voltage_to_physical_voltage(V_norm, omega_p_rad_s):
    """Convert code voltage <dphi/dtau> to physical voltage in volts."""
    return np.asarray(V_norm, dtype=float) * PHI0_SI * omega_p_rad_s / (2.0 * np.pi)


def normalized_current_to_physical_current(i_norm, Ic_A):
    """Convert normalized bias current i=I/Ic to physical current in amperes."""
    return np.asarray(i_norm, dtype=float) * Ic_A


def plasma_angular_frequency_from_jc_cspec(Jc_A_per_m2, Cspec_F_per_m2):
    """
    Plasma angular frequency for a tunnel process with given technology:
        omega_p = sqrt(2*pi*Jc/(Phi0*Cspec)).

    Jc and Cspec are per unit area, so omega_p does not depend on the
    selected junction area.
    """
    if Jc_A_per_m2 <= 0 or Cspec_F_per_m2 <= 0:
        raise ValueError("Jc_A_per_m2 and Cspec_F_per_m2 must be positive")
    return float(np.sqrt(2.0 * np.pi * Jc_A_per_m2 / (PHI0_SI * Cspec_F_per_m2)))


def plasma_angular_frequency(Ic_A, Cj_F):
    """RCSJ plasma angular frequency omega_p = sqrt(2*pi*Ic/(Phi0*Cj))."""
    if Ic_A <= 0 or Cj_F <= 0:
        raise ValueError("Ic_A and Cj_F must be positive")
    return float(np.sqrt(2.0 * np.pi * Ic_A / (PHI0_SI * Cj_F)))


def normalized_voltage_for_josephson_frequency(f_josephson_Hz, omega_p_rad_s):
    """
    In this code time is normalized as tau = omega_p*t, so
        V_norm = <dphi/dtau>.

    The ordinary AC Josephson relation is:
        f_J = V_phys/Phi0.

    Also:
        V_phys = (Phi0/2*pi)*omega_p*V_norm.

    Therefore:
        f_J = omega_p*V_norm/(2*pi),
        V_norm = 2*pi*f_J/omega_p = f_J/f_p.
    """
    return np.asarray(f_josephson_Hz, dtype=float) * 2.0 * np.pi / omega_p_rad_s


def josephson_frequency_from_normalized_voltage(V_norm, omega_p_rad_s):
    """Convert code voltage <dphi/dtau> to Josephson frequency in Hz."""
    return np.asarray(V_norm, dtype=float) * omega_p_rad_s / (2.0 * np.pi)


def physical_voltage_for_josephson_frequency(f_josephson_Hz):
    """AC Josephson relation: V_phys = Phi0*f_J."""
    return float(PHI0_SI * f_josephson_Hz)


def first_step_vnorm_from_length(N, l, u_norm=1.0):
    """Approximate first fluxon/ZFS step voltage: V_norm ~= 2*pi*u_norm/L_norm."""
    L_norm = (N - 1) * np.sqrt(l)
    return float(2.0 * np.pi * u_norm / L_norm)


def tune_l_for_first_step_josephson(N, f_josephson_Hz, omega_p_rad_s, u_norm=1.0):
    """Choose l for a requested Josephson frequency on the first step at fixed N."""
    V_target = normalized_voltage_for_josephson_frequency(f_josephson_Hz, omega_p_rad_s)
    L_target = 2.0 * np.pi * u_norm / V_target
    return float((L_target / (N - 1)) ** 2)


def tune_N_for_first_step_josephson(l, f_josephson_Hz, omega_p_rad_s, u_norm=1.0):
    """Choose integer N for a requested Josephson frequency on the first step at fixed l."""
    V_target = normalized_voltage_for_josephson_frequency(f_josephson_Hz, omega_p_rad_s)
    L_target = 2.0 * np.pi * u_norm / V_target
    return int(round(L_target / np.sqrt(l))) + 1


def shunt_alpha_from_area(area_um2, R_cell_ohm, omega_p_rad_s, Cspec_pF_per_um2):
    """alpha = 1/(omega_p*R_cell*C_cell), C_cell=Cspec*area."""
    C_cell_F = Cspec_pF_per_um2 * 1e-12 * area_um2
    return float(1.0 / (omega_p_rad_s * R_cell_ohm * C_cell_F))


def cell_area_for_alpha(alpha, R_cell_ohm, omega_p_rad_s, Cspec_pF_per_um2):
    """Cell area in um^2 needed to obtain the requested damping alpha."""
    return float(1.0 / (omega_p_rad_s * R_cell_ohm * alpha * Cspec_pF_per_um2 * 1e-12))


def find_closest_to_target(currents, V_norm, V_target_norm):
    """Return the current point closest to the requested normalized voltage."""
    V_norm = np.asarray(V_norm, dtype=float)
    idx = int(np.argmin(np.abs(V_norm - V_target_norm)))
    return idx, float(currents[idx]), float(V_norm[idx])


if __name__ == "__main__":

    # ------------------------------------------------------------
    # Technology parameters
    # ------------------------------------------------------------
    # Given fabrication data:
    #   Jc = 0.1 kA/cm^2 = 100 A/cm^2 = 1 uA/um^2
    #   Cspec = 0.06 pF/um^2
    #   shunt sheet resistance = 1 Ohm/square
    Jc_kA_per_cm2 = 0.1
    Cspec_pF_per_um2 = 0.06
    R_sheet_ohm_per_square = 1.0

    Jc_A_per_m2 = Jc_kA_per_cm2 * 1e3 / 1e-4
    Cspec_F_per_m2 = Cspec_pF_per_um2 * 1e-12 / 1e-12

    omega_p = plasma_angular_frequency_from_jc_cspec(Jc_A_per_m2, Cspec_F_per_m2)
    f_p_Hz = omega_p / (2.0 * np.pi)

    # Desired radiation / fluxon round-trip frequency on the first step:
    #     f_rad = u/(2L).
    # For one-fluxon zero-field step the Josephson frequency is approximately:
    #     f_J = 2*f_rad.
    # Therefore the corresponding step voltage is:
    #     V_step = Phi0*f_J = 2*Phi0*f_rad.
    f_rad_target_Hz = 5.0e9
    f_target_Hz = 2.0 * f_rad_target_Hz
    V_target_norm = normalized_voltage_for_josephson_frequency(f_target_Hz, omega_p)
    V_target_phys = physical_voltage_for_josephson_frequency(f_target_Hz)

    # ------------------------------------------------------------
    # Numerical geometry tuned so that the FIRST step has Josephson frequency near 5 GHz
    # ------------------------------------------------------------
    # First-step estimate: V_norm ~= 2*pi*u_norm/L_norm,
    # L_norm=(N-1)*sqrt(l). For u_norm ~= 1 and f_J=5 GHz,
    # this gives L_norm ~= 45.0.
    # OPTION 1: keep l=0.1 and choose N automatically.
    l = 0.1
    N = 65  # tuned for f_rad = u/(2L) ~ 5 GHz when u_norm ~ 1

    # Damping used in the normalized equations. Below we choose a real
    # LJJ geometry and then calculate the physical Ic, C and shunt R
    # corresponding to this alpha.
    alpha = 0.2

    # ------------------------------------------------------------
    # Real physical LJJ geometry
    # ------------------------------------------------------------
    # d_eff is the effective magnetic thickness. You should replace it
    # with the value from your fabrication stack when it is known.
    d_eff_nm = 200.0
    W_LJJ_um = 10.0

    lambda_J_m = josephson_penetration_depth(Jc_A_per_m2, d_eff_nm * 1e-9)
    dx_norm = np.sqrt(l)
    dx_phys_m = dx_norm * lambda_J_m
    L_LJJ_phys_m = (N - 1) * dx_phys_m

    cell_area_um2 = W_LJJ_um * dx_phys_m * 1e6
    total_area_um2 = W_LJJ_um * L_LJJ_phys_m * 1e6

    Ic_cell_A = Jc_A_per_m2 * cell_area_um2 * 1e-12
    C_cell_F = Cspec_pF_per_um2 * 1e-12 * cell_area_um2

    Ic_total_A = Jc_A_per_m2 * total_area_um2 * 1e-12
    C_total_F = Cspec_pF_per_um2 * 1e-12 * total_area_um2

    # Required shunt resistance per cell for the selected alpha:
    # alpha = 1/(omega_p * R_cell * C_cell).
    R_cell_ohm = 1.0 / (omega_p * alpha * C_cell_F)
    R_squares_per_cell = R_cell_ohm / R_sheet_ohm_per_square

    # Physical target voltage corresponding to f_J=5 GHz.
    V_target_uV = V_target_phys * 1e6


    f_ext = 0.0
    a = 2.0 * np.pi * f_ext

    # -------------------------
    # Fluxon initial conditions
    # -------------------------
    fluxon_counts = [1]

    fluxon_cell = 1
    v0 = 0.9
    direction = "right"

    placement_mode = "spread"
    spacing_cells = 12

    
    fluxon_cells_map = None

   
    i_min = 0.0
    i_max = 1.2
    di = 0.01

   
    sweep_direction = "up"   
    currents = make_current_array(i_min, i_max, di, sweep_direction=sweep_direction)

    t_end = 1500.0
    n_eval = 5000
    method = "RK45"             
    transient_fraction = 0.8

    voltage_mode = "node"
    node_index = N // 2

    reset_each_current = True

    # -------------------------
    # Spectrum settings
    # -------------------------
    # To check radiation frequency f_rad = u/(2L), use an edge signal.
    # A middle node can be crossed twice per round trip and may show a strong 2*f_rad component.
    make_spectrum = True
    spectrum_signal_mode = "right_edge"   # "right_edge", "left_edge", "node", or "global"
    spectrum_node_index = None             # used only if spectrum_signal_mode == "node"
    spectrum_transient_fraction = 0.5      # use the final 50% of the time trace for FFT
    spectrum_max_freq_GHz = 25.0

    # Load used to convert the FFT voltage spectrum to power in watts:
    #     P_W = V_rms^2 / R_load.
    # Use 50 Ohm if you want the usual microwave load estimate.
    spectrum_load_resistance_ohm = 50.0

    spectrum_max_freq_tau = spectrum_max_freq_GHz * 1e9 / omega_p


    cpu = os.cpu_count() or 2
    max_workers = min(len(fluxon_counts), max(1, cpu - 1))

    jobs = [
        (
            int(n_fluxons),
            N,
            alpha,
            l,
            currents,
            a,
            fluxon_cell,
            v0,
            direction,
            placement_mode,
            spacing_cells,
            fluxon_cells_map,
            t_end,
            n_eval,
            method,
            transient_fraction,
            voltage_mode,
            node_index,
            reset_each_current,
            make_spectrum,
            spectrum_signal_mode,
            spectrum_node_index,
            spectrum_transient_fraction,
            spectrum_max_freq_tau,
        )
        for n_fluxons in fluxon_counts
    ]

    print(
        f"Running {len(jobs)} IV curves in parallel with max_workers={max_workers}: "
        f"N_fluxons={fluxon_counts}"
    )
    print(
        f"N={N}, alpha={alpha}, l={l}, sweep={sweep_direction}, "
        f"I range: {currents[0]:g} -> {currents[-1]:g}, points={len(currents)}"
    )
    print(f"placement_mode={placement_mode}, fluxon_cell={fluxon_cell}, v0={v0}, direction={direction}")
    print(
        "Physical technology scale: "
        f"Jc={Jc_kA_per_cm2:.6g} kA/cm^2, "
        f"Cspec={Cspec_pF_per_um2:.6g} pF/um^2, "
        f"omega_p={omega_p:.6g} rad/s, f_p={f_p_Hz/1e9:.6g} GHz"
    )
    print(
        "Physical LJJ geometry: "
        f"d_eff={d_eff_nm:.6g} nm, W={W_LJJ_um:.6g} um, "
        f"lambda_J={lambda_J_m*1e6:.6g} um, "
        f"dx={dx_phys_m*1e6:.6g} um, L={L_LJJ_phys_m*1e3:.6g} mm"
    )
    print(
        "Physical cell/total parameters: "
        f"A_cell={cell_area_um2:.6g} um^2, Ic_cell={Ic_cell_A*1e6:.6g} uA, "
        f"C_cell={C_cell_F*1e12:.6g} pF; "
        f"A_total={total_area_um2:.6g} um^2, Ic_total={Ic_total_A*1e3:.6g} mA, "
        f"C_total={C_total_F*1e12:.6g} pF"
    )
    print(
        "Target in physical units: "
        f"f_rad={f_rad_target_Hz/1e9:.6g} GHz, "
        f"corresponding f_J={f_target_Hz/1e9:.6g} GHz -> "
        f"V_step=Phi0*f_J={V_target_uV:.6g} uV"
    )
    print(
        "Tuned physical/numerical length: "
        f"L_norm=(N-1)*sqrt(l)={(N-1)*np.sqrt(l):.6g}, "
        f"first_step_est={normalized_voltage_to_physical_voltage(first_step_vnorm_from_length(N, l), omega_p)*1e6:.6g} uV, "
        f"f_J_est={josephson_frequency_from_normalized_voltage(first_step_vnorm_from_length(N, l), omega_p)/1e9:.6g} GHz"
    )
    print(
        "Physical shunt estimate: "
        f"alpha={alpha:.6g}, R_cell={R_cell_ohm:.6g} Ohm, "
        f"R_sheet={R_sheet_ohm_per_square:.6g} Ohm/sq -> "
        f"R_squares_per_cell={R_squares_per_cell:.6g} squares"
    )

    t0 = time.time()
    results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_one_fluxon_number, job) for job in jobs]
        for done_count, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            n_fluxons = res["n_fluxons"]
            results[n_fluxons] = res
            elapsed = time.time() - t0
            print(
                f"Done {done_count}/{len(jobs)}: "
                f"N_fl={n_fluxons}, cells={res['cells']}, "
                f"initial winding={res['q_fluxon_initial']:.6g}, elapsed={elapsed:.1f}s"
            )


    out_dir = Path("iv_fluxon_results")
    out_dir.mkdir(exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = (
        f"IV_radiation_5GHz_with_spectrum_"
        f"N{N}_alpha{safe_float_tag(alpha)}_l{safe_float_tag(l)}_"
        f"I{safe_float_tag(i_min)}to{safe_float_tag(i_max)}_dI{safe_float_tag(di)}_"
        f"{sweep_direction}_{placement_mode}_{stamp}"
    )
    data_path = out_dir / f"{base}.csv"
    png_path = out_dir / f"{base}.png"

    n_sorted = sorted(results)

    # Physical bias-current outputs.
    # I_cell is the current corresponding to one discrete LJJ element.
    # I_total is the total current density integrated over the whole LJJ area.
    I_cell_uA = normalized_current_to_physical_current(currents, Ic_cell_A) * 1e6
    I_total_mA = normalized_current_to_physical_current(currents, Ic_total_A) * 1e3
    J_bias_A_per_cm2 = currents * (Jc_A_per_m2 / 1e4)

    data_columns = [I_cell_uA, I_total_mA, J_bias_A_per_cm2]
    header_cols = ["I_cell_uA", "I_total_mA", "J_bias_A_per_cm2"]
    for n_fluxons in n_sorted:
        Vn = results[n_fluxons]["V"]
        V_uV = normalized_voltage_to_physical_voltage(Vn, omega_p) * 1e6
        fn_GHz = josephson_frequency_from_normalized_voltage(Vn, omega_p) / 1e9
        frad_GHz = 0.5 * fn_GHz
        data_columns.append(V_uV)
        header_cols.append(f"V_uV_Nfl_{n_fluxons}")
        data_columns.append(fn_GHz)
        header_cols.append(f"josephson_frequency_GHz_Nfl_{n_fluxons}")
        data_columns.append(frad_GHz)
        header_cols.append(f"radiation_frequency_GHz_Nfl_{n_fluxons}")

    data = np.column_stack(data_columns)
    np.savetxt(
        data_path,
        data,
        delimiter=",",
        header=",".join(header_cols),
        comments="",
    )

    if 1 in results:
        idx, I_closest, V_closest = find_closest_to_target(
            currents, results[1]["V"], V_target_norm
        )
        f_closest_GHz = josephson_frequency_from_normalized_voltage(V_closest, omega_p) / 1e9
        V_closest_uV = normalized_voltage_to_physical_voltage(V_closest, omega_p) * 1e6
        I_cell_closest_uA = normalized_current_to_physical_current(I_closest, Ic_cell_A) * 1e6
        I_total_closest_mA = normalized_current_to_physical_current(I_closest, Ic_total_A) * 1e3
        J_closest_A_per_cm2 = I_closest * (Jc_A_per_m2 / 1e4)
        print(
            "Closest point on N_fl=1 curve to target radiation frequency: "
            f"index={idx}, I_cell={I_cell_closest_uA:.6g} uA, "
            f"I_total={I_total_closest_mA:.6g} mA, "
            f"J_bias={J_closest_A_per_cm2:.6g} A/cm^2, "
            f"V={V_closest_uV:.6g} uV, "
            f"f_J={f_closest_GHz:.6g} GHz, f_rad={0.5*f_closest_GHz:.6g} GHz"
        )

    plt.figure(figsize=(8, 6))

    for n_fluxons in n_sorted:
        V_uV = normalized_voltage_to_physical_voltage(results[n_fluxons]["V"], omega_p) * 1e6
        cells = results[n_fluxons]["cells"]
        label = rf"$N_{{fl}}={n_fluxons}$, cells={cells}"
        plt.plot(V_uV, I_total_mA, "o", ms=3, lw=1.0, label=label)

    plt.axvline(
        V_target_uV,
        ls="--",
        lw=1.2,
        label=rf"target $f_{{rad}}={f_rad_target_Hz/1e9:g}$ GHz"
    )

    ax = plt.gca()
    ax.set_xlabel(r"$V_\mathrm{phys}$ ($\mu$V)")
    ax.set_ylabel(r"$I_\mathrm{bias,total}$ (mA)")
    ax.set_title("Physical I-V characteristics for different initial fluxon numbers")
    ax.grid(True)

    def uV_to_rad_ghz(v_uV):
        # For a one-fluxon zero-field step: f_rad = f_J/2 = V/(2*Phi0).
        return 0.5 * np.asarray(v_uV) * 1e-6 / PHI0_SI / 1e9

    def rad_ghz_to_uV(f_ghz):
        return 2.0 * PHI0_SI * np.asarray(f_ghz) * 1e9 * 1e6

    secax = ax.secondary_xaxis("top", functions=(uV_to_rad_ghz, rad_ghz_to_uV))
    secax.set_xlabel(r"Radiation / fluxon frequency $f_{rad}=f_J/2$ (GHz)")

    def total_mA_to_current_density(y_mA):
        return np.asarray(y_mA) * 1e-3 / (total_area_um2 * 1e-12) / 1e4

    def current_density_to_total_mA(j_A_cm2):
        return np.asarray(j_A_cm2) * 1e4 * (total_area_um2 * 1e-12) * 1e3

    secay = ax.secondary_yaxis("right", functions=(total_mA_to_current_density, current_density_to_total_mA))
    secay.set_ylabel(r"$J_\mathrm{bias}$ (A/cm$^2$)")

    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")

    # ============================================================
    # Spectrum heatmap: frequency vs bias current
    # ============================================================
    if make_spectrum:
        for n_fluxons in n_sorted:
            res = results[n_fluxons]
            if "spectrum_power" not in res:
                continue

            freq_tau = np.asarray(res["spectrum_freq_tau"], dtype=float)
            freq_GHz = freq_tau * omega_p / 1e9

            # res["spectrum_power"] is V_norm,rms^2 for each FFT bin.
            # Convert it to physical load power:
            #     V_phys = V_norm * (Phi0/2pi) * omega_p
            #     P_W = V_phys,rms^2 / R_load.
            power_norm_rms2 = np.asarray(res["spectrum_power"], dtype=float)
            V_scale = PHI0_SI * omega_p / (2.0 * np.pi)
            power_W = power_norm_rms2 * V_scale**2 / spectrum_load_resistance_ohm

            peak_indices = np.argmax(power_W, axis=1)
            peak_freq_GHz = freq_GHz[peak_indices]
            peak_power_W = power_W[np.arange(power_W.shape[0]), peak_indices]

            spectrum_npz_path = out_dir / f"{base}_spectrum_Nfl{n_fluxons}.npz"
            np.savez(
                spectrum_npz_path,
                currents_norm=currents,
                I_cell_uA=I_cell_uA,
                I_total_mA=I_total_mA,
                frequency_GHz=freq_GHz,
                power_W=power_W,
                peak_frequency_GHz=peak_freq_GHz,
                peak_power_W=peak_power_W,
                load_resistance_ohm=spectrum_load_resistance_ohm,
                spectrum_signal_mode=res.get("spectrum_signal_mode", spectrum_signal_mode),
            )

            spectrum_peak_csv_path = out_dir / f"{base}_spectrum_peaks_Nfl{n_fluxons}.csv"
            np.savetxt(
                spectrum_peak_csv_path,
                np.column_stack([currents, I_cell_uA, I_total_mA, peak_freq_GHz, peak_power_W]),
                delimiter=",",
                header="i_norm,I_cell_uA,I_total_mA,peak_frequency_GHz,peak_power_W",
                comments="",
            )

            # Heatmap in the style:
            #     x-axis: frequency
            #     y-axis: bias current
            #     color: load power in W
            fig, ax_hm = plt.subplots(figsize=(7.2, 5.8))
            extent = [freq_GHz[0], freq_GHz[-1], I_cell_uA[0], I_cell_uA[-1]]
            im = ax_hm.imshow(
                power_W,
                aspect="auto",
                origin="lower",
                extent=extent,
            )
            ax_hm.set_xlabel("Frequency, GHz")
            ax_hm.set_ylabel(r"Bias, $\mu$A")
            ax_hm.set_title(
                rf"Spectrum heatmap, $N_{{fl}}={n_fluxons}$, signal={res.get('spectrum_signal_mode', spectrum_signal_mode)}"
            )
            cbar = fig.colorbar(im, ax=ax_hm)
            cbar.set_label("Load power, W")
            ax_hm.grid(False)
            fig.tight_layout()

            spectrum_png_path = out_dir / f"{base}_spectrum_heatmap_Nfl{n_fluxons}.png"
            fig.savefig(spectrum_png_path, dpi=200, bbox_inches="tight")

            print("Saved spectrum data:", spectrum_npz_path.resolve())
            print("Saved spectrum peaks:", spectrum_peak_csv_path.resolve())
            print("Saved spectrum heatmap:", spectrum_png_path.resolve())

    print("Saved data:", data_path.resolve())
    print("Saved plot:", png_path.resolve())

    plt.show()
