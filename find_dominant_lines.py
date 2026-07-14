from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def load_response_curve(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load response curve from CSV file with Wavelength_nm and Response columns."""
    wavelengths = []
    responses = []
    
    with file_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip()
        if not header.startswith("Wavelength_nm"):
            raise ValueError("Expected CSV with 'Wavelength_nm' header")
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(",")
            if len(parts) < 2:
                continue
            
            try:
                wl = float(parts[0])
                response = float(parts[1])
                wavelengths.append(wl)
                responses.append(response)
            except ValueError:
                continue
    
    if not wavelengths:
        raise ValueError("No response data found in CSV file")
    
    return np.array(wavelengths, dtype=float), np.array(responses, dtype=float)


def correct_spectrum_by_response(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    response_wavelengths: np.ndarray,
    response_values: np.ndarray,
) -> np.ndarray:
    """Divide spectrum by interpolated response curve values."""
    # Interpolate response values to match spectrum wavelengths
    interp_response = interp1d(
        response_wavelengths,
        response_values,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )
    
    response_at_spectrum = interp_response(wavelengths)
    
    # Avoid division by zero
    response_at_spectrum = np.where(response_at_spectrum != 0, response_at_spectrum, 1.0)
    
    corrected = spectrum / response_at_spectrum
    return corrected


def load_spectrum_csv(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    wavelengths = []
    intensities = []
    in_data = False

    with file_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line == "[Data]":
                in_data = True
                continue

            if not in_data:
                continue

            if line.startswith("#"):
                continue

            parts = line.split(";")
            if len(parts) < 2:
                continue

            try:
                wl = float(parts[0])
                inten = float(parts[1])
            except ValueError:
                continue

            wavelengths.append(wl)
            intensities.append(inten)

    if not wavelengths:
        raise ValueError("No numeric spectrum data found after [Data] section.")

    return np.array(wavelengths, dtype=float), np.array(intensities, dtype=float)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        raise ValueError("Smoothing window must be odd.")

    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def _find_local_peak_candidates(intensities: np.ndarray) -> np.ndarray:
    """Return indices of local maxima, including flat-top maxima centers."""
    n = len(intensities)
    if n < 3:
        return np.arange(n, dtype=int)

    peaks: list[int] = []
    i = 1
    while i < n - 1:
        if intensities[i - 1] < intensities[i]:
            j = i
            while j + 1 < n and intensities[j + 1] == intensities[i]:
                j += 1
            if j + 1 < n and intensities[j] > intensities[j + 1]:
                peaks.append((i + j) // 2)
            i = j + 1
            continue
        i += 1

    if not peaks:
        return np.arange(n, dtype=int)
    return np.array(peaks, dtype=int)


def _estimate_peak_prominences(intensities: np.ndarray, peak_indices: np.ndarray) -> np.ndarray:
    """Estimate prominence for each peak index using nearest higher-bound intervals."""
    n = len(intensities)
    prominences = np.zeros(len(peak_indices), dtype=float)

    for k, peak_idx in enumerate(peak_indices):
        peak_height = intensities[peak_idx]

        left = peak_idx
        while left > 0 and intensities[left - 1] <= peak_height:
            left -= 1

        right = peak_idx
        while right < n - 1 and intensities[right + 1] <= peak_height:
            right += 1

        left_min = float(np.min(intensities[left : peak_idx + 1]))
        right_min = float(np.min(intensities[peak_idx : right + 1]))
        baseline = max(left_min, right_min)
        prominences[k] = max(0.0, float(peak_height - baseline))

    return prominences


def find_dominant_lines(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    count: int | None = None,
    min_separation_nm: float = 1.0,
    min_prominence_ratio: float = 0.03,
) -> list[int]:
    if len(wavelengths) < 3:
        order = np.argsort(intensities)[::-1]
        if count is None:
            return order.tolist()
        return order[:count].tolist()

    peak_indices = _find_local_peak_candidates(intensities)
    prominences = _estimate_peak_prominences(intensities, peak_indices)

    max_signal = float(np.max(intensities))
    min_prominence = max(0.0, min_prominence_ratio * max_signal)
    keep_mask = prominences >= min_prominence

    if np.any(keep_mask):
        peak_indices = peak_indices[keep_mask]
        prominences = prominences[keep_mask]

    # Prefer peaks with higher prominence; use intensity as tiebreaker.
    sort_order = np.lexsort((-intensities[peak_indices], -prominences))
    peak_indices = peak_indices[sort_order]

    selected = []
    for idx in peak_indices:
        wl = wavelengths[idx]
        if any(abs(wl - wavelengths[s]) < min_separation_nm for s in selected):
            continue
        selected.append(int(idx))
        if count is not None and len(selected) == count:
            break

    if count is not None and len(selected) < count:
        remaining = np.argsort(intensities)[::-1]
        for idx in remaining:
            if idx in selected:
                continue
            wl = wavelengths[idx]
            if any(abs(wl - wavelengths[s]) < min_separation_nm for s in selected):
                continue
            selected.append(int(idx))
            if len(selected) == count:
                break

    selected.sort(key=lambda i: intensities[i], reverse=True)
    return selected


def refine_peaks_on_raw(
    candidate_indices: list[int],
    raw_intensities: np.ndarray,
    min_separation_nm: float,
    wavelengths: np.ndarray,
    search_half_window: int = 6,
) -> list[int]:
    """Snap each candidate to a raw-data maximum; center 1.0 flat tops."""
    refined: list[int] = []
    n = len(raw_intensities)

    for idx in candidate_indices:
        start = max(0, idx - search_half_window)
        end = min(n, idx + search_half_window + 1)
        local_raw = raw_intensities[start:end]
        snapped = start + int(np.argmax(local_raw))

        # For saturated peaks at exactly 1.0, use the center of the full plateau.
        if raw_intensities[snapped] == 1.0:
            left = snapped
            while left > 0 and raw_intensities[left - 1] == 1.0:
                left -= 1

            right = snapped
            while right < n - 1 and raw_intensities[right + 1] == 1.0:
                right += 1

            snapped = (left + right) // 2

        if any(abs(wavelengths[snapped] - wavelengths[s]) < min_separation_nm for s in refined):
            continue
        refined.append(snapped)

    refined.sort(key=lambda i: raw_intensities[i], reverse=True)
    return refined


def print_results(wavelengths: np.ndarray, intensities: np.ndarray, indices: list[int]) -> None:
    print(f"Detected {len(indices)} relevant spectral lines:")
    print(f"{'Rank':<6}{'Wavelength (nm)':<20}{'Intensity':<15}")
    for rank, idx in enumerate(indices, start=1):
        print(f"{rank:<6}{wavelengths[idx]:<20.6f}{intensities[idx]:<15.6e}")


def plot_results(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    indices: list[int],
    title: str,
    smoothed: np.ndarray | None = None,
) -> None:
    plt.figure(figsize=(11, 5))
    if smoothed is not None:
        plt.plot(wavelengths, smoothed, lw=1.5, color="tab:green", label="Smoothed (after correction)", alpha=0.8)

    peak_wl = wavelengths[indices]
    if smoothed is not None:
        peak_int = smoothed[indices]
    else:
        peak_int = intensities[indices]
    plt.scatter(peak_wl, peak_int, color="tab:red", zorder=3, label=f"Detected peaks ({len(indices)})")

    for rank, idx in enumerate(indices, start=1):
        y = float(smoothed[idx]) if smoothed is not None else float(intensities[idx])
        plt.annotate(
            f"{rank}",
            xy=(wavelengths[idx], y),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color="tab:red",
        )

    plt.title(title)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.xlim(370, 830)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()


def sample_spectrum_at_wavelengths(
    spectrum_wavelengths_nm: np.ndarray,
    spectrum_intensities: np.ndarray,
    query_wavelengths_nm: np.ndarray,
) -> np.ndarray:
    """Sample spectrum intensity at given wavelengths using linear interpolation.

    Values outside the spectrum range return NaN.
    """
    spectrum_wavelengths_nm = np.asarray(spectrum_wavelengths_nm, dtype=float)
    spectrum_intensities = np.asarray(spectrum_intensities, dtype=float)
    query_wavelengths_nm = np.asarray(query_wavelengths_nm, dtype=float)

    valid = np.isfinite(spectrum_wavelengths_nm) & np.isfinite(spectrum_intensities)
    x = spectrum_wavelengths_nm[valid]
    y = spectrum_intensities[valid]
    if x.size == 0:
        return np.full_like(query_wavelengths_nm, np.nan, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Handle duplicate wavelength samples by averaging their intensities.
    unique_x, inv = np.unique(x, return_inverse=True)
    if unique_x.size != x.size:
        sums = np.bincount(inv, weights=y)
        counts = np.bincount(inv)
        x = unique_x
        y = sums / counts

    return np.interp(query_wavelengths_nm, x, y, left=np.nan, right=np.nan)


def main(
    csv_file: Path,
    response_file: Path | None = None,
    peak_count: int | None = None,
    smooth_window: int = 1,
    min_separation: float = 1.0,
    min_prominence_ratio: float = 0.01,
    save: Path | None = None,
    show_plot: bool = True,
    plot_smoothed: bool = True,
) -> None:
    wavelengths, intensities = load_spectrum_csv(csv_file)

    # Apply response curve correction to the raw spectrum first (if provided),
    # then apply smoothing to the corrected spectrum.
    corrected = intensities.copy()
    if response_file is not None and response_file.exists():
        response_wl, response_values = load_response_curve(response_file)
        corrected = correct_spectrum_by_response(
            wavelengths, corrected, response_wl, response_values
        )
        print(f"Applied response correction from: {response_file.name}")

    smoothed = moving_average(corrected, smooth_window)
    
    candidate_indices = find_dominant_lines(
        wavelengths,
        smoothed,
        count=peak_count,
        min_separation_nm=min_separation,
        min_prominence_ratio=min_prominence_ratio,
    )
    indices = refine_peaks_on_raw(
        candidate_indices,
        smoothed,
        min_separation_nm=min_separation,
        wavelengths=wavelengths,
    )

    # Show results using the smoothed/corrected values
    # Compute shift so that the 6th detected peak is placed at 486.1 nm
    shift_offset = 0.0
    if len(indices) >= 6:
        peak7_idx = indices[6]
        shift_offset = float(wavelengths[peak7_idx] - 486.1)
        print(f"Shifting wavelengths by {shift_offset:.6f} nm (peak7 at {wavelengths[peak7_idx]:.6f} -> 486.1)")
    else:
        print("Less than 6 peaks detected — no wavelength shift applied.")

    wavelengths_shifted = wavelengths - shift_offset

    print_results(wavelengths_shifted, smoothed, indices)

    title = f"Dominant spectral lines: {csv_file.name}"
    plot_results(
        wavelengths_shifted, smoothed, indices, title,
        smoothed=smoothed if plot_smoothed else None
    )

    if save is not None:
        plt.savefig(save, dpi=160)
        print(f"Saved plot to: {save}")

    if show_plot:
        plt.show()


    HI_lines = [434.34, 486.1]
    HI_Ag = [1.2652e+08, 2.6942e+08]
    HI_E = [13.0545017, 12.7485393]

    CII_lines = [426.66, 723.59, 589.1, 514.03, 407]
    CII_Ag = [1.87e+09, 2.51e+08, 1.26e+08, 1.51e+08, 6.04e+08]
    CII_E = [20.950654, 18.045987, 20.150477, 23.114048, 27.412155]

    OI_lines = [776.94, 615.394617, 533.58]
    OI_Ag = [5.53e+08, 1.91e+08, 6.778e+07]
    OI_E = [10.7409314, 12.7537, 13.0661186]

    FI_lines = [685.58, 690.23, 731.08, 775.6, 703.73, 634.69, 641.37, 740]
    FI_Ag = [3.90e+08, 2.11e+08, 9.00e+07, 1.94e+08, 1.69e+08, 9.28e+07, 5.44e+07, 2.12e+08]
    FI_E = [14.504588, 14.526486, 14.680382, 14.583383, 14.746284, 14.683178, 14.683178, 14.371989]
    #FI_lines = [685.58, 690.23, 775.6]
    #FI_Ag = [5.57e+08, 2.93e+08, 1.94e+08]
    #FI_E = [14.504588, 14.526486, 14.583383]

    SI_lines = [685.58, 690.23, 775.6]
    SI_Ag = [5.57e+08, 2.93e+08, 1.94e+08]
    SI_E = [14.504588, 14.526486, 14.583383]

    SiII_lines = [505.5, 634.7, 636.5]
    SiII_Ag = [8.7e+08, 2.34e+08, 1.36e+08]
    SiII_E = [12.524588, 10.07, 10.066]

    FII_lines = [444.68, 429.95]
    FII_Ag = [2.12e+09, 1.2e+09]
    FII_E = [31.559698, 29.548333]

    # Run Boltzmann plots for all ions
    print("\n" + "="*60)
    print("BOLTZMANN PLOT ANALYSIS")
    print("="*60)
    
    ions = [
        ("HI", HI_lines, HI_Ag, HI_E),
        ("CII", CII_lines, CII_Ag, CII_E),
        ("OI", OI_lines, OI_Ag, OI_E),
        ("FI", FI_lines, FI_Ag, FI_E),
        ("SI", SI_lines, SI_Ag, SI_E),
        ("SiII", SiII_lines, SiII_Ag, SiII_E),
        ("FII", FII_lines, FII_Ag, FII_E),
    ]
    
    for ion_name, lines, Ag, E in ions:
        print(f"\n{ion_name}:")
        fig, ax = plt.subplots(figsize=(7, 5))
        try:
            boltzmann_plot(
                line_wavelengths_nm=lines,
                spectrum_wavelengths_nm=wavelengths_shifted,
                spectrum_intensities=smoothed,
                Ag=Ag,
                E=E,
                ax=ax,
                show=False,
            )
        except ValueError as exc:
            print(f"Skipping {ion_name} Boltzmann plot: {exc}")
            plt.close(fig)
            continue
        save_path = Path(f"Boltzmann_{ion_name}.png")
        plt.savefig(save_path, dpi=160)
        print(f"Saved Boltzmann plot to: {save_path}")
        plt.close()


def boltzmann_plot(
    line_wavelengths_nm: np.ndarray,
    spectrum_wavelengths_nm: np.ndarray,
    spectrum_intensities: np.ndarray,
    Ag: np.ndarray,
    E: np.ndarray,
    ax: plt.Axes | None = None,
    show: bool = True,
) -> tuple[float, float]:
    """Compute Boltzmann plot from line data and perform linear regression.

    Intensities are always derived by sampling the provided spectrum at the
    given line wavelengths.

    Args:
        line_wavelengths_nm: array-like of wavelengths (lambda) in nm.
        spectrum_wavelengths_nm: spectrum wavelength axis in nm.
        spectrum_intensities: spectrum intensity values (same length as spectrum_wavelengths_nm).
        Ag: array-like of A*g values for each line.
        E: array-like of excitation energies (eV) for each line.
        ax: optional Matplotlib Axes to draw on; creates one if None.
        show: whether to call `plt.show()` after plotting.

    Returns:
        (slope, intercept) of the fitted line (log_val = slope * E + intercept).
    """
    line_wavelengths_nm = np.asarray(line_wavelengths_nm, dtype=float)
    Ag = np.asarray(Ag, dtype=float)
    E = np.asarray(E, dtype=float)

    if not (line_wavelengths_nm.shape == Ag.shape == E.shape):
        raise ValueError("`line_wavelengths_nm`, `Ag`, and `E` must have the same shape")

    intensities = sample_spectrum_at_wavelengths(
        spectrum_wavelengths_nm=spectrum_wavelengths_nm,
        spectrum_intensities=spectrum_intensities,
        query_wavelengths_nm=line_wavelengths_nm,
    )

    # Compute log values for Boltzmann plot: ln(I * lambda / (A*g))
    print("Boltzmann input wavelengths:", line_wavelengths_nm)
    print("Boltzmann input intensities:", intensities)
    ratio = intensities * line_wavelengths_nm / Ag
    
    ratio = np.where(np.isfinite(ratio) & (ratio > 0.0), ratio, np.nan)
    log_val = np.log(ratio)

    finite_mask = np.isfinite(log_val) & np.isfinite(E)
    finite_count = int(np.sum(finite_mask))
    if finite_count < 2:
        raise ValueError("not enough finite line intensities inside spectrum range")

    # Fit the line for all valid points; estimate covariance only when there are
    # enough points to support an uncertainty calculation.
    coeffs = np.polyfit(E[finite_mask], log_val[finite_mask], 1)
    if finite_count > 2:
        _, pcov = np.polyfit(E[finite_mask], log_val[finite_mask], 1, cov=True)
    else:
        pcov = None
    slope, intercept = coeffs
    if pcov is not None:
        slope_std = float(np.sqrt(pcov[0, 0])) if np.isfinite(pcov[0, 0]) and pcov[0, 0] >= 0 else float("nan")
        intercept_std = float(np.sqrt(pcov[1, 1])) if np.isfinite(pcov[1, 1]) and pcov[1, 1] >= 0 else float("nan")
    else:
        slope_std = float("nan")
        intercept_std = float("nan")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        created_fig = True

    ax.scatter(E, log_val, color="tab:blue", label="data")
    for e_value, log_value, wavelength_nm in zip(E[finite_mask], log_val[finite_mask], line_wavelengths_nm[finite_mask]):
        ax.annotate(
            f"{wavelength_nm:.2f} nm",
            xy=(e_value, log_value),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="tab:blue",
        )
    E_line = np.linspace(np.nanmin(E[finite_mask]), np.nanmax(E[finite_mask]), 200)
    fit_line = slope * E_line + intercept
    ax.plot(E_line, fit_line, color="tab:red", label=f"fit: y={slope:.4e}x+{intercept:.4e}")
    if pcov is not None:
        fit_var = (
            (E_line ** 2) * pcov[0, 0]
            + 2.0 * E_line * pcov[0, 1]
            + pcov[1, 1]
        )
        fit_std = np.sqrt(np.maximum(fit_var, 0.0))
        ax.fill_between(E_line, fit_line - fit_std, fit_line + fit_std, color="tab:red", alpha=0.15, label="fit ±1σ")
    ax.set_xlabel("E (eV)")
    ax.set_ylabel("ln(I * lambda / A_g)")
    ax.grid(alpha=0.25)
    ax.legend()

    # Compute temperature from slope. For a Boltzmann plot slope = -1/(k_B * T).
    k_B_eV_per_K = 8.617333262145e-5  # Boltzmann constant in eV/K
    if slope == 0 or np.isnan(slope):
        T_eV = float('nan')
        T_K = float('nan')
        T_eV_std = float('nan')
        T_K_std = float('nan')
    else:
        # Use physical relation: slope = -1/(k_B * T)
        # therefore T_K = -1 / (k_B * slope) and T_eV = k_B * T_K = -1 / slope
        T_K = -1.0 / (k_B_eV_per_K * slope)
        T_eV = -1.0 / slope
        if np.isfinite(slope_std):
            T_K_std = abs(1.0 / (k_B_eV_per_K * slope ** 2)) * slope_std
            T_eV_std = abs(1.0 / (slope ** 2)) * slope_std
        else:
            T_K_std = float('nan')
            T_eV_std = float('nan')

    # Annotate temperature on the plot
    try:
        if np.isfinite(T_K) and np.isfinite(T_eV):
            if np.isfinite(T_K_std) and np.isfinite(T_eV_std):
                temp_text = f"T = {T_K:.0f} ± {T_K_std:.0f} K ({T_eV:.4f} ± {T_eV_std:.4f} eV)"
                print(f"Temperature: {T_K:.0f} ± {T_K_std:.0f} K ({T_eV:.4f} ± {T_eV_std:.4f} eV)")
            else:
                temp_text = f"T = {T_K:.0f} K ({T_eV:.4f} eV)"
                print(f"Temperature: {T_K:.0f} K ({T_eV:.4f} eV)")
            ax.annotate(temp_text, xy=(0.05, 0.95), xycoords="axes fraction",
                        fontsize=10, verticalalignment="top")
            if np.isfinite(slope_std):
                print(f"Fit covariance slope σ: {slope_std:.4e}, intercept σ: {intercept_std:.4e}")
    except Exception:
        pass

    if show and created_fig:
        plt.show()

    return float(slope), float(intercept)

if __name__ == "__main__":
    # Edit these parameters directly instead of passing command-line CSV arguments.
    main(
        csv_file=Path("S2332.csv"),
        response_file=Path("response_curve.csv"),
        peak_count=None,
        smooth_window=1,
        min_separation=1.0,
        min_prominence_ratio=0.015,
        save=Path("S2332_dominant_lines.png"),
        show_plot=True,
    )