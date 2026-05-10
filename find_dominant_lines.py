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


def main(
    csv_file: Path,
    response_file: Path | None = None,
    peak_count: int | None = None,
    smooth_window: int = 7,
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
        peak6_idx = indices[5]
        shift_offset = float(wavelengths[peak6_idx] - 486.1)
        print(f"Shifting wavelengths by {shift_offset:.6f} nm (peak6 at {wavelengths[peak6_idx]:.6f} -> 486.1)")
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


if __name__ == "__main__":
    # Edit these parameters directly instead of passing command-line CSV arguments.
    main(
        csv_file=Path("S2332.csv"),
        response_file=Path("response_curve.csv"),
        peak_count=None,
        smooth_window=7,
        min_separation=1.0,
        min_prominence_ratio=0.015,
        save=Path("S2332_dominant_lines.png"),
        show_plot=True,
    )
