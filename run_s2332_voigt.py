import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def load_functions_from_file(module_path: Path, function_names: set[str]) -> dict[str, object]:
    """Load selected functions from a Python file without executing top-level analysis code."""
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))

    kept_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            kept_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in function_names:
            kept_nodes.append(node)

    mini_module = ast.Module(body=kept_nodes, type_ignores=[])
    compiled = compile(mini_module, str(module_path), "exec")
    namespace: dict[str, object] = {}
    exec(compiled, namespace, namespace)

    missing = function_names - namespace.keys()
    if missing:
        raise RuntimeError(f"Missing expected functions in {module_path.name}: {sorted(missing)}")

    return {name: namespace[name] for name in function_names}


def load_s2332_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse Thorlabs CSV that contains a metadata header and a [Data] section."""
    lines = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()

    try:
        data_start = lines.index("[Data]") + 1
    except ValueError as exc:
        raise RuntimeError("Could not find [Data] section in CSV file") from exc

    data_lines = [ln for ln in lines[data_start:] if ln.strip() and ";" in ln]
    data = np.genfromtxt(data_lines, delimiter=";")

    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError("CSV data section does not look like 2-column numeric data")

    wavelength_nm = data[:, 0]
    intensity = data[:, 1]
    return wavelength_nm, intensity


def fit_voigt_on_peak(wavelength_nm: np.ndarray, intensity: np.ndarray, voigt_func):
    baseline = float(np.median(intensity))
    peak_idx = int(np.argmax(intensity))
    center_guess = float(wavelength_nm[peak_idx])

    # Fit on a local window around the strongest peak for stability.
    window_nm = 3.0
    mask = (wavelength_nm >= center_guess - window_nm) & (wavelength_nm <= center_guess + window_nm)
    x_fit = wavelength_nm[mask]
    y_fit = intensity[mask]

    amp_guess = float(np.max(y_fit) - np.min(y_fit))
    sigma_guess = 0.05
    gamma_guess = 0.05

    p0 = [amp_guess, center_guess, sigma_guess, gamma_guess, baseline]
    bounds = (
        [0.0, x_fit.min(), 1e-9, 1e-9, -np.inf],
        [np.inf, x_fit.max(), np.inf, np.inf, np.inf],
    )

    popt, pcov = curve_fit(voigt_func, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=20000)
    y_model = voigt_func(x_fit, *popt)
    return x_fit, y_fit, y_model, popt, pcov


def compute_signal_xlim(
    wavelength_nm: np.ndarray,
    intensity: np.ndarray,
    snr_threshold: float = 4.0,
    padding_nm: float = 2.0,
) -> tuple[float, float]:
    """Return x-limits that bracket signal points above the estimated noise floor."""
    baseline = float(np.median(intensity))
    mad = float(np.median(np.abs(intensity - baseline)))
    noise_sigma = 1.4826 * mad

    full_left = float(wavelength_nm.min())
    full_right = float(wavelength_nm.max())
    if noise_sigma <= 0.0 or not np.isfinite(noise_sigma):
        return full_left, full_right

    threshold = baseline + snr_threshold * noise_sigma
    signal_idx = np.where(intensity > threshold)[0]
    if signal_idx.size == 0:
        return full_left, full_right

    left = float(wavelength_nm[signal_idx[0]] - padding_nm)
    right = float(wavelength_nm[signal_idx[-1]] + padding_nm)
    return max(left, full_left), min(right, full_right)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    source_module = project_dir / "H_Voigt_Yair.py"
    csv_file = project_dir / "S2332.csv"
    h_alpha_nm = 656.28
    h_beta_nm = 486.13
    h_gamma_nm = 434.05

    functions = load_functions_from_file(source_module, {"voigt"})
    voigt = functions["voigt"]

    wavelength_nm, intensity = load_s2332_csv(csv_file)
    x_fit, y_fit, y_model, popt, _ = fit_voigt_on_peak(wavelength_nm, intensity, voigt)

    amplitude, center, sigma, gamma, offset = popt

    print("Voigt fit results on S2332.csv")
    print(f"amplitude = {amplitude:.6g}")
    print(f"center (nm) = {center:.6f}")
    print(f"sigma (nm) = {sigma:.6f}")
    print(f"gamma (nm) = {gamma:.6f}")
    print(f"offset = {offset:.6g}")

    plt.figure(figsize=(9, 5))
    plt.plot(wavelength_nm, intensity, label="S2332 data", alpha=0.75)
    plt.plot(x_fit, y_model, "r--", linewidth=2, label="Voigt fit (local peak)")
    plt.axvline(h_alpha_nm, color="tab:green", linestyle=":", linewidth=1.5, label=f"H alpha ({h_alpha_nm:.2f} nm)")
    plt.axvline(h_beta_nm, color="tab:purple", linestyle=":", linewidth=1.5, label=f"H beta ({h_beta_nm:.2f} nm)")
    plt.axvline(h_gamma_nm, color="tab:orange", linestyle=":", linewidth=1.5, label=f"H gamma ({h_gamma_nm:.2f} nm)")
    x_left, x_right = compute_signal_xlim(wavelength_nm, intensity)
    plt.xlim(x_left, x_right)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title("S2332 Peak Fit using voigt() from H_Voigt_Yair.py")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = project_dir / "S2332_voigt_fit.png"
    plt.savefig(out_path, dpi=160)
    print(f"Saved plot: {out_path.name}")

    plt.figure(figsize=(9, 5))
    plt.plot(wavelength_nm, intensity, label="S2332 data", alpha=0.75)
    plt.xlim(x_left, x_right)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title("S2332 Spectrum")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path_clean = project_dir / "S2332_spectrum_only.png"
    plt.savefig(out_path_clean, dpi=160)
    print(f"Saved plot: {out_path_clean.name}")


if __name__ == "__main__":
    main()
