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


def print_spectrums(
    experiments: list[tuple[str, int, list[tuple[str, list[float], list[float], list[float]]]]],
    experiments_ions: list[tuple[str, list[str]]],
    response_file: Path | None = None,
    save: Path | None = None,
    xlim: tuple[float, float] | None = None,
    show_plot: bool = True,
) -> None:
    """Plot all requested experiments on the same graph.

    Each experiment is expected to have a matching spectrum file in the current directory or in a
    `spectra/` subfolder. The spectrum is optionally corrected by a response curve and shifted by
    the requested number of nm. For the requested ions, the corresponding wavelengths are marked on
    the graph with dashed vertical lines and colors specific to each (experiment, ion) pair.
    """
    response_wavelengths = None
    response_values = None
    if response_file is not None:
        response_wavelengths, response_values = load_response_curve(response_file)

    requested_ions_by_experiment = {exp_id: set(ions) for exp_id, ions in experiments_ions}
    fallback_colors = plt.rcParams.get("axes.prop_cycle").by_key().get("color", ["tab:blue"])
    ion_marker_colors: dict[tuple[str, str], str] = {}

    fig, ax = plt.subplots(figsize=(11, 7))

    requested_experiment_ids = {exp_id for exp_id, _, _ in experiments}
    if experiments_ions:
        requested_experiment_ids = requested_experiment_ids.intersection(
            {exp_id for exp_id, _ in experiments_ions}
        )

    for exp_id, shift, ion_entries in experiments:
        if experiments_ions and exp_id not in requested_experiment_ids:
            continue

        candidate_paths = [
            Path(f"{exp_id}.csv"),
            Path(f"{exp_id}.txt"),
            Path(f"{exp_id}.dat"),
            Path("yahli_experiments") / f"{exp_id}.csv",
            Path("yahli_experiments") / f"{exp_id}.txt",
            Path("yahli_experiments") / f"{exp_id}.dat",
        ]

        spectrum_file = next((p for p in candidate_paths if p.exists()), None)
        if spectrum_file is None:
            print(f"Skipping experiment {exp_id}: no spectrum file found.")
            continue

        wavelengths, intensities = load_spectrum_csv(spectrum_file)

        if response_file is not None:
            intensities = correct_spectrum_by_response(
                wavelengths,
                intensities,
                response_wavelengths,
                response_values,
            )

        shifted_wavelengths = np.asarray(wavelengths, dtype=float) + float(shift)
        ax.plot(shifted_wavelengths, intensities, label=exp_id, alpha=0.9)
        min_wl = float(np.min(shifted_wavelengths))
        max_wl = float(np.max(shifted_wavelengths))

        requested_ions = requested_ions_by_experiment.get(exp_id, set())

        for ion_name, ion_wavelengths, _, _ in ion_entries:
            if ion_name not in requested_ions:
                continue

            marker_key = (exp_id, ion_name)
            if marker_key not in ion_marker_colors:
                color_index = len(ion_marker_colors) % len(fallback_colors)
                ion_marker_colors[marker_key] = fallback_colors[color_index]
            marker_color = ion_marker_colors[marker_key]

            for wl in ion_wavelengths:
                mark_wl = float(wl)
                ax.axvline(mark_wl, color="black", linestyle="--", linewidth=0.9, alpha=0.7)

                if not (min_wl <= mark_wl <= max_wl):
                    continue

                marker_intensity = float(np.interp(mark_wl, shifted_wavelengths, intensities))
                ax.scatter([mark_wl], [marker_intensity], color=marker_color, marker="o", s=18, zorder=5)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Corrected intensity")
    ax.set_title("Experiments after response correction and wavelength shift")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.grid(alpha=0.25)
    ax.legend()

    if save is not None:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
        print(f"Saved plot to: {save_path}")

    if show_plot:
        plt.show()


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


def _boltzmann_fit_on_axis(
    line_wavelengths_nm: np.ndarray,
    spectrum_wavelengths_nm: np.ndarray,
    spectrum_intensities: np.ndarray,
    Ag: np.ndarray,
    E: np.ndarray,
    ax: plt.Axes,
) -> tuple[float, float]:
    """Compute Boltzmann fit and draw it on an existing axis."""
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

    ratio = intensities * line_wavelengths_nm / Ag
    ratio = np.where(np.isfinite(ratio) & (ratio > 0.0), ratio, np.nan)
    log_val = np.log(ratio)

    finite_mask = np.isfinite(log_val) & np.isfinite(E)
    finite_count = int(np.sum(finite_mask))
    if finite_count < 2:
        raise ValueError("not enough finite line intensities inside spectrum range")

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

    ax.scatter(E[finite_mask], log_val[finite_mask], color="tab:blue", label="data")
    E_line = np.linspace(np.nanmin(E[finite_mask]), np.nanmax(E[finite_mask]), 200)
    fit_line = slope * E_line + intercept
    ax.plot(E_line, fit_line, color="tab:red", label=f"fit: y={slope:.4e}x+{intercept:.4e}")
    if pcov is not None:
        fit_var = (E_line ** 2) * pcov[0, 0] + 2.0 * E_line * pcov[0, 1] + pcov[1, 1]
        fit_std = np.sqrt(np.maximum(fit_var, 0.0))
        ax.fill_between(E_line, fit_line - fit_std, fit_line + fit_std, color="tab:red", alpha=0.15, label="fit ±1σ")

    ax.set_xlabel("E (eV)")
    ax.set_ylabel("ln(I * lambda / A_g)")
    ax.grid(alpha=0.25)
    ax.legend()

    k_B_eV_per_K = 8.617333262145e-5
    if slope == 0 or np.isnan(slope):
        T_eV = float("nan")
        T_K = float("nan")
        T_eV_std = float("nan")
        T_K_std = float("nan")
    else:
        T_K = -1.0 / (k_B_eV_per_K * slope)
        T_eV = -1.0 / slope
        if np.isfinite(slope_std):
            T_K_std = abs(1.0 / (k_B_eV_per_K * slope ** 2)) * slope_std
            T_eV_std = abs(1.0 / (slope ** 2)) * slope_std
        else:
            T_K_std = float("nan")
            T_eV_std = float("nan")

    if np.isfinite(T_K) and np.isfinite(T_eV):
        if np.isfinite(T_K_std) and np.isfinite(T_eV_std):
            temp_text = f"T = {T_K:.0f} ± {T_K_std:.0f} K ({T_eV:.4f} ± {T_eV_std:.4f} eV)"
        else:
            temp_text = f"T = {T_K:.0f} K ({T_eV:.4f} eV)"
        ax.annotate(temp_text, xy=(0.05, 0.95), xycoords="axes fraction", fontsize=10, verticalalignment="top")

    if np.isfinite(slope_std):
        print(f"Fit covariance slope σ: {slope_std:.4e}, intercept σ: {intercept_std:.4e}")

    print(f"Temperature: {T_K:.0f} K ({T_eV:.4f} eV)")
    return float(slope), float(intercept)


def boltzmann_plot(
    experiments: list[tuple[str, int, list[tuple[str, list[float], list[float], list[float]]]]],
    experiments_ions: list[tuple[str, list[str]]],
    response_file: Path | None = None,
    save: Path | None = None,
    show: bool = True,
) -> None:
    """Generate and save one Boltzmann plot per selected ion.

    Selection follows `experiments_ions`. For an experiment, an empty ion list means
    "all ions in that experiment".
    """
    response_wavelengths = None
    response_values = None
    if response_file is not None:
        response_wavelengths, response_values = load_response_curve(response_file)

    requested_ions_by_experiment = {exp_id: set(ions) for exp_id, ions in experiments_ions}

    for exp_id, shift, ion_entries in experiments:
        if experiments_ions and exp_id not in requested_ions_by_experiment:
            continue

        candidate_paths = [
            Path(f"{exp_id}.csv"),
            Path(f"{exp_id}.txt"),
            Path(f"{exp_id}.dat"),
            Path("yahli_experiments") / f"{exp_id}.csv",
            Path("yahli_experiments") / f"{exp_id}.txt",
            Path("yahli_experiments") / f"{exp_id}.dat",
        ]
        spectrum_file = next((p for p in candidate_paths if p.exists()), None)
        if spectrum_file is None:
            print(f"Skipping experiment {exp_id}: no spectrum file found.")
            continue

        spectrum_wavelengths_nm, spectrum_intensities = load_spectrum_csv(spectrum_file)
        if response_file is not None:
            spectrum_intensities = correct_spectrum_by_response(
                spectrum_wavelengths_nm,
                spectrum_intensities,
                response_wavelengths,
                response_values,
            )

        shifted_spectrum_wavelengths_nm = np.asarray(spectrum_wavelengths_nm, dtype=float) + float(shift)
        requested_ions = requested_ions_by_experiment.get(exp_id, set())

        for ion_name, ion_wavelengths, Ag, E in ion_entries:
            if requested_ions and ion_name not in requested_ions:
                continue

            fig, ax = plt.subplots(figsize=(7, 5))
            # Ion wavelengths are expected to already be shift-corrected.
            line_wavelengths_nm = np.asarray(ion_wavelengths, dtype=float)
            try:
                _boltzmann_fit_on_axis(
                    line_wavelengths_nm=line_wavelengths_nm,
                    spectrum_wavelengths_nm=shifted_spectrum_wavelengths_nm,
                    spectrum_intensities=spectrum_intensities,
                    Ag=np.asarray(Ag, dtype=float),
                    E=np.asarray(E, dtype=float),
                    ax=ax,
                )
                ax.set_title(f"Boltzmann plot: {exp_id} - {ion_name}")
            except ValueError as exc:
                plt.close(fig)
                print(f"Skipping Boltzmann plot for {exp_id}/{ion_name}: {exc}")
                continue

            if save is not None:
                save_base = Path(save)
                if save_base.suffix:
                    root_dir = save_base.parent
                    file_stem = save_base.stem
                    file_suffix = save_base.suffix
                else:
                    root_dir = save_base
                    file_stem = "Boltzmann"
                    file_suffix = ".png"

                out_path = root_dir / exp_id / f"{file_stem}_{exp_id}_{ion_name}{file_suffix}"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_path, dpi=160)
                print(f"Saved Boltzmann plot to: {out_path}")

            if show:
                plt.show()
            else:
                plt.close(fig)


if __name__ == "__main__":

    #experiment - (ID, shift, ions)
    #ion - (name, wavelengths, Ag, E)
    experiments = [
            ("S2318", -1.9, [
                ("CII", [426.65, 657.9, 723.36, 589.09], [3.19e9, 2.19e8, 3.9e8, 2.1e8], [20.950654, 16.33, 18.045987, 20.15]), #expected temp 1.6 eV
                ("OI", [615.39, 777.16], [1.91e+08, 5.53e+08], [12.7537, 10.7409]),
                ("FI", [685.65, 690.2, 739.94, 775.3, 703.72, 624.1, 731.06, 733.18], [5.57e+08, 2.93e+08, 2.12e+08, 1.94e+08, 1.69e+08, 1.16e+08, 9.00e+07, 9.68e+07], [14.504588, 14.526486, 14.371989, 14.583383, 14.746284, 14.683178, 14.680382, 14.387235]),
                #("FI", [739.94, 703.72, 733.18], [2.12e+08, 1.69e+08, 9.68e+07], [14.371989, 14.746284, 14.387235]),
                ("NII", [500.36, 567.7], [2e+09, 4.31e+08], [23.141959, 20.665518]),
                ("HI", [656.3, 486.08], [1.575e+09, 2.6942e+08], [12.0875, 12.7485393]) #expected temp 0.5 eV
                ]),
            ("S2319", -1.85, [ #probably contains Na and B
                ("CII", [426.7, 658, 723.4, 589.1, 514.5, 461.7], [3.19e9, 2.19e8, 3.9e8, 2.1e8, 5.32e8, 3.6e9], [20.950654, 16.33, 18.045987, 20.15, 23.118, 27.47]), #expected temp 1.6 eV. 589nm line mixed with NaI?
                ("OI", [615.43, 777.2], [1.91e+08, 5.53e+08], [12.7537, 10.7409]),
                #("FI", [685.7, 690.2, 740.0, 775.32, 703.66, 624.12, 683.5, 742.6, 641.4, 731.15, 733.2, 779.56], [5.57e+08, 2.93e+08, 2.12e+08, 1.94e+08, 1.69e+08, 1.16e+08, 8.72e+07, 7.86e+07, 5.44e+07, 9.00e+07, 9.68e+07, 1.01e+08], [14.504588, 14.526486, 14.371989, 14.583383, 14.746284, 14.683178, 14.544408, 14.399969, 14.683178, 14.680382, 14.387235, 14.614380]), #some lines mixed with BI around 680nm
                ("FI", [740.0, 775.32, 703.66, 742.6, 641.4, 733.2], [2.12e+08, 1.94e+08, 1.69e+08, 7.86e+07, 5.44e+07, 9.68e+07], [14.371989, 14.583383, 14.746284, 14.399969, 14.683178, 14.387235]),
                ("FII", [444.7, 429.8], [4.93e+09, 1.2e+09], [31.559723, 29.548333]), #expected temp 2 eV
                ("NII", [500.4, 567.8, 463.0], [2e+09, 4.31e+08, 3.74e+08], [23.141959, 20.665518, 21.159916]), #568nm line mixed with NaI?
                ("HI", [656.2, 486.14, 434.16, 410.3], [1.575e+09, 2.6942e+08, 2.342e+08, 1.386e+08], [12.0875, 12.7485393, 13.0545, 13.22070369]) #expected temp 0.5 eV. 434nm mixed with NaII?
                #("HI", [656.2, 434.16, 410.3], [1.575e+09, 2.342e+08, 1.386e+08], [12.0875, 13.0545, 13.22070369])
                ]),
            ("S2325", -1.88, [
                ("CII", [426.67, 658, 723.37, 588.9, 514.5, 461.7], [3.19e9, 2.19e8, 3.9e8, 2.1e8, 5.32e8, 3.6e9], [20.950654, 16.33, 18.045987, 20.15, 23.118, 27.47]), #expected temp 1.6 eV
                ("OI", [615.4, 777.2], [1.91e+08, 5.53e+08], [12.7537, 10.7409]),
                ("FI", [685.64, 690.24, 740.0, 775.3, 703.66, 624.12, 683.5, 742.6, 641.35, 731.12, 733.2, 779.54], [5.57e+08, 2.93e+08, 2.12e+08, 1.94e+08, 1.69e+08, 1.16e+08, 8.72e+07, 7.86e+07, 5.44e+07, 9.00e+07, 9.68e+07, 1.01e+08], [14.504588, 14.526486, 14.371989, 14.583383, 14.746284, 14.683178, 14.544408, 14.399969, 14.683178, 14.680382, 14.387235, 14.614380]),
                ("FII", [444.7, 429.77], [4.93e+09, 1.2e+09], [31.559723, 29.548333]), #expected temp 2 eV
                ("NII", [500.39, 567.87, 462.95], [2e+09, 4.31e+08, 3.74e+08], [23.141959, 20.665518, 21.159916]),
                ("HI", [656.2, 486.1, 434.14, 410.26], [1.575e+09, 2.6942e+08, 2.342e+08, 1.386e+08], [12.0875, 12.7485393, 13.0545, 13.22070369]) #expected temp 0.5 eV
                ]),
            ("S2326", -1.88, [
                ("CII", [426.67, 658, 723.37, 588.9, 514.47, 461.72], [3.19e9, 2.19e8, 3.9e8, 2.1e8, 5.32e8, 3.6e9], [20.950654, 16.33, 18.045987, 20.15, 23.118, 27.47]), #expected temp 1.6 eV
                ("OI", [615.4, 777.2], [1.91e+08, 5.53e+08], [12.7537, 10.7409]),
                ("FI", [685.64, 690.24, 740.0, 775.3, 703.66, 624.12, 683.5, 742.6, 641.35, 731.12, 733.2, 779.54], [5.57e+08, 2.93e+08, 2.12e+08, 1.94e+08, 1.69e+08, 1.16e+08, 8.72e+07, 7.86e+07, 5.44e+07, 9.00e+07, 9.68e+07, 1.01e+08], [14.504588, 14.526486, 14.371989, 14.583383, 14.746284, 14.683178, 14.544408, 14.399969, 14.683178, 14.680382, 14.387235, 14.614380]),
                ("FII", [444.7, 429.77], [4.93e+09, 1.2e+09], [31.559723, 29.548333]), #expected temp 2 eV
                ("NII", [500.39, 567.87, 462.95], [2e+09, 4.31e+08, 3.74e+08], [23.141959, 20.665518, 21.159916]),
                ("HI", [656.2, 486.1, 434.14, 410.26], [1.575e+09, 2.6942e+08, 2.342e+08, 1.386e+08], [12.0875, 12.7485393, 13.0545, 13.22070369]) #expected temp 0.5 eV
                ]),
            ("S2330", -1.88, [
                ("CII", [426.67, 658.2, 723.37, 588.9, 514.47, 461.72, 437.2, 441.17], [3.19e9, 2.19e8, 3.9e8, 2.1e8, 5.32e8, 3.6e9, 1.052e9, 2.72e9], [20.950654, 16.33, 18.045987, 20.15, 23.118, 27.47, 27.4909, 27.412]), #expected temp 1.6 eV
                ("OI", [615.4, 777.2], [1.91e+08, 5.53e+08], [12.7537, 10.7409]),
                ("FI", [685.64, 690.24, 740.0, 775.3, 703.66, 624.12, 683.5, 742.6, 641.35, 731.12, 733.2, 779.78, 720.3, 748.22, 755, 757.1, 760.2],
                [5.57e+08, 2.93e+08, 2.12e+08, 1.94e+08, 1.69e+08, 1.16e+08, 8.72e+07, 7.86e+07, 5.44e+07, 9.00e+07, 9.68e+07, 1.01e+08, 2.60e+07, 5.3e+07, 5.28e+07, 3.6e+07],
                [14.504588, 14.526486, 14.371989, 14.583383, 14.746284, 14.683178, 14.544408, 14.399969, 14.683178, 14.680382, 14.387235, 14.614380, 14.387235, 14.371989, 14.387235, 14.614380]),
                ("FII", [444.7, 429.77], [4.93e+09, 1.2e+09], [31.559723, 29.548333]), #expected temp 2 eV
                ("NII", [500.39, 567.87, 462.95], [2e+09, 4.31e+08, 3.74e+08], [23.141959, 20.665518, 21.159916]),
                #("HI", [656.2, 486.1, 434.14, 410.26], [1.575e+09, 2.6942e+08, 2.342e+08, 1.386e+08], [12.0875, 12.7485393, 13.0545, 13.22070369]) #expected temp 0.5 eV
                ("HI", [486.1, 434.14, 410.26], [2.6942e+08, 2.342e+08, 1.386e+08], [12.7485393, 13.0545, 13.22070369])
                ]),
            ("S2331", -1.89, [
                ("CII", [426.67, 658, 723.37, 588.9, 514.47, 461.72], [3.19e9, 2.19e8, 3.9e8, 2.1e8, 5.32e8, 3.6e9], [20.950654, 16.33, 18.045987, 20.15, 23.118, 27.47]), #expected temp 1.6 eV
                ("OI", [615.4, 777.2], [1.91e+08, 5.53e+08], [12.7537, 10.7409]),
                ("FI", [685.64, 690.24, 740.0, 775.3, 703.66, 624.12, 683.5, 742.6, 641.35, 731.12, 733.2, 779.54], [5.57e+08, 2.93e+08, 2.12e+08, 1.94e+08, 1.69e+08, 1.16e+08, 8.72e+07, 7.86e+07, 5.44e+07, 9.00e+07, 9.68e+07, 1.01e+08], [14.504588, 14.526486, 14.371989, 14.583383, 14.746284, 14.683178, 14.544408, 14.399969, 14.683178, 14.680382, 14.387235, 14.614380]),
                ("FII", [444.7, 429.77], [4.93e+09, 1.2e+09], [31.559723, 29.548333]), #expected temp 2 eV
                ("NII", [500.39, 567.87, 462.95], [2e+09, 4.31e+08, 3.74e+08], [23.141959, 20.665518, 21.159916]),
                ("HI", [656.2, 486.1, 434.14, 410.26], [1.575e+09, 2.6942e+08, 2.342e+08, 1.386e+08], [12.0875, 12.7485393, 13.0545, 13.22070369]) #expected temp 0.5 eV
                ]),
            ("S2336", -1.8, [
                ("CII", [426.67, 658.35, 723.37, 589, 514.5, 461.7], [3.19e9, 2.19e8, 3.9e8, 2.1e8, 5.32e8, 3.6e9], [20.950654, 16.33, 18.045987, 20.15, 23.118, 27.47]), #expected temp 1.6 eV
                ("OI", [615.4, 777.2], [1.91e+08, 5.53e+08], [12.7537, 10.7409]),
                ("FI", [685.64, 690.24, 740.0, 775.3, 703.66, 624.12, 683.5, 742.6, 641.35, 731.12, 733.07], [5.57e+08, 2.93e+08, 2.12e+08, 1.94e+08, 1.69e+08, 1.16e+08, 8.72e+07, 7.86e+07, 5.44e+07, 9.00e+07, 9.68e+07], [14.504588, 14.526486, 14.371989, 14.583383, 14.746284, 14.683178, 14.544408, 14.399969, 14.683178, 14.680382, 14.387235]),
                ("FII", [444.7, 429.77], [4.93e+09, 1.2e+09], [31.559723, 29.548333]), #expected temp 2 eV
                ("NII", [500.39, 567.87, 462.95], [2e+09, 4.31e+08, 3.74e+08], [23.141959, 20.665518, 21.159916]),
                ("HI", [656.2, 486.1, 434.14, 410.26], [1.575e+09, 2.6942e+08, 2.342e+08, 1.386e+08], [12.0875, 12.7485393, 13.0545, 13.22070369]) #expected temp 0.5 eV
                ]),
            ("S2337", -1.9, [
                ("CII", [426.7, 658.22, 723.58, 589.1, 514.5, 461.7], [3.19e9, 2.19e8, 3.9e8, 2.1e8, 5.32e8, 3.6e9], [20.950654, 16.33, 18.045987, 20.15, 23.118, 27.47]), #expected temp 1.6 eV. 589nm line mixed with NaI?
                ("OI", [615.43, 777.2], [1.91e+08, 5.53e+08], [12.7537, 10.7409]),
                ("FI", [685.7, 690.2, 740.0, 775.32, 703.66, 624.12, 683.5, 742.6, 641.4, 731.15, 733.2, 779.56], [5.57e+08, 2.93e+08, 2.12e+08, 1.94e+08, 1.69e+08, 1.16e+08, 8.72e+07, 7.86e+07, 5.44e+07, 9.00e+07, 9.68e+07, 1.01e+08], [14.504588, 14.526486, 14.371989, 14.583383, 14.746284, 14.683178, 14.544408, 14.399969, 14.683178, 14.680382, 14.387235, 14.614380]), #some lines mixed with BI around 680nm
                ("FII", [444.7, 429.8], [4.93e+09, 1.2e+09], [31.559723, 29.548333]), #expected temp 2 eV
                ("NII", [500.4, 567.8, 463.0], [2e+09, 4.31e+08, 3.74e+08], [23.141959, 20.665518, 21.159916]), #568nm line mixed with NaI?
                ("HI", [656.2, 486.14, 434.16, 410.3], [1.575e+09, 2.6942e+08, 2.342e+08, 1.386e+08], [12.0875, 12.7485393, 13.0545, 13.22070369]) #expected temp 0.5 eV. 434nm mixed with NaII?
                ]),
            ("S2338", -1.89, [
                ("CII", [426.65, 657.9, 723.36, 589.09], [3.19e9, 2.19e8, 3.9e8, 2.1e8], [20.950654, 16.33, 18.045987, 20.15]), #expected temp 1.6 eV
                ("FI", [685.65, 690.2, 739.94, 775.5, 703.72, 624.1, 731.06, 733.18], [5.57e+08, 2.93e+08, 2.12e+08, 1.94e+08, 1.69e+08, 1.16e+08, 9.00e+07, 9.68e+07], [14.504588, 14.526486, 14.371989, 14.583383, 14.746284, 14.683178, 14.680382, 14.387235]),
                ("NII", [500.36, 567.7], [2e+09, 4.31e+08], [23.141959, 20.665518]),
                ("HI", [656.3, 486.08], [1.575e+09, 2.6942e+08], [12.0875, 12.7485393]) #expected temp 0.5 eV
                ]),
            ("S2339", -1.88, [
                ("CII", [426.7, 658, 723.4, 589.1, 514.5], [3.19e9, 2.19e8, 3.9e8, 2.1e8, 5.32e8], [20.950654, 16.33, 18.045987, 20.15, 23.118]), #expected temp 1.6 eV. 589nm line mixed with NaI?
                ("OI", [615.43, 777.2], [1.91e+08, 5.53e+08], [12.7537, 10.7409]),
                ("FI", [685.7, 690.2, 740.0, 775.32, 703.75, 624.12, 683.5, 742.6, 641.4, 731.15, 733.2, 779.56], [5.57e+08, 2.93e+08, 2.12e+08, 1.94e+08, 1.69e+08, 1.16e+08, 8.72e+07, 7.86e+07, 5.44e+07, 9.00e+07, 9.68e+07, 1.01e+08], [14.504588, 14.526486, 14.371989, 14.583383, 14.746284, 14.683178, 14.544408, 14.399969, 14.683178, 14.680382, 14.387235, 14.614380]), #some lines mixed with BI around 680nm
                ("FII", [444.7, 429.8], [4.93e+09, 1.2e+09], [31.559723, 29.548333]), #expected temp 2 eV
                ("NII", [500.4, 567.8], [2e+09, 4.31e+08], [23.141959, 20.665518]), #568nm line mixed with NaI?
                ("HI", [656.2, 486.14, 434.16, 410.3], [1.575e+09, 2.6942e+08, 2.342e+08, 1.386e+08], [12.0875, 12.7485393, 13.0545, 13.22070369]) #expected temp 0.5 eV. 434nm mixed with NaII?
                ]),
            ("S2346", -1.89, [
                ("CII", [426.65, 657.9, 723.59, 589.09], [3.19e9, 2.19e8, 3.9e8, 2.1e8], [20.950654, 16.33, 18.045987, 20.15]), #expected temp 1.6 eV
                ("NII", [500.36, 567.9], [2e+09, 4.31e+08], [23.141959, 20.665518]),
                ("HI", [656.3, 486.08], [1.575e+09, 2.6942e+08], [12.0875, 12.7485393]) #expected temp 0.5 eV
                ]),
        ]

    print_spectrums(
        experiments=experiments,
        experiments_ions=[("S2346", ["CII", "NII", "HI"])],
        response_file=Path("response_curve.csv"),
        save=Path("yahli_experiments\workInProgress.png"),
        xlim=(400, 800),
        show_plot=True,
    )

    boltzmann_plot(
        experiments=experiments,
        experiments_ions=[("S2346", ["CII", "NII", "HI"]),],
        response_file=Path("response_curve.csv"),
        save=Path("yahli_experiments"),
        show=True,
    )