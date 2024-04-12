from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np

from src.direct_model.interferometer import FabryPerotInterferometer
from src.outputs.visualization import SubplotsOptions


def calculate_opd(l: int, opd_step: float) -> float:
    # If l = opd_num - 1:
    #  opd_max = (opd_num - 1) * opd_step
    return l * opd_step


def calculate_wn(k: int, wn_step: float) -> float:
    # If k = wn_num - 1:
    #  wn_max = (wn_num - 1/2) * wn_step
    #         = (wn_num - 1/2) / (2 * wn_num * opd_step)
    #         = 1 / (2 * opd_step) - 1 / (4 * wn_num * opd_step)
    # If opd_max is constant and wn_num = opd_max, then wn_max is dependent only on the opd_step.
    # As opd_step changes, wn_max changes inversely.
    return (k + 1/2) * wn_step


def calculate_wn_step(wn_num: int, opd_step: float) -> float:
    return 1 / (2 * wn_num * opd_step)


def print_info(array: np.ndarray, unit: str):
    print(
        f"\tInterval: [{array.min():.4f}, {array.max():7.4f}]."
        f" Step size: {np.mean(np.diff(array)):.4f}."
        f" No. samples: {array.size}."
        f" Unit: {unit}."
    )


def print_info_opds_wns(opds, wns, opd_unit="um", wn_unit="1/um"):
    print("OPD Information:")
    print_info(opds, opd_unit)

    print("Wavenumber Information:")
    print_info(wns, wn_unit)

    print()


def crop_limits(array, min_lim = None, max_lim = None):
    if min_lim is not None:
        array = array[array >= min_lim]
    if max_lim is not None:
        array = array[array <= max_lim]
    return array


def calculate_opd_step_max(wn_max_lim: float):
    sampling_frequency_min = 2 * wn_max_lim
    return 1 / sampling_frequency_min


def calculate_opd_num_min(opd_max: float, opd_step_max: float, opd_min: float = 0) -> int:
    ratio = (opd_max - opd_min) / opd_step_max + 1
    return int(np.ceil(ratio))


def estimate_wave_model(reflectance: float, wave_model_overrider: int = -1) -> int:
    if wave_model_overrider == -1:
        reflectivity_order_mapper = {
            0.001: 2,
            0.01: 3,
            0.05: 3,
            0.1: 4,
            0.2: 5,
            0.5: 10,
            0.8: 27,
        }
        return reflectivity_order_mapper[reflectance]
    else:
        return wave_model_overrider


def main():
    reflectance = 0.5
    fp_wave_model = estimate_wave_model(reflectance=reflectance, wave_model_overrider=2)
    wl_min_lim, wl_max_lim = 0.35, 1.  # in um. Chosen for the UV + RGB + NIR range [350, 1000] nm

    wn_min_lim, wn_max_lim = 1/wl_max_lim, 1/wl_min_lim  # in 1/um
    opd_step_max = calculate_opd_step_max(wn_max_lim)

    opd_min = 0
    opd_max = 55.65  # Larger opd_max, smaller spectral step size (better spectral resolution)
    opd_num_min = calculate_opd_num_min(opd_max, opd_step_max, opd_min)
    opd_num = 319  # Higher num, smaller step, larger spectral bandwidth
    opds = np.linspace(opd_min, opd_max, opd_num, endpoint=True)
    opd_step = np.mean(np.diff(opds))

    wn_num = opd_num * 1 * (fp_wave_model - 1)
    wn_step = calculate_wn_step(wn_num=wn_num, opd_step=opd_step)
    wn_min = calculate_wn(k=0, wn_step=wn_step)
    wn_max = calculate_wn(k=wn_num-1, wn_step=wn_step)
    wns = np.linspace(wn_min, wn_max, wn_num, endpoint=True)

    opd_unit = "um"
    wn_unit = "1/um"
    print_info_opds_wns(opds, wns, opd_unit, wn_unit)

    wns_cropped = crop_limits(wns, min_lim=wn_min_lim, max_lim=wn_max_lim)
    print_info_opds_wns(opds, wns_cropped, opd_unit, wn_unit)

    reflectance = np.array([reflectance])
    transmittance = 1. - reflectance
    fp = FabryPerotInterferometer(
        transmittance_coefficients=transmittance,
        opds=opds,
        phase_shift=np.array([0.]),
        reflectance_coefficients=reflectance,
        order=fp_wave_model,
    )
    transmat = fp.transmittance_response(wavenumbers=wns_cropped)

    opd_idx = opd_num // 18
    fig, axes = plt.subplots(nrows=2, ncols=2)
    transmat.visualize(fig=fig, axs=axes[0, 0], aspect=transmat.data.shape[1] / transmat.data.shape[0])
    transmat.visualize_singular_values(axs=axes[0, 1])
    transmat.visualize_opd_response(axs=axes[1, 0], opd_idx=opd_idx)
    transmat.visualize_dct(axs=axes[1, 1], opd_idx=opd_idx)
    plt.show()


if __name__ == "__main__":
    main()
