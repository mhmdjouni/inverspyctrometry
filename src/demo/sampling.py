from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from src.common_utils.transmittance_response import TransmittanceResponse
from src.direct_model.interferometer import FabryPerotInterferometer


def calculate_opd(l: int, opd_step: float) -> float:
    return l * opd_step


def calculate_wn(k: int, wn_step: float) -> float:
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


def crop_limits(array, min_lim=None, max_lim=None):
    if min_lim is not None:
        array = array[array >= min_lim]
    if max_lim is not None:
        array = array[array <= max_lim]
    return array


def estimate_harmonic_order(reflectance: float, override: int = -1) -> int:
    if override == -1:
        reflectivity_order_mapper = {
            0.001: 2,
            0.01: 3,
            0.05: 3,
            0.1: 4,
            0.2: 5,
            0.4: 8,
            0.5: 10,
            0.7: 19,
            0.8: 27,
        }
        return reflectivity_order_mapper[reflectance]
    else:
        return override


def orthogonalize(transmat: TransmittanceResponse, reflectance: float) -> TransmittanceResponse:
    matrix: np.ndarray = transmat.data
    matrix = matrix / ((1-reflectance)/(1+reflectance)) - 1
    matrix /= np.sqrt(2 * matrix.shape[0])
    matrix[0] /= np.sqrt(2)
    return replace(transmat, data=matrix)


def main():
    reflectance_scalar = 0.8
    harmonic_order = estimate_harmonic_order(reflectance=reflectance_scalar, override=-1)

    opds = np.arange(0, 51) * 0.2
    wn_num = opds.size * (harmonic_order - 1)
    wn_step = calculate_wn_step(wn_num, np.mean(np.diff(opds)))
    wn_min = calculate_wn(0, wn_step)
    wn_max = calculate_wn(wn_num - 1, wn_step)
    wavenumbers = np.linspace(wn_min, wn_max, wn_num, endpoint=True)

    opd_unit = "um"
    wn_unit = "1/um"
    print_info_opds_wns(opds, wavenumbers, opd_unit, wn_unit)

    wavenumbers = crop_limits(wavenumbers, min_lim=1., max_lim=2.5)
    print_info_opds_wns(opds, wavenumbers, opd_unit, wn_unit)

    reflectance = np.array([reflectance_scalar])
    transmittance = 1. - reflectance
    fp = FabryPerotInterferometer(
        transmittance_coefficients=transmittance,
        opds=opds,
        phase_shift=np.array([0.]),
        reflectance_coefficients=reflectance,
        order=0,
    )
    transmat = fp.transmittance_response(wavenumbers=wavenumbers)
    transmat = orthogonalize(transmat, reflectance_scalar)

    opd_idx = 10
    fig, axes = plt.subplots(nrows=2, ncols=2)
    # transmat.visualize(fig=fig, axs=axes[0, 0], aspect=transmat.data.shape[1] / transmat.data.shape[0])
    transmat.visualize(fig=fig, axs=axes[0, 0], aspect="auto", x_ticks_decimals=1, y_ticks_decimals=0)
    transmat.visualize_singular_values(axs=axes[0, 1])
    transmat.visualize_opd_response(axs=axes[1, 0], opd_idx=opd_idx)
    transmat.visualize_dct(axs=axes[1, 1], opd_idx=opd_idx)
    plt.show()


if __name__ == "__main__":
    main()
