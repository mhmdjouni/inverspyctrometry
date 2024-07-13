from dataclasses import replace, dataclass, asdict

import matplotlib.pyplot as plt
import numpy as np

from src.common_utils.custom_vars import InterferometerType
from src.common_utils.transmittance_response import TransmittanceResponse
from src.direct_model.interferometer import interferometer_factory
from src.interface.configuration import load_config
from src.outputs.visualization import SubplotsOptions, RcParamsOptions, savefig_dir_list


def calculate_opd(l: int, opd_step: float) -> float:
    return l * opd_step


def calculate_wn(k: int, wn_step: float) -> float:
    return (k + 1 / 2) * wn_step


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


def estimate_harmonic_order(device_type: InterferometerType, reflectance: float) -> int:
    if device_type == InterferometerType.MICHELSON:
        return 2
    elif device_type == InterferometerType.FABRY_PEROT:
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
        raise ValueError(f"Option {device_type} is not supported.")


def compensate(
        device_type: InterferometerType,
        transmat: TransmittanceResponse,
        reflectance: float,
) -> TransmittanceResponse:
    matrix: np.ndarray = transmat.data
    if device_type == InterferometerType.MICHELSON:
        matrix = matrix - 2 * (1 - reflectance)
    elif device_type == InterferometerType.FABRY_PEROT:
        matrix = matrix / ((1 - reflectance) / (1 + reflectance)) - 1
    matrix /= np.sqrt(2 * matrix.shape[0])
    matrix[0] /= np.sqrt(2)
    return replace(transmat, data=matrix)


def main_per_case(
        exp_title: str,
        device_type: InterferometerType,
        reflectance_scalar: float,
        opd_idx: int,
        is_show: bool,
        override_harmonic_order: int,
):
    # PROCESS
    opds = np.arange(0, 51) * 0.2
    if override_harmonic_order < 2:
        harmonic_order = estimate_harmonic_order(device_type=device_type, reflectance=reflectance_scalar)
    else:
        harmonic_order = override_harmonic_order
    wn_num = opds.size * (harmonic_order - 1)
    wn_step = calculate_wn_step(wn_num, np.mean(np.diff(opds)))
    wn_min = calculate_wn(0, wn_step)
    wn_max = calculate_wn(wn_num - 1, wn_step)
    wavenumbers = np.linspace(wn_min, wn_max, wn_num, endpoint=True)

    opd_unit = "um"
    wn_unit = "1/um"
    # print_info_opds_wns(opds, wavenumbers, opd_unit, wn_unit)

    wavenumbers = crop_limits(wavenumbers, min_lim=1., max_lim=2.5)
    # print_info_opds_wns(opds, wavenumbers, opd_unit, wn_unit)

    reflectance = np.array([reflectance_scalar])
    transmittance = 1. - reflectance
    device = interferometer_factory(
        option=device_type,
        transmittance_coefficients=transmittance,
        opds=opds,
        phase_shift=np.array([0.]),
        reflectance_coefficients=reflectance,
        order=0,
    )

    transmat = device.transmittance_response(wavenumbers=wavenumbers)
    transmat_compensated = compensate(device_type, transmat, reflectance_scalar)

    # VISUALIZE
    rc_params = RcParamsOptions(fontsize=21)
    subplots_opts = SubplotsOptions(figsize=(6.4, 4.8))
    plt.rcParams['font.size'] = str(rc_params.fontsize)

    figs, axes = zip(*[plt.subplots(**asdict(subplots_opts)) for _ in range(4)])
    transmat.visualize(
        fig=figs[0],
        axs=axes[0][0, 0],
        title="",
        is_colorbar=True,
        x_ticks_num=5,
        x_ticks_decimals=1,
        y_ticks_decimals=0,
        aspect="auto",
    )
    condition_number = transmat_compensated.condition_number()
    condition_number_str = f"{condition_number:.0f}" if condition_number < 1e10 else r"$\infty$"
    print(f"\n{exp_title}\nCondition number: {condition_number_str}")
    transmat_compensated.visualize_singular_values(
        axs=axes[1][0, 0],
        title=f"Condition number = {condition_number_str}",
        linewidth=3,
        marker="o",
        markevery=5,
    )
    transmat.visualize_opd_response(
        axs=axes[2][0, 0],
        opd_idx=opd_idx,
        title=None,
        show_full_title=False,
        linewidth=3,
    )
    transmat.visualize_dct(
        axs=axes[3][0, 0],
        opd_idx=opd_idx,
        title=None,
        show_full_title=False,
        linewidth=3,
    )

    if is_show:
        plt.show()

    # SAVE
    filenames = [
        "transfer_matrix.pdf",
        "singular_values.pdf",
        "opd_response.pdf",
        "opd_dct.pdf",
    ]
    project_dir = load_config().directory_paths.project
    paper_dir = project_dir.parents[1] / "latex" / "20249999_ieee_tsp_inversion_v4"
    figures_dir_list = [
        paper_dir / "figures" / "direct_model",
    ]
    save_subdir = f"{exp_title}/transfer_matrices"
    for filename, fig in zip(filenames, figs):
        savefig_dir_list(
            fig=fig,
            filename=filename,
            directories_list=figures_dir_list,
            subdirectory=save_subdir,
        )


def visualize_together(
        transmat,
        transmat_compensated,
        opd_idx,
):
    fig, axes = plt.subplots(nrows=2, ncols=2)
    transmat.visualize(fig=fig, axs=axes[0, 0], aspect="auto", x_ticks_decimals=1, y_ticks_decimals=0)
    transmat_compensated.visualize_singular_values(axs=axes[0, 1])
    transmat.visualize_opd_response(axs=axes[1, 0], opd_idx=opd_idx)
    transmat.visualize_dct(axs=axes[1, 1], opd_idx=opd_idx)
    plt.show()


@dataclass(frozen=True)
class Case:
    exp_title: str
    device_type: InterferometerType
    reflectance_scalar: float
    override_harmonic_order: int


def main():
    cases = [
        Case(
            exp_title="mich",
            device_type=InterferometerType.MICHELSON,
            reflectance_scalar=0.5,
            override_harmonic_order=0,
        ),
        Case(
            exp_title="mich_oversampled",
            device_type=InterferometerType.MICHELSON,
            reflectance_scalar=0.5,
            override_harmonic_order=5,
        ),
        Case(
            exp_title="fp_0_low_r",
            device_type=InterferometerType.FABRY_PEROT,
            reflectance_scalar=0.2,
            override_harmonic_order=0,
        ),
        Case(
            exp_title="fp_0_med_r",
            device_type=InterferometerType.FABRY_PEROT,
            reflectance_scalar=0.5,
            override_harmonic_order=0,
        ),
        Case(
            exp_title="fp_0_high_r",
            device_type=InterferometerType.FABRY_PEROT,
            reflectance_scalar=0.8,
            override_harmonic_order=0,
        ),
    ]
    opd_idx = 10
    is_show = False

    for case in cases:
        main_per_case(
            exp_title=case.exp_title,
            device_type=case.device_type,
            reflectance_scalar=case.reflectance_scalar,
            override_harmonic_order=case.override_harmonic_order,
            opd_idx=opd_idx,
            is_show=is_show,
        )


if __name__ == "__main__":
    main()
