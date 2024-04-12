from dataclasses import replace, asdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.common_utils.utils import generate_wavenumbers_from_opds
from src.demo.paper.snr.visualize import experiment_figures_subdir_convention
from src.interface.configuration import load_config
from src.outputs.visualization import SubplotsOptions, RcParamsOptions, savefig_dir_list


def visualize_transfer_matrices(
        interferometer_id: int,
        rc_params: RcParamsOptions,
        subplots_options: SubplotsOptions,
        transmat_imshow_options: dict,
        singvals_plot_options: dict,
        opd_response_plot_options: dict,
        dct_opd_plot_options: dict,
        opd_idx: int,
        is_plt_show: bool,
):
    config = load_config()
    paper_dir = config.directory_paths.project.parents[1] / "latex" / "20249999_ieee_tsp_inversion_v4"

    db = config.database()
    figures_dir_list = [
        paper_dir / "figures" / "direct_model",
    ]
    save_subdir = f"{db.interferometers[interferometer_id].title}/transfer_matrices"

    # Load
    interferometer = db.interferometer(ifm_id=interferometer_id)
    opds = np.linspace(0, 10, 300)
    interferometer = replace(interferometer, opds=opds, phase_shift=np.array([0.]))
    wavenumbers = np.linspace(1, 2.85, opds.size)
    # opds = interferometer.opds
    # wavenumbers = generate_wavenumbers_from_opds(
    #     wavenumbers_num=opds.size,
    #     del_opd=np.mean(np.diff(opds)),
    #     wavenumbers_start=1.,
    #     wavenumbers_stop=2.85,
    # )
    transmittance_response = interferometer.transmittance_response(wavenumbers=wavenumbers)
    transmittance_response = transmittance_response.rescale(new_max=1.)

    # Visualize
    plt.rcParams['font.size'] = str(rc_params.fontsize)
    fig_1, axes_1 = plt.subplots(**asdict(subplots_options))
    fig_2, axes_2 = plt.subplots(**asdict(subplots_options))
    fig_3, axes_3 = plt.subplots(**asdict(subplots_options))
    fig_4, axes_4 = plt.subplots(**asdict(subplots_options))
    transmittance_response.visualize(fig=fig_1, axs=axes_1[0, 0], **transmat_imshow_options)
    transmittance_response.visualize_singular_values(axs=axes_2[0, 0], **singvals_plot_options)
    transmittance_response.visualize_opd_response(axs=axes_3[0, 0], opd_idx=opd_idx, **opd_response_plot_options)
    transmittance_response.visualize_dct(axs=axes_4[0, 0], opd_idx=opd_idx, **dct_opd_plot_options)

    if is_plt_show:
        plt.show()

    # Save
    filename = "transfer_matrix.pdf"
    savefig_dir_list(
        fig=fig_1,
        filename=filename,
        directories_list=figures_dir_list,
        subdirectory=save_subdir,
    )
    filename = "singular_values.pdf"
    savefig_dir_list(
        fig=fig_2,
        filename=filename,
        directories_list=figures_dir_list,
        subdirectory=save_subdir,
    )
    filename = f"opd_response_idx_{opd_idx}.pdf"
    savefig_dir_list(
        fig=fig_3,
        filename=filename,
        directories_list=figures_dir_list,
        subdirectory=save_subdir,
    )
    filename = f"dct_opd_idx_{opd_idx}.pdf"
    savefig_dir_list(
        fig=fig_4,
        filename=filename,
        directories_list=figures_dir_list,
        subdirectory=save_subdir,
    )


def main():
    interferometer_ids = [0, 4, 5, 6]

    inputs_dict = {
        "rc_params": RcParamsOptions(fontsize=21),
        "subplots_options": SubplotsOptions(),
        "transmat_imshow_options": {
            "title": "",
            "is_colorbar": False,
            "x_ticks_num": 4,
            "y_ticks_decimals": 0,
        },
        "singvals_plot_options": {
            "title": "",
            "linewidth": 3,
        },
        "opd_response_plot_options": {
            "title": None,
            "show_full_title": False,
            "linewidth": 3,
        },
        "dct_opd_plot_options": {
            "title": None,
            "show_full_title": False,
        },
        "opd_idx": 50,
        "is_plt_show": True,
    }

    for i_ifm, interferometer_id in enumerate(interferometer_ids):
        visualize_transfer_matrices(
            interferometer_id=interferometer_id,
            **inputs_dict,
        )
    plt.show()


if __name__ == "__main__":
    main()
