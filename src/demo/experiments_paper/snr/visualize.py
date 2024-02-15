from dataclasses import asdict

from matplotlib import pyplot as plt

from src.demo.experiments_paper.snr.utils import experiment_subdir_convention, experiment_dir_convention
from src.interface.configuration import load_config
from src.outputs.visualization import savefig_dir_list, RcParamsOptions, SubplotsOptions


def experiment_figures_subdir_convention(
        dataset_id: int,
        interferometer_id: int,
        folder_name: str,
) -> str:
    save_subdir = experiment_subdir_convention(
        dataset_id=dataset_id,
        interferometer_id=interferometer_id,
    )
    save_subdir = f"{save_subdir}/{folder_name}"
    return save_subdir


def visualize_transfer_matrix(
        experiment_id: int,
        dataset_id: int,
        interferometer_id: int,
        rc_params: RcParamsOptions,
        subplots_options: SubplotsOptions,
        plot_options: dict,
):
    config = load_config()
    paper_dir = config.directory_paths.project.parents[1] / "latex" / "20249999_ieee_tsp_inversion_v4"

    db = config.database()
    figures_dir_list = [
        experiment_dir_convention(dir_type="figures", experiment_id=experiment_id),
        experiment_dir_convention(dir_type="paper_figures", experiment_id=experiment_id, custom_dir=paper_dir),
    ]
    save_subdir = experiment_figures_subdir_convention(
        dataset_id=dataset_id,
        interferometer_id=interferometer_id,
        folder_name="transfer_matrix",
    )

    spectra_ref = db.dataset_spectrum(ds_id=dataset_id)
    interferometer = db.interferometer(ifm_id=interferometer_id)
    transfer_matrix = interferometer.transmittance_response(wavenumbers=spectra_ref.wavenumbers)

    plt.rcParams['font.size'] = str(rc_params.fontsize)
    fig, axes = plt.subplots(**asdict(subplots_options))
    transfer_matrix.visualize(fig=fig, axs=axes[0, 0], **plot_options)

    filename = "transfer_matrix.pdf"
    savefig_dir_list(
        fig=fig,
        filename=filename,
        directories_list=figures_dir_list,
        subdirectory=save_subdir,
    )
