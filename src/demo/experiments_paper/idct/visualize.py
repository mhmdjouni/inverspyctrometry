from dataclasses import asdict, replace

import numpy as np
from matplotlib import pyplot as plt

from src.demo.experiments_paper.snr.utils import experiment_dir_convention, experiment_subdir_convention
from src.demo.experiments_paper.snr.visualize import experiment_figures_subdir_convention
from src.interface.configuration import load_config
from src.outputs.visualization import RcParamsOptions, SubplotsOptions, savefig_dir_list


def main():
    config = load_config()
    db = config.database()

    experiment_id = 3
    experiment_config = db.experiments[experiment_id]

    rc_params = RcParamsOptions(fontsize=17)
    subplots_options = SubplotsOptions()
    plot_options = {"ylim": [-0.3, 1.3]}
    acquisition_indices = [0, 13]

    for i_ds, dataset_id in enumerate(experiment_config.dataset_ids):
        print(f"Dataset: {db.datasets[dataset_id].title.upper()}")

        paper_dir = config.directory_paths.project.parents[1] / "latex" / "20249999_ieee_tsp_inversion_v4"
        reconstruction_dir = experiment_dir_convention(dir_type="reconstruction", experiment_id=experiment_id)
        figures_dir_list = [
            experiment_dir_convention(dir_type="figures", experiment_id=experiment_id),
            experiment_dir_convention(dir_type="paper_figures", experiment_id=experiment_id, custom_dir=paper_dir),
        ]
        save_subdir = experiment_figures_subdir_convention(
            dataset_id=dataset_id,
            interferometer_id=-1,
            noise_level_index=-1,
            folder_name="spectrum_comparison",
        )

        spectra_ref_scaled = db.dataset_spectrum(ds_id=dataset_id).rescale(new_max=1, axis=-2)

        plt.rcParams['font.size'] = str(rc_params.fontsize)
        fig, axes = plt.subplots(**asdict(subplots_options))

        spectra_ref_scaled.visualize(
            axs=axes[0, 0],
            acq_ind=acquisition_indices[i_ds],
            label="Reference",
            color="C0",
            linewidth=3,
            title="",
            **plot_options,
        )

        for i_ifm, interferometer_id in enumerate(experiment_config.interferometer_ids):
            print(f"\tInterferometer: {db.interferometers[interferometer_id].title.upper()}")

            noise_level_index = experiment_config.noise_level_indices[0]

            inversion_protocol_id = experiment_config.inversion_protocol_ids[0]

            load_subdir = experiment_subdir_convention(
                dataset_id=dataset_id,
                interferometer_id=interferometer_id,
                noise_level_index=noise_level_index,
                inversion_protocol_id=inversion_protocol_id,
            )

            spectra_rec = replace(
                spectra_ref_scaled,
                data=np.load(file=reconstruction_dir / load_subdir / "spectra_rec.npy"),
            )
            spectra_rec_matched, _ = spectra_rec.match_stats(reference=spectra_ref_scaled, axis=-2)

            delta_min_str = r"$\delta_{min}$"
            spectra_rec_matched.visualize(
                axs=axes[0, 0],
                acq_ind=acquisition_indices[i_ds],
                label=(
                    f"{delta_min_str}={db.interferometers[interferometer_id].opds.start:.2f}"
                ),
                color=f"C{i_ifm + 1}",
                linewidth=1.3,
                title=f"{db.inversion_protocols[inversion_protocol_id].title.upper()}",
                **plot_options,
            )

        filename = f"acquisition_{acquisition_indices[i_ds]:03}.pdf"
        plt.show()
        savefig_dir_list(
            fig=fig,
            filename=filename,
            directories_list=figures_dir_list,
            subdirectory=save_subdir,
        )


if __name__ == "__main__":
    main()
