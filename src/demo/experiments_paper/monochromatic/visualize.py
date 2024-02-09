from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from src.common_utils.light_wave import Spectrum
from src.common_utils.utils import match_stats
from src.interface.configuration import load_config


def visualize_transfer_matrix(
        experiment_id: int,
        plot_options: dict,
        is_save: bool,
        save_options: dict,
):
    pass


def main():
    config = load_config()
    db = config.database()

    reports_folder = config.directory_paths.reports

    experiment_id_options = [1, 2]

    for experiment_id in experiment_id_options:
        experiment_params = db.experiments[experiment_id]
        pprint(dict(experiment_params))
        print()
        experiment_dir = reports_folder / f"experiment_{experiment_id}"

        for ds_id in experiment_params.dataset_ids:
            wavenumbers_ifgm = db.dataset_central_wavenumbers(dataset_id=ds_id)
            print(f"Dataset: {db.datasets[ds_id].title.upper()}")
            dataset_dir = experiment_dir / f"invert_{db.datasets[ds_id].title}"

            spectra_ref = Spectrum(data=np.eye(wavenumbers_ifgm.size), wavenumbers=wavenumbers_ifgm)

            for char_id in experiment_params.interferometer_ids:
                print(f"\tCharacterization: {db.characterizations[char_id].title.upper()}")
                characterization_dir = dataset_dir / f"{db.characterizations[char_id].title}"

                for ip_id in experiment_params.inversion_protocol_ids:
                    lambdaas = db.inversion_protocol_lambdaas(inv_protocol_id=ip_id)
                    print(f"\t\tInversion Protocol: {db.inversion_protocols[ip_id].title.upper()}")
                    inverter_dir = characterization_dir / f"{db.inversion_protocols[ip_id].title}"

                    spectra_rec_all = np.load(file=inverter_dir / "spectra_rec_all.npy")

                    spectra_rec_all_matched, reference = match_stats(
                        array=spectra_rec_all,
                        reference=spectra_ref.data,
                        axis=-2
                    )
                    rmse_full = np.load(file=inverter_dir / "rmse_full.npy")
                    rmse_diagonal = np.load(file=inverter_dir / "rmse_diagonal.npy")
                    rmcw = np.load(file=inverter_dir / "rmcw.npy")


if __name__ == "__main__":
    main()
