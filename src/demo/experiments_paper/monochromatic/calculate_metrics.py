from pprint import pprint

import numpy as np

from src.common_utils.light_wave import Spectrum
from src.common_utils.utils import match_stats, calculate_rmse, calculate_rmcw
from src.demo.experiments_paper.monochromatic.utils import print_metrics
from src.interface.configuration import load_config


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
                    spectra_rec_all_matched_stats, reference = match_stats(
                        array=spectra_rec_all,
                        reference=spectra_ref.data,
                        axis=-2
                    )
                    rmse_full = calculate_rmse(
                        array=spectra_rec_all_matched_stats,
                        reference=reference
                    )
                    rmse_diagonal = calculate_rmse(
                        array=np.diagonal(spectra_rec_all_matched_stats, offset=0, axis1=1, axis2=2),
                        reference=np.diag(reference)
                    )
                    rmcw = calculate_rmcw(monochromatic_array=spectra_rec_all_matched_stats)

                    # print(f"\t\t\t----")
                    nb_tabs = int(2)
                    category = "RMSE_MIN"
                    best_idx = np.argmin(rmse_diagonal)
                    print_metrics(nb_tabs, category, best_idx, lambdaas[best_idx], rmse_full[best_idx], rmse_diagonal[best_idx], rmcw[best_idx], wavenumbers_ifgm.size)
                    category = "RMCW_MAX"
                    best_idx = np.argmax(rmcw)
                    print_metrics(nb_tabs, category, best_idx, lambdaas[best_idx], rmse_full[best_idx], rmse_diagonal[best_idx], rmcw[best_idx], wavenumbers_ifgm.size)
                    print("")

                    # Save the metrics wrt lambda, per inv_protocol per dataset
                    if not inverter_dir.exists():
                        inverter_dir.mkdir(parents=True, exist_ok=True)
                    np.save(file=inverter_dir / "rmse_full", arr=rmse_full)
                    np.save(file=inverter_dir / "rmse_diagonal", arr=rmse_diagonal)
                    np.save(file=inverter_dir / "rmcw", arr=rmcw)


if __name__ == "__main__":
    main()
