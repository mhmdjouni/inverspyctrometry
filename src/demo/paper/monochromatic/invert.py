from dataclasses import dataclass
from pprint import pprint

import numpy as np

from src.interface.configuration import load_config


# TODO: Consider cropping the wavenumbers of mc651 starting from 1.1
# TODO: Dump the experiment configuration in the reports/experiment/ folder (could also be a separate function to call)


@dataclass(frozen=True)
class ExtrapCase:
    case: int
    description: str
    opds_resampler: str
    extrap_kind: str
    extrap_fill: any
    transmat_extrap: str


def run_one_experiment(
        experiment_id_options: list,
        extrap_case: ExtrapCase,
):
    config = load_config()
    db = config.database()

    reports_folder = config.directory_paths.reports

    for experiment_id in experiment_id_options:
        experiment_config = db.experiments[experiment_id]
        pprint(dict(experiment_config))
        experiment_dir = reports_folder / f"experiment_{experiment_id}" / "reconstruction"

        for ds_id in experiment_config.dataset_ids:
            interferograms_ref = db.dataset_interferogram(ds_id=ds_id)
            wavenumbers_ifgm = db.dataset_central_wavenumbers(dataset_id=ds_id)
            print(f"Dataset: {db.datasets[ds_id].title.upper()}")
            dataset_dir = experiment_dir / f"invert_{db.datasets[ds_id].title}"

            interferograms_ref = interferograms_ref.rescale(new_max=1, axis=-2)

            for char_id in experiment_config.interferometer_ids:
                characterization = db.characterization(char_id=char_id)
                print(f"\tCharacterization: {db.characterizations[char_id].title.upper()}")
                characterization_dir = dataset_dir / f"{db.characterizations[char_id].title}"

                transfer_matrix = characterization.transmittance_response(wavenumbers=wavenumbers_ifgm)
                transfer_matrix = transfer_matrix.rescale(new_max=1, axis=None)

                for ip_id in experiment_config.inversion_protocol_ids:
                    lambdaas = db.inversion_protocol_lambdaas(inv_protocol_id=ip_id)
                    print(f"\t\tInversion Protocol: {db.inversion_protocols[ip_id].title.upper()}")
                    inverter_dir = characterization_dir / f"{db.inversion_protocols[ip_id].title}"

                    spectra_rec_all = np.zeros(
                        shape=(lambdaas.size, wavenumbers_ifgm.size, interferograms_ref.data.shape[-1]))

                    for il, lambdaa in enumerate(lambdaas):
                        inverter = db.inversion_protocol(
                            inv_protocol_id=ip_id,
                            lambdaa=lambdaa,
                            is_compute_and_save_cost=False,
                            experiment_id=-1,
                        )
                        spectra_rec = inverter.reconstruct_spectrum(
                            interferogram=interferograms_ref, transmittance_response=transfer_matrix
                        )
                        spectra_rec_all[il] = spectra_rec.data

                    # Save all spectral reconstructions wrt lambda, per inv_protocol per dataset
                    if not inverter_dir.exists():
                        inverter_dir.mkdir(parents=True, exist_ok=True)
                    np.save(file=inverter_dir / "spectra_rec_all", arr=spectra_rec_all)

            # Save the wavenumbers for the sake of independent plots
            if not dataset_dir.exists():
                dataset_dir.mkdir(parents=True, exist_ok=True)
            np.save(file=dataset_dir / "wavenumbers", arr=wavenumbers_ifgm)

        # Dump the experiment's JSON
        # if not experiment_dir.exists():
        #     experiment_dir.mkdir(parents=True, exist_ok=True)
        # experiment_config.model_dump_json()


def main():
    extrap_cases = [
        ExtrapCase(
            case=1,
            description="Concatenate lowest OPDs but set the corresponding interferogram values to zero",
            opds_resampler="concatenate_missing",
            extrap_kind="linear",
            extrap_fill=(0., 0.),
            transmat_extrap="model"
        ),
        ExtrapCase(
            case=2,
            description="Concatenate lowest OPDs but extrapolate the interferogram values using conventional methods",
            opds_resampler="concatenate_missing",
            extrap_kind="linear",
            extrap_fill="extrapolate",
            transmat_extrap="model"
        ),
        ExtrapCase(
            case=3,
            description="Concatenate lowest OPDs but extrapolate the interferogram values using fourier series",
            opds_resampler="concatenate_missing",
            extrap_kind="linear",
            extrap_fill="fourier",
            transmat_extrap="model"
        ),
    ]

    experiment_id_options = [1]  # 1, 2, 8

    for extrap_case in extrap_cases:
        run_one_experiment(
            experiment_id_options=experiment_id_options,
            extrap_case=extrap_case,
        )


if __name__ == "__main__":
    main()
