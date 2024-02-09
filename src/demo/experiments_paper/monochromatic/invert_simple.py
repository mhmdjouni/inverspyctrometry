from dataclasses import replace
from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt

from src.common_utils.light_wave import Spectrum
from src.common_utils.utils import match_stats
from src.interface.configuration import load_config


def main():
    config = load_config()
    db = config.database()

    experiment_id = 1
    experiment_params = db.experiments[experiment_id]
    pprint(dict(experiment_params))
    print()

    for ds_id in experiment_params.dataset_ids:
        interferograms_ref = db.dataset_interferogram(ds_id=ds_id)
        wavenumbers_ifgm = db.dataset_central_wavenumbers(dataset_id=ds_id)
        print(f"Dataset: {db.datasets[ds_id].title.upper()}")

        interferograms_ref = interferograms_ref.rescale(new_max=1, axis=-2)

        spectra_ref = Spectrum(data=np.eye(wavenumbers_ifgm.size), wavenumbers=wavenumbers_ifgm)

        for char_id in experiment_params.interferometer_ids:
            characterization = db.characterization(char_id=char_id)
            print(f"\tCharacterization: {db.characterizations[char_id].title.upper()}")

            transfer_matrix = characterization.transmittance_response(wavenumbers=wavenumbers_ifgm)
            transfer_matrix = transfer_matrix.rescale(new_max=1, axis=None)

            for ip_id in experiment_params.inversion_protocol_ids:
                lambdaas = db.inversion_protocol_lambdaas(inv_protocol_id=ip_id)
                print(f"\t\tInversion Protocol: {db.inversion_protocols[ip_id].title.upper()}")

                for lambdaa in lambdaas:
                    inverter = db.inversion_protocol(inv_protocol_id=ip_id, lambdaa=lambdaa)
                    spectra_rec = inverter.reconstruct_spectrum(
                        interferogram=interferograms_ref, transmittance_response=transfer_matrix
                    )

                    # Visualize results
                    reconstruction_matched, _ = match_stats(
                        array=spectra_rec.data,
                        reference=spectra_ref.data,
                        axis=-2,
                    )
                    fig, axs = plt.subplots(1, 2, squeeze=True)
                    spectra_ref.visualize_matrix(axs=axs[0])
                    replace(spectra_rec, data=reconstruction_matched).visualize_matrix(axs=axs[1])
                plt.show()


if __name__ == "__main__":
    main()
