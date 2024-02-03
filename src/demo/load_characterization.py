import matplotlib.pyplot as plt
import numpy as np

from src.common_utils.transmittance_response import TransmittanceResponse
from src.common_utils.utils import PlotOptions
from src.interface.configuration import load_config


def main():
    config = load_config()
    db = config.database()

    vis_kwargs = [
        {
            "char_id": 0,
            "vmin": 0,
            "vmax": 2.3e10,
        },
        {
            "char_id": 1,
            "vmin": 0,
            "vmax": 115,
        },
    ]

    characterization_id = 1
    characterization_schema = db.characterizations[characterization_id]
    print(characterization_schema)

    characterization = db.characterization(char_id=characterization_id)

    ds_id = db.characterizations[characterization_id].source_dataset_id
    interferograms = db.dataset_interferogram(ds_id=ds_id)

    central_wavenumbers = np.load(db.datasets[ds_id].wavenumbers_path)
    transmittance_response = characterization.transmittance_response(wavenumbers=central_wavenumbers)

    plot_options = PlotOptions(figsize=(8, 6), fontsize=25)
    plt.rcParams['font.size'] = str(plot_options.fontsize)

    fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=plot_options.figsize, tight_layout=True)
    interferograms.visualize_matrix(
        axs=axs[0, 0],
        vmin=vis_kwargs[characterization_id]["vmin"],
        vmax=vis_kwargs[characterization_id]["vmax"],
    )

    fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=plot_options.figsize, tight_layout=True)
    transmittance_response.visualize(
        axs=axs[0, 0],
        vmin=vis_kwargs[characterization_id]["vmin"],
        vmax=vis_kwargs[characterization_id]["vmax"],
    )

    plt.show()


if __name__ == "__main__":
    main()
