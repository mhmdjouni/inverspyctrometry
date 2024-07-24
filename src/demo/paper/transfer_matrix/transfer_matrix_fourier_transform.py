import matplotlib.pyplot as plt

from src.common_utils.custom_vars import InterferometerType
from src.demo.paper.transfer_matrix.utils import SamplingOptionsSchema


def main():
    sampling_options_schema_list = [
        {
            "experiment_title": "Fabry-Perot",
            "device": {
                "type": InterferometerType.FABRY_PEROT,
                "reflectance_scalar": 0.1,
                "opds": {
                    "num": 50 + 1,
                    "step": 0.2,
                },
            },
            "spectral_range": {
                "min": 0,
                "max": 2.5,
                "override_harmonic_order": None,
            },
        },
    ]

    for sampling_options_schema in sampling_options_schema_list:
        options = SamplingOptionsSchema(**sampling_options_schema)
        experiment = options.create_experiment()

        transfer_matrix = experiment.transfer_matrix()
        transfer_matrix_decomposition = experiment.transfer_matrix_decomposition()

        fig, axs = plt.subplots(nrows=2, ncols=len(transfer_matrix_decomposition) - 1, squeeze=False, tight_layout=True, figsize=(12.8, 9.6))
        transfer_matrix.visualize(
            fig=fig,
            axs=axs[0, 0],
            aspect="auto",
            title=f"{experiment.experiment_title} R={options.device.reflectance_scalar:.2f}",
        )
        transfer_matrix.visualize_dct(axs=axs[0, 1], opd_idx=-1, title="DCT of A")
        for idx in range(1, len(transfer_matrix_decomposition)):
            transfer_matrix_decomposition[idx].visualize_dct(
                axs=axs[1, idx - 1],
                opd_idx=-1,
                title="DCT of A" + r"$^{(" + f"{idx}" + r")}$",
                ylim=axs[0, 1].get_ylim(),
            )
        plt.show()


if __name__ == "__main__":
    main()
