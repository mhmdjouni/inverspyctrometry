import matplotlib.pyplot as plt
import numpy as np

from src.common_utils.utils import generate_wavenumbers_from_opds
from src.interface.configuration import load_config
from src.outputs.visualization import plot_custom


def main():
    interferometer_ids = [0, 4, 5, 6]
    opd_idx = 40

    db = load_config().database()

    fig_1, axes_1 = plt.subplots(nrows=2, ncols=2, squeeze=False)
    fig_2, axes_2 = plt.subplots(nrows=2, ncols=2, squeeze=False)
    fig_3, axes_3 = plt.subplots(nrows=2, ncols=2, squeeze=False)
    fig_4, axes_4 = plt.subplots(nrows=2, ncols=2, squeeze=False)
    fig_5, axes_5 = plt.subplots(nrows=1, ncols=1, squeeze=False)

    for i_ifm, interferometer_id in enumerate(interferometer_ids):
        interferometer = db.interferometer(ifm_id=interferometer_id)
        opds = interferometer.opds
        wavenumbers = generate_wavenumbers_from_opds(
            wavenumbers_num=opds.size,
            del_opd=np.mean(np.diff(opds)),
            wavenumbers_start=1.,
            wavenumbers_stop=2.85,
        )

        transmittance_response = interferometer.transmittance_response(wavenumbers=wavenumbers)
        reflectance = interferometer.reflectance(wavenumbers=wavenumbers)

        row_index = i_ifm // 2
        col_index = i_ifm % 2

        transmittance_response.visualize(fig=fig_1, axs=axes_1[row_index, col_index])
        transmittance_response.visualize_opd_response(axs=axes_2[row_index, col_index], opd_idx=opd_idx)
        transmittance_response.visualize_dct(axs=axes_3[row_index, col_index], opd_idx=opd_idx)
        transmittance_response.visualize_singular_values(axs=axes_4[row_index, col_index])
        plot_custom(axs=axes_5[0, 0], x_array=wavenumbers, array=reflectance[0], color=f"C{i_ifm}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
