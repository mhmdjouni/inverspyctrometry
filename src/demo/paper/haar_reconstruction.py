import numpy as np
from matplotlib import pyplot as plt

from src.common_utils.interferogram import Interferogram
from src.inverse_model.inverspectrometer import FabryPerotInverSpectrometerHaar


def main():
    wn_max_target = 1 / 350 * 1000  # um
    opd_step = 1 / (2 * wn_max_target)
    opd_num = 319
    opds = opd_step * np.arange(opd_num)
    acquisition = np.zeros_like(opds)[:, None]
    acquisition[15] = 1.
    interferogram = Interferogram(
        data=acquisition,
        opds=opds,
    )

    reflectance = np.array([0.7])
    transmittance = np.array([1.])  # The values in the paper seem to be normalized by the transmittance
    order = 10
    wn_min = 1/1000 * 1000
    wn_max = 1/350 * 1000
    wn_num = 319
    wavenumbers = np.linspace(start=wn_min, stop=wn_max, num=wn_num)  # um
    fp_haar = FabryPerotInverSpectrometerHaar(
        transmittance=transmittance,
        wavenumbers=wavenumbers,
        reflectance=reflectance,
        order=order,
        is_mean_center=True,
    )

    coefficients = fp_haar.kernel_fourier_coefficients()
    print(np.around(coefficients, decimals=3))
    print()

    # fig, axs = plt.subplots(squeeze=False)
    # interferogram.visualize(axs=axs[0, 0], acq_ind=0)
    # plt.show()

    spectrum = fp_haar.reconstruct_spectrum(interferogram=interferogram)
    fig, axs = plt.subplots(squeeze=False)
    spectrum.visualize(axs=axs[0, 0], acq_ind=0)
    plt.show()


if __name__ == "__main__":
    main()
