import matplotlib.pyplot as plt
import numpy as np

from src.common_utils.custom_vars import InterferometerType
from src.common_utils.utils import generate_shifted_dirac
from src.common_utils.interferogram import Interferogram
from src.inverse_model.inverspectrometer import MichelsonInverSpectrometer
from src.inverse_model.protocols import PseudoInverse
from src.loaders.transmittance_response import transmittance_response_factory


def main():
    # Observation
    opds = np.linspace(start=0, stop=56, num=320, endpoint=False)
    observation = generate_shifted_dirac(array=opds, shift=0.175*4)  # shift can be a list of indices or a list of OPD values
    interferogram = Interferogram(data=observation, opds=opds)

    # Given or loaded from some Characterization
    wavenumbers = np.linspace(start=0, stop=1/(2*np.mean(np.diff(opds))), num=opds.size, endpoint=False)
    reflectance = 0.13 * np.ones_like(a=wavenumbers, dtype=np.float_)
    transmittance = 1 - reflectance

    # Generate or load a transmittance response (Generated using Interferometer || Loaded from disk)
    transmittance_response = transmittance_response_factory(
        option="generate",
        kwargs={
            "type": InterferometerType.MICHELSON,
            "opds": opds,
            "transmittance": transmittance,
            "order": 0,
            "wavenumbers": wavenumbers,
        }
    )

    inverter_pinv = PseudoInverse()
    spectrum_pinv = inverter_pinv.reconstruct_spectrum(
        interferogram=interferogram,
        transmittance_response=transmittance_response,
    )

    inverspectrometer_michelson = MichelsonInverSpectrometer(transmittance=transmittance)
    spectrum_mich = inverspectrometer_michelson.reconstruct_spectrum(interferogram=interferogram)

    fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False)
    transmittance_response.visualize(axs=axs[0, 0])
    interferogram.visualize(axs=axs[0, 1])
    spectrum_pinv.visualize(axs=axs[0, 2])

    fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False)
    transmittance_response.visualize(axs=axs[0, 0])
    interferogram.visualize(axs=axs[0, 1])
    spectrum_mich.visualize(axs=axs[0, 2])

    plt.show()


if __name__ == "__main__":
    main()
