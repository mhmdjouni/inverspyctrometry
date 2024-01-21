import matplotlib.pyplot as plt
import numpy as np

from src.common_utils.custom_vars import InterferometerType, InversionProtocolType
from src.common_utils.utils import generate_shifted_dirac, generate_sampled_opds, generate_wavenumbers_from_opds
from src.common_utils.interferogram import Interferogram
from src.inverse_model.inverspectrometer import MichelsonInverSpectrometer
from src.inverse_model.protocols import PseudoInverse, inversion_protocol_factory
from src.loaders.transmittance_response import transmittance_response_factory


def main():
    nb_opd, del_opd = 320, 0.175
    opds = generate_sampled_opds(nb_opd=nb_opd, del_opd=del_opd)

    # For controlled experiment: Reference interferometer and spectrum assumptions
    reflectance_cst = 0.13
    transmittance_cst = 1 - reflectance_cst
    nb_wn = nb_opd

    # Observation
    observation = (transmittance_cst * nb_wn) * generate_shifted_dirac(array=opds, shift=del_opd*4)
    interferogram = Interferogram(data=observation, opds=opds)

    # Generate or load a transmittance response (Generated using Interferometer || Loaded from disk)
    wavenumbers = generate_wavenumbers_from_opds(nb_wn=nb_wn, del_opd=del_opd)
    reflectance = reflectance_cst * np.ones_like(a=wavenumbers)
    transmittance = 1 - reflectance
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

    # Michelson reconstruction using IDCT
    inverspectrometer_michelson = MichelsonInverSpectrometer(transmittance=transmittance)
    spectrum_mich = inverspectrometer_michelson.reconstruct_spectrum(interferogram=interferogram)
    fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False)
    interferogram.visualize(axs=axs[0, 0])
    spectrum_mich.visualize(axs=axs[0, 1])

    # Plain IDCT
    inverter_idct = inversion_protocol_factory(option=InversionProtocolType.IDCT, kwargs={})
    spectrum_idct = inverter_idct.reconstruct_spectrum(
        interferogram=interferogram,
        transmittance_response=transmittance_response,
    )
    fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False)
    interferogram.visualize(axs=axs[0, 0])
    spectrum_idct.visualize(axs=axs[0, 1])

    # Pseudo-inverse
    inverter_pinv = inversion_protocol_factory(option=InversionProtocolType.PSEUDO_INVERSE, kwargs={})
    spectrum_pinv = inverter_pinv.reconstruct_spectrum(
        interferogram=interferogram,
        transmittance_response=transmittance_response,
    )
    fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False)
    interferogram.visualize(axs=axs[0, 0])
    spectrum_pinv.visualize(axs=axs[0, 1])

    # TSVD
    inverter_tsvd = inversion_protocol_factory(option=InversionProtocolType.TSVD, kwargs={"penalization_ratio": 1})
    spectrum_tsvd = inverter_tsvd.reconstruct_spectrum(
        interferogram=interferogram,
        transmittance_response=transmittance_response,
    )
    fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False)
    interferogram.visualize(axs=axs[0, 0])
    spectrum_tsvd.visualize(axs=axs[0, 1])

    # Ridge Regression
    inverter_rr = inversion_protocol_factory(option=InversionProtocolType.RIDGE_REGRESSION, kwargs={"penalization": 0})
    spectrum_rr = inverter_rr.reconstruct_spectrum(
        interferogram=interferogram,
        transmittance_response=transmittance_response,
    )
    fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False)
    interferogram.visualize(axs=axs[0, 0])
    spectrum_rr.visualize(axs=axs[0, 1])

    plt.show()


if __name__ == "__main__":
    main()
