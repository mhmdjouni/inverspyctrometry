from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from src.common_utils.custom_vars import InterferometerType, InversionProtocolType, NormOperatorType, \
    LinearOperatorMethod
from src.common_utils.transmittance_response import TransmittanceResponse
from src.common_utils.utils import generate_shifted_dirac, generate_sampled_opds, generate_wavenumbers_from_opds
from src.common_utils.interferogram import Interferogram
from src.inverse_model.inverspectrometer import MichelsonInverSpectrometer
from src.inverse_model.operators import CTVOperator, NormOperator, LinearOperator
from src.inverse_model.protocols import inversion_protocol_factory, InversionProtocol
from src.loaders.transmittance_response import transmittance_response_factory


@dataclass(frozen=True)
class InversionProtocolDemo:
    """
    Basic demo for inversion protocols
    """
    inverter: InversionProtocol

    def test(
            self,
            interferogram: Interferogram,
            transmittance_response: TransmittanceResponse,
    ):
        spectrum = self.inverter.reconstruct_spectrum(
            interferogram=interferogram,
            transmittance_response=transmittance_response,
        )
        fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False)
        interferogram.visualize(axs=axs[0, 0])
        spectrum.visualize(axs=axs[0, 1])


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
    idct_inverter = inversion_protocol_factory(option=InversionProtocolType.IDCT, kwargs={})
    demo_idct = InversionProtocolDemo(inverter=idct_inverter)
    demo_idct.test(interferogram=interferogram, transmittance_response=transmittance_response)

    # Pseudo-inverse
    pinv_inverter = inversion_protocol_factory(option=InversionProtocolType.PSEUDO_INVERSE, kwargs={})
    demo_pinv = InversionProtocolDemo(inverter=pinv_inverter)
    demo_pinv.test(interferogram=interferogram, transmittance_response=transmittance_response)

    # Truncated SVD
    tsvd_inverter = inversion_protocol_factory(option=InversionProtocolType.TSVD, kwargs={"penalization_ratio": 0.9})
    demo_tsvd = InversionProtocolDemo(inverter=tsvd_inverter)
    demo_tsvd.test(interferogram=interferogram, transmittance_response=transmittance_response)

    # Ridge Regression
    rr_inverter = inversion_protocol_factory(option=InversionProtocolType.RIDGE_REGRESSION, kwargs={"penalization": 10})
    demo_rr = InversionProtocolDemo(inverter=rr_inverter)
    demo_rr.test(interferogram=interferogram, transmittance_response=transmittance_response)

    # Loris-Verhoeven Primal-Dual
    lv_inverter = inversion_protocol_factory(
        option=InversionProtocolType.LORIS_VERHOEVEN,
        kwargs={
            # Sparsity on the spectrum = L1 norm + Identity
            "regularization_parameter": int(3e2),
            "prox_functional": CTVOperator(norm=NormOperator.from_norm(norm=NormOperatorType.L1O)),  # TODO: Load directly from a given norm, e.g., CTVOperator.from_norm(etc, etc), or from an Enum, or from a factory function using an Enum
            "domain_transform": LinearOperator.from_method(method=LinearOperatorMethod.IDENTITY),  # TODO: Load directly from a given Enum, or from a factory function using an Enum
            "nb_iters": int(3e3),
        }
    )
    demo_lv = InversionProtocolDemo(inverter=lv_inverter)
    demo_lv.test(interferogram=interferogram, transmittance_response=transmittance_response)

    plt.show()


if __name__ == "__main__":
    main()
