"""
Load a transmittance response:
- Generate it from Interferometer class, given some Characterization (Reflectance, OPDs)
- Load it from disk (raw calibration data), with list of OPDs and Wavenumbers
"""
import numpy as np

from src.common_utils.transmittance_response import TransmittanceResponse
from src.direct_model.interferometer import interferometer_factory


def transmittance_response_factory(
        option: str,  # Might switch to Enum, Generate from Interferometer or Load from disk
        kwargs: dict,
) -> TransmittanceResponse:
    # TODO: Parameters to be optimized?
    if option == "generate":
        print("Generating a TransmittanceResponse from Interferometer class given a Characterization")
        interferometer = interferometer_factory(
            option=kwargs["type"],
            transmittance=kwargs["transmittance"],
            opds=kwargs["opds"],
            order=kwargs["order"],
        )
        transmittance_response = interferometer.generate_transmittance_response(wavenumbers=kwargs["wavenumbers"])

    elif option == "load":
        # TODO: Make a loader for this using path
        print(f"Loading a TransmittanceResponse (raw calibration data) from disk given a path {kwargs['path']}")
        transmittance_response = TransmittanceResponse(
            data=np.eye(N=kwargs["opds"].size, M=kwargs["wavenumbers"].size),  # dummy identity matrix
            wavenumbers=kwargs["wavenumbers"],
            opds=kwargs["opds"],
        )

    else:
        raise ValueError(f"Option '{option}' is not supported.")

    return transmittance_response
