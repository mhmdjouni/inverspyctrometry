from dataclasses import replace, dataclass
from types import SimpleNamespace

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum
from src.common_utils.utils import calculate_rmse
from src.demo.paper.haar.utils import generate_synthetic_spectrum, generate_interferogram, compute_wavenumbers, \
    oversample_wavenumbers, oversample_spectrum, invert_haar, load_spectrum, invert_protocols
from src.direct_model.characterization import Characterization
from src.direct_model.interferometer import FabryPerotInterferometer
from src.interface.configuration import load_config


@dataclass(frozen=True)
class Extrapolation:
    case: int
    description: str
    opds_resampler: str
    extrap_kind: str
    extrap_fill: any
    transmat_extrap: str


def load_interferogram(option: str) -> tuple[Interferogram, np.ndarray]:
    if option == "mc451":
        dataset_id = 3
        acq_id = 200
    elif option == "mc651":
        dataset_id = 4
        acq_id = 300
    elif option == "cc_green":
        dataset_id = 5
        acq_id = 0
    else:
        raise ValueError(f"Option {option} is not supported.")

    db = load_config().database()
    interferogram = db.dataset_interferogram(ds_id=dataset_id)
    central_wavenumbers = db.dataset_central_wavenumbers(dataset_id=dataset_id)
    interferogram = replace(interferogram, data=interferogram.data[:, acq_id:acq_id + 1])
    return interferogram, central_wavenumbers


def get_interferogram(ifm_type: str, extrap: Extrapolation):
    """
    Load the Interferogram dataset of interest and the central wavenumbers
    Sort the OPDs
    Extrapolate the OPDs starting from the Zero-OPD
    """
    interferogram, central_wavenumbers = load_interferogram(option=ifm_type)
    interferogram = interferogram.sort_opds()
    interferogram = interferogram.extrapolate(
        support_resampler=extrap.opds_resampler,
        kind=extrap.extrap_kind,
        fill_value=extrap.extrap_fill,
    )
    return interferogram


def invert_haar_real(wavenumbers_central, characterization_id, haar_order, interferogram_sim):
    db = load_config().database()
    characterization = db.characterization(char_id=characterization_id)
    characterization = characterization.sort_opds()
    # transmittance = np.mean(characterization.transmittance(wavenumbers=wavenumbers_central))[None]
    reflectance = np.mean(characterization.reflectance(wavenumbers=wavenumbers_central))[None]
    fp_obj = SimpleNamespace(
        transmittance=1.,
        reflectance=reflectance,
    )
    spectrum = invert_haar(wavenumbers_central, fp_obj, haar_order, interferogram_sim)
    return spectrum


def invert_protocols_real(protocols, wavenumbers, characterization_id, interferogram, spectrum_ref, extrap: Extrapolation):
    db = load_config().database()
    characterization = db.characterization(char_id=characterization_id)
    characterization = characterization.sort_opds()
    characterization = characterization.extrapolate_opds(support_resampler=extrap.opds_resampler)
    fp_obj = SimpleNamespace(
        transmittance=characterization.transmittance_coefficients,
        phase_shift=characterization.phase_shift,
        reflectance=characterization.reflectance_coefficients,
        order=characterization.order,
    )
    spectrum_protocols = invert_protocols(protocols, wavenumbers, fp_obj, interferogram, spectrum_ref=spectrum_ref)
    return spectrum_protocols


def main():
    # OPTIONS

    opts = SimpleNamespace(
        ifm_type="cc_green",  # "mc451", "mc651", "cc_green"
        extrap = Extrapolation(
            case=3,
            description="Concatenate lowest OPDs but extrapolate the interferogram values using fourier series",
            opds_resampler="concatenate_missing",
            extrap_kind="linear",
            extrap_fill="fourier",
            transmat_extrap="model",
        )
    )
    char_id = 0
    haar_order = 20
    protocols = [
        SimpleNamespace(id=0, label="IDCT", color="green"),
        SimpleNamespace(id=5, label="TSVD", color="pink"),
        SimpleNamespace(id=6, label="RR", color="orange"),
        SimpleNamespace(id=10, label="LV-L1", color="purple"),
    ]

    # OBSERVATION (SIMULATION)

    print("\n\nOBSERVATION")
    interferogram_sim = get_interferogram(opts.ifm_type, opts.extrap)
    interferogram_sim = interferogram_sim.rescale(new_max=1., axis=-2)

    # REFERENCE SPECTRUM

    print("\n\nREFERENCE SPECTRUM")
    spectrum_ref = load_spectrum(option=opts.ifm_type)
    spectrum_ref = spectrum_ref.rescale(new_max=1., axis=-2)

    # INVERSION

    print("\n\nINVERSION")
    wavenumbers = spectrum_ref.wavenumbers
    spectrum_haar = invert_haar_real(wavenumbers, char_id, haar_order, interferogram_sim)
    spectrum_protocols = invert_protocols_real(protocols, wavenumbers, char_id, interferogram_sim, spectrum_ref, opts.extrap)

    # METRICS

    print("\n\nMETRICS")
    rmse = calculate_rmse(
        array=spectrum_haar.data,
        reference=spectrum_ref.data,
        is_match_axis=-2,
        is_match_stats=True,
        is_rescale_reference=True,
    )
    print(f"\t{'HAAR:':6} RMSE = {rmse:.4f}")
    for spectrum_protocol, protocol in zip(spectrum_protocols, protocols):
        rmse = calculate_rmse(
            array=spectrum_protocol.data,
            reference=spectrum_ref.data,
            is_match_axis=-2,
            is_match_stats=True,
            is_rescale_reference=True,
        )
        print(f"\t{protocol.label + ':':6} RMSE = {rmse:.4f}")

    # VISUALIZATION

    print("\n\nVISUALIZATION")
    acq_idx = 0
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    axs_spc, axs_ifm, axs_rec = axs
    spc_ylim = [-0.1, 1.4]

    spectrum_ref.visualize(
        axs=axs_spc,
        acq_ind=acq_idx,
        label="Reference",
        color="red",
        ylim=spc_ylim,
    )

    interferogram_sim.visualize(
        axs=axs_ifm,
        acq_ind=acq_idx,
        title="Simulated Interferogram",
        color="red",
    )

    spectrum_ref.visualize(
        axs=axs_rec,
        acq_ind=acq_idx,
        label="Reference",
        color="red",
        ylim=spc_ylim,
    )

    for spectrum_protocol, protocol in zip(spectrum_protocols, protocols):
        spectrum_protocol, _ = spectrum_protocol.match_stats(reference=spectrum_ref)
        spectrum_protocol.visualize(
            axs=axs_rec,
            acq_ind=acq_idx,
            label=protocol.label,
            color=protocol.color,
            linestyle="--",
            ylim=spc_ylim,
        )

    spectrum_haar, _ = spectrum_haar.match_stats(reference=spectrum_ref, axis=-2)
    spectrum_haar.visualize(
        axs=axs_rec,
        acq_ind=acq_idx,
        label="HAAR",
        color="blue",
        linestyle=":",
        marker="x",
        markevery=40,
        ylim=spc_ylim,
    )

    plt.show()


if __name__ == "__main__":
    main()
