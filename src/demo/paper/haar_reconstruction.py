from dataclasses import replace

import numpy as np
from matplotlib import pyplot as plt

from src.common_utils.custom_vars import Opd, Deg
from src.common_utils.function_generator import GaussianGenerator
from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum
from src.direct_model.interferometer import FabryPerotInterferometer
from src.inverse_model.inverspectrometer import FabryPerotInverSpectrometerHaar
from src.inverse_model.protocols import IDCT, PseudoInverse


def main():
    is_imshow = True
    fp_orders = [0]
    reflectaces = [0.7]
    haar_orders = [10]
    opd_samples_skips = [0, 4, 10]
    extrapolation_options = [
        # {"kind": "none", "fill_value": ""},
        {"kind": "quadratic", "fill_value": (0., 0.)},
        # {"kind": "quadratic", "fill_value": "extrapolate"},
        # {"kind": "cubic", "fill_value": "extrapolate"},
        # {"kind": "linear", "fill_value": "extrapolate"},
        # {"kind": "nearest", "fill_value": "extrapolate"},
        # {"kind": "nearest-up", "fill_value": "extrapolate"},
        # {"kind": "zero", "fill_value": "extrapolate"},
        # {"kind": "slinear", "fill_value": "extrapolate"},
        # {"kind": "previous", "fill_value": "extrapolate"},
        # {"kind": "next", "fill_value": "extrapolate"},
    ]

    loop_inputs(
        is_imshow=is_imshow,
        fp_orders=fp_orders,
        reflectaces=reflectaces,
        haar_orders=haar_orders,
        extrapolation_options=extrapolation_options,
        opd_samples_skips=opd_samples_skips,
    )


def loop_inputs(
        is_imshow: bool,
        fp_orders: list,
        reflectaces: list,
        haar_orders: list,
        extrapolation_options: list,
        opd_samples_skips: list,
):
    for fp_order in fp_orders:
        for reflectance in reflectaces:
            for haar_order in haar_orders:
                for opd_samples_skip in opd_samples_skips:
                    for extrapolation in extrapolation_options:
                        (
                            interferogram,
                            spectrum_ref,
                            spectra_rec,
                        ) = paper_test(
                            gauss_coeffs=np.array([1., 0.9, 0.75]),
                            gauss_means=np.array([2000, 4250, 6500]),  # cm
                            gauss_stds=np.array([300, 1125, 400]),  # cm
                            opd_step=100 * 1e-7,  # cm
                            opd_num=2048,
                            reflectance=np.array([[reflectance]]),
                            # wn_min=24.31 * 1,  # cm
                            # wn_max=24.41 * 7,  # cm
                            wn_min=20.,  # cm
                            wn_max=20000.1,  # cm
                            haar_order=haar_order,
                            fp_order=fp_order,
                            wn_num_factor=10,
                            idct_correction=1.4,
                            opd_samples_skip=opd_samples_skip,
                            extrapolation=extrapolation,
                        )

                        colors = [
                            "green",
                            "blue",
                            "orange",
                        ]
                        labels = [
                            "IDCT",
                            "Haar",
                            "PINV",
                        ]
                        if is_imshow:
                            visualize_test(
                                interferogram=interferogram,
                                spectrum_ref=spectrum_ref,
                                spectra_rec=spectra_rec,
                                reflectance=reflectance,
                                haar_order=haar_order,
                                acq_ind=0,
                                ylim=[-0.5, 1.2],
                                labels=labels,
                                colors=colors,
                            )
        plt.show()


def continuous_spectrum(
        gauss_coeffs,
        gauss_means,
        gauss_stds,
        opd_step: float,
        opd_num: int,
        wn_min: float,
        wn_max: float,
        wn_num_factor: float,
):
    wn_max_dct = 1 / (2 * opd_step)
    wn_num = int(opd_num * (wn_max - wn_min) / wn_max_dct * wn_num_factor)
    wavenumbers = np.linspace(start=wn_min, stop=wn_max, num=wn_num, endpoint=False)  # um-1

    gaussian_gen = GaussianGenerator(
        coefficients=gauss_coeffs[:, None],
        means=gauss_means[:, None],
        stds=gauss_stds[:, None],
    )
    spectrum_data = gaussian_gen.generate_data(variable=wavenumbers)
    spectrum_ref = Spectrum(data=spectrum_data, wavenumbers=wavenumbers)

    return spectrum_ref


def direct_model(
        spectrum_ref: Spectrum,
        opd_step: float,
        opd_num: int,
        reflectance: np.ndarray[tuple[Opd, Deg], np.dtype[np.float_]],
        fp_order: int,
        wn_num_factor: float,
        idct_correction: float,
        opd_samples_skip: int = 0,
):
    opds = opd_step * (opd_samples_skip + np.arange(opd_num - opd_samples_skip))
    ifm = FabryPerotInterferometer(
        transmittance_coefficients=np.array([1.]),
        opds=opds,
        phase_shift=np.array([0]),
        reflectance_coefficients=reflectance,
        order=fp_order,
    )
    interferogram = ifm.acquire_interferogram(spectrum=spectrum_ref)
    interferogram = replace(interferogram, data=interferogram.data / wn_num_factor / idct_correction)

    return interferogram


def inverse_model(
        interferogram: Interferogram,
        opd_step: float,
        opd_num: int,
        reflectance: np.ndarray[tuple[Opd, Deg], np.dtype[np.float_]],
        wn_min: float,
        wn_max: float,
        haar_order: int,
        fp_order: int,
        wn_num_factor: float,
):
    wn_step = 1 / (2 * interferogram.opds.max())  # Delta x
    wn_max_bandwidth = 1 / (2 * opd_step)  # full Nyquist-Shannon bandwidth of wavenumbers [0, wn_max_bandwidth]
    wn_num = int(opd_num * (wn_max - wn_min) / wn_max_bandwidth)
    wavenumbers = wn_min + wn_step * np.arange(wn_num)  # um-1

    haar_inv = FabryPerotInverSpectrometerHaar(
        transmittance=np.array([1.]),
        wavenumbers=wavenumbers,
        reflectance=reflectance[0],
        order=haar_order,
        is_mean_center=True,
    )
    spectrum_haar = haar_inv.reconstruct_spectrum(interferogram=interferogram)

    wavenumbers = np.linspace(start=wn_min, stop=wn_max, num=int(wn_num * wn_num_factor), endpoint=False)  # um-1

    interferometer = FabryPerotInterferometer(
        transmittance_coefficients=np.array([1.]),
        opds=interferogram.opds,
        phase_shift=np.array([0]),
        reflectance_coefficients=reflectance,
        order=fp_order,
    )
    transmittance_response = interferometer.transmittance_response(wavenumbers=wavenumbers)

    idct_inv = IDCT(is_mean_center=True)
    spectrum_idct = idct_inv.reconstruct_spectrum(
        interferogram=interferogram,
        transmittance_response=transmittance_response
    )

    pinv_inv = PseudoInverse()
    spectrum_pinv = pinv_inv.reconstruct_spectrum(
        interferogram=interferogram,
        transmittance_response=transmittance_response
    )

    return (
        spectrum_idct,
        spectrum_haar,
        spectrum_pinv,
    )


def paper_test(
        gauss_coeffs,
        gauss_means,
        gauss_stds,
        opd_step: float,
        opd_num: int,
        reflectance: np.ndarray[tuple[Opd, Deg], np.dtype[np.float_]],
        wn_min: float,
        wn_max: float,
        haar_order: int,
        fp_order: int,
        wn_num_factor: float,
        idct_correction: float,
        extrapolation: dict,
        opd_samples_skip: int = 0,
):
    spectrum_ref = continuous_spectrum(
        gauss_coeffs=gauss_coeffs,
        gauss_means=gauss_means,
        gauss_stds=gauss_stds,
        opd_step=opd_step,
        opd_num=opd_num,
        wn_min=wn_min,
        wn_max=wn_max,
        wn_num_factor=wn_num_factor,
    )

    interferogram = direct_model(
        spectrum_ref=spectrum_ref,
        opd_step=opd_step,
        opd_num=opd_num,
        reflectance=reflectance,
        fp_order=fp_order,
        wn_num_factor=wn_num_factor,
        idct_correction=idct_correction,
        opd_samples_skip=opd_samples_skip,
    )

    if extrapolation["kind"] != "none":
        interferogram = interferogram.extrapolate(
            kind=extrapolation["kind"],
            fill_value=extrapolation["fill_value"],
        )

    spectra_rec = inverse_model(
        interferogram=interferogram,
        opd_step=opd_step,
        opd_num=opd_num,
        reflectance=reflectance,
        wn_min=wn_min,
        wn_max=wn_max,
        haar_order=haar_order,
        fp_order=fp_order,
        wn_num_factor=wn_num_factor,
    )

    return (
        interferogram,
        spectrum_ref,
        spectra_rec,
    )


def visualize_test(
        reflectance,
        haar_order,
        interferogram,
        spectrum_ref,
        spectra_rec,
        acq_ind,
        ylim,
        colors,
        labels,
):
    fig, axs = plt.subplots(1, 2, squeeze=False, figsize=(9, 5))

    axs_current = axs[0, 0]
    interferogram.visualize(axs=axs_current, acq_ind=acq_ind)
    axs_current.set_title(f'Simulated Interferogram, R = {reflectance}')
    axs_current.set_xlabel('OPDs [cm]')
    axs_current.set_ylabel('Intensity')

    axs_current = axs[0, 1]
    spectrum_ref.visualize(axs=axs_current, acq_ind=acq_ind, color="red", label='Reference', ylim=ylim)
    for spectrum_rec, color, label in zip(spectra_rec, colors, labels):
        spectrum_rec_eq, _ = spectrum_rec.match_stats(reference=spectrum_ref, axis=-2)
        spectrum_rec_eq.visualize(
            axs=axs_current, acq_ind=acq_ind, linestyle="dashed", color=color, label=label, ylim=ylim
        )
    axs_current.set_title(f'Reconstructed Spectrum, M = {haar_order}')
    axs_current.set_xlabel('Wavenumbers [cm-1]')
    axs_current.set_ylabel('Intensity')

    plt.legend()
    plt.grid(True)


if __name__ == "__main__":
    main()
