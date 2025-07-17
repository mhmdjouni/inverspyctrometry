import time
from dataclasses import dataclass
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from inverspyctrometry.common_utils.custom_vars import LinearOperatorMethod
from inverspyctrometry.common_utils.interferogram import Interferogram
from inverspyctrometry.direct_model.characterization import Characterization
from inverspyctrometry.interface.configuration import load_config
from inverspyctrometry.inverse_model.operators import wavelet_transform


# TODO: Consider cropping the wavenumbers of mc651 starting from 1.1
# TODO: Dump the experiment configuration in the reports/experiment/ folder (could also be a separate function to call)


@dataclass(frozen=True)
class ExtrapCase:
    case: int
    description: str
    opds_resampler: str
    extrap_kind: str
    extrap_fill: any
    transmat_extrap: str


def visualize_extraps(
        axs,
        extrap_support,
        extrap_data,
        ref_support,
        ref_data,
        extrap_case,
        data_title,
        support_label,
        extrap_legend=None,
        ref_legend=None,
):
    line_extrap, = axs.plot(extrap_support, extrap_data, color="C1")
    line_ref, = axs.plot(ref_support, ref_data, color="C0")
    axs.set_title(
        f"{data_title}"
        # f"\nCase: {extrap_case.case}; Fill: {extrap_case.extrap_fill}"
    )
    axs.grid("visible")
    axs.set_xlabel(support_label)
    axs.set_ylabel("Intensity")
    if ref_legend is None:
        ref_legend = "Reference"
    if extrap_legend is None:
        extrap_legend = "Extrapolated"
    axs.legend([line_ref, line_extrap], [ref_legend, extrap_legend])


def visualize_ifgm_extraps(
        axs,
        extrapolated: Interferogram,
        reference: Interferogram,
        extrap_case: ExtrapCase,
        acq_idx: int,
):
    visualize_extraps(
        axs=axs,
        extrap_support=extrapolated.opds,
        extrap_data=extrapolated.data[:, acq_idx],
        ref_support=reference.opds,
        ref_data=reference.data[:, acq_idx],
        extrap_case=extrap_case,
        data_title=f"Interferogram Data, Acq idx = {acq_idx}",
        support_label="OPDs [um]",
    )


def visualize_char_extraps(
        fig,
        axs,
        extrapolated: Characterization,
        reference: Characterization,
        extrap_case: ExtrapCase,
        wavenumbers: np.ndarray,
        acq_idx: int,
        opd_idx: int,
):
    transmat_ref = reference.transmittance_response(wavenumbers=wavenumbers)
    transmat_extrap = extrapolated.transmittance_response(wavenumbers=wavenumbers)
    visualize_extraps(
        axs=axs[0, 1],
        extrap_support=extrapolated.opds,
        extrap_data=transmat_extrap.data[:, acq_idx],
        ref_support=reference.opds,
        ref_data=transmat_ref.data[:, acq_idx],
        extrap_case=extrap_case,
        data_title=f"Transfer Matrix, Wvn idx = {acq_idx}",
        support_label="OPDs [um]",
    )

    transmat_extrap.visualize(
        fig=fig,
        axs=axs[0, 2],
        title="Transfer Matrix (Extrapolated)",
    )

    visualize_extraps(
        axs=axs[1, 0],
        extrap_support=extrapolated.opds,
        extrap_data=extrapolated.phase_shift,
        ref_support=reference.opds,
        ref_data=reference.phase_shift,
        extrap_case=extrap_case,
        data_title=f"Phase Shift",
        support_label="OPDs [um]",
    )

    transmittance_extrap = extrapolated.transmittance(wavenumbers=wavenumbers)
    transmittance_ref = reference.transmittance(wavenumbers=wavenumbers)
    visualize_extraps(
        axs=axs[1, 1],
        extrap_support=wavenumbers,
        extrap_data=transmittance_extrap[opd_idx, :],
        ref_support=wavenumbers,
        ref_data=transmittance_ref[opd_idx, :],
        extrap_case=extrap_case,
        data_title=f"Transmittance",
        support_label="Wavenumbers [um-1]",
        ref_legend=f"OPD = {reference.opds[opd_idx]:.2f} um (real polyn)",
        extrap_legend=f"OPD = {extrapolated.opds[opd_idx]:.2f} um (mean polyn)",
    )

    reflectance_extrap = extrapolated.reflectance(wavenumbers=wavenumbers)
    reflectance_ref = reference.reflectance(wavenumbers=wavenumbers)
    visualize_extraps(
        axs=axs[1, 2],
        extrap_support=wavenumbers,
        extrap_data=reflectance_extrap[opd_idx, :],
        ref_support=wavenumbers,
        ref_data=reflectance_ref[opd_idx, :],
        extrap_case=extrap_case,
        data_title=f"Reflectance",
        support_label="Wavenumbers [um-1]",
        ref_legend=f"OPD = {reference.opds[opd_idx]:.2f} um (real polyn)",
        extrap_legend=f"OPD = {extrapolated.opds[opd_idx]:.2f} um (mean polyn)",
    )


def run_one_experiment(
        experiment_id_options: list,
        extrap_case: ExtrapCase,
        acq_idx: int,
):
    # fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=(20, 10))

    config = load_config()
    db = config.database()

    reports_folder = config.directory_paths.reports

    for experiment_id in experiment_id_options:
        experiment_config = db.experiments[experiment_id]
        pprint(dict(experiment_config))
        experiment_dir = reports_folder / f"experiment_{experiment_id}" / "reconstruction"

        for ds_id in experiment_config.dataset_ids:
            interferograms_ref = db.dataset_interferogram(ds_id=ds_id)
            wavenumbers_ifgm = db.dataset_central_wavenumbers(dataset_id=ds_id)
            print(f"Dataset: {db.datasets[ds_id].title.upper()}")
            dataset_dir = experiment_dir / f"invert_{db.datasets[ds_id].title}"
            interferograms_ref = interferograms_ref.extrapolate(
                support_resampler=extrap_case.opds_resampler,
                kind=extrap_case.extrap_kind,
                fill_value=extrap_case.extrap_fill,
            )

            interferograms_ref = interferograms_ref.rescale(new_max=1, axis=-2)
            # visualize_ifgm_extraps(
            #     axs=axs[0, 0],
            #     extrapolated=interferograms_extrap,
            #     reference=interferograms_ref,
            #     extrap_case=extrap_case,
            #     acq_idx=acq_idx,
            # )

            for char_id in experiment_config.interferometer_ids:
                characterization = db.characterization(characterization_id=char_id)
                print(f"\tCharacterization: {db.characterizations[char_id].title.upper()}")
                characterization_dir = dataset_dir / f"{db.characterizations[char_id].title}"

                characterization = characterization.extrapolate_opds(support_resampler=extrap_case.opds_resampler)
                # visualize_char_extraps(
                #     fig=fig,
                #     axs=axs,
                #     extrapolated=characterization,
                #     reference=characterization,
                #     extrap_case=extrap_case,
                #     wavenumbers=wavenumbers_ifgm,
                #     acq_idx=acq_idx,
                #     opd_idx=0,
                # )

                transfer_matrix = characterization.transmittance_response(wavenumbers=wavenumbers_ifgm)
                transfer_matrix = transfer_matrix.rescale(new_max=1, axis=None)

                for ip_id in experiment_config.inversion_protocol_ids:
                    lambdaas = db.inversion_protocol_lambdaas(inv_protocol_id=ip_id)
                    print(f"\t\tInversion Protocol: {db.inversion_protocols[ip_id].title.upper()}")
                    inverter_dir = characterization_dir / f"{db.inversion_protocols[ip_id].title}"

                    spectra_rec_all = np.zeros(shape=(lambdaas.size, wavenumbers_ifgm.size, interferograms_ref.data.shape[-1]))
                    execution_times_all = np.zeros(shape=lambdaas.size)
                    cost_progress_all = np.zeros(shape=(lambdaas.size, db.inversion_protocols[ip_id].nb_iters))

                    kwargs = {}
                    if db.inversion_protocols[ip_id].linear_operator == LinearOperatorMethod.DWT:
                        coeff_slices = wavelet_transform(x=wavenumbers_ifgm[..., None], wavelet="db8", level=3)[1]
                        kwargs = {
                            "wavelet": "db8",
                            "level": 3,
                            "coeff_slices": coeff_slices,
                        }

                    for il, lambdaa in enumerate(lambdaas):
                        inverter = db.inversion_protocol(
                            inv_protocol_id=ip_id,
                            lambdaa=lambdaa,
                            is_compute_and_save_cost=True,
                            experiment_id=-1,
                            **kwargs
                        )

                        start_time = time.time()
                        spectra_rec, cost_progress = inverter.reconstruct_spectrum(
                            interferogram=interferograms_ref, transmittance_response=transfer_matrix
                        )
                        end_time = time.time()
                        execution_time = end_time - start_time

                        spectra_rec_all[il] = spectra_rec.data
                        execution_times_all[il] = execution_time
                        cost_progress_all[il] = cost_progress

                    # Save all spectral reconstructions wrt lambda, per inv_protocol per dataset
                    if not inverter_dir.exists():
                        inverter_dir.mkdir(parents=True, exist_ok=True)
                    np.save(file=inverter_dir / "spectra_rec_all.npy", arr=spectra_rec_all)
                    np.save(file=inverter_dir / "execution_times_all.npy", arr=execution_times_all)
                    np.save(file=inverter_dir / "cost_progress_all.npy", arr=cost_progress_all)

            # Save the wavenumbers for the sake of independent plots
            if not dataset_dir.exists():
                dataset_dir.mkdir(parents=True, exist_ok=True)
            np.save(file=dataset_dir / "wavenumbers", arr=wavenumbers_ifgm)

        # Dump the experiment's JSON
        # if not experiment_dir.exists():
        #     experiment_dir.mkdir(parents=True, exist_ok=True)
        # experiment_config.model_dump_json()


def main():
    extrap_cases = [
        # ExtrapCase(
        #     case=1,
        #     description="Concatenate lowest OPDs but set the corresponding interferogram values to zero",
        #     opds_resampler="concatenate_missing",
        #     extrap_kind="linear",
        #     extrap_fill=(0., 0.),
        #     transmat_extrap="model"
        # ),
        # ExtrapCase(
        #     case=2,
        #     description="Concatenate lowest OPDs but extrapolate the interferogram values using conventional methods",
        #     opds_resampler="concatenate_missing",
        #     extrap_kind="linear",
        #     extrap_fill="extrapolate",
        #     transmat_extrap="model"
        # ),
        ExtrapCase(
            case=3,
            description="Concatenate lowest OPDs but extrapolate the interferogram values using fourier series",
            opds_resampler="concatenate_missing",
            extrap_kind="linear",
            extrap_fill="fourier",
            transmat_extrap="model"
        ),
    ]

    experiment_id_options = [1, 2]

    for extrap_case in extrap_cases:
        run_one_experiment(
            experiment_id_options=experiment_id_options,
            extrap_case=extrap_case,
            acq_idx=200,
        )
    plt.show()


if __name__ == "__main__":
    main()
