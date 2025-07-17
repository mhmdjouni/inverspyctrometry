import matplotlib.pyplot as plt
import numpy as np

from inverspyctrometry.interface.configuration import load_config
from inverspyctrometry.inverse_model.operators import wavelet_transform, inverse_wavelet_transform


def main():
    config = load_config()
    db = config.database()

    spectra = db.dataset(dataset_id=1).data[:, :]  # shape K x N
    print(spectra.shape)
    coeff_slices = wavelet_transform(x=spectra, wavelet="db8", level=3)[1]

    # Apply DWT to each signal / column of the spectra array and return the concatenated DWT coefficients
    spectra_dwt = wavelet_transform(x=spectra, wavelet="db8", level=3)[0]

    # Reconstruct the signals using IDWT
    spectra_rec = inverse_wavelet_transform(u=spectra_dwt, wavelet="db8", coeff_slices=coeff_slices)

    # Ensure reconstruction is close to the original
    error = np.linalg.norm(spectra - spectra_rec) / np.linalg.norm(spectra)
    print(f"Reconstruction error: {error:.6e}")


if __name__ == "__main__":
    main()
