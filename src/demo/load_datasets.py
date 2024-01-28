from src.interface.configuration import load_config


def main():
    db = load_config().database()

    for spectrum_id in range(3):
        dataset = db.datasets[spectrum_id]
        print(dataset)

        spectrum = db.dataset_spectrum(ds_id=spectrum_id)
        print(spectrum.data.shape)
        print(spectrum.wavenumbers)
        print(spectrum.wavenumbers_unit)

        print()


if __name__ == "__main__":
    main()
