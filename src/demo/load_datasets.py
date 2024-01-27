from src.interface.configuration import load_config


def main():
    db = load_config().database()
    spectrum_id = 0

    dataset = db.datasets[spectrum_id]
    print(dataset)
    print()

    spectrum = db.dataset_spectrum(ds_id=spectrum_id)
    print(spectrum)


if __name__ == "__main__":
    main()
