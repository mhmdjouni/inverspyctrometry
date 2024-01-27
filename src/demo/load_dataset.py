from pprint import pprint

from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum
from src.interface.configuration import load_config


def main():
    config = load_config()
    db = config.database()
    pprint(db.datasets)
    pprint(db.interferometers)
    pprint(db.inversion_protocols)
    pprint(db.noise_levels)

    print()

    spectrum_id = 0  # solar dataset
    interferogram_id = 3  # mc-451 dataset
    pprint(db.datasets[spectrum_id])
    pprint(db.datasets[interferogram_id])


if __name__ == "__main__":
    main()
