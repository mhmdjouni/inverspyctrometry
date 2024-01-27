from src.interface.configuration import load_config


def main():
    config = load_config()

    path = config.directory_paths
    print(path)
    print()

    db_paths = config.database_paths
    print(db_paths)
    print()

    db = config.database()
    print(db.datasets)
    print(db.interferometers)
    print(db.inversion_protocols)
    print(db.noise_levels)


if __name__ == "__main__":
    main()
