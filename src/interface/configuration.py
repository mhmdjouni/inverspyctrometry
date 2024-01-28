import json
from pathlib import Path

from omegaconf import OmegaConf
from pydantic import BaseModel, DirectoryPath, FilePath

from src.database.database import DatabaseSchema


class DirectoryPathSchema(BaseModel):
    data: DirectoryPath
    datasets: DirectoryPath
    notebooks: DirectoryPath
    reports: DirectoryPath
    src: DirectoryPath


class DatabasePathSchema(BaseModel):
    datasets: FilePath
    interferometers: FilePath
    inversion_protocols: FilePath
    noise_levels: FilePath

    def open(self) -> DatabaseSchema:
        datasets_dict = self._load_dict_from_json(path=self.datasets)
        interferometers_dict = self._load_dict_from_json(path=self.interferometers)
        inversion_protocols_dict = self._load_dict_from_json(path=self.inversion_protocols)
        noise_levels_dict = self._load_dict_from_json(path=self.noise_levels)
        return DatabaseSchema(
            **datasets_dict,
            **interferometers_dict,
            **inversion_protocols_dict,
            **noise_levels_dict,
        )

    @staticmethod
    def _load_dict_from_json(path: Path) -> dict:
        with open(path) as json_opened:
            database_dict = json.load(json_opened)
        database_obj = OmegaConf.create(database_dict)
        root_dict = OmegaConf.to_container(database_obj, resolve=True)
        return root_dict


class ConfigSchema(BaseModel):
    directory_paths: DirectoryPathSchema
    database_paths: DatabasePathSchema

    def database(self) -> DatabaseSchema:
        return self.database_paths.open()


def load_config(config_sub_path: Path | str = "data/database/config.json") -> ConfigSchema:
    project_path = Path(__file__).resolve().parents[2]
    config_path = project_path / config_sub_path
    with open(config_path) as config_file:
        config_dict = json.load(fp=config_file)
    return ConfigSchema(**config_dict)

