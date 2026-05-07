
from __future__ import annotations

from citeextract import paths


def test_repo_root_contains_pyproject():
    assert (paths.repo_root() / "pyproject.toml").is_file()


def test_repo_root_contains_dockerfile():
    assert (paths.repo_root() / "Dockerfile").is_file()


def test_config_path_resolves_to_existing_file():
    assert paths.config_path().is_file()


def test_data_dir_default_is_repo_data():
    assert paths.data_dir() == paths.repo_root() / "data"


def test_data_dir_respects_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CITEEXTRACT_DATA_DIR", str(tmp_path))
    assert paths.data_dir() == tmp_path
