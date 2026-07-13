from __future__ import annotations

from pathlib import Path

from app.services.metal_composition import config as config_module


def _patch_config_roots(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    """Point config loading at an isolated fake repository and return its roots."""
    fake_project_root = tmp_path / "repo"
    fake_api_root = fake_project_root / "api"
    fake_api_root.mkdir(parents=True, exist_ok=True)
    (fake_api_root / ".env").write_text("", encoding="utf-8")

    monkeypatch.setattr(config_module, "PROJECT_ROOT", fake_project_root)
    monkeypatch.setattr(config_module, "API_ROOT", fake_api_root)
    return fake_project_root, fake_api_root


def test_get_settings_defaults_to_uploaded_document_root(monkeypatch, tmp_path):
    """Settings should default uploaded PDFs to the API cache upload directory."""
    _project_root, fake_api_root = _patch_config_roots(monkeypatch, tmp_path)
    monkeypatch.delenv("METAL_COMPOSITION_UPLOADED_DOCUMENT_ROOT", raising=False)

    config_module.get_settings.cache_clear()
    try:
        settings = config_module.get_settings()
    finally:
        config_module.get_settings.cache_clear()

    assert settings.uploaded_document_root == (
        fake_api_root / ".cache" / "metal_composition" / "uploads"
    ).resolve()
    assert not hasattr(settings, "document_root")


def test_get_settings_uploaded_document_root_env_override_wins(monkeypatch, tmp_path):
    """Relative upload-root overrides should resolve from the API directory."""
    _project_root, fake_api_root = _patch_config_roots(monkeypatch, tmp_path)
    monkeypatch.setenv("METAL_COMPOSITION_UPLOADED_DOCUMENT_ROOT", "./custom-uploads")

    config_module.get_settings.cache_clear()
    try:
        settings = config_module.get_settings()
    finally:
        config_module.get_settings.cache_clear()

    assert settings.uploaded_document_root == (fake_api_root / "custom-uploads").resolve()


def test_get_settings_classification_ownership_table_env_override(monkeypatch, tmp_path):
    """Settings should load the HANA ownership table name for active job supersession."""
    _patch_config_roots(monkeypatch, tmp_path)
    monkeypatch.setenv("METAL_COMPOSITION_UI_STATE_CLASSIFICATION_OWNERSHIP_TABLE", "CUSTOM_OWNERSHIP")

    config_module.get_settings.cache_clear()
    try:
        settings = config_module.get_settings()
    finally:
        config_module.get_settings.cache_clear()

    assert settings.ui_state_classification_ownership_table == "CUSTOM_OWNERSHIP"
