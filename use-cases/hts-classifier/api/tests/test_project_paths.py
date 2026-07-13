from __future__ import annotations

from pathlib import Path

from app.services.metal_composition.project_paths import _derive_roots


def test_derive_roots_for_local_repo_layout():
    service_dir = Path("/workspace/section-232-agent/api/app/services/metal_composition")

    api_root, project_root = _derive_roots(service_dir)

    assert api_root == Path("/workspace/section-232-agent/api")
    assert project_root == Path("/workspace/section-232-agent")


def test_derive_roots_for_cloud_foundry_layout():
    service_dir = Path("/home/vcap/app/app/services/metal_composition")

    api_root, project_root = _derive_roots(service_dir)

    assert api_root == Path("/home/vcap/app")
    assert project_root == Path("/home/vcap/app")
