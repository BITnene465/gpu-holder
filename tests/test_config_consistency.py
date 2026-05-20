from __future__ import annotations

from pathlib import Path

from gpu_holder.config import (
    DEFAULT_CONFIG_TEMPLATE,
    config_template,
    config_reference,
    recipe_reference,
    recipe_template,
    load_config_file,
    validate_config_keys,
)


ROOT = Path(__file__).resolve().parents[1]


def test_example_config_matches_default_template() -> None:
    example = (ROOT / "examples" / "gpu-holder.toml").read_text(encoding="utf-8")

    assert example == DEFAULT_CONFIG_TEMPLATE


def test_default_config_template_uses_documented_config_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "gpu-holder.toml"
    config_path.write_text(DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")
    parsed = load_config_file(config_path)

    assert validate_config_keys(parsed) == []


def test_minimal_profile_template_does_not_override_profile_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "gpu-holder.toml"
    config_path.write_text(config_template(profile="quota", minimal=True), encoding="utf-8")
    parsed = load_config_file(config_path)

    assert parsed == {"profile": "quota"}
    assert validate_config_keys(parsed) == []


def test_default_config_template_covers_non_optional_reference_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "gpu-holder.toml"
    config_path.write_text(DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")
    parsed = load_config_file(config_path)
    present_keys = set(parsed)
    optional_keys = {"max_held_gpus", "pause_file"}
    reference_keys = {str(field["key"]) for field in config_reference()}

    assert reference_keys - optional_keys <= present_keys


def test_recipe_templates_use_documented_config_keys(tmp_path: Path) -> None:
    recipe_names = [str(recipe["name"]) for recipe in recipe_reference()]

    assert {"first-run", "strict-quota", "busy-shared", "compute-only"} <= set(recipe_names)

    for recipe_name in recipe_names:
        config_path = tmp_path / f"{recipe_name}.toml"
        config_path.write_text(recipe_template(recipe_name), encoding="utf-8")
        parsed = load_config_file(config_path)

        assert parsed["profile"]
        assert validate_config_keys(parsed) == []
