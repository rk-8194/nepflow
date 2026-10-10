from pathlib import Path

import pytest

from nepflow.config import (
    GenerationConfig,
    find_config_path,
    load_config,
    load_legacy_config,
    render_default_config,
)
from nepflow.errors import ConfigurationError

BASE_CONFIG = """
[project]
name = demo
schema_version = 1
random_seed = 7

[composition]
elements = W, Cr
gas_elements = He
composition_step = 0.125

[generation]
crystal_structures = bcc, fcc
target_n_atoms = 64
use_liquid = false
elastic_strain_amplitudes = -0.02, -0.01, 0.01, 0.02

[hpc]
vasp_command = vasp_std
"""


def write_config(tmp_path: Path, text: str = BASE_CONFIG) -> Path:
    path = tmp_path / "project.config"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def test_loader_builds_immutable_typed_root_and_applies_defaults(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))

    assert config.project.random_seed == 7
    assert config.composition.elements == ("W", "Cr")
    assert config.composition.gas_elements == ("He",)
    assert config.generation.n_gas_interstitials == 10
    assert config.generation.surface_enabled is False
    assert config.generation.n_surfaces == 0
    assert config.hpc.vasp_command == "vasp_std"
    with pytest.raises(AttributeError):
        config.project = config.project


def test_canonical_point_defect_defaults_round_trip_without_chemistry_inference(
    tmp_path: Path,
) -> None:
    expected = GenerationConfig()
    assert expected.target_n_atoms == 128
    assert expected.vacancy_min == 0.008
    assert expected.vacancy_max == 0.025
    assert expected.interstitial_d_min == 1.65
    assert expected.interstitial_min == 0.008
    assert expected.interstitial_max == 0.025
    assert expected.defect_defect_d_min == 1.65
    assert expected.periodic_image_d_min == 6.0
    assert expected.interstitial_max_attempts == 1000
    assert expected.substitution_min == 0.008
    assert expected.substitution_max == 0.025
    assert expected.antisite_min == 0.008
    assert expected.antisite_max == 0.025
    assert expected.n_substitutions == 0
    assert expected.n_antisites == 0

    fields = (
        "target_n_atoms",
        "vacancy_min",
        "vacancy_max",
        "interstitial_d_min",
        "interstitial_min",
        "interstitial_max",
        "defect_defect_d_min",
        "periodic_image_d_min",
        "interstitial_max_attempts",
        "substitution_min",
        "substitution_max",
        "antisite_min",
        "antisite_max",
    )
    fallback = load_config(write_config(tmp_path, BASE_CONFIG)).generation
    for field in fields:
        assert getattr(fallback, field) == getattr(expected, field)

    rendered = render_default_config(
        "demo",
        {
            "elements": "W,Cr",
            "gas_elements": "",
            "crystal_structures": "bcc",
            "target_n_atoms": "128",
            "scp_address": "",
            "ntasks": "1",
        },
    )
    path = write_config(tmp_path, rendered)
    loaded = load_config(path).generation
    for field in fields:
        assert getattr(loaded, field) == getattr(expected, field)
    for field in (
        "vacancy_species",
        "substitution_pairs",
        "antisite_pairs",
        "interstitial_sites",
        "crystallographic_interstitial_sites",
    ):
        assert getattr(loaded, field) == ()
        assert f"{field}=\n" in rendered


def test_generation_source_scope_defaults_are_explicit_and_stable(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    scope_fields = (
        "volume_sources",
        "elastic_sources",
        "rattle_sources",
        "liquid_sources",
        "vacancy_sources",
        "interstitial_sources",
        "gas_interstitial_sources",
        "vacancy_interstitial_sources",
        "gas_in_vacancy_sources",
    )

    assert all(getattr(config.generation, field) == ("all",) for field in scope_fields)
    effective_generation = config.effective_mapping()["generation"]
    assert all(effective_generation[field] == ["all"] for field in scope_fields)


def test_generation_source_scope_round_trip_preserves_effective_mapping(tmp_path: Path) -> None:
    first_text = BASE_CONFIG.replace(
        "[generation]",
        "[generation]\nvolume_sources = mp_phase, mp_gas_phase\nrattle_sources = sqs",
        1,
    )
    second_text = BASE_CONFIG.replace(
        "[generation]",
        "[generation]\nvolume_sources = MP_PHASE,MP_GAS_PHASE\nrattle_sources=sqs",
        1,
    )

    first = load_config(write_config(tmp_path, first_text))
    second = load_config(write_config(tmp_path, second_text))

    assert first.generation.volume_sources == ("mp_phase", "mp_gas_phase")
    assert first.generation.rattle_sources == ("sqs",)
    assert first.effective_mapping() == second.effective_mapping()


def test_surface_orientation_semantics_round_trip_without_count_reinterpretation(
    tmp_path: Path,
) -> None:
    text = BASE_CONFIG.replace(
        "[generation]",
        (
            "[generation]\n"
            "surface_enabled = true\n"
            "n_surfaces = 1\n"
            "surface_miller_indices = 1,0,0;1,1,0;1,1,1\n"
            "surface_target_n_atoms = 96\n"
            "surface_target_tolerance = 0.15\n"
            "surface_max_n_atoms = 192\n"
            "surface_max_in_plane_repeat = 3,4\n"
            "surface_max_normal_repeat = 12\n"
            "surface_termination_policy = first"
        ),
        1,
    )

    config = load_config(write_config(tmp_path, text))
    legacy = load_legacy_config(write_config(tmp_path, text))

    assert config.generation.surface_enabled is True
    assert config.generation.n_surfaces == 1
    assert config.generation.surface_miller_indices == (
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
    )
    assert config.generation.surface_termination_policy == "first"
    assert config.generation.surface_target_n_atoms == 96
    assert config.generation.surface_target_tolerance == pytest.approx(0.15)
    assert config.generation.surface_max_n_atoms == 192
    assert config.generation.surface_max_in_plane_repeat == (3, 4)
    assert config.generation.surface_max_normal_repeat == 12
    assert legacy.get("generation", "n_surfaces") == "1"
    assert legacy.get("generation", "surface_miller_indices") == "1,0,0;1,1,0;1,1,1"
    assert legacy.get("generation", "surface_target_n_atoms") == "96"
    assert legacy.get("generation", "surface_max_in_plane_repeat") == "3,4"


def test_loader_rejects_unsupported_surface_orientation_even_when_disabled(tmp_path: Path) -> None:
    text = BASE_CONFIG.replace(
        "[generation]",
        "[generation]\nsurface_miller_indices = 1,0,1",
        1,
    )

    with pytest.raises(ConfigurationError, match="unsupported orientation"):
        load_config(write_config(tmp_path, text))


def test_config_path_resolution_is_canonical_and_deterministic(tmp_path: Path) -> None:
    project_dir = tmp_path / "project_demo"
    canonical = project_dir / "config" / "project.config"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(BASE_CONFIG, encoding="utf-8")

    assert (
        find_config_path(project_dir, explicit_path=project_dir / "config" / "demo.ini")
        == canonical
    )

    canonical.unlink()
    legacy = project_dir / "config" / "demo.ini"
    legacy.write_text(BASE_CONFIG, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Only project.config"):
        find_config_path(project_dir, explicit_path=legacy)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("random_seed", "not-an-int"),
        ("composition_step", "not-a-float"),
        ("use_liquid", "maybe"),
        ("elastic_strain_amplitudes", "not-a-list"),
    ],
)
def test_loader_rejects_bad_types(tmp_path: Path, key: str, value: str) -> None:
    defaults = {
        "random_seed": "7",
        "composition_step": "0.125",
        "use_liquid": "false",
        "elastic_strain_amplitudes": "-0.02, -0.01, 0.01, 0.02",
    }
    text = BASE_CONFIG.replace(f"{key} = {defaults[key]}", f"{key} = {value}")

    with pytest.raises(ConfigurationError):
        load_config(write_config(tmp_path, text))


@pytest.mark.parametrize(
    "value",
    [
        "unknown_source",
        "",
        "all,mp_phase",
        "mp_phase,mp_phase",
        "mp_phase,,sqs",
    ],
)
def test_loader_rejects_invalid_generation_source_scopes(tmp_path: Path, value: str) -> None:
    text = BASE_CONFIG.replace(
        "[generation]",
        f"[generation]\nvacancy_sources = {value}",
        1,
    )

    with pytest.raises(ConfigurationError, match="generation.vacancy_sources"):
        load_config(write_config(tmp_path, text))


def test_loader_rejects_bad_enum(tmp_path: Path) -> None:
    text = BASE_CONFIG + "\n[selection]\ndescriptor_type = unsupported\n"

    with pytest.raises(ConfigurationError, match="descriptor_type"):
        load_config(write_config(tmp_path, text))


def test_loader_rejects_unsupported_atomic_descriptor_mode(tmp_path: Path) -> None:
    text = BASE_CONFIG + "\n[selection]\ndescriptor_type = atomic\n"

    with pytest.raises(ConfigurationError, match="descriptor_type"):
        load_config(write_config(tmp_path, text))


@pytest.mark.parametrize(
    "text, message",
    [
        (BASE_CONFIG.replace("elements = W, Cr", ""), "composition.elements"),
        (
            BASE_CONFIG.replace("crystal_structures = bcc, fcc", ""),
            "generation.crystal_structures",
        ),
        (BASE_CONFIG.replace("vasp_command = vasp_std", ""), "hpc.vasp_command"),
    ],
)
def test_loader_rejects_missing_required_fields(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        load_config(write_config(tmp_path, text))


def test_loader_rejects_unknown_keys_and_legacy_project_dir(tmp_path: Path) -> None:
    unknown = BASE_CONFIG.replace(
        "target_n_atoms = 64", "target_n_atoms = 64\nfuture_switch = true"
    )
    with pytest.raises(ConfigurationError, match="Unknown configuration key"):
        load_config(write_config(tmp_path, unknown))

    unsupported_legacy = BASE_CONFIG.replace(
        "target_n_atoms = 64", "target_n_atoms = 64\nn_strained = 3"
    )
    with pytest.raises(ConfigurationError, match="Unknown configuration key"):
        load_config(write_config(tmp_path, unsupported_legacy))

    project_dir = BASE_CONFIG + "\n[paths]\nproject_dir = /tmp/demo\n"
    with pytest.raises(ConfigurationError, match="paths.project_dir"):
        load_config(write_config(tmp_path, project_dir))


def test_loader_rejects_conflicting_canonical_and_legacy_aliases(tmp_path: Path) -> None:
    conflicting = BASE_CONFIG.replace("gas_elements = He", "gas_elements = He\ngasElements = Ne")

    with pytest.raises(ConfigurationError, match="canonical and legacy"):
        load_config(write_config(tmp_path, conflicting))


def test_legacy_gas_elements_and_generation_elements_are_explicit_adapters(
    tmp_path: Path,
) -> None:
    legacy = BASE_CONFIG.replace("elements = W, Cr", "").replace(
        "gas_elements = He", "gasElements = He"
    )
    legacy = legacy.replace(
        "[generation]\ncrystal_structures = bcc, fcc",
        "[generation]\nelements = W, Cr\ncrystal_structures = bcc, fcc",
    )

    config = load_config(write_config(tmp_path, legacy))

    assert config.composition.elements == ("W", "Cr")
    assert config.composition.gas_elements == ("He",)


def test_loader_enforces_one_vasp_command_and_adapter_preserves_no_project_dir(
    tmp_path: Path,
) -> None:
    slurm_source = BASE_CONFIG.replace("[hpc]\nvasp_command = vasp_std", "[hpc]\n")
    slurm_source += "\n[slurm]\nvasp_command = vasp_std\n"
    with pytest.raises(ConfigurationError, match="single hpc.vasp_command"):
        load_config(write_config(tmp_path, slurm_source))

    adapted = load_legacy_config(write_config(tmp_path))
    assert adapted.get("hpc", "vasp_command") == "vasp_std"
    assert not adapted.has_option("paths", "project_dir")


def test_loader_uses_snake_case_model_fields_and_does_not_load_environment_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MP_API_KEY", "must-not-be-loaded")
    text = (
        BASE_CONFIG
        + "\n[materialsproject]\napi_key = historical-secret\n"
        + "\n[train_nep]\nouterZBL = 2\n"
    )

    config = load_config(write_config(tmp_path, text))

    assert config.train_nep.outer_zbl == 2.0
    assert config.materials_project.api_key is None
    assert config.effective_mapping()["materials_project"]["api_key"] is None


def test_equivalent_config_forms_have_the_same_effective_mapping(tmp_path: Path) -> None:
    first = BASE_CONFIG + "\n[train_nep]\nweights = 1, 1, 1\nouterZBL = 2.0\n"
    second = BASE_CONFIG + "\n[train_nep]\nweights = 1.0 1.0 1.0\nouterzbl = 2\n"

    assert load_config(write_config(tmp_path, first)).effective_mapping() == (
        load_config(write_config(tmp_path, second)).effective_mapping()
    )


def test_model_run_id_remains_in_its_declared_legacy_section(tmp_path: Path) -> None:
    gpumd = BASE_CONFIG + "\n[gpumd]\nmodel_run_id = run-1\n"
    adapted = load_legacy_config(write_config(tmp_path, gpumd))

    assert adapted.get("gpumd", "model_run_id") == "run-1"
    assert not adapted.has_section("validate")


def test_loader_exposes_deterministic_typed_training_sweep(tmp_path: Path) -> None:
    text = BASE_CONFIG + "\n[training_sweep]\nlambda_f = 1.0|2.0\ncutoff = 6 5|7 5\n"

    config = load_config(write_config(tmp_path, text))

    assert config.train_nep.sweep_mapping() == {
        "cutoff": (("6", "5"), ("7", "5")),
        "lambda_f": (1.0, 2.0),
    }
