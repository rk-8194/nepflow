from pathlib import Path

import pytest

from nepflow.config import load_config
from nepflow.errors import ConfigurationError

BASE_CONFIG = """
[project]
name = demo
schema_version = 1

[composition]
elements = Fe, Cr

[generation]
crystal_structures = bcc

[hpc]
vasp_command = vasp_std
"""


def _load(tmp_path: Path, magnetism: str = ""):
    path = tmp_path / "project.config"
    path.write_text(BASE_CONFIG + magnetism, encoding="utf-8")
    return load_config(path)


def test_magnetism_defaults_are_typed_and_stable(tmp_path: Path) -> None:
    config = _load(tmp_path)

    assert config.magnetism.enabled is False
    assert config.magnetism.include_non_magnetic is True
    assert config.magnetism.include_ferromagnetic is False
    assert config.magnetism.include_antiferromagnetic is False
    assert config.magnetism.magnetic_sources == ("all",)
    assert config.effective_mapping()["magnetism"]["magnetic_sources"] == ["all"]


def test_magnetism_section_parses_named_moment_sets_and_scopes(tmp_path: Path) -> None:
    config = _load(
        tmp_path,
        """

[magnetism]
enabled = true
target_potential_magnetic = true
include_non_magnetic = true
include_ferromagnetic = true
include_antiferromagnetic = true
moment_sets = {"low":{"Fe":2.5,"Cr":1.5},"high":{"Fe":3.5,"Cr":2.5}}
symmetry_tolerance = 0.002
phase_tolerance = 1e-7
max_afm_orderings = 12
unmapped_site_policy = reject
magnetic_sources = SQS
defect_families = vacancy, substitution
max_defect_parents = 4
max_magnetic_variants_per_parent = 8
max_magnetic_variants_per_defect = 3
""",
    )

    assert [item.name for item in config.magnetism.moment_sets] == ["low", "high"]
    assert config.magnetism.moment_sets[0].element_moments["Fe"] == 2.5
    assert config.magnetism.magnetic_sources == ("sqs",)
    assert config.magnetism.defect_families == ("vacancy", "substitution")
    assert config.magnetism.max_afm_orderings == 12


@pytest.mark.parametrize(
    "section, message",
    [
        (
            """
[magnetism]
enabled = true
include_ferromagnetic = true
moment_sets = {"low":{"Fe":2.5}}
""",
            "target-potential magnetic capability",
        ),
        (
            """
[magnetism]
enabled = true
target_potential_magnetic = true
include_ferromagnetic = true
moment_sets = {"low":{"Fe":2.5},"high":{"Fe":3.5,"Cr":1.0}}
""",
            "same complete element set",
        ),
        (
            """
[magnetism]
enabled = true
target_potential_magnetic = true
include_antiferromagnetic = true
unmapped_site_policy = guess
""",
            "moment_sets",
        ),
    ],
)
def test_magnetism_validation_rejects_incompatible_or_incomplete_settings(
    tmp_path: Path,
    section: str,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        _load(tmp_path, section)


def test_magnetism_validation_rejects_unknown_defect_family_and_source(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="defect_families"):
        _load(
            tmp_path,
            """

[magnetism]
enabled = true
defect_families = vacancy, unknown
""",
        )
    with pytest.raises(ConfigurationError, match="magnetism.magnetic_sources"):
        _load(
            tmp_path,
            """

[magnetism]
magnetic_sources = unknown_source
""",
        )
