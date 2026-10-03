from configparser import ConfigParser

import pytest

from nepflow.mlip.nep.inputs import NepHyperparameters, NepInputRenderer


def make_train_config(**overrides: object) -> ConfigParser:
    config = ConfigParser()
    config["composition"] = {"elements": "Si,Ge", "gasElements": ""}
    config["train_nep"] = {
        "cutoff": "6 5",
        "n_max": "4 4",
        "basis_size": "8 8",
        "l_max": "4 2 1",
        "neuron": "80",
        "population": "50",
        "batch": "3000",
        "generation": "250000",
        "outerZBL": "2.0",
        "charge_mode": "0",
        "weights": "1,1",
        "lambda_e": "1.0",
        "lambda_f": "1.0",
        "lambda_v": "1.0",
        "lambda_shear": "1.0",
    }
    for key, value in overrides.items():
        config["train_nep"][key] = str(value)
    return config


def render_and_identify(
    *,
    composition_elements: str = "Si,Ge",
    composition_gas_elements: str = "",
    **overrides: object,
) -> tuple[str, str]:
    config = make_train_config(**overrides)
    config["composition"]["elements"] = composition_elements
    config["composition"]["gasElements"] = composition_gas_elements
    hyperparameters = NepHyperparameters.from_legacy_config(config)
    rendered = NepInputRenderer().render_content(hyperparameters)
    return rendered, hyperparameters.identity_hash()


def identity_cases() -> list[tuple[str, str, str]]:
    return [
        ("cutoff", "6 5", "7 5"),
        ("n_max", "4 4", "5 4"),
        ("basis_size", "8 8", "9 8"),
        ("l_max", "4 2 1", "5 2 1"),
        ("neuron", "80", "96"),
        ("population", "50", "60"),
        ("batch", "3000", "4000"),
        ("generation", "250000", "300000"),
        ("outerZBL", "2.0", "3.0"),
        ("charge_mode", "0", "1"),
        ("lambda_e", "1.0", "2.0"),
        ("lambda_f", "1.0", "2.0"),
        ("lambda_v", "1.0", "2.0"),
        ("lambda_shear", "1.0", "2.0"),
        ("weights", "1,1", "2,1"),
    ]


@pytest.mark.parametrize("key,baseline,variant", identity_cases(), ids=lambda case: case[0])
def test_identity_bearing_config_changes_render_and_run_identity(
    key: str, baseline: str, variant: str
) -> None:
    baseline_rendered, baseline_identity = render_and_identify(**{key: baseline})
    variant_rendered, variant_identity = render_and_identify(**{key: variant})

    assert baseline_rendered != variant_rendered, key
    assert baseline_identity != variant_identity, key


def test_equivalent_effective_inputs_have_stable_identity() -> None:
    comma_rendered, comma_identity = render_and_identify(weights="1, 2")
    spaced_rendered, spaced_identity = render_and_identify(weights="1.0 2.0")

    assert comma_rendered == spaced_rendered
    assert comma_identity == spaced_identity


def test_effective_type_list_and_order_changes_identity() -> None:
    baseline_rendered, baseline_identity = render_and_identify()
    extra_rendered, extra_identity = render_and_identify(
        composition_elements="Si,Ge,O", weights="1,1,1"
    )
    reordered_rendered, reordered_identity = render_and_identify(composition_elements="Ge,Si")

    assert baseline_rendered != extra_rendered
    assert baseline_identity != extra_identity
    assert baseline_rendered != reordered_rendered
    assert baseline_identity != reordered_identity


def test_solid_gas_partition_is_effectively_identical() -> None:
    solid_rendered, solid_identity = render_and_identify(composition_elements="Si,Ge")
    partitioned_rendered, partitioned_identity = render_and_identify(
        composition_elements="Si", composition_gas_elements="Ge"
    )

    assert solid_rendered == partitioned_rendered
    assert solid_identity == partitioned_identity


def test_lambda_shear_is_rendered_from_typed_settings() -> None:
    rendered, _identity = render_and_identify(lambda_shear="5.0")
    assert "lambda_shear 5.0" in rendered


def test_invalid_nonempty_weights_are_not_replaced_by_equal_weights() -> None:
    config = make_train_config(weights="1")
    with pytest.raises(ValueError, match="weights"):
        NepHyperparameters.from_legacy_config(config)
