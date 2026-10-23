"""Tests of the `condition` section of src/config.py: derived values, validation, model schemas."""

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError

from src.config import CONFIGS_DIR, PROJECT_DIR, config_from_cli, load_config, load_saved_config, save_config

DEFAULT_YAML = CONFIGS_DIR / "configuration_0.yaml"


def as_dict(node) -> dict:
    return OmegaConf.to_container(node, resolve=True)


# ============================== DERIVED VALUES ==============================

def test_condition_reads_the_weather_the_dit_reads():
    cfg = load_config(DEFAULT_YAML)
    cond, data = cfg.condition, cfg.data.params
    assert (cond.T, cond.n_predict, cond.n_context) == (data.T, data.noise_image_num,
                                                        data.T - data.noise_image_num)
    assert (cond.n_context, cond.n_predict) == (10, 20)
    assert (cond.n_channels, cond.steps_per_frame, cond.center_crop) == (
        data.meso_channels, data.meso_steps_per_frame, data.meso_center_crop)
    assert cond.series_dir == f"{cfg.paths.preprocessed_dir}/condition_mean"
    assert cond.outputs_dir == "registry/outputs/condition"


def test_derived_values_follow_their_source():
    cfg = load_config(DEFAULT_YAML, overrides=["data.params.noise_image_num=15", "condition.meso_agg=last"])
    assert (cfg.condition.n_context, cfg.condition.n_predict) == (15, 15)
    assert cfg.condition.series_dir.endswith("condition_last")


def test_derived_values_set_by_hand_must_agree():
    with pytest.raises(ValueError, match="condition.n_context"):
        load_config(DEFAULT_YAML, overrides=["condition.n_context=7"])


# ================================ VALIDATION ================================

@pytest.mark.parametrize("override, message", [
    ("condition.meso_agg=median", "condition.meso_agg"),
    ("condition.test_splits=[iid,future]", "condition.test_splits"),
    ("condition.lr=0", "condition.lr"),
    ("condition.batch_size=0", "condition.batch_size"),
    ("condition.early_stopping_patience=-1", "condition.early_stopping_patience"),
    ("condition.model.params.activation=tanh", "activation"),
])
def test_invalid_values_are_rejected(override, message):
    with pytest.raises(ValueError, match=message):
        load_config(DEFAULT_YAML, overrides=[override])


def test_a_different_number_of_weather_channels_does_not_block_the_other_phases():
    """The variable names are labels: with other channels generic names are used, no error."""
    from src.dataset.meso_series import variable_names
    cfg = load_config(DEFAULT_YAML, overrides=["data.params.meso_channels=4"])
    assert cfg.condition.n_channels == 4
    assert variable_names(cfg) == ["channel_0", "channel_1", "channel_2", "channel_3"]
    assert variable_names(load_config(DEFAULT_YAML))[0] == "precipitation"


def test_mamba_convolution_width_is_checked():
    with pytest.raises(ValueError, match="d_conv"):
        load_config(DEFAULT_YAML, overrides=["condition.model.params.d_conv=5"])


def test_errors_are_reported_together():
    with pytest.raises(ValueError) as error:
        load_config(DEFAULT_YAML, overrides=["condition.meso_agg=median", "condition.lr=0"])
    assert "condition.meso_agg" in str(error.value) and "condition.lr" in str(error.value)


def test_patchmixer_patches_must_fit_the_context():
    with pytest.raises(ValueError, match="patch_len"):
        load_config(DEFAULT_YAML, overrides=["condition.model.name=PatchMixer",
                                             "condition.model.params.patch_len=12"])


# =============================== MODEL SCHEMAS ===============================

def test_switching_model_replaces_its_parameters():
    cfg = load_config(DEFAULT_YAML, overrides=["condition.model.name=PatchMixer"])
    params = cfg.condition.model.params
    assert "patch_len" in params and "d_state" not in params


def test_parameters_of_the_other_model_are_rejected():
    with pytest.raises(ConfigKeyError):
        load_config(DEFAULT_YAML, overrides=["condition.model.name=PatchMixer",
                                             "condition.model.params.d_state=8"])


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="condition.model.name"):
        load_config(DEFAULT_YAML, overrides=["condition.model.name=Transformer"])


def test_model_switch_is_the_same_from_load_config_and_from_the_command_line(monkeypatch):
    """condition.model.name chosen by an override works the same way in load_config() and with --set."""
    monkeypatch.chdir(PROJECT_DIR)
    override = "condition.model.name=PatchMixer"
    from_load = load_config(DEFAULT_YAML, overrides=[override])
    from_cli = config_from_cli(["--cfgdir", str(DEFAULT_YAML), "--project_name", "test", "--set", override])
    assert as_dict(from_load.condition) == as_dict(from_cli.condition)


def test_saved_patchmixer_run_reloads_with_its_parameters(tmp_path):
    """load_saved_config keeps the parameters of the architecture the run was trained with."""
    cfg = load_config(DEFAULT_YAML, overrides=["condition.model.name=PatchMixer",
                                               "condition.model.params.patch_len=3"])
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    assert load_saved_config(path).condition.model.params.patch_len == 3


# =============================== SAVE / RELOAD ===============================

def test_saved_config_reloads_identical(tmp_path):
    cfg = load_config(DEFAULT_YAML, overrides=["condition.model.name=PatchMixer", "condition.num_epochs=7"])
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    assert as_dict(load_saved_config(path)) == as_dict(cfg)
