import pytest

from trainloop import settings


def test_clean_drops_unset_and_converts_types():
    out = settings.clean({"epochs": "2", "learning_rate": "3e-4", "seed": None, "optimizer": "", "lora_rank": "auto"})
    assert out == {"epochs": 2, "learning_rate": 3e-4}


@pytest.mark.parametrize("raw", [{"epochs": 0}, {"optimizer": "sgd"}, {"target_modules": ["mlp"]}, {"nope": 1},
                                 {"learning_rate": "fast"}])
def test_clean_rejects_bad_values(raw):
    with pytest.raises(settings.SettingsError):
        settings.clean(raw)


def test_split_reports_what_a_trainer_ignores():
    s = {"epochs": 2, "lora_alpha": 64, "weight_decay": 0.01, "method": "qlora"}
    used, ignored = settings.split("tinker", s)
    assert used == {"epochs": 2}
    assert ignored == ["lora_alpha", "weight_decay", "method"]
    used, ignored = settings.split("unsloth", s)
    assert used == s and ignored == []


def test_split_enforces_rank_caps():
    with pytest.raises(settings.SettingsError):
        settings.split("fireworks", {"lora_rank": 64})
    assert settings.split("unsloth", {"lora_rank": 64})[0] == {"lora_rank": 64}


def test_load_folds_in_settings_saved_the_old_way():
    state = {"train_epochs": 3, "learning_rate": 1e-4, "settings": {"epochs": 2}}
    assert settings.load(state) == {"epochs": 2, "learning_rate": 1e-4}
