import pytest
from config import ConfigError, describe_config, load_config, mask_secret


def test_missing_telegram_token_gives_readable_error(clean_env):
    with pytest.raises(ConfigError) as error:
        load_config()
    assert "TELEGRAM_BOT_TOKEN" in str(error.value)


def test_defaults_are_ollama(clean_env):
    clean_env.setenv("TELEGRAM_BOT_TOKEN", "123456:token")

    config = load_config()

    assert config.llm_provider == "ollama"
    assert config.llm_base_url == "http://127.0.0.1:11434/v1"
    assert config.llm_model == "qwen3:1.7b"
    assert config.llm_api_key == "ollama"
    assert config.llm_timeout_seconds == 180.0


def test_anthropic_profile_uses_anthropic_api_key(clean_env):
    clean_env.setenv("TELEGRAM_BOT_TOKEN", "123456:token")
    clean_env.setenv("LLM_PROVIDER", "anthropic")
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")

    config = load_config()

    assert config.llm_model == "claude-opus-5"
    assert config.llm_api_key == "sk-ant-secret-value"


def test_unknown_provider_gives_readable_error(clean_env):
    clean_env.setenv("TELEGRAM_BOT_TOKEN", "123456:token")
    clean_env.setenv("LLM_PROVIDER", "openai")

    with pytest.raises(ConfigError) as error:
        load_config()
    assert "LLM_PROVIDER" in str(error.value)


def test_secrets_are_masked_in_logs(clean_env):
    clean_env.setenv("TELEGRAM_BOT_TOKEN", "123456:super-secret-token")

    dump = describe_config(load_config())

    assert "123456:super-secret-token" not in dump
    assert mask_secret("123456:super-secret-token") in dump
