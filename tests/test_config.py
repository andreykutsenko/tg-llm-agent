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


def test_agent_defaults(clean_env):
    clean_env.setenv("TELEGRAM_BOT_TOKEN", "123456:token")

    config = load_config()

    assert config.agent_max_steps == 8
    assert config.exec_allowlist == ("curl", "cat", "ls", "date", "head", "tail", "wc", "grep")
    assert config.exec_timeout_seconds == 20.0
    assert config.exec_max_output == 4000
    assert config.skills_max_chars == 8000
    assert config.telegram_allowed_ids == ()


def test_agent_max_steps_outside_range_is_rejected(clean_env):
    clean_env.setenv("TELEGRAM_BOT_TOKEN", "123456:token")
    clean_env.setenv("AGENT_MAX_STEPS", "42")

    with pytest.raises(ConfigError) as error:
        load_config()
    assert "AGENT_MAX_STEPS" in str(error.value)


def test_allowed_ids_are_parsed_as_numbers(clean_env):
    clean_env.setenv("TELEGRAM_BOT_TOKEN", "123456:token")
    clean_env.setenv("TELEGRAM_ALLOWED_IDS", " 100, 200 ,300")

    assert load_config().telegram_allowed_ids == (100, 200, 300)


def test_non_numeric_allowed_id_gives_readable_error(clean_env):
    clean_env.setenv("TELEGRAM_BOT_TOKEN", "123456:token")
    clean_env.setenv("TELEGRAM_ALLOWED_IDS", "100,@username")

    with pytest.raises(ConfigError) as error:
        load_config()
    assert "TELEGRAM_ALLOWED_IDS" in str(error.value)
