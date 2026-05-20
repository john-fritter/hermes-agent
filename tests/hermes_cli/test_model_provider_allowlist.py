"""Tests for limiting /model provider discovery to configured providers."""

from __future__ import annotations


def test_authenticated_provider_list_respects_model_enabled_providers(monkeypatch):
    from hermes_cli import model_switch

    monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake")
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama_fake")
    monkeypatch.setattr(model_switch, "_load_model_enabled_providers", lambda: {"openai-codex", "ollama-cloud"})
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.models.provider_model_ids", lambda provider: ["gpt-5.5"] if provider == "openai-codex" else ["copilot-model"])

    providers = model_switch.list_authenticated_providers(
        current_provider="openai-codex",
        current_model="gpt-5.5",
        custom_providers=[
            {
                "name": "ollama-cloud",
                "base_url": "https://ollama.com/v1",
                "key_env": "OLLAMA_API_KEY",
                "model": "glm-5.1",
                "models": {"glm-5.1": {}, "kimi-k2.6": {}},
            }
        ],
        max_models=5,
    )

    slugs = {p["slug"] for p in providers}
    assert "ollama-cloud" in slugs
    assert "copilot" not in slugs


def test_provider_allowlist_empty_means_no_filter(monkeypatch):
    from hermes_cli import model_switch

    monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake")
    monkeypatch.setattr(model_switch, "_load_model_enabled_providers", lambda: set())
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.models.provider_model_ids", lambda provider: ["copilot-model"])

    providers = model_switch.list_authenticated_providers(
        current_provider="openai-codex",
        current_model="gpt-5.5",
        max_models=5,
    )

    assert "copilot" in {p["slug"] for p in providers}
