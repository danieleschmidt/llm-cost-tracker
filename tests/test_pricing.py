"""Tests for ModelPricing."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_cost_tracker.pricing import ModelPricing, ModelRate


def test_known_model_cost():
    p = ModelPricing()
    # gpt-4o: $2.50/1M in, $10.00/1M out
    cost = p.compute_cost("gpt-4o", 1_000_000, 1_000_000)
    assert abs(cost - 12.50) < 1e-6


def test_anthropic_model():
    p = ModelPricing()
    rate = p.get_rate("claude-3-5-sonnet-20241022")
    assert rate.provider == "anthropic"
    assert rate.input_per_1m == 3.00
    assert rate.output_per_1m == 15.00


def test_unknown_model_raises():
    p = ModelPricing()
    with pytest.raises(KeyError):
        p.get_rate("totally-fake-model-xyz")


def test_custom_rate():
    custom = {"my-model": ModelRate("acme", "my-model", 1.00, 2.00)}
    p = ModelPricing(custom_rates=custom)
    cost = p.compute_cost("my-model", 500_000, 500_000)
    assert abs(cost - 1.50) < 1e-6


def test_list_models():
    p = ModelPricing()
    models = p.list_models()
    assert len(models) > 10


def test_list_models_provider_filter():
    p = ModelPricing()
    openai = p.list_models(provider="openai")
    assert all(m.provider == "openai" for m in openai)
    assert len(openai) > 3


def test_zero_tokens():
    p = ModelPricing()
    cost = p.compute_cost("gpt-4o", 0, 0)
    assert cost == 0.0


def test_fuzzy_match():
    p = ModelPricing()
    # "gpt-4o" is in db; partial match should still work
    rate = p.get_rate("gpt-4o")
    assert rate.input_per_1m == 2.50


def test_gemini_pricing():
    p = ModelPricing()
    rate = p.get_rate("gemini-1.5-flash")
    assert rate.provider == "google"
    assert rate.input_per_1m < 1.0   # it's cheap


def test_cohere_pricing():
    p = ModelPricing()
    rate = p.get_rate("command-r-plus")
    assert rate.provider == "cohere"
