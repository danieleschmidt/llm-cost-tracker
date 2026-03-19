"""
ModelPricing: LLM provider pricing database.

Prices are in USD per 1,000,000 tokens (input / output).
Updated: early 2026. Sources: official provider pricing pages.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelRate:
    """Input and output cost in USD per 1M tokens."""
    provider: str
    model: str
    input_per_1m: float   # $ per 1M input tokens
    output_per_1m: float  # $ per 1M output tokens
    notes: str = ""


# ── Pricing database ────────────────────────────────────────────────────────
# Prices in USD / 1M tokens, as of early 2026.
_PRICING: dict[str, ModelRate] = {}

def _add(*rates: ModelRate) -> None:
    for r in rates:
        _PRICING[r.model] = r
        # also register without provider prefix
        short = r.model.split("/")[-1] if "/" in r.model else None
        if short and short not in _PRICING:
            _PRICING[short] = r


_add(
    # ── OpenAI ──────────────────────────────────────────────────────────────
    ModelRate("openai", "gpt-4o",                  2.50,  10.00),
    ModelRate("openai", "gpt-4o-mini",             0.15,   0.60),
    ModelRate("openai", "gpt-4o-2024-11-20",       2.50,  10.00),
    ModelRate("openai", "gpt-4-turbo",            10.00,  30.00),
    ModelRate("openai", "gpt-4",                  30.00,  60.00),
    ModelRate("openai", "gpt-3.5-turbo",           0.50,   1.50),
    ModelRate("openai", "o1",                     15.00,  60.00),
    ModelRate("openai", "o1-mini",                 3.00,  12.00),
    ModelRate("openai", "o3-mini",                 1.10,   4.40),
    ModelRate("openai", "o3",                     10.00,  40.00),

    # ── Anthropic ───────────────────────────────────────────────────────────
    ModelRate("anthropic", "claude-3-5-sonnet-20241022",  3.00, 15.00),
    ModelRate("anthropic", "claude-3-5-haiku-20241022",   0.80,  4.00),
    ModelRate("anthropic", "claude-3-opus-20240229",     15.00, 75.00),
    ModelRate("anthropic", "claude-3-sonnet-20240229",    3.00, 15.00),
    ModelRate("anthropic", "claude-3-haiku-20240307",     0.25,  1.25),
    ModelRate("anthropic", "claude-sonnet-4-5",           3.00, 15.00),
    ModelRate("anthropic", "claude-sonnet-4-6",           3.00, 15.00),
    ModelRate("anthropic", "claude-opus-4",              15.00, 75.00),

    # ── Google ──────────────────────────────────────────────────────────────
    ModelRate("google", "gemini-1.5-pro",          3.50,  10.50, "up to 128k ctx"),
    ModelRate("google", "gemini-1.5-flash",        0.075,  0.30),
    ModelRate("google", "gemini-2.0-flash",        0.10,   0.40),
    ModelRate("google", "gemini-2.0-flash-lite",   0.075,  0.30),
    ModelRate("google", "gemini-2.0-pro",          3.50,  10.50),

    # ── Cohere ──────────────────────────────────────────────────────────────
    ModelRate("cohere", "command-r-plus",          2.50,  10.00),
    ModelRate("cohere", "command-r",               0.15,   0.60),
    ModelRate("cohere", "command-r7b",             0.0375, 0.15),

    # ── Meta / Llama (via API providers) ────────────────────────────────────
    ModelRate("meta", "llama-3.3-70b-instruct",    0.23,   0.40, "avg hosted price"),
    ModelRate("meta", "llama-3.1-405b-instruct",   3.00,   3.00, "avg hosted price"),
    ModelRate("meta", "llama-3.1-70b-instruct",    0.20,   0.20, "avg hosted price"),

    # ── Mistral ─────────────────────────────────────────────────────────────
    ModelRate("mistral", "mistral-large-2411",     2.00,   6.00),
    ModelRate("mistral", "mistral-small-2501",     0.10,   0.30),
    ModelRate("mistral", "codestral-2501",         0.10,   0.30),
)


class ModelPricing:
    """Lookup and compute costs for LLM API calls."""

    def __init__(self, custom_rates: Optional[dict[str, ModelRate]] = None):
        self._db = dict(_PRICING)
        if custom_rates:
            self._db.update(custom_rates)

    def get_rate(self, model: str) -> ModelRate:
        """Return rate for model. Raises KeyError if unknown."""
        key = model.lower().strip()
        if key in self._db:
            return self._db[key]
        # fuzzy: try partial match
        for k, v in self._db.items():
            if key in k or k in key:
                return v
        raise KeyError(
            f"Unknown model: {model!r}. "
            f"Known models: {sorted(self._db)}"
        )

    def compute_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Return USD cost for the given token counts."""
        rate = self.get_rate(model)
        return (
            rate.input_per_1m * input_tokens / 1_000_000
            + rate.output_per_1m * output_tokens / 1_000_000
        )

    def list_models(self, provider: Optional[str] = None) -> list[ModelRate]:
        seen = set()
        result = []
        for r in self._db.values():
            key = (r.provider, r.model)
            if key in seen:
                continue
            seen.add(key)
            if provider is None or r.provider.lower() == provider.lower():
                result.append(r)
        return sorted(result, key=lambda r: (r.provider, r.model))

    def add_custom(self, rate: ModelRate) -> None:
        self._db[rate.model] = rate
