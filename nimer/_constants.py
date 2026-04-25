"""Model identifiers used across the SDK.

Update these constants when Anthropic releases new models or you want
to change the routing targets. Keeping them centralized means the
router, pricing, and tests all stay in sync.
"""

# Current Claude 4.x family (verify against Anthropic's docs before release)
MODEL_HAIKU = "claude-haiku-4-5-20251001"
MODEL_SONNET = "claude-sonnet-4-6"
MODEL_OPUS = "claude-opus-4-7"

# What we treat as the "naive default" when computing savings.
# Most developers reach for Sonnet by default, so that's our baseline.
BASELINE_MODEL = MODEL_SONNET

# Every model we actively route to.
SUPPORTED_MODELS = (MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS)
