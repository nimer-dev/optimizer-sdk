from nimer.retry_policy import RetryPolicy, circuit_is_open, record_provider_failure


def test_circuit_opens_after_threshold() -> None:
    policy = RetryPolicy(circuit_breaker_threshold=2, circuit_breaker_ttl_secs=60)
    record_provider_failure("anthropic", policy)
    record_provider_failure("anthropic", policy)
    assert circuit_is_open("anthropic", policy) is True
