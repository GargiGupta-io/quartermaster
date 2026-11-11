"""Usage limits for the open demo, with a fake clock."""

from procurement_agent.limits import DAY, UsageLimits, visitor_id


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


def test_a_visitor_is_limited_within_the_window_then_allowed_again():
    clock = Clock()
    limits = UsageLimits(per_visitor=2, window_seconds=600, per_day=0, clock=clock)
    assert limits.check("a") is None
    assert limits.check("a") is None
    refusal = limits.check("a")
    assert refusal and refusal.retry_after == 601
    assert "2 requests" in refusal.message
    clock.now += 601
    assert limits.check("a") is None


def test_visitors_are_counted_separately():
    limits = UsageLimits(per_visitor=1, window_seconds=600, per_day=0, clock=Clock())
    assert limits.check("a") is None
    assert limits.check("b") is None
    assert limits.check("a") is not None


def test_the_daily_cap_covers_everyone():
    clock = Clock()
    limits = UsageLimits(per_visitor=0, window_seconds=600, per_day=3, clock=clock)
    for visitor in "abc":
        assert limits.check(visitor) is None
    refusal = limits.check("d")
    assert refusal and "daily limit" in refusal.message
    clock.now += DAY
    assert limits.check("d") is None


def test_refusals_are_not_counted():
    clock = Clock()
    limits = UsageLimits(per_visitor=1, window_seconds=600, per_day=0, clock=clock)
    limits.check("a")
    for _ in range(5):
        limits.check("a")
    clock.now += 601
    assert limits.check("a") is None


def test_zero_turns_limits_off():
    limits = UsageLimits(per_visitor=0, window_seconds=1, per_day=0, clock=Clock())
    assert all(limits.check("a") is None for _ in range(100))


def test_visitor_address_prefers_the_cloudflare_header():
    headers = {"cf-connecting-ip": "1.1.1.1", "x-forwarded-for": "9.9.9.9, 2.2.2.2"}
    assert visitor_id(headers, "10.0.0.1") == "1.1.1.1"
    assert visitor_id({"x-forwarded-for": "9.9.9.9, 2.2.2.2"}, "10.0.0.1") == "9.9.9.9"
    assert visitor_id({}, "10.0.0.1") == "10.0.0.1"
