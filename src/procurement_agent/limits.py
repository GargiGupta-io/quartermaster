"""Usage limits, so an open demo cannot use up the free AI allowance.

Two limits apply to anything that calls the AI: a few requests per visitor in a short window,
and a cap for the whole service per day. Counts live in memory, so a restart resets them.
"""

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass

DAY = 24 * 60 * 60


@dataclass
class Refusal:
    message: str
    retry_after: int  # seconds


class UsageLimits:
    def __init__(
        self,
        per_visitor: int,
        window_seconds: int,
        per_day: int,
        clock: Callable[[], float] = time.time,
    ):
        self.per_visitor = per_visitor
        self.window = window_seconds
        self.per_day = per_day
        self._clock = clock
        self._visitors: dict[str, deque[float]] = defaultdict(deque)
        self._day: deque[float] = deque()
        self._lock = threading.Lock()

    def check(self, visitor: str) -> Refusal | None:
        """Count one use by this visitor, or say why it is refused. Refusals are not counted."""
        now = self._clock()
        with self._lock:
            _drop_older_than(self._day, now - DAY)
            if self.per_day and len(self._day) >= self.per_day:
                wait = int(self._day[0] + DAY - now) + 1
                return Refusal(
                    "The demo has reached its daily limit. Please try again tomorrow.", wait
                )
            recent = self._visitors[visitor]
            _drop_older_than(recent, now - self.window)
            if self.per_visitor and len(recent) >= self.per_visitor:
                wait = int(recent[0] + self.window - now) + 1
                minutes = max(1, round(wait / 60))
                return Refusal(
                    f"You have sent {self.per_visitor} requests in a short time. "
                    f"Please try again in about {minutes} minute{'s' if minutes != 1 else ''}.",
                    wait,
                )
            recent.append(now)
            self._day.append(now)
            return None


def _drop_older_than(times: deque[float], cutoff: float) -> None:
    while times and times[0] <= cutoff:
        times.popleft()


def visitor_id(headers, fallback: str) -> str:
    """The visitor's address. Render sits behind Cloudflare, which sets CF-Connecting-IP;
    X-Forwarded-For can be faked by the sender, so it is only a fallback. The daily cap
    protects the allowance even if an address is faked."""
    for name in ("cf-connecting-ip", "true-client-ip"):
        if headers.get(name):
            return headers[name].strip()
    forwarded = headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or fallback
