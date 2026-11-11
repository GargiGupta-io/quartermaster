"""Checks on what comes in, before the AI sees it.

Requests are screened for text that is not a request at all, and for attempts to give the
assistant new instructions. Documents are screened for the same instructions, so a planted
line in a PDF cannot steer the AI. These are cheap first lines of defence: the AI is also told
to treat everything it reads as data, and code still decides every figure and approval.
"""

import re

_INSTRUCTION_PATTERNS = [
    r"\b(?:ignore|disregard) (?:all |any )?(?:the |your |previous |prior |above |earlier )*"
    + r"(?:instructions|rules|policies|prompts?)\b",
    r"\bforget (?:all |your |the )*(?:instructions|rules)\b",
    r"\b(?:system|developer) (?:prompt|message|instructions)\b",
    r"\byou are now\b",
    r"\bact as (?:an? )?(?:admin|administrator|developer|cfo|approver)\b",
    r"\b(?:jailbreak|developer mode|dan mode)\b",
    r"\b(?:approve|authori[sz]e) (?:this|it|the order) (?:automatically|without (?:any )?approval)\b",
    r"\bskip (?:the |all )?(?:approvals?|checks?|policy)\b",
    r"\breveal (?:your |the )?(?:prompt|instructions|api key|secrets?)\b",
]
_INSTRUCTIONS = re.compile("|".join(_INSTRUCTION_PATTERNS), re.IGNORECASE)

NOT_A_REQUEST = (
    "This does not look like a purchase request or a question. Please describe what you need, "
    "for example: 20 laptops for the Pune engineering team, budget 15 lakh."
)
INSTRUCTIONS_REFUSED = (
    "This request contains instructions aimed at the assistant itself, such as asking it to "
    "ignore its rules or skip approval, so it was not processed. Please send only what you need "
    "and why."
)


def instruction_found(text: str) -> str | None:
    """The first phrase that tries to give the assistant new instructions, if any."""
    match = _INSTRUCTIONS.search(text)
    return match.group(0) if match else None


def screen_request(text: str) -> str | None:
    """Why the request should not reach the AI, or None if it may."""
    letters = sum(ch.isalpha() for ch in text)
    if letters < 3 or letters < len(text.strip()) * 0.4:
        return NOT_A_REQUEST
    if instruction_found(text):
        return INSTRUCTIONS_REFUSED
    return None
