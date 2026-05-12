"""
Configuration for few-shot attribute detection.
Defines which (attribute, demographic) pairs require few-shot examples for each model.
"""

# GPT-4o attribute+demo pairs requiring few-shot examples
# These pairs showed improved accuracy with few-shot prompting
GPT4O_FEW_SHOT_PAIRS = [
    ("smile", "asian_woman"),
    ("smile", "black_woman"),
    ("smile", "indian_man"),
    ("smile", "indian_woman"),
    ("shoulder_length_hair", "asian_woman"),
    ("shoulder_length_hair", "black_woman"),
    ("shoulder_length_hair", "indian_woman"),
    ("shoulder_length_hair", "white_woman"),
]

# Claude skin tone few-shot - needed for all demographics
# Based on evaluation showing Claude benefits from few-shot for skin tone comparison
CLAUDE_SKIN_TONE_FEW_SHOT_DEMOS = [
    "asian_man",
    "asian_woman",
    "black_man",
    "black_woman",
    "white_man",
    "white_woman",
    "indian_man",
    "indian_woman",
]

# All demographics list for reference
ALL_DEMOGRAPHICS = [
    "asian_man",
    "asian_woman",
    "black_man",
    "black_woman",
    "white_man",
    "white_woman",
    "indian_man",
    "indian_woman",
]


def needs_gpt4o_few_shot(attribute: str, demographic: str) -> bool:
    """Check if this (attribute, demographic) pair needs GPT-4o few-shot."""
    return (attribute, demographic) in GPT4O_FEW_SHOT_PAIRS


def needs_claude_skin_tone_few_shot(demographic: str) -> bool:
    """Check if this demographic needs Claude skin tone few-shot."""
    return demographic in CLAUDE_SKIN_TONE_FEW_SHOT_DEMOS


def get_gpt4o_few_shot_attributes() -> list:
    """Get unique attributes that need GPT-4o few-shot."""
    return list(set(attr for attr, _ in GPT4O_FEW_SHOT_PAIRS))
