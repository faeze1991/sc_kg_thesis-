"""
Location alias map for normalizing the `made_in` field.
Maps common variant spellings -> canonical form.
Draft based on observed top-20 values in the full extraction results;
review and extend before running on full corpus.
"""

LOCATION_ALIASES = {
    # United States
    "usa": "United States",
    "u.s.a.": "United States",
    "u.s.a": "United States",
    "us": "United States",
    "u.s.": "United States",
    "united states": "United States",
    "united states of america": "United States",
    "america": "United States",

    # United Kingdom
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "united kingdom": "United Kingdom",
    "england": "United Kingdom",  # NOTE: England is technically a constituent country, not synonymous with UK -
    "great britain": "United Kingdom",  # flagging this as a simplification worth confirming with supervisor
    "britain": "United Kingdom",
    "scotland": "United Kingdom",  # same caveat as England
    "wales": "United Kingdom",

    # Common non-specific / regional - kept distinct, NOT merged into a country
    "imported": "Imported (unspecified)",
    "europe": "Europe (unspecified)",
    "eu": "Europe (unspecified)",

    # Other common variants seen or expected
    "china": "China",
    "prc": "China",
    "people's republic of china": "China",

    "italy": "Italy",
    "italia": "Italy",

    "germany": "Germany",
    "deutschland": "Germany",

    "france": "France",

    "australia": "Australia",

    "india": "India",

    "japan": "Japan",

    "canada": "Canada",

    "spain": "Spain",
    "espana": "Spain",
    "españa": "Spain",

    "portugal": "Portugal",

    "turkey": "Turkey",
    "turkiye": "Turkey",
    "türkiye": "Turkey",

    "new zealand": "New Zealand",
    "nz": "New Zealand",
}


def normalize_location(raw: str) -> str:
    """
    Normalize a raw made_in string to a canonical location name.
    Falls back to title-cased original if no alias match found.
    """
    if not raw or raw.strip().lower() == "unknown":
        return "unknown"
    key = raw.strip().lower()
    key = key.rstrip(".")  # trailing periods e.g. "U.S.A."
    if key in LOCATION_ALIASES:
        return LOCATION_ALIASES[key]
    # No match - return cleaned original (title case for consistency)
    return raw.strip()
