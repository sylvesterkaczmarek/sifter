"""Rules for text that came from a repository and is on its way to a terminal.

A project file's own strings are quoted back to whoever ran sifter in the clone,
so they are the one place where what the screen says and what the tools receive
can be made to disagree.
"""

from __future__ import annotations

# Long enough for any real registry name or path, short enough that one cannot
# push the lines around it off a terminal by wrapping.
DISPLAY_CHARS = 200


def reject_control_characters(label: str, value: str) -> None:
    """Refuse ``value`` if a terminal would obey it rather than show it."""
    if any(not char.isprintable() for char in value):
        raise ValueError(
            f"{label} contains a control character: {value!r}. What the screen "
            "shows and what oras and cosign receive would not be the same text."
        )


def reject_path_component(label: str, value: str) -> None:
    """Refuse a value that can escape the directory it is joined beneath."""
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a single filename component, not a path: {value!r}")


def show_control_characters(value: str) -> str:
    """Make terminal instructions visible while preserving log lines and tabs."""
    return "".join(
        char if char in {"\n", "\t"} or char.isprintable() else ascii(char)[1:-1] for char in value
    )


def bound(value: str, limit: int = DISPLAY_CHARS) -> str:
    """Truncate ``value`` to a length that cannot scroll its neighbours away."""
    if len(value) <= limit:
        return value
    return value[:limit] + "…"
