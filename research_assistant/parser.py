"""Multi-command parser for the human-in-the-loop review step.

Supported commands (case-insensitive):
    approve 1,3,5       — approve specific subtopics by ID
    approve all          — approve every subtopic
    reject 2,4           — reject specific subtopics by ID
    add 'Topic Name'     — add a brand-new subtopic
    modify 3 to 'New'    — rename subtopic #3
    done                 — finish editing and proceed

Multiple commands can be separated by ``;`` on a single input line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ParsedCommand:
    """A single parsed command from user input."""

    action: str  # "approve", "approve_all", "reject", "add", "modify", "done"
    targets: list[int] = field(default_factory=list)
    value: str = ""


class ParseError(Exception):
    """Raised when a command string cannot be understood."""


def parse_single_command(text: str) -> ParsedCommand:
    """Parse one command string into a ``ParsedCommand``.

    Raises ``ParseError`` with a user-friendly message on failure.
    """
    text = text.strip()
    if not text:
        raise ParseError("Empty command — type 'done' to finish or 'approve all' to approve everything.")

    # done
    if re.match(r"^done$", text, re.IGNORECASE):
        return ParsedCommand(action="done")

    # approve all
    if re.match(r"^approve\s+all$", text, re.IGNORECASE):
        return ParsedCommand(action="approve_all")

    # approve 1,3,5  or  approve 1 3 5
    m = re.match(r"^approve\s+([\d,\s]+)$", text, re.IGNORECASE)
    if m:
        ids = _parse_ids(m.group(1))
        if not ids:
            raise ParseError("'approve' needs at least one subtopic ID (e.g. 'approve 1,3').")
        return ParsedCommand(action="approve", targets=ids)

    # reject 2,4
    m = re.match(r"^reject\s+([\d,\s]+)$", text, re.IGNORECASE)
    if m:
        ids = _parse_ids(m.group(1))
        if not ids:
            raise ParseError("'reject' needs at least one subtopic ID (e.g. 'reject 2').")
        return ParsedCommand(action="reject", targets=ids)

    # add 'topic'  or  add "topic"
    m = re.match(r"^add\s+['\"](.+?)['\"]$", text, re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        if not name:
            raise ParseError("Subtopic name cannot be empty.")
        return ParsedCommand(action="add", value=name)

    # modify 3 to 'new name'
    m = re.match(r"^modify\s+(\d+)\s+to\s+['\"](.+?)['\"]$", text, re.IGNORECASE)
    if m:
        tid = int(m.group(1))
        name = m.group(2).strip()
        if not name:
            raise ParseError("New subtopic name cannot be empty.")
        return ParsedCommand(action="modify", targets=[tid], value=name)

    raise ParseError(
        f"Unrecognised command: '{text}'\n"
        "Try: approve 1,3 | reject 2 | add 'Topic' | modify 3 to 'New Name' | approve all | done"
    )


def _parse_ids(raw: str) -> list[int]:
    """Extract integer IDs from a comma-or-space-separated string."""
    return [int(x) for x in re.split(r"[,\s]+", raw.strip()) if x.strip().isdigit()]


def parse_command_line(line: str) -> list[ParsedCommand]:
    """Parse a line that may contain multiple ``;``-separated commands.

    Returns a list of ``ParsedCommand`` objects.
    Raises ``ParseError`` on the first unparseable segment.
    """
    segments = [seg.strip() for seg in line.split(";") if seg.strip()]
    if not segments:
        raise ParseError("Empty input — type 'done' to finish or 'approve all' to approve everything.")
    return [parse_single_command(seg) for seg in segments]


def validate_ids(ids: list[int], valid_ids: set[int]) -> list[str]:
    """Return a list of warning strings for any IDs not in *valid_ids*."""
    return [
        f"Subtopic #{tid} does not exist — ignored."
        for tid in ids
        if tid not in valid_ids
    ]
