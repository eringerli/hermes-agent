"""Invariant tests for the _compressed_summary merge guard in both merge passes.

LCM-X emits one user row per frontier DAG node with ``_compressed_summary=True``.
The host's consecutive-user merge passes must NOT collapse these rows: they are
structural cache anchors whose byte-stability is required for provider prompt
cache prefix retention.

Pass 2 (``drop_thinking_only_and_merge_users``) is the critical guard: it has no
``split_user_originated_turn`` pre-check, so without the inline flag test it
merges flagged pairs into a single blob (proven red on base).

Pass 1 (``_merge_consecutive_users``) already blocks flagged rows via
``split_user_originated_turn(prev)[0] is None``; the inline flag check there is
defensive redundancy. Both are tested to pin the invariant.
"""

from agent.agent_runtime_helpers import (
    _merge_consecutive_users,
    drop_thinking_only_and_merge_users,
)


def _row(content: str, flagged: bool = False) -> dict:
    m = {"role": "user", "content": content}
    if flagged:
        m["_compressed_summary"] = True
    return m


# --- Pass 2: drop_thinking_only_and_merge_users (critical — red on base) ---

def test_pass2_flagged_pair_not_merged():
    """Two consecutive flagged user rows must survive as separate messages.

    This is the guard that is NOT redundant: pass 2 has no
    split_user_originated_turn pre-check, so without the inline
    ``_compressed_summary`` flag test it collapses the pair to one blob.
    """
    msgs = [_row("LCM frontier node alpha", True), _row("LCM frontier node beta", True)]
    out = drop_thinking_only_and_merge_users(msgs)
    assert len(out) == 2, f"Expected 2 separate rows, got {len(out)}: {out}"
    assert out[0]["content"] == "LCM frontier node alpha"
    assert out[1]["content"] == "LCM frontier node beta"


def test_pass2_plain_pair_merged():
    """Regression guard: plain (unflagged) consecutive user rows still merge."""
    msgs = [_row("plain user msg one"), _row("plain user msg two")]
    out = drop_thinking_only_and_merge_users(msgs)
    assert len(out) == 1, f"Expected merged pair, got {len(out)}: {out}"
    assert "plain user msg one" in out[0]["content"]
    assert "plain user msg two" in out[0]["content"]


def test_pass2_mixed_pair_not_merged():
    """A flagged row adjacent to a plain row must not be merged."""
    msgs = [_row("LCM summary row", True), _row("new human prompt")]
    out = drop_thinking_only_and_merge_users(msgs)
    assert len(out) == 2, f"Expected 2 rows (flagged protects plain too), got {len(out)}: {out}"


# --- Pass 1: _merge_consecutive_users (defensive — already guarded by split check) ---

def test_pass1_flagged_pair_not_merged():
    """Flagged pair survives pass 1. Redundant guard: split_user_originated_turn
    already blocks this, but the inline flag check documents the intent."""
    msgs = [_row("LCM frontier node alpha", True), _row("LCM frontier node beta", True)]
    out, repairs = _merge_consecutive_users(msgs)
    assert len(out) == 2, f"Expected 2 separate rows, got {len(out)}: {out}"


def test_pass1_plain_pair_merged():
    """Regression guard: plain consecutive user rows still merge in pass 1."""
    msgs = [_row("plain user msg one"), _row("plain user msg two")]
    out, repairs = _merge_consecutive_users(msgs)
    assert len(out) == 1, f"Expected merged pair, got {len(out)}: {out}"
    assert repairs == 1
