from __future__ import annotations

from innards._text import IncrementalDecoder, strip_stop

VOCAB = {1: "Hel", 2: "lo", 3: " wor", 4: "ld", 5: "\nUs", 6: "er:", 7: " more", 8: "\xe2", 9: "\x82\xac"}


def decode(ids: list[int]) -> str:
    raw = "".join(VOCAB[i] for i in ids)
    # Simulate byte-level tokens: ids 8+9 together form "€"; alone they are incomplete.
    raw = raw.replace("\xe2\x82\xac", "€").replace("\xe2", "�")
    return raw


def test_strip_stop() -> None:
    assert strip_stop("abc\nUser: x", ["\nUser:"]) == ("abc", True)
    assert strip_stop("abc", ["\nUser:"]) == ("abc", False)


def test_deltas_join_to_full_text() -> None:
    d = IncrementalDecoder(decode)
    out = [d.push([i]) for i in (1, 2, 3, 4)]
    out.append(d.flush())
    assert "".join(out) == "Hello world"
    assert d.text == "Hello world"


def test_incomplete_utf8_is_held_back() -> None:
    d = IncrementalDecoder(decode)
    assert d.push([8]) == ""
    assert d.push([9]) == "€"


def test_stop_string_is_held_then_cut() -> None:
    d = IncrementalDecoder(decode, ["\nUser:"])
    pieces = [d.push([i]) for i in (1, 2, 3, 4, 5)]
    assert "".join(pieces) == "Hello world"  # "\nUs" held back: could start a stop string
    assert d.push([6]) == ""
    assert d.stopped
    assert d.push([7]) == ""
    assert d.flush() == ""
    assert d.text == "Hello world"


def test_held_text_released_when_not_a_stop() -> None:
    d = IncrementalDecoder(decode, ["\nUser:"])
    d.push([1, 2])
    d.push([5])
    assert d.push([7]) == "\nUs more"
