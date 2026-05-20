from __future__ import annotations

from gpu_holder.events import read_events, read_events_since, write_event


def test_read_events_skips_corrupt_jsonl_lines(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"timestamp": 1, "type": "controller_start"}',
                '{"timestamp": ',
                '{"timestamp": 2, "type": "decision", "gpu_index": 0}',
                "not-json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    events = read_events(path)

    assert [event["type"] for event in events] == ["controller_start", "decision"]


def test_read_events_since_skips_corrupt_jsonl_lines_and_keeps_offset(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    write_event(path, "controller_start", pid=1)
    good_offset = path.stat().st_size
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"timestamp": \n')
        handle.write('{"timestamp": 2, "type": "decision", "gpu_index": 0}\n')

    events, next_offset = read_events_since(path, offset=good_offset)

    assert [event["type"] for event in events] == ["decision"]
    assert next_offset == path.stat().st_size


def test_read_events_filters_after_skipping_corrupt_lines(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"timestamp": 1, "type": "decision", "gpu_index": 0}',
                "not-json",
                '{"timestamp": 2, "type": "decision", "gpu_index": 1}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    events = read_events(path, event_types={"decision"}, gpu_indices={1})

    assert len(events) == 1
    assert events[0]["gpu_index"] == 1
