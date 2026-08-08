from memory.notifications import NotificationCenter, classify_event


def _event(event_id, summary, importance, source="camera", timestamp=1000):
    return {
        "event_id": event_id,
        "summary": summary,
        "importance": importance,
        "source": source,
        "timestamp": timestamp,
        "application": "Front door" if source == "camera" else "Editor",
    }


def test_safety_language_is_critical_even_below_score_threshold():
    severity, category = classify_event(
        _event("e1", "Smoke appeared near the kitchen ceiling", .6))

    assert severity == "critical"
    assert category == "safety"


def test_negated_safety_language_does_not_trigger():
    severity, category = classify_event(
        _event("e1", "No smoke or unusual activity was visible", .2))

    assert severity is None
    assert category is None


def test_high_importance_activity_enters_important_inbox(tmp_path):
    center = NotificationCenter(path=str(tmp_path / "notifications.json"))

    alert = center.consider_event(
        _event("e1", "Completed the camera reconnect fix", .82))
    inbox = center.list()

    assert alert["severity"] == "important"
    assert inbox["unread_count"] == 1
    assert inbox["notifications"][0]["event_id"] == "e1"


def test_same_event_is_not_notified_twice(tmp_path):
    center = NotificationCenter(path=str(tmp_path / "notifications.json"))
    event = _event("e1", "Unknown person at the door", .8)

    assert center.consider_event(event) is not None
    assert center.consider_event(event) is None
    assert len(center.list()["notifications"]) == 1


def test_event_can_escalate_from_important_to_critical(tmp_path):
    center = NotificationCenter(path=str(tmp_path / "notifications.json"))

    center.consider_event(
        _event("e1", "Unknown person at the door", .8, timestamp=1000))
    escalated = center.consider_event(
        _event("e1", "Possible forced entry at the door", .98, timestamp=1010))

    assert escalated["severity"] == "critical"
    assert len(center.list()["notifications"]) == 2


def test_notifications_can_be_marked_read(tmp_path):
    center = NotificationCenter(path=str(tmp_path / "notifications.json"))
    alert = center.consider_event(
        _event("e1", "Package delivered at the door", .8))

    center.mark_read(alert["id"])

    assert center.list()["unread_count"] == 0


def test_explicit_task_notification_keeps_speech_and_action_metadata(tmp_path):
    center = NotificationCenter(path=str(tmp_path / "notifications.json"))
    item = center.publish(
        "Task overdue · reminder 1/3",
        "Submit chapter was due. Reply with a reason or reschedule it.",
        category="task_deadline", room_id="agent:tomorrow-planner",
        speak=True, metadata={"task_id": "task-1", "action": "delay_response"},
        timestamp=1000,
    )

    restored = NotificationCenter(
        path=str(tmp_path / "notifications.json")).list()["notifications"][0]
    assert item["speak"] is True
    assert restored["metadata"]["task_id"] == "task-1"
    assert restored["category"] == "task_deadline"
