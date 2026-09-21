import json
import runpy
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from aiops_pipeline import load_data, run_pipeline  # noqa: E402
from anomaly_detector import AnomalyDetector  # noqa: E402
from calculations import area_of_circle, get_nth_fibonacci  # noqa: E402
from event_consumer import EventConsumer  # noqa: E402
from event_producer import EventProducer  # noqa: E402
from event_topic import EventTopic  # noqa: E402


def test_event_topic_stores_and_returns_messages():
    topic = EventTopic("events")
    event = {"type": "anomaly", "value": 42}

    topic.publish(event)

    assert topic.name == "events"
    assert topic.get_messages() == [event]


def test_event_topic_returns_a_copy_of_messages():
    topic = EventTopic("events")
    topic.publish({"id": 1})

    messages = topic.get_messages()
    messages.append({"id": 2})

    assert topic.get_messages() == [{"id": 1}]


def test_event_topic_clear_removes_messages():
    topic = EventTopic("events")
    topic.publish({"id": 1})

    topic.clear()

    assert topic.get_messages() == []


def test_event_producer_publishes_valid_event():
    topic = EventTopic("events")
    producer = EventProducer(topic)
    event = {"type": "anomaly"}

    assert producer.publish(event) is True
    assert topic.get_messages() == [event]


def test_event_producer_rejects_empty_events():
    topic = EventTopic("events")
    producer = EventProducer(topic)

    assert producer.publish(None) is False
    assert producer.publish({}) is False
    assert topic.get_messages() == []


def test_anomaly_detector_reports_all_conditions():
    record = {
        "timestamp": "2026-09-20T10:00:00",
        "service": "payment-service",
        "response_time_ms": 501,
        "cpu_percent": 81,
        "memory_percent": 81,
        "log_level": "WARNING",
    }

    event = AnomalyDetector().detect(record)

    assert event["reasons"] == [
        "High response time",
        "High CPU utilization",
        "High memory utilization",
        "Error log detected",
    ]
    assert event["source"] == record


def test_calculations_reject_negative_inputs():
    with pytest.raises(ValueError, match="Radius cannot be negative"):
        area_of_circle(-1)

    with pytest.raises(ValueError, match="n cannot be negative"):
        get_nth_fibonacci(-1)


def test_get_nth_fibonacci_calculates_later_values():
    assert get_nth_fibonacci(10) == 55


def test_pipeline_loads_data_and_processes_anomalies(tmp_path):
    data_file = tmp_path / "service_data.json"
    records = [
        {
            "timestamp": "2026-09-20T10:00:00",
            "service": "payment-service",
            "response_time_ms": 100,
            "cpu_percent": 40,
            "memory_percent": 50,
            "log_level": "INFO",
        },
        {
            "timestamp": "2026-09-20T10:01:00",
            "service": "payment-service",
            "response_time_ms": 600,
            "cpu_percent": 40,
            "memory_percent": 50,
            "log_level": "INFO",
        },
    ]
    data_file.write_text(json.dumps(records), encoding="utf-8")

    assert load_data(data_file) == records
    result = run_pipeline(data_file)

    assert result["records_processed"] == 2
    assert len(result["anomalies_detected"]) == 1
    assert result["events_consumed"] == []


def test_pipeline_script_runs_from_project_root(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)
    consumed_event = {
        "timestamp": "2026-09-20T10:00:00",
        "service": "payment-service",
        "type": "ANOMALY",
        "reasons": ["High response time"],
    }
    monkeypatch.setattr(EventConsumer, "consume", lambda self: [consumed_event])

    runpy.run_path(str(SRC_DIR / "aiops_pipeline.py"), run_name="__main__")