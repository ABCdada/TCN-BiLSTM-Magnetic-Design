from __future__ import annotations

import pytest

from app import app


@pytest.fixture()
def client():
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def test_resources_page_lists_evidence_classes(client):
    response = client.get("/resources")
    assert response.status_code == 200
    assert b"Resources and results" in response.data
    assert b"Independent measured validation" in response.data
    assert b"Synthetic software verification" in response.data


def test_allowlisted_resource_download(client):
    response = client.get("/downloads/pfc-validation-csv")
    assert response.status_code == 200
    assert "attachment" in response.headers["Content-Disposition"]
    assert b"known_samples,predicted_samples" in response.data


def test_test_center_resource_is_one_line_b_only_csv(client):
    response = client.get("/downloads/synthetic-upload-csv")
    assert response.status_code == 200
    assert "B_TestCenter.csv" in response.headers["Content-Disposition"]
    text = response.get_data(as_text=True).strip()
    assert "B,H" not in text
    assert len(text.splitlines()) == 1
    values = [float(value) for value in text.split(",")]
    assert len(values) == 1_000
    assert max(abs(value) for value in values) < 1.0


def test_unknown_resource_is_not_a_path_download(client):
    assert client.get("/downloads/../../app.py").status_code == 404
    assert client.get("/downloads/not-registered").status_code == 404
