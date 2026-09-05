from pathlib import Path

from fastapi.testclient import TestClient

import app
import database


ROOT = Path(__file__).resolve().parents[1]


def test_database_delete_removes_children_before_parent(monkeypatch):
    calls = []

    def fake_request(endpoint, method="GET", data=None, query_params=None):
        calls.append((endpoint, method, query_params))
        if endpoint == "omr_results" and method == "GET":
            if query_params.get("id") == "eq.42":
                return [{"id": 42, "exam_id": 9}]
            return []
        return None

    monkeypatch.setattr(database, "is_db_configured", lambda: True)
    monkeypatch.setattr(database, "_supabase_request", fake_request)

    assert database.delete_omr_result_from_db(42) == {"id": 42, "deleted": True}
    assert calls[1][:2] == ("question_results", "DELETE")
    assert calls[2][:2] == ("scans", "DELETE")
    assert calls[3][:2] == ("omr_results", "DELETE")
    assert calls[-1][:2] == ("exams", "DELETE")


def test_delete_api_requires_teacher_key(monkeypatch):
    monkeypatch.setenv("TEACHER_DASHBOARD_DELETE_KEY", "test-secret")
    monkeypatch.setattr(
        app,
        "delete_omr_result_from_db",
        lambda result_id: {"id": result_id, "deleted": True},
    )
    client = TestClient(app.app)

    assert client.delete("/api/omr-results/42").status_code == 403
    response = client.delete(
        "/api/omr-results/42",
        headers={"X-Dashboard-Delete-Key": "test-secret"},
    )
    assert response.status_code == 200
    assert response.json() == {"id": 42, "deleted": True}


def test_result_page_uses_scan_cache_and_dashboard_has_delete_button():
    scanner_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    result_js = (ROOT / "static" / "result.js").read_text(encoding="utf-8")
    dashboard_js = (ROOT / "static" / "dashboard.js").read_text(encoding="utf-8")

    assert "data?.scan_id || data?.id || null" in scanner_js
    assert "sessionStorage.getItem(`omr-result:${resultId}`)" in result_js
    assert 'sessionStorage.getItem("omr-result:latest")' in result_js
    assert 'class="action-delete-btn"' in dashboard_js
    assert 'method: "DELETE"' in dashboard_js
