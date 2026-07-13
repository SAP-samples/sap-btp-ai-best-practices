from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)
client.headers.update({"X-API-Key": "test-api-key"})


def test_single_account_endpoint_returns_deterministic_recommendation() -> None:
    """The account endpoint should still return the deterministic baseline response."""
    client.delete("/api/answers")
    response = client.post(
        "/api/accounts/evaluate",
        json={"billing_account": "104"},
    )

    assert response.status_code == 200

    payload = response.json()
    assert payload["billing_account"] == "104"
    assert payload["final_offer"]["display_name"] == "Plan Savings Review"
    assert payload["final_offer"]["explanation"]["polish_status"] == "fallback"
    assert payload["final_offer"]["explanation"]["facts_used"]
    assert payload["explanation"]["summary"].startswith(
        "Plan Savings Review is the current primary recommendation"
    )
    assert payload["explanation"]["facts_used"]
    assert payload["source_documents"]
    assert payload["routing_stage"] == "primary_offer_with_followup"
    assert payload["questions"][0]["explanation"]["blockers"] == [
        payload["questions"][0]["expected_fact"]
    ]
    assert any(
        offer["explanation"]["summary"]
        for offer in payload["blocked_offers"]
        if offer["program_id"] == "income_qualified_discount"
    )


def test_commercial_account_endpoint_returns_deterministic_recommendation() -> None:
    """The account endpoint should still support commercial account evaluation."""
    client.delete("/api/answers")
    response = client.post(
        "/api/accounts/evaluate",
        json={"billing_account": "1004"},
    )

    assert response.status_code == 200

    payload = response.json()
    assert payload["billing_account"] == "1004"
    assert payload["customer_type"] == "COMMERCIAL"
    assert payload["final_offer"] is None
    assert payload["workflow_stage"] == "needs_core_facts"
    assert payload["questions"]


def test_chat_thread_accepts_account_in_free_text_and_persists_state() -> None:
    """The chat API should keep durable thread state after account capture."""
    create_response = client.post("/api/chat/threads")
    assert create_response.status_code == 201

    thread_id = create_response.json()["thread_id"]

    message_response = client.post(
        f"/api/chat/threads/{thread_id}/messages",
        json={"message": "my billing account number is 104"},
    )
    assert message_response.status_code == 200
    assert message_response.json()["billing_account"] == "104"

    fetch_response = client.get(f"/api/chat/threads/{thread_id}")
    assert fetch_response.status_code == 200
    assert fetch_response.json()["billing_account"] == "104"


def test_chat_thread_returns_not_found_for_unknown_thread_id() -> None:
    """Unknown chat threads should return a 404."""
    response = client.get("/api/chat/threads/unknown-thread")

    assert response.status_code == 404
    assert "unknown-thread" in response.json()["detail"]


def test_chat_decline_returns_bad_request_for_ineligible_program() -> None:
    """Declining an offer that is not eligible in the thread should fail."""
    client.delete("/api/answers")
    create_response = client.post("/api/chat/threads")
    thread_id = create_response.json()["thread_id"]

    message_response = client.post(
        f"/api/chat/threads/{thread_id}/messages",
        json={"message": "my billing account number is 104"},
    )
    assert message_response.status_code == 200

    decline_response = client.post(
        f"/api/chat/threads/{thread_id}/decline",
        json={"program_id": "income_qualified_discount"},
    )

    assert decline_response.status_code == 400
    assert "income_qualified_discount" in decline_response.json()["detail"]


def test_batch_run_endpoint_returns_summary_and_artifacts() -> None:
    """Batch runs should return metadata and generated artifact paths."""
    client.delete("/api/answers")
    response = client.post("/api/batch/runs")

    assert response.status_code == 202

    payload = response.json()
    assert payload["summary"]["total_accounts"] > 0
    assert payload["artifacts"]["excel_path"].endswith(".xlsx")
    assert payload["artifacts"]["json_path"].endswith(".json")


def test_single_account_endpoint_reuses_and_persists_saved_answers() -> None:
    """Lookup evaluation should persist request answers and reuse them later."""
    client.delete("/api/answers")

    first_response = client.post(
        "/api/accounts/evaluate",
        json={
            "billing_account": "6001",
            "user_answers": {
                "customer_wants_followup": True,
                "account_name_type": "PERSONAL",
            },
        },
    )
    assert first_response.status_code == 200

    answers_response = client.get("/api/answers/6001")
    assert answers_response.status_code == 200
    assert answers_response.json()["answers"] == {
        "customer_wants_followup": True,
        "account_name_type": "PERSONAL",
    }

    second_response = client.post(
        "/api/accounts/evaluate",
        json={"billing_account": "6001"},
    )
    assert second_response.status_code == 200
    question_facts = [
        question["expected_fact"]
        for question in second_response.json()["questions"]
    ]
    assert "customer_wants_followup" not in question_facts
    assert "account_name_type" not in question_facts


def test_saved_answer_reset_endpoints_clear_account_and_global_answers() -> None:
    """The answers API should reset one account or all saved answer overlays."""
    client.delete("/api/answers")
    client.post(
        "/api/accounts/evaluate",
        json={"billing_account": "6001", "user_answers": {"customer_wants_followup": True}},
    )
    client.post(
        "/api/accounts/evaluate",
        json={"billing_account": "104", "user_answers": {"customer_wants_followup": False}},
    )

    account_reset = client.delete("/api/answers/6001")
    assert account_reset.status_code == 200
    assert account_reset.json()["deleted_count"] == 1
    assert client.get("/api/answers/6001").json()["answers"] == {}
    assert client.get("/api/answers/104").json()["answers"] == {
        "customer_wants_followup": False
    }

    global_reset = client.delete("/api/answers")
    assert global_reset.status_code == 200
    assert global_reset.json()["deleted_count"] == 1
    assert client.get("/api/answers/104").json()["answers"] == {}
