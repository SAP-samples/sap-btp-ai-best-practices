"""Offline contract tests for the reusable direct SAP RPT API client."""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
sys.path.insert(0, str(ASSET_DIR))

from rpt_client_api import (  # noqa: E402 - reusable asset is added to sys.path above
    RPTAPIError,
    RPTClientAPI,
    flatten_predictions,
    map_explanation_rows,
)

ENVIRONMENT = {
    "AICORE_AUTH_URL": "https://auth.example.test",
    "AICORE_CLIENT_ID": "client-id",
    "AICORE_CLIENT_SECRET": "never-print-this-secret",
    "AICORE_BASE_URL": "https://api.example.test/v2/",
    "AICORE_RESOURCE_GROUP": "default",
}


class FakeResponse:
    """Return a controlled HTTP response without reaching an external service."""

    def __init__(self, payload: dict, status_code: int = 200) -> None:
        """Store the response JSON and HTTP status returned to the client."""
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> dict:
        """Return the configured JSON payload."""
        return self._payload

    def raise_for_status(self) -> None:
        """Raise the same HTTP error type used by requests for failures."""
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeSession:
    """Emulate OAuth, AI Core discovery, and inference HTTP calls."""

    def __init__(
        self,
        *,
        configurations: list[dict] | None = None,
        deployments: list[dict] | None = None,
        token_expires_in: int = 3600,
        prediction: dict | None = None,
        prediction_status: int = 200,
    ) -> None:
        """Configure fake discovery resources, token lifetime, and prediction."""
        self.configurations = configurations or [
            {
                "id": "configuration-1",
                "parameterBindings": [
                    {"key": "modelName", "value": "sap-rpt-1.6"},
                    {"key": "modelVersion", "value": "1"},
                ],
            }
        ]
        self.deployments = deployments or [
            {
                "id": "deployment-1",
                "configurationId": "configuration-1",
                "status": "RUNNING",
                "deploymentUrl": "https://inference.example.test/v2/inference/deployments/deployment-1",
            }
        ]
        self.token_expires_in = token_expires_in
        self.prediction = prediction or {
            "id": "response-1",
            "status": {"code": 0, "message": "OK"},
            "predictions": [{"target": [{"prediction": "A", "confidence": 0.9}]}],
        }
        self.prediction_status = prediction_status
        self.auth_calls = 0
        self.get_calls: list[dict] = []
        self.inference_calls: list[dict] = []

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        """Return a token or prediction and retain the observable request."""
        if url.endswith("/oauth/token"):
            self.auth_calls += 1
            return FakeResponse(
                {
                    "access_token": f"token-{self.auth_calls}",
                    "expires_in": self.token_expires_in,
                }
            )

        captured = {"url": url, **kwargs}
        body = kwargs.get("data")
        if hasattr(body, "to_string"):
            captured["multipart_body"] = body.to_string()
        self.inference_calls.append(captured)
        return FakeResponse(self.prediction, self.prediction_status)

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        """Return configurations or deployments and retain query parameters."""
        self.get_calls.append({"url": url, **kwargs})
        if url.endswith("/lm/configurations"):
            return FakeResponse({"resources": self.configurations})
        if url.endswith("/lm/deployments"):
            return FakeResponse({"resources": self.deployments})
        return FakeResponse({"error": "unexpected URL"}, status_code=404)


def make_client(
    fake_session: FakeSession, model_name: str = "sap-rpt-1.6"
) -> RPTClientAPI:
    """Construct a client with isolated environment variables and HTTP transport."""
    for configuration in fake_session.configurations:
        for binding in configuration.get("parameterBindings", []):
            if binding.get("key") == "modelName":
                binding["value"] = model_name
    with (
        patch.dict(os.environ, ENVIRONMENT, clear=True),
        patch("rpt_client_api.requests.Session", return_value=fake_session),
    ):
        return RPTClientAPI(model_name=model_name)


class RPTClientAPIContractTests(unittest.TestCase):
    """Protect the public environment, transport, and response-helper contracts."""

    def test_loads_dotenv_from_the_callers_working_directory(self) -> None:
        """A copied client must load the consuming project's local .env file."""
        fake = FakeSession()
        with tempfile.TemporaryDirectory() as temporary_directory:
            dotenv_path = Path(temporary_directory) / ".env"
            dotenv_path.write_text(
                "\n".join(f"{key}={value}" for key, value in ENVIRONMENT.items())
                + "\n",
                encoding="utf-8",
            )
            original_directory = Path.cwd()
            try:
                os.chdir(temporary_directory)
                with (
                    patch.dict(os.environ, {}, clear=True),
                    patch("rpt_client_api.requests.Session", return_value=fake),
                ):
                    client = RPTClientAPI()
            finally:
                os.chdir(original_directory)

        self.assertEqual(client.deployment_url, fake.deployments[0]["deploymentUrl"])

    def test_requires_all_environment_variables(self) -> None:
        """A missing credential fails before any HTTP request is attempted."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            original_directory = Path.cwd()
            try:
                os.chdir(temporary_directory)
                with (
                    patch.dict(os.environ, {}, clear=True),
                    self.assertRaisesRegex(RPTAPIError, "AICORE_AUTH_URL"),
                ):
                    RPTClientAPI()
            finally:
                os.chdir(original_directory)

    def test_normalizes_v2_and_resolves_exact_running_deployment(self) -> None:
        """A base URL already ending in /v2 must never become /v2/v2."""
        fake = FakeSession()
        client = make_client(fake)

        self.assertEqual(client.deployment_url, fake.deployments[0]["deploymentUrl"])
        self.assertEqual(
            [call["url"] for call in fake.get_calls],
            [
                "https://api.example.test/v2/lm/configurations",
                "https://api.example.test/v2/lm/deployments",
            ],
        )
        self.assertEqual(fake.get_calls[0]["params"]["scenarioId"], "foundation-models")

    def test_rejects_ambiguous_deployment_matches(self) -> None:
        """Two running deployments for one model must not be selected arbitrarily."""
        deployments = [
            {
                "id": suffix,
                "configurationId": "configuration-1",
                "status": "RUNNING",
                "deploymentUrl": f"https://inference.example.test/{suffix}",
            }
            for suffix in ("deployment-1", "deployment-2")
        ]
        with self.assertRaisesRegex(RPTAPIError, "multiple"):
            make_client(FakeSession(deployments=deployments))

    def test_rejects_missing_running_deployment(self) -> None:
        """A matching configuration without a running deployment is actionable."""
        stopped = [
            {
                "id": "deployment-1",
                "configurationId": "configuration-1",
                "status": "STOPPED",
            }
        ]
        with self.assertRaisesRegex(RPTAPIError, "No running"):
            make_client(FakeSession(deployments=stopped))

    def test_json_keeps_advanced_fields_and_supports_gzip(self) -> None:
        """Raw multi-target and 1.5/1.6 fields must reach AI Core unchanged."""
        fake = FakeSession()
        client = make_client(fake)
        payload = {
            "prediction_config": {
                "target_columns": [
                    {
                        "name": "class_target",
                        "prediction_placeholder": "[PREDICT]",
                        "task_type": "classification",
                        "top_k": 2,
                    },
                    {
                        "name": "value_target",
                        "prediction_placeholder": "[PREDICT]",
                        "task_type": "regression",
                    },
                ],
                "explanations": {
                    "top_column_scores": 3,
                    "top_relevant_context_rows": 2,
                },
                "context_mode": "default",
            },
            "columns": {
                "feature": [1, 2],
                "class_target": ["A", "[PREDICT]"],
                "value_target": [1.5, "[PREDICT]"],
            },
        }

        result = client.predict_json(payload, compress=True)
        sent = fake.inference_calls[-1]

        self.assertIs(result, fake.prediction)
        self.assertEqual(json.loads(gzip.decompress(sent["data"])), payload)
        self.assertEqual(sent["headers"]["Content-Encoding"], "gzip")
        self.assertEqual(sent["url"], client.deployment_url + "/predict")

    def test_async_json_uses_the_same_contract(self) -> None:
        """The async entrypoint must return the same unmodified response."""
        fake = FakeSession()
        client = make_client(fake)
        payload = {
            "prediction_config": {
                "target_columns": [
                    {"name": "target", "prediction_placeholder": "[PREDICT]"}
                ]
            },
            "rows": [{"target": "A"}, {"target": "[PREDICT]"}],
        }

        result = asyncio.run(client.apredict_json(payload))

        self.assertIs(result, fake.prediction)
        self.assertEqual(fake.inference_calls[-1]["json"], payload)

    def test_async_parquet_uses_the_same_contract(self) -> None:
        """The async Parquet entrypoint delegates without changing arguments."""
        fake = FakeSession()
        client = make_client(fake)
        config = {"target_columns": [{"name": "target"}]}

        with tempfile.NamedTemporaryFile(suffix=".parquet") as parquet_file:
            parquet_file.write(b"PAR1-test-data")
            parquet_file.flush()
            result = asyncio.run(
                client.apredict_parquet(
                    parquet_file.name,
                    config,
                    index_column="row_id",
                    parse_data_types=False,
                )
            )

        self.assertIs(result, fake.prediction)
        self.assertIn(b"false", fake.inference_calls[-1]["multipart_body"])

    def test_refreshes_an_expired_oauth_token(self) -> None:
        """A short-lived token is refreshed before the inference request."""
        fake = FakeSession(token_expires_in=1)
        client = make_client(fake)

        client.predict_json(
            {
                "prediction_config": {"target_columns": [{"name": "target"}]},
                "rows": [{"target": "[PREDICT]"}],
            }
        )

        self.assertGreaterEqual(fake.auth_calls, 2)
        self.assertEqual(
            fake.inference_calls[-1]["headers"]["Authorization"],
            f"Bearer token-{fake.auth_calls}",
        )

    def test_parquet_uses_documented_multipart_fields(self) -> None:
        """Parquet requests include file, prediction config, parsing, and index fields."""
        fake = FakeSession()
        client = make_client(fake)
        config = {
            "target_columns": [
                {
                    "name": "target",
                    "prediction_placeholder": "[PREDICT]",
                    "task_type": "regression",
                }
            ],
            "explanations": {"top_column_scores": 2},
        }

        with tempfile.NamedTemporaryFile(suffix=".parquet") as parquet_file:
            parquet_file.write(b"PAR1-test-data")
            parquet_file.flush()
            client.predict_parquet(
                parquet_file.name,
                config,
                index_column="row_id",
                parse_data_types=True,
            )

        sent = fake.inference_calls[-1]
        multipart = sent["multipart_body"]
        self.assertIn(b'name="file"', multipart)
        self.assertIn(json.dumps(config).encode("utf-8"), multipart)
        self.assertIn(b'name="index_column"', multipart)
        self.assertIn(b"row_id", multipart)
        self.assertIn(b"true", multipart)

    def test_deep_context_is_restricted_to_rpt_16_large(self) -> None:
        """Deep mode must fail locally on models where SAP does not support it."""
        client = make_client(FakeSession())
        payload = {
            "prediction_config": {
                "target_columns": [{"name": "target"}],
                "context_mode": "deep",
            },
            "rows": [{"target": "[PREDICT]"}],
        }

        with self.assertRaisesRegex(RPTAPIError, "sap-rpt-1.6-large"):
            client.predict_json(payload)

    def test_context_modes_are_restricted_to_rpt_16(self) -> None:
        """A 1.5 request cannot accidentally send the 1.6 context-mode field."""
        client = make_client(FakeSession(), model_name="sap-rpt-1.5")
        payload = {
            "prediction_config": {
                "target_columns": [{"name": "target"}],
                "context_mode": "default",
            },
            "rows": [{"target": "[PREDICT]"}],
        }

        with self.assertRaisesRegex(RPTAPIError, "RPT-1.6"):
            client.predict_json(payload)

    def test_application_errors_are_safe_and_keep_the_payload(self) -> None:
        """Application status 2/3 errors retain details without exposing credentials."""
        failure = {
            "id": "response-error",
            "status": {"code": 2, "message": "invalid input"},
            "detail": "target column is missing",
        }
        client = make_client(FakeSession(prediction=failure))

        with self.assertRaises(RPTAPIError) as caught:
            client.predict_json(
                {
                    "prediction_config": {"target_columns": [{"name": "target"}]},
                    "rows": [{"target": "[PREDICT]"}],
                }
            )

        self.assertIs(caught.exception.response_payload, failure)
        self.assertNotIn(ENVIRONMENT["AICORE_CLIENT_SECRET"], str(caught.exception))

    def test_http_error_redacts_credentials_echoed_by_the_server(self) -> None:
        """An upstream error body cannot copy configured secrets into exception text."""
        failure = {"error": f"invalid credential {ENVIRONMENT['AICORE_CLIENT_SECRET']}"}
        client = make_client(FakeSession(prediction=failure, prediction_status=400))

        with self.assertRaises(RPTAPIError) as caught:
            client.predict_json(
                {
                    "prediction_config": {"target_columns": [{"name": "target"}]},
                    "rows": [{"target": "[PREDICT]"}],
                }
            )

        self.assertNotIn(ENVIRONMENT["AICORE_CLIENT_SECRET"], str(caught.exception))
        self.assertIn("[REDACTED]", str(caught.exception))
        self.assertEqual(
            caught.exception.response_payload,
            {"error": "invalid credential [REDACTED]"},
        )

    def test_warning_response_is_returned_unchanged(self) -> None:
        """Application status 1 remains available to callers as a raw response."""
        warning = {
            "id": "response-warning",
            "status": {"code": 1, "message": "completed with warning"},
            "predictions": [],
        }
        client = make_client(FakeSession(prediction=warning))

        result = client.predict_json(
            {
                "prediction_config": {"target_columns": [{"name": "target"}]},
                "rows": [{"target": "[PREDICT]"}],
            }
        )

        self.assertIs(result, warning)

    def test_malformed_response_is_rejected(self) -> None:
        """A nominal HTTP success without RPT status is not treated as success."""
        client = make_client(FakeSession(prediction={"predictions": []}))

        with self.assertRaisesRegex(RPTAPIError, "status.code"):
            client.predict_json(
                {
                    "prediction_config": {"target_columns": [{"name": "target"}]},
                    "rows": [{"target": "[PREDICT]"}],
                }
            )

    def test_malformed_status_redacts_credentials_from_diagnostics(self) -> None:
        """Malformed success payloads cannot retain credentials in error details."""
        failure = {
            "status": {"code": "not-a-number"},
            "detail": {"nested": ENVIRONMENT["AICORE_CLIENT_SECRET"]},
        }
        client = make_client(FakeSession(prediction=failure))

        with self.assertRaises(RPTAPIError) as caught:
            client.predict_json(
                {
                    "prediction_config": {"target_columns": [{"name": "target"}]},
                    "rows": [{"target": "[PREDICT]"}],
                }
            )

        self.assertNotIn(
            ENVIRONMENT["AICORE_CLIENT_SECRET"],
            str(caught.exception.response_payload),
        )
        self.assertEqual(
            caught.exception.response_payload["detail"]["nested"], "[REDACTED]"
        )

    def test_http_error_redacts_secret_before_truncating_excerpt(self) -> None:
        """A credential crossing the excerpt boundary cannot leak a partial prefix."""
        failure = {
            "error": "x" * 1_980 + ENVIRONMENT["AICORE_CLIENT_SECRET"]
        }
        client = make_client(FakeSession(prediction=failure, prediction_status=400))

        with self.assertRaises(RPTAPIError) as caught:
            client.predict_json(
                {
                    "prediction_config": {"target_columns": [{"name": "target"}]},
                    "rows": [{"target": "[PREDICT]"}],
                }
            )

        self.assertNotIn(
            ENVIRONMENT["AICORE_CLIENT_SECRET"][:5], str(caught.exception)
        )

    def test_flattens_multi_target_candidates_and_intervals(self) -> None:
        """Each target candidate becomes one complete long-form record."""
        response = {
            "predictions": [
                {
                    "row_id": "q-1",
                    "class_target": [
                        {"prediction": "A", "confidence": 0.7},
                        {"prediction": "B", "confidence": 0.2},
                    ],
                    "value_target": [
                        {
                            "prediction": 12.5,
                            "confidence": 0.8,
                            "confidence_interval": [10.0, 15.0],
                        }
                    ],
                }
            ]
        }

        flattened = flatten_predictions(response, index_column="row_id")

        self.assertEqual(len(flattened), 3)
        self.assertEqual(
            flattened[2],
            {
                "query_position": 0,
                "index": "q-1",
                "target": "value_target",
                "rank": 1,
                "prediction": 12.5,
                "confidence": 0.8,
                "confidence_interval": [10.0, 15.0],
            },
        )

    def test_maps_explanations_for_rows_and_columns(self) -> None:
        """Relevant positions map to the full original input in both JSON layouts."""
        response = {
            "explanations": {
                "top_column_scores": [{"feature": 0.75}],
                "top_relevant_context_rows": [[1, 0]],
            }
        }
        rows = [
            {"row_id": "r-0", "feature": 10},
            {"row_id": "r-1", "feature": 20},
        ]
        columns = {"row_id": ["r-0", "r-1"], "feature": [10, 20]}

        row_result = map_explanation_rows(response, rows)
        column_result = map_explanation_rows(response, columns)

        self.assertEqual(row_result, column_result)
        self.assertEqual(
            row_result[0]["relevant_context_rows"],
            [
                {"position": 1, "row": rows[1]},
                {"position": 0, "row": rows[0]},
            ],
        )

    def test_maps_explanation_positions_for_rows_and_columns(self) -> None:
        """Relevant context indices resolve against the complete original input."""
        response = {
            "explanations": {
                "top_column_scores": [{"feature": 0.8}],
                "top_relevant_context_rows": [[2, 0]],
            }
        }
        rows = [
            {"row_id": "r0", "feature": 10},
            {"row_id": "r1", "feature": 20},
            {"row_id": "r2", "feature": 30},
        ]
        columns = {
            "row_id": ["r0", "r1", "r2"],
            "feature": [10, 20, 30],
        }

        mapped_rows = map_explanation_rows(response, rows)
        mapped_columns = map_explanation_rows(response, columns)

        expected = [
            {
                "query_position": 0,
                "column_scores": {"feature": 0.8},
                "relevant_context_rows": [
                    {"position": 2, "row": rows[2]},
                    {"position": 0, "row": rows[0]},
                ],
            }
        ]
        self.assertEqual(mapped_rows, expected)
        self.assertEqual(mapped_columns, expected)


if __name__ == "__main__":
    unittest.main()
