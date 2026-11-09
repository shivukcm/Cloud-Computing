#!/usr/bin/env python3
"""
Unit tests for the Inference Service API and Schema Validation.
Tested using pytest and FastAPI TestClient.
"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

# Configure environment before importing app
os.environ["MODEL_VERSION"] = "v1"
os.environ["FAULT_INJECT"] = "0"
os.environ["HOSTNAME"] = "test-pod-7744"

import service.app as service_app


class TestInferenceService(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(service_app.app)

    def test_healthz_endpoint(self):
        """Test that /healthz returns HTTP 200 while process is alive."""
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "OK")

    def test_readyz_endpoint(self):
        """Test that /readyz returns 503 before ready and 200 once ready."""
        service_app.is_ready = False
        resp_unready = self.client.get("/readyz")
        self.assertEqual(resp_unready.status_code, 503)

        service_app.is_ready = True
        resp_ready = self.client.get("/readyz")
        self.assertEqual(resp_ready.status_code, 200)

    def test_parse_schema_validation(self):
        """Test validation helper against valid and invalid schemas."""
        valid_json = (
            '{"intent": "weather_query", "date": null, "time": null, "place":'
            ' "london"}'
        )
        is_val, obj = service_app.validate_schema(valid_json)
        self.assertTrue(is_val)
        self.assertEqual(obj["intent"], "weather_query")
        self.assertEqual(obj["place"], "london")

        # Invalid: extra key
        invalid_keys = (
            '{"intent": "weather_query", "date": null, "time": null, "place":'
            ' "london", "extra": 123}'
        )
        is_val, _ = service_app.validate_schema(invalid_keys)
        self.assertFalse(is_val)

        # Invalid: non-existent intent
        bad_intent = (
            '{"intent": "order_pizza", "date": null, "time": null, "place":'
            " null}"
        )
        is_val, _ = service_app.validate_schema(bad_intent)
        self.assertFalse(is_val)

    def test_parse_success_and_headers(self):
        """Test successful POST /parse returns HTTP 200 with required headers."""
        service_app.is_ready = True

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": MagicMock(shape=[1, 5])}
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode.return_value = (
            '{"intent": "alarm_set", "date": "tomorrow", "time": "seven am",'
            ' "place": null}'
        )

        mock_model = MagicMock()
        mock_model.generate.return_value = [[0, 1, 2, 3, 4, 5, 6]]

        with (
            patch.object(service_app, "model", mock_model),
            patch.object(service_app, "tokenizer", mock_tokenizer),
        ):
            resp = self.client.post(
                "/parse", json={"text": "set an alarm for seven am tomorrow"}
            )
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.headers.get("X-Model-Version"), "v1")
            self.assertEqual(resp.headers.get("X-Served-By"), "test-pod-7744")
            data = resp.json()
            self.assertEqual(data["intent"], "alarm_set")
            self.assertEqual(data["date"], "tomorrow")
            self.assertEqual(data["time"], "seven am")
            self.assertIsNone(data["place"])

    def test_fault_injection(self):
        """Test that FAULT_INJECT=1 forces HTTP 500 with exact error body."""
        service_app.is_ready = True
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": MagicMock(shape=[1, 5])}
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode.return_value = (
            '{"intent": "alarm_set", "date": null, "time": null, "place": null}'
        )

        mock_model = MagicMock()
        mock_model.generate.return_value = [[0, 1, 2]]

        with (
            patch.dict(os.environ, {"FAULT_INJECT": "1"}),
            patch.object(service_app, "model", mock_model),
            patch.object(service_app, "tokenizer", mock_tokenizer),
        ):
            resp = self.client.post(
                "/parse", json={"text": "set an alarm for seven am tomorrow"}
            )
            self.assertEqual(resp.status_code, 500)
            self.assertEqual(resp.json(), {"error": "invalid_model_output"})
            self.assertEqual(resp.headers.get("X-Model-Version"), "v1")


if __name__ == "__main__":
    unittest.main()
