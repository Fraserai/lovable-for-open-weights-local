"""Tests for Fireworks large dataset upload (signed-URL flow)."""

import json

import httpx
import pytest

from trainloop.fireworks import Fireworks, MAX_SIMPLE_UPLOAD_BYTES


def test_small_file_uses_simple_upload(tmp_path):
    """Files <= 150 MB use the simple :upload endpoint."""
    calls = []

    def handler(request: httpx.Request):
        calls.append(request)
        path = request.url.path
        if path.endswith("/datasets"):
            return httpx.Response(200, json={})
        if path.endswith(":upload"):
            return httpx.Response(200, json={"id": "ds1", "bytes": 100})
        return httpx.Response(200, json={})

    data = tmp_path / "train.jsonl"
    data.write_text('{"messages": [{"role": "user", "content": "hi"}]}\n')

    fw = Fireworks("key", "acct", transport=httpx.MockTransport(handler), poll_seconds=0)
    fw.upload_dataset("ds1", data, 1)
    fw.close()

    # Should NOT call getUploadEndpoint
    endpoints = [c.url.path for c in calls]
    assert any(":upload" in p for p in endpoints)
    assert not any("getUploadEndpoint" in p for p in endpoints)
    assert not any("validateUpload" in p for p in endpoints)


def test_large_file_uses_signed_url_flow(tmp_path, monkeypatch):
    """Files > 150 MB use getUploadEndpoint -> PUT -> validateUpload."""
    # Lower the threshold for testing without a 150 MB file
    monkeypatch.setattr("trainloop.fireworks.MAX_SIMPLE_UPLOAD_BYTES", 100)

    put_calls = []
    api_calls = []

    def handler(request: httpx.Request):
        if "fireworks.ai" in request.url.host:
            api_calls.append(request)
            path = request.url.path
            if path.endswith("/datasets"):
                return httpx.Response(200, json={})
            if path.endswith("getUploadEndpoint"):
                return httpx.Response(200, json={
                    "filenameToSignedUrls": {"train.jsonl": "https://storage.googleapis.com/signed-url"}
                })
            if path.endswith("validateUpload"):
                return httpx.Response(200, json={})
            return httpx.Response(200, json={})
        else:
            # This is the signed URL PUT
            put_calls.append(request)
            return httpx.Response(200, json={})

    # Create a file larger than our patched threshold (100 bytes)
    data = tmp_path / "train.jsonl"
    data.write_text("x" * 200)

    fw = Fireworks("key", "acct", transport=httpx.MockTransport(handler), poll_seconds=0)
    fw.upload_dataset("ds1", data, 1)
    fw.close()

    # Verify API call sequence
    api_paths = [c.url.path for c in api_calls]
    assert "/datasets" in api_paths[0]
    assert "getUploadEndpoint" in api_paths[1]
    assert "validateUpload" in api_paths[2]

    # Verify the PUT to signed URL
    assert len(put_calls) == 1
    put = put_calls[0]
    assert put.method == "PUT"
    assert put.url == "https://storage.googleapis.com/signed-url"
    assert put.headers.get("Content-Type") == "application/octet-stream"
    # No Authorization header on signed URL
    assert "authorization" not in {k.lower() for k in put.headers}


def test_signed_url_put_failure_raises_clear_error(tmp_path, monkeypatch):
    """Failed PUT to signed URL raises FireworksError with status and body."""
    monkeypatch.setattr("trainloop.fireworks.MAX_SIMPLE_UPLOAD_BYTES", 100)

    def handler(request: httpx.Request):
        if "fireworks.ai" in request.url.host:
            path = request.url.path
            if path.endswith("/datasets"):
                return httpx.Response(200, json={})
            if path.endswith("getUploadEndpoint"):
                return httpx.Response(200, json={
                    "filenameToSignedUrls": {"train.jsonl": "https://storage.googleapis.com/signed-url"}
                })
            if path.endswith("validateUpload"):
                return httpx.Response(200, json={})
        return httpx.Response(403, text="Forbidden: signature expired")

    data = tmp_path / "train.jsonl"
    data.write_text("x" * 200)

    fw = Fireworks("key", "acct", transport=httpx.MockTransport(handler), poll_seconds=0)
    with pytest.raises(Exception, match="403: Forbidden"):
        fw.upload_dataset("ds1", data, 1)
    fw.close()


def test_missing_signed_url_raises_clear_error(tmp_path, monkeypatch):
    """If getUploadEndpoint returns no URL, raise a clear error."""
    monkeypatch.setattr("trainloop.fireworks.MAX_SIMPLE_UPLOAD_BYTES", 100)

    def handler(request: httpx.Request):
        if "fireworks.ai" in request.url.host:
            path = request.url.path
            if path.endswith("/datasets"):
                return httpx.Response(200, json={})
            if path.endswith("getUploadEndpoint"):
                return httpx.Response(200, json={"filenameToSignedUrls": {}})
        return httpx.Response(200, json={})

    data = tmp_path / "train.jsonl"
    data.write_text("x" * 200)

    fw = Fireworks("key", "acct", transport=httpx.MockTransport(handler), poll_seconds=0)
    with pytest.raises(Exception, match="no signed upload URL"):
        fw.upload_dataset("ds1", data, 1)
    fw.close()


def test_validate_upload_called_after_put(tmp_path, monkeypatch):
    """validateUpload is called after successful PUT."""
    monkeypatch.setattr("trainloop.fireworks.MAX_SIMPLE_UPLOAD_BYTES", 100)

    calls = []

    def handler(request: httpx.Request):
        calls.append(request)
        if "fireworks.ai" in request.url.host:
            path = request.url.path
            if path.endswith("/datasets"):
                return httpx.Response(200, json={})
            if path.endswith("getUploadEndpoint"):
                return httpx.Response(200, json={
                    "filenameToSignedUrls": {"train.jsonl": "https://storage.googleapis.com/signed-url"}
                })
            if path.endswith("validateUpload"):
                return httpx.Response(200, json={})
        return httpx.Response(200, json={})

    data = tmp_path / "train.jsonl"
    data.write_text("x" * 200)

    fw = Fireworks("key", "acct", transport=httpx.MockTransport(handler), poll_seconds=0)
    fw.upload_dataset("ds1", data, 1)
    fw.close()

    # validateUpload should be the last API call
    api_calls = [c for c in calls if "fireworks.ai" in c.url.host]
    assert api_calls[-1].url.path.endswith("validateUpload")