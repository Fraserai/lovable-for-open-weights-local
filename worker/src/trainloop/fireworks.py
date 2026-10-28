"""Minimal Fireworks REST client for managed SFT (verified against the Gateway API docs, Sept 2026)."""

import time
from pathlib import Path

import httpx

API = "https://api.fireworks.ai/v1"
MAX_SIMPLE_UPLOAD_BYTES = 150 * 1024 * 1024  # :upload limit; larger files need the signed-URL flow

TERMINAL_STATES = {
    "JOB_STATE_COMPLETED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_EARLY_STOPPED",
    "JOB_STATE_DELETED",
}


class FireworksError(RuntimeError):
    pass


class Fireworks:
    def __init__(self, api_key: str, account_id: str, transport: httpx.BaseTransport | None = None,
                 poll_seconds: float = 30):
        self.account = account_id
        self._transport = transport
        self._client = httpx.Client(
            base_url=f"{API}/accounts/{account_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120,
            transport=transport,
        )
        self._poll = poll_seconds

    def close(self) -> None:
        self._client.close()

    def _call(self, method: str, path: str, **kw) -> dict:
        resp = self._client.request(method, path, **kw)
        if resp.is_error:
            raise FireworksError(f"{method} {path} -> {resp.status_code}: {resp.text[:500]}")
        return resp.json() if resp.content else {}

    def dataset_name(self, dataset_id: str) -> str:
        return f"accounts/{self.account}/datasets/{dataset_id}"

    def model_name(self, model_id: str) -> str:
        return f"accounts/{self.account}/models/{model_id}"

    # --- datasets ---

    def upload_dataset(self, dataset_id: str, jsonl: Path, example_count: int) -> str:
        size = jsonl.stat().st_size
        self._call("POST", "/datasets", json={
            "datasetId": dataset_id,
            "dataset": {"displayName": dataset_id, "exampleCount": str(example_count),
                        "format": "CHAT", "userUploaded": {}},
        })
        if size <= MAX_SIMPLE_UPLOAD_BYTES:
            # Simple multipart upload for files <= 150 MB
            with jsonl.open("rb") as f:
                self._call("POST", f"/datasets/{dataset_id}:upload",
                           files={"file": (jsonl.name, f, "application/jsonl")})
        else:
            # Signed-URL flow for large files
            self._upload_large_dataset(dataset_id, jsonl, size)
        return self.dataset_name(dataset_id)

    def _upload_large_dataset(self, dataset_id: str, jsonl: Path, size: int) -> None:
        """Upload a large dataset file via signed URL (GCS)."""
        # Step 2: Get the signed upload URL
        resp = self._call("POST", f"/datasets/{dataset_id}:getUploadEndpoint",
                          json={"filenameToSize": {jsonl.name: str(size)}})
        signed_url = resp.get("filenameToSignedUrls", {}).get(jsonl.name)
        if not signed_url:
            raise FireworksError(f"no signed upload URL returned for {jsonl.name}")

        # Step 3: PUT the file to the signed URL (no Authorization header, stream from disk)
        # Use the same transport for testability; signed URL needs no auth, just Content-Type
        with jsonl.open("rb") as f:
            with httpx.Client(timeout=None, transport=self._transport) as client:
                put_resp = client.put(
                    signed_url,
                    content=f,
                    headers={"Content-Type": "application/octet-stream"},
                )
                if put_resp.is_error:
                    raise FireworksError(
                        f"PUT {signed_url[:80]}... -> {put_resp.status_code}: {put_resp.text[:500]}"
                    )

        # Step 4: Validate the upload
        self._call("POST", f"/datasets/{dataset_id}:validateUpload", json={})

    def wait_dataset_ready(self, dataset_id: str, timeout: float = 900) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            ds = self._call("GET", f"/datasets/{dataset_id}")
            if ds.get("state") == "READY":
                return ds
            if time.monotonic() > deadline:
                raise FireworksError(f"dataset {dataset_id} not READY after {timeout:.0f}s (state={ds.get('state')})")
            time.sleep(min(self._poll, 10))

    # --- supervised fine-tuning ---

    def create_sft_job(self, job_id: str, dataset: str, base_model: str, output_model_id: str,
                       epochs: int, lora_rank: int, learning_rate: float | None = None,
                       max_context_length: int | None = None) -> dict:
        body = {
            "displayName": job_id,
            "dataset": dataset,
            "baseModel": base_model,
            "outputModel": self.model_name(output_model_id),
            "epochs": epochs,
            "loraRank": lora_rank,
            "evalAutoCarveout": True,
        }
        if learning_rate is not None:
            body["learningRate"] = learning_rate
        if max_context_length is not None:
            body["maxContextLength"] = max_context_length
        return self._call("POST", "/supervisedFineTuningJobs",
                          params={"supervisedFineTuningJobId": job_id}, json=body)

    def get_sft_job(self, job_id: str) -> dict:
        return self._call("GET", f"/supervisedFineTuningJobs/{job_id}")

    def wait_sft_job(self, job_id: str, on_update=None) -> dict:
        while True:
            job = self.get_sft_job(job_id)
            if on_update:
                on_update(job)
            if job.get("state") in TERMINAL_STATES:
                return job
            time.sleep(self._poll)

    def fetch_metrics(self, job: dict) -> bytes | None:
        """Raw metrics file the trainer returns (signed URL on the job), if any."""
        url = job.get("metricsFileSignedUrl")
        if not url:
            return None
        resp = httpx.get(url, timeout=120)
        resp.raise_for_status()
        return resp.content

    # --- handback ---

    def download_model(self, model_id: str, dest: Path) -> list[Path]:
        """Downloads the trained model's files (a LoRA adapter for GLM-5.3-Flash)."""
        urls = self._call("GET", f"/models/{model_id}:getDownloadEndpoint").get("filenameToSignedUrls", {})
        if not urls:
            raise FireworksError(f"no downloadable files for model {model_id}")
        written = []
        for filename, url in urls.items():
            out = dest / filename
            if not out.resolve().is_relative_to(dest.resolve()):
                raise FireworksError(f"refusing unsafe filename from API: {filename}")
            out.parent.mkdir(parents=True, exist_ok=True)
            with httpx.stream("GET", url, timeout=None) as resp:
                resp.raise_for_status()
                with out.open("wb") as f:
                    for block in resp.iter_bytes():
                        f.write(block)
            written.append(out)
        return written
