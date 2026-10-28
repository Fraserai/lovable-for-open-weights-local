"""The remote MCP server: the seven tools a coding agent uses to train a model and own it.

Agents connect by URL (https://<host>/mcp) with "Authorization: Bearer <Weightroom API key>".
Tool descriptions use the words people actually use, because that's how an agent picks a tool.
"""

import json

import anyio
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

from . import service
from .errors import WRError

INSTRUCTIONS = """Weightroom trains a custom model on the user's own data and hands back weights they own.
The user brings training-ready data; you drive the tools. The order:
1. upload_data (for a local .jsonl or .csv; run the curl command it returns) - or skip it for a public Hugging Face dataset
2. check_data (free): fix anything it reports; pass its suggested_instructions if it gives some
3. quote (free): show the user the price in credits and the time, and get their yes
4. train with the quote_id (this spends credits; retrying with the same quote never charges twice)
5. status until done (it says when to check back; list_jobs finds jobs from earlier sessions)
6. Show the user the scorecard, then deliver for the weights download, a private Hugging Face repo, or a live endpoint.
Every error says what to do next in its "fix" field."""

mcp = MCPServer(name="weightroom", title="Weightroom", instructions=INSTRUCTIONS,
                description="Train or fine-tune a custom model on your own data and get weights you own.")


def _auth(ctx: Context) -> str | None:
    headers = ctx.headers or {}
    return headers.get("authorization") or headers.get("Authorization")


async def _run(fn, *args, **kwargs) -> dict:
    try:
        return await anyio.to_thread.run_sync(lambda: fn(*args, **kwargs))
    except WRError as e:
        raise ToolError(json.dumps(e.to_dict())) from e


@mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False))
async def upload_data(filename: str, ctx: Context) -> dict:
    """Start uploading a training dataset from the user's computer (.jsonl or .csv).

    Returns a dataset_id and a one-line curl command; run that command from the folder holding the file,
    then call check_data. For a public Hugging Face dataset, skip this and call check_data with hf_dataset.
    Free."""
    return await _run(service.upload_data, _auth(ctx), filename)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True))
async def check_data(ctx: Context, dataset_id: str | None = None, hf_dataset: str | None = None,
                     hf_config: str | None = None, hf_split: str | None = None, limit: int = 5000,
                     user_column: str | None = None, assistant_column: str | None = None,
                     instructions: str | None = None) -> dict:
    """Check a dataset before training (free, instant): rows, which columns are the input and the answer,
    labels and how balanced they are, and errors with the exact fix. Holds back ~10% of rows so the
    scorecard can test the trained model on examples it never saw.

    Give dataset_id (from upload_data) or hf_dataset ('owner/name', public Hugging Face datasets, with
    optional hf_config, hf_split and a row limit). Name the columns with user_column and assistant_column
    if they aren't detected. instructions becomes the system prompt on every example (for label tasks,
    pass the suggested_instructions it returns). Calling it again re-checks with new options."""
    return await _run(service.check_data, _auth(ctx), dataset_id=dataset_id, hf_dataset=hf_dataset,
                      hf_config=hf_config, hf_split=hf_split, limit=limit, user_column=user_column,
                      assistant_column=assistant_column, instructions=instructions)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False))
async def quote(dataset_id: str, ctx: Context, base_model: str | None = None, epochs: int | None = None) -> dict:
    """Get the exact price (in credits) and time to train a custom model on a checked dataset. Free; spends
    nothing. Returns a quote_id that train needs; quotes last an hour.

    base_model is a Tinker model id (default Qwen/Qwen3.5-4B; e.g. Qwen/Qwen3-8B, moonshotai/Kimi-K2.6).
    Leave epochs unset to size training to the data. Show the user the price and get their approval
    before calling train."""
    return await _run(service.quote, _auth(ctx), dataset_id, base_model, epochs)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True))
async def train(quote_id: str, ctx: Context) -> dict:
    """Start training (fine-tuning) a custom model. THIS SPENDS CREDITS: only call it after the user has
    approved the price from quote. Needs the quote_id.

    Returns a job_id straight away; training runs on its own (minutes to an hour). Safe to retry: calling
    it again with the same quote_id returns the same job and never charges twice. Credits are refunded in
    full if training fails."""
    return await _run(service.train, _auth(ctx), quote_id)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def status(job_id: str, ctx: Context) -> dict:
    """Check a training job: state (queued, training, scoring, done, failed), progress, and when to check
    back. When done it includes the scorecard: the original model against the trained one on held-back
    rows (e.g. accuracy 67.5% -> 82.5%). Show the user the scorecard."""
    return await _run(service.status, _auth(ctx), job_id)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def list_jobs(ctx: Context, limit: int = 20) -> dict:
    """List this account's training jobs, newest first: use it to find jobs started in an earlier session."""
    return await _run(service.list_jobs, _auth(ctx), limit)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True))
async def deliver(job_id: str, ctx: Context, download: bool = True, hf_repo: str | None = None,
                  endpoint: bool = True) -> dict:
    """Get the trained model the user owns: a download link for the LoRA adapter (download=true), a push to
    their private Hugging Face repo (hf_repo='name' or 'owner/name'; they add a Hugging Face token in the
    Weightroom dashboard first), and a live OpenAI-compatible endpoint with an example call (endpoint=true).

    Downloads and repo pushes take a minute: call deliver again with the same arguments to get the links."""
    return await _run(service.deliver, _auth(ctx), job_id, download, hf_repo, endpoint)
