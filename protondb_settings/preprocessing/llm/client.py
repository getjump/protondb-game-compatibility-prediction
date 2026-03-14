"""OpenAI-compatible LLM client with retry and concurrency support.

Works with any provider: local llama.cpp, OpenRouter, OpenAI, Anthropic, etc.
Supports structured outputs (JSON Schema) with fallback to json_object mode.

Auto-detects ollama and uses native /api/chat to pass think=false, which
disables reasoning tokens for thinking models (Qwen 3.x, DeepSeek-R1 etc.)
giving ~7x speedup on structured output tasks. The /v1 OpenAI-compat
endpoint in ollama does not yet support think parameter (PR #11249 pending).
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    import threading

import httpx
import openai
from pydantic import BaseModel, ValidationError

from protondb_settings.config import (
    DEFAULT_LLM_CONCURRENCY,
    DEFAULT_LLM_MAX_RETRIES,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_TEMPERATURE,
)

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _detect_ollama(base_url: str) -> str | None:
    """Return ollama base URL (without /v1) if endpoint is ollama, else None."""
    probe_url = base_url.rstrip("/")
    if probe_url.endswith("/v1"):
        probe_url = probe_url[:-3]
    try:
        r = httpx.get(f"{probe_url}/api/version", timeout=3)
        if r.status_code == 200 and "version" in r.text:
            return probe_url
    except Exception:
        pass
    return None


class LLMClient:
    """LLM client with multiple backend support.

    Backends:
    - "claude-cli": Uses `claude -p` for extraction (Claude Code CLI)
    - "openai": OpenAI-compatible API (default)
    - Auto-detected ollama: native /api/chat with think=false

    Configuration priority:
    1. Constructor arguments
    2. Environment variables (OPENAI_BASE_URL, OPENAI_API_KEY, MODEL, LLM_BACKEND)
    """

    def __init__(
        self,
        *,
        backend: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_concurrency: int = DEFAULT_LLM_CONCURRENCY,
        max_retries: int = DEFAULT_LLM_MAX_RETRIES,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
    ) -> None:
        self.backend = backend or os.environ.get("LLM_BACKEND", "openai")

        # OpenRouter shortcut: backend=openrouter auto-configures base_url
        if self.backend == "openrouter":
            self.base_url = "https://openrouter.ai/api/v1"
            self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
            self.model = model or os.environ.get("MODEL", "anthropic/claude-sonnet-4")
            if not self.api_key:
                raise ValueError("OPENROUTER_API_KEY env var or --api-key required for openrouter backend")
        else:
            self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8090/v1")
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
            self.model = model or os.environ.get("MODEL", "qwen2.5-7b-instruct")
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.temperature = temperature
        self._structured_supported: bool | None = None
        self._cancel: threading.Event | None = None  # set externally for early abort

        if self.backend in ("claude-cli", "openrouter"):
            if self.backend == "claude-cli":
                log.info("Using Claude CLI backend (claude -p)")
            else:
                log.info("Using OpenRouter backend (model=%s)", self.model)
            self._ollama_base = None
            self._http = None
            if self.backend == "claude-cli":
                self._client = None
                return
            # OpenRouter uses standard OpenAI client
            self._client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=120.0,
            )
            return

        # Detect ollama for native API (think=false support)
        self._ollama_base = _detect_ollama(self.base_url)
        if self._ollama_base:
            log.info("Detected ollama at %s — using native API (think=false)", self._ollama_base)
            self._http = httpx.Client(timeout=120.0)
        else:
            self._http = None

        # OpenAI client as fallback / for non-ollama providers
        self._client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=120.0,
        )

    @property
    def is_local(self) -> bool:
        """Heuristic: consider local if base_url points to localhost."""
        if self.backend in ("openrouter", "claude-cli"):
            return False
        return "localhost" in self.base_url or "127.0.0.1" in self.base_url

    def _build_response_format(
        self, schema_model: type[BaseModel] | None, schema_name: str | None,
    ) -> dict[str, Any]:
        """Build response_format dict, trying json_schema first."""
        if schema_model is not None and self._structured_supported is not False:
            name = schema_name or schema_model.__name__
            json_schema = schema_model.model_json_schema()
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "schema": json_schema,
                    "strict": True,
                },
            }
        return {"type": "json_object"}

    def _strip_code_fences(self, content: str) -> str:
        """Strip markdown code fences if present."""
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)
        return content

    def _call_claude_cli(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        json_schema: dict | None = None,
    ) -> str | None:
        """Call Claude via CLI: `claude -p --output-format json`."""
        import subprocess

        # Build prompt from messages
        system = ""
        user = ""
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            elif m["role"] == "user":
                user = m["content"]

        prompt = f"{system}\n\n{user}" if system else user

        cmd = [
            "claude", "-p",
            "--output-format", "json",
            "--max-turns", "1",
            "--no-session-persistence",
        ]
        if json_schema:
            cmd += ["--json-schema", json.dumps(json_schema)]
        if self.model and self.model != "qwen2.5-7b-instruct":
            cmd += ["--model", self.model]

        try:
            result = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                log.warning("Claude CLI error: %s", result.stderr[:200])
                return None

            # claude --output-format json wraps response in {"type":"result","result":"..."}
            output = json.loads(result.stdout)
            if isinstance(output, dict) and "result" in output:
                return output["result"]
            return result.stdout
        except subprocess.TimeoutExpired:
            log.warning("Claude CLI timeout (120s)")
            return None
        except Exception as e:
            log.warning("Claude CLI error: %s", e)
            return None

    def _call_llm(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        response_format: dict[str, Any],
    ) -> str | None:
        """Call LLM and return content string. Routes to appropriate backend."""
        if self.backend == "claude-cli":
            # Extract json_schema if available
            schema = None
            if response_format.get("type") == "json_schema":
                schema = response_format.get("json_schema", {}).get("schema")
            return self._call_claude_cli(messages, max_tokens=max_tokens, json_schema=schema)

        if self._ollama_base and self._http:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": max_tokens,
                },
                "format": "json",
            }
            r = self._http.post(f"{self._ollama_base}/api/chat", json=payload)
            if r.status_code == 404:
                raise openai.NotFoundError(
                    message=f"Model '{self.model}' not found",
                    response=r, body=None,
                )
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content")

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        return response.choices[0].message.content

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        schema: type[T] | None = None,
        schema_name: str | None = None,
    ) -> dict | list | None:
        """Send a chat completion request and parse the response as JSON.

        If `schema` is provided, attempts structured output (json_schema mode)
        first, falling back to json_object + Pydantic validation.

        Returns parsed dict/list on success, None if all retries exhausted.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_format = self._build_response_format(schema, schema_name)
        used_structured = response_format["type"] == "json_schema"

        for attempt in range(self.max_retries):
            if self._cancel is not None and self._cancel.is_set():
                log.info("LLM request cancelled")
                return None
            try:
                content = self._call_llm(
                    messages, max_tokens=max_tokens,
                    response_format=response_format,
                )
                if not content or not content.strip():
                    log.warning("LLM returned empty content (attempt %d/%d)", attempt + 1, self.max_retries)
                    # If json_schema mode returned empty twice, fall back to json_object
                    if used_structured and attempt >= 1 and self._structured_supported is None:
                        log.info("Falling back to json_object mode (empty responses with json_schema)")
                        self._structured_supported = False
                        response_format = {"type": "json_object"}
                        used_structured = False
                    continue

                content = self._strip_code_fences(content)
                parsed = json.loads(content)

                # Validate with Pydantic if schema provided
                if schema is not None:
                    try:
                        schema.model_validate(parsed)
                    except ValidationError as e:
                        log.warning(
                            "Schema validation failed (attempt %d/%d): %s",
                            attempt + 1, self.max_retries, e,
                        )
                        continue

                if used_structured and self._structured_supported is None:
                    self._structured_supported = True
                    log.info("Structured outputs (json_schema) confirmed working")

                return parsed

            except json.JSONDecodeError as e:
                preview = (content[:200] + "...") if len(content) > 200 else content
                log.warning(
                    "LLM returned invalid JSON (attempt %d/%d): %s\n  Response: %r",
                    attempt + 1, self.max_retries, e, preview,
                )
            except openai.BadRequestError as e:
                if used_structured and self._structured_supported is None:
                    log.info(
                        "Provider doesn't support json_schema, falling back to json_object: %s",
                        e.message,
                    )
                    self._structured_supported = False
                    response_format = {"type": "json_object"}
                    used_structured = False
                    continue
                log.error("Bad request: %s", e.message)
                return None
            except openai.RateLimitError:
                wait = 2 ** attempt
                log.info("Rate limited, waiting %ds (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
            except openai.NotFoundError as e:
                log.error("Model/endpoint not found (404): %s", e.message)
                raise
            except (openai.APIStatusError, httpx.HTTPStatusError) as e:
                status = getattr(e, "status_code", None)
                if status is None:
                    resp = getattr(e, "response", None)
                    status = resp.status_code if resp else None
                if status and status >= 500:
                    wait = 2 ** attempt
                    log.warning("Server error %d, retrying in %ds (attempt %d)", status, wait, attempt + 1)
                    time.sleep(wait)
                else:
                    msg = getattr(e, "message", str(e))
                    log.error("API error %s: %s", status, msg)
                    return None
            except Exception as e:
                wait = 2 ** attempt
                log.warning(
                    "Unexpected error, retrying in %ds (attempt %d): %s",
                    wait, attempt + 1, e,
                )
                time.sleep(wait)

        log.error("All %d retries exhausted", self.max_retries)
        return None

    def batch_complete_json(
        self,
        tasks: list[tuple[str, str]],
        *,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        schema: type[T] | None = None,
        schema_name: str | None = None,
        on_result: Callable[[int, Any], None] | None = None,
    ) -> list[dict | list | None]:
        """Run multiple (system_prompt, user_prompt) pairs concurrently.

        For local LLM servers, limits concurrency to 1 to avoid
        queuing requests that will be processed sequentially anyway.

        Returns results in the same order as input tasks.
        """
        results: list[dict | list | None] = [None] * len(tasks)
        workers = min(self.max_concurrency, 1 if self.is_local else self.max_concurrency)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {}
            for i, (sys_prompt, user_prompt) in enumerate(tasks):
                future = pool.submit(
                    self.complete_json,
                    sys_prompt,
                    user_prompt,
                    max_tokens=max_tokens,
                    schema=schema,
                    schema_name=schema_name,
                )
                future_to_idx[future] = i

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                    if on_result:
                        on_result(idx, result)
                except Exception as e:
                    log.error("Task %d failed: %s", idx, e)

        return results
