from __future__ import annotations

import contextlib
import io
import json
import os
import re
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from openai import OpenAI


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "").strip()
AIPIPE_BASE_URL = os.getenv(
    "AIPIPE_BASE_URL",
    "https://aipipe.org/openai/v1",
).strip()
AIPIPE_MODEL = os.getenv("AIPIPE_MODEL", "gpt-5-nano").strip()
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing from .env")

if not AIPIPE_TOKEN:
    raise RuntimeError("AIPIPE_TOKEN is missing from .env")


TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
LOG_FILE = Path("run.jsonl")

MAX_HISTORY_MESSAGES = 20
MAX_AGENT_STEPS = 10
QUESTION_TIME_LIMIT = 210
MAX_TOOL_OUTPUT = 8000

client = OpenAI(
    api_key=AIPIPE_TOKEN,
    base_url=AIPIPE_BASE_URL,
)

app = FastAPI(title="TDS Data Analyst Telegram Bot")

chat_histories: dict[int, list[dict[str, str]]] = defaultdict(list)
history_lock = threading.Lock()
log_lock = threading.Lock()


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_log(event: dict[str, Any]) -> None:
    record = {
        "timestamp": utc_timestamp(),
        **event,
    }

    with log_lock:
        with LOG_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


# -------------------------------------------------------------------
# FastAPI routes
# -------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": AIPIPE_MODEL,
        "timestamp": utc_timestamp(),
    }


@app.get("/run.jsonl", response_class=PlainTextResponse)
def run_log() -> str:
    if not LOG_FILE.exists():
        return ""

    return LOG_FILE.read_text(encoding="utf-8")


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "TDS Data Analyst Telegram Bot",
    }


# -------------------------------------------------------------------
# Python analysis tool
# -------------------------------------------------------------------

def run_python(code: str) -> str:
    """
    Execute model-generated Python and capture stdout.

    The environment deliberately includes common analysis libraries.
    """
    output_buffer = io.StringIO()

    tool_globals: dict[str, Any] = {
        "__builtins__": __builtins__,
    }

    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code, tool_globals, tool_globals)

        output = output_buffer.getvalue().strip()

        if not output:
            output = "Code executed successfully with no printed output."

        return output[-MAX_TOOL_OUTPUT:]

    except Exception:
        error = traceback.format_exc()
        return error[-MAX_TOOL_OUTPUT:]


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run Python code for downloading, reading, cleaning, analysing "
                "and calculating results from public datasets. Print all useful "
                "results because only stdout is returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python code to execute.",
                    }
                },
                "required": ["code"],
            },
        },
    }
]


# -------------------------------------------------------------------
# JSON response handling
# -------------------------------------------------------------------

def remove_code_fences(text: str) -> str:
    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```$", "", cleaned)

    return cleaned.strip()


def first_balanced_json_object(text: str) -> str | None:
    start = text.find("{")

    if start == -1:
        return None

    depth = 0
    inside_string = False
    escaped = False

    for index in range(start, len(text)):
        character = text[index]

        if inside_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                inside_string = False
            continue

        if character == '"':
            inside_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1

            if depth == 0:
                return text[start:index + 1]

    return None


def normalise_model_reply(raw_reply: str) -> dict[str, Any]:
    cleaned = remove_code_fences(raw_reply)
    candidate = first_balanced_json_object(cleaned)

    parsed: Any

    try:
        if candidate is None:
            parsed = json.loads(cleaned)
        else:
            parsed = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        parsed = {"answer": cleaned or "Unable to produce an answer"}

    if not isinstance(parsed, dict):
        parsed = {"answer": parsed}

    if "answer" not in parsed:
        parsed = {"answer": parsed}

    parsed["log_url"] = f"{BASE_URL}/run.jsonl"

    return parsed


# -------------------------------------------------------------------
# LLM agent
# -------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a careful data-analysis agent responding through Telegram.

Rules:
1. Answer the user's latest message. Earlier messages are context for a
   multi-turn question.
2. Reply with exactly one JSON object and nothing else.
3. The outer JSON must contain exactly:
   - "answer": shaped exactly as the user's question requests
   - "log_url": "LOG_URL_PLACEHOLDER"
4. Do not add explanations, markdown, code fences or extra keys.
5. Match requested key names, nesting, lists, strings and numbers exactly.
6. Use run_python whenever a result can be downloaded, calculated, verified
   or extracted from a public dataset. Do not guess computable values.
7. Python may use requests, pandas, numpy, BeautifulSoup and openpyxl.
8. Print useful values from Python so they are returned to you.
9. If a public-data download fails, try one reasonable alternative source or
   method. Do not waste the entire time budget.
10. If the user sends a setup-only message in a multi-turn conversation,
    still respond with a small valid JSON acknowledgement, for example:
    {"answer": "acknowledged", "log_url": "LOG_URL_PLACEHOLDER"}
11. Never expose API keys, environment variables or private system details.
"""


def build_messages(chat_id: int, current_message: str) -> list[dict[str, str]]:
    with history_lock:
        previous = list(chat_histories[chat_id])

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *previous,
        {"role": "user", "content": current_message},
    ]


def solve_question(chat_id: int, question: str) -> dict[str, Any]:
    started_at = time.monotonic()
    deadline = started_at + QUESTION_TIME_LIMIT

    messages: list[dict[str, Any]] = build_messages(chat_id, question)

    write_log(
        {
            "event": "question_received",
            "chat_id": chat_id,
            "question": question,
        }
    )

    final_text = ""

    for step_number in range(1, MAX_AGENT_STEPS + 1):
        remaining = deadline - time.monotonic()

        if remaining <= 15:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "The time limit is almost reached. Do not call tools. "
                        "Return your best final JSON answer immediately."
                    ),
                }
            )

            response = client.chat.completions.create(
                model=AIPIPE_MODEL,
                messages=messages,
            )

            final_text = response.choices[0].message.content or ""
            break

        response = client.chat.completions.create(
            model=AIPIPE_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message
        assistant_dict = assistant_message.model_dump(exclude_none=True)
        messages.append(assistant_dict)

        tool_calls = assistant_message.tool_calls or []

        if not tool_calls:
            final_text = assistant_message.content or ""
            break

        for tool_call in tool_calls:
            if time.monotonic() >= deadline:
                break

            if tool_call.function.name != "run_python":
                tool_output = "Unknown tool requested."
                tool_code = ""
            else:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    tool_code = str(arguments.get("code", ""))
                except json.JSONDecodeError:
                    tool_code = ""
                    tool_output = "Invalid tool arguments."

                if tool_code:
                    tool_output = run_python(tool_code)

            write_log(
                {
                    "event": "tool_call",
                    "chat_id": chat_id,
                    "step": step_number,
                    "tool": tool_call.function.name,
                    "code": tool_code,
                    "output": tool_output,
                }
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_output,
                }
            )

    if not final_text:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Return your best final answer now as exactly one JSON "
                    "object. Do not call tools."
                ),
            }
        )

        response = client.chat.completions.create(
            model=AIPIPE_MODEL,
            messages=messages,
        )
        final_text = response.choices[0].message.content or ""

    final_reply = normalise_model_reply(final_text)

    elapsed = round(time.monotonic() - started_at, 3)

    write_log(
        {
            "event": "final_reply",
            "chat_id": chat_id,
            "elapsed_seconds": elapsed,
            "raw_model_reply": final_text,
            "reply": final_reply,
        }
    )

    with history_lock:
        history = chat_histories[chat_id]
        history.append({"role": "user", "content": question})
        history.append(
            {
                "role": "assistant",
                "content": json.dumps(final_reply, ensure_ascii=False),
            }
        )
        chat_histories[chat_id] = history[-MAX_HISTORY_MESSAGES:]

    return final_reply


# -------------------------------------------------------------------
# Telegram API
# -------------------------------------------------------------------

def telegram_request(
    method: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    response = requests.post(
        f"{TELEGRAM_API}/{method}",
        json=payload or {},
        timeout=timeout,
    )
    response.raise_for_status()

    data = response.json()

    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error: {data}")

    return data


def send_telegram_message(chat_id: int, reply: dict[str, Any]) -> None:
    text = json.dumps(
        reply,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    telegram_request(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": text,
        },
        timeout=30,
    )


def handle_message(chat_id: int, text: str) -> None:
    try:
        reply = solve_question(chat_id, text)
    except Exception:
        write_log(
            {
                "event": "handler_error",
                "chat_id": chat_id,
                "question": text,
                "error": traceback.format_exc(),
            }
        )

        reply = {
            "answer": "internal error",
            "log_url": f"{BASE_URL}/run.jsonl",
        }

    try:
        send_telegram_message(chat_id, reply)
    except Exception:
        write_log(
            {
                "event": "telegram_send_error",
                "chat_id": chat_id,
                "error": traceback.format_exc(),
            }
        )


def polling_loop() -> None:
    offset: int | None = None

    write_log({"event": "polling_started"})

    while True:
        try:
            payload: dict[str, Any] = {
                "timeout": 50,
                "allowed_updates": ["message"],
            }

            if offset is not None:
                payload["offset"] = offset

            response = telegram_request(
                "getUpdates",
                payload,
                timeout=60,
            )

            for update in response.get("result", []):
                offset = int(update["update_id"]) + 1

                message = update.get("message") or {}
                text = message.get("text")
                chat = message.get("chat") or {}
                chat_id = chat.get("id")

                if not isinstance(text, str) or chat_id is None:
                    continue

                worker = threading.Thread(
                    target=handle_message,
                    args=(int(chat_id), text),
                    daemon=True,
                )
                worker.start()

        except Exception:
            write_log(
                {
                    "event": "polling_error",
                    "error": traceback.format_exc(),
                }
            )
            time.sleep(5)


# -------------------------------------------------------------------
# Render keep-awake
# -------------------------------------------------------------------

def keep_awake_loop() -> None:
    while True:
        time.sleep(600)

        if not BASE_URL.startswith("http"):
            continue

        try:
            requests.get(
                f"{BASE_URL}/health",
                timeout=30,
            )
        except Exception as error:
            write_log(
                {
                    "event": "keep_awake_error",
                    "error": str(error),
                }
            )


@app.on_event("startup")
def startup() -> None:
    polling_thread = threading.Thread(
        target=polling_loop,
        daemon=True,
    )
    polling_thread.start()

    keep_awake_thread = threading.Thread(
        target=keep_awake_loop,
        daemon=True,
    )
    keep_awake_thread.start()