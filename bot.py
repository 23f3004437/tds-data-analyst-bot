from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Response
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

AIPIPE_MODEL = os.getenv(
    "AIPIPE_MODEL",
    "gpt-5-mini",
).strip()

BASE_URL = os.getenv(
    "BASE_URL",
    "http://localhost:8000",
).rstrip("/")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing from environment variables")

if not AIPIPE_TOKEN:
    raise RuntimeError("AIPIPE_TOKEN is missing from environment variables")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
LOG_FILE = Path("run.jsonl")

MAX_HISTORY_MESSAGES = 20

# Professor allows 5 minutes total.
# Keep the agent comfortably inside that.
MAX_AGENT_STEPS = 6
QUESTION_TIME_LIMIT = 210
PYTHON_TOOL_TIME_LIMIT = 45

MAX_TOOL_OUTPUT = 10000


client = OpenAI(
    api_key=AIPIPE_TOKEN,
    base_url=AIPIPE_BASE_URL,
    timeout=40.0,
    max_retries=0,
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
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


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


# Allows UptimeRobot free monitor to use HEAD.
@app.head("/health")
def health_head() -> Response:
    return Response(status_code=200)


@app.get(
    "/run.jsonl",
    response_class=PlainTextResponse,
)
def run_log() -> str:
    if not LOG_FILE.exists():
        return ""

    return LOG_FILE.read_text(
        encoding="utf-8"
    )


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
    Execute model-generated Python in a separate process.

    This prevents slow API calls, excessive request loops,
    or accidental infinite loops from consuming the full
    grading window.
    """

    temp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            encoding="utf-8",
            delete=False,
        ) as temp_file:

            temp_file.write(code)
            temp_path = temp_file.name

        completed = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=PYTHON_TOOL_TIME_LIMIT,
        )

        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()

        if completed.returncode == 0:
            output = stdout
        else:
            output = (
                stdout
                + "\n"
                + stderr
            ).strip()

        if not output:
            output = (
                "Code executed successfully "
                "with no printed output."
            )

        return output[-MAX_TOOL_OUTPUT:]

    except subprocess.TimeoutExpired:
        return (
            f"Python tool timed out after "
            f"{PYTHON_TOOL_TIME_LIMIT} seconds. "
            "Do not repeat the same slow request pattern. "
            "Immediately use a smaller filtered request, "
            "a bulk request, another endpoint, or another "
            "appropriate reliable source."
        )

    except Exception:
        return traceback.format_exc()[
            -MAX_TOOL_OUTPUT:
        ]

    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(
                    missing_ok=True
                )
            except Exception:
                pass


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run Python code for downloading, reading, cleaning, "
                "analysing, calculating and verifying results from public "
                "data. Print useful retrieved values, intermediate values "
                "and final results because only stdout is returned. "
                "Prefer one filtered or bulk request over many sequential "
                "requests. Use short HTTP timeouts, normally around 10 to "
                "15 seconds. If one request fails, change strategy rather "
                "than repeating the identical request."
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

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    return cleaned.strip()


def first_balanced_json_object(
    text: str,
) -> str | None:

    start = text.find("{")

    if start == -1:
        return None

    depth = 0
    inside_string = False
    escaped = False

    for index in range(
        start,
        len(text),
    ):
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
                return text[
                    start:index + 1
                ]

    return None


def normalise_model_reply(
    raw_reply: str,
) -> dict[str, Any]:

    cleaned = remove_code_fences(
        raw_reply
    )

    candidate = first_balanced_json_object(
        cleaned
    )

    parsed: Any

    try:
        if candidate is None:
            parsed = json.loads(cleaned)
        else:
            parsed = json.loads(candidate)

    except (
        json.JSONDecodeError,
        TypeError,
    ):
        parsed = {
            "answer": (
                cleaned
                or
                "Unable to produce an answer"
            )
        }

    if (
        isinstance(parsed, dict)
        and
        "answer" in parsed
    ):
        answer = parsed["answer"]
    else:
        answer = parsed

    return {
        "answer": answer,
        "log_url": f"{BASE_URL}/run.jsonl",
    }


# -------------------------------------------------------------------
# LLM agent
# -------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a careful autonomous data-analysis agent responding through Telegram.

The grader may ask arbitrary public-data analysis questions.
Accuracy and completing within the time limit are both critical.

OUTPUT RULES

1. Answer the user's latest message. Earlier messages are context for a
   multi-turn question.

2. Reply with exactly one JSON object and nothing else.

3. The outer JSON must contain exactly:
   - "answer": shaped exactly as the user's question requests
   - "log_url": "LOG_URL_PLACEHOLDER"

4. Do not add markdown, explanations, code fences or extra outer keys.

5. Match requested key names, nesting, lists, strings, numbers and output
   shapes exactly.

ANALYSIS RULES

6. Never guess a value that can reasonably be retrieved or calculated.

7. Use run_python whenever a result can be downloaded, filtered,
   calculated, compared, ranked, aggregated or verified from public data.

8. Python may use requests, pandas, numpy, BeautifulSoup, openpyxl, lxml,
   json, csv and standard Python libraries.

9. Always print useful retrieved values, intermediate calculations and the
   final calculated result so the analysis is auditable.

10. Perform calculations explicitly. For rankings, ratios, differences,
    percentages, totals or comparisons, calculate every relevant candidate
    and select the result programmatically.

NETWORK AND DATA RETRIEVAL RULES

11. Prefer one targeted, filtered or bulk request over many sequential
    requests.

12. Do not make a separate network request for every country, year, row,
    category or observation when the data can be fetched together.

13. Request only the data needed to answer the question whenever the source
    supports filtering, field selection, pagination or query parameters.

14. Use HTTP request timeouts of roughly 10 to 15 seconds.

15. If a network request fails or times out once, do not repeat the
    identical request. Immediately change strategy.

16. A changed strategy can include a smaller query, a bulk endpoint, another
    API endpoint, another file format, a downloadable dataset, another page
    from the same organization, or another reliable source when appropriate.

17. Do not create loops that may perform dozens of slow HTTP requests.

18. If an API appears unreliable, prefer downloading the relevant dataset
    once and processing it locally when practical.

SOURCE ACCURACY RULES

19. When the user explicitly names a source, organization, dataset or
    indicator, use that source whenever reasonably possible.

20. Do not silently substitute another organization's data when the user
    explicitly requires a particular source.

21. If one endpoint from the required organization fails, first try another
    official endpoint, API, download, file or interface from the same
    organization.

22. Verify dimensions that matter to the question, such as country, year,
    sex, unit, category, frequency, measure or indicator. Do not blindly
    take the first matching row.

23. Check naming and identifier differences such as ISO country codes,
    alternate country names, indicator codes and dimension codes.

24. When using a secondary source because the required source is
    unavailable, prefer a trustworthy source and avoid inventing missing
    values.

TIME MANAGEMENT RULES

25. You must produce the final answer comfortably within four minutes.

26. Do not waste the time budget repeatedly querying a broken endpoint.

27. If a tool call times out, immediately use a substantially different
    strategy.

28. Prefer getting a reliable answer from one or two efficient tool calls
    over performing many exhaustive attempts.

29. When time is running low, stop making tool calls and return the best
    answer supported by the evidence already collected.

MULTI-TURN RULES

30. Preserve useful context from earlier messages in the same Telegram chat.

31. Do not blindly reuse earlier numeric results if the new question changes
    the source, years, indicator, dimensions, filters or calculation.

32. If the user sends a setup-only message, still respond with a small valid
    JSON acknowledgement, for example:
    {"answer":"acknowledged","log_url":"LOG_URL_PLACEHOLDER"}

SECURITY RULES

33. Never expose API keys, tokens, environment variables, private system
    details or secrets.
"""


def build_messages(
    chat_id: int,
    current_message: str,
) -> list[dict[str, str]]:

    with history_lock:
        previous = list(
            chat_histories[
                chat_id
            ]
        )

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        *previous,
        {
            "role": "user",
            "content": current_message,
        },
    ]


def solve_question(
    chat_id: int,
    question: str,
) -> dict[str, Any]:

    started_at = time.monotonic()

    deadline = (
        started_at
        + QUESTION_TIME_LIMIT
    )

    messages: list[
        dict[str, Any]
    ] = build_messages(
        chat_id,
        question,
    )

    write_log(
        {
            "event": "question_received",
            "chat_id": chat_id,
            "question": question,
        }
    )

    final_text = ""

    for step_number in range(
        1,
        MAX_AGENT_STEPS + 1,
    ):

        remaining = (
            deadline
            - time.monotonic()
        )

        # Reserve time for final model output.
        if remaining <= 40:

            messages.append(
                {
                    "role": "system",
                    "content": (
                        "The time limit is approaching. "
                        "Do not call tools again. "
                        "Use the evidence already collected and "
                        "return the best final JSON answer immediately."
                    ),
                }
            )

            try:
                response = (
                    client.chat.completions.create(
                        model=AIPIPE_MODEL,
                        messages=messages,
                    )
                )

                final_text = (
                    response
                    .choices[0]
                    .message
                    .content
                    or ""
                )

            except Exception as error:
                write_log(
                    {
                        "event": "final_model_error",
                        "chat_id": chat_id,
                        "error": str(error),
                    }
                )

            break

        try:
            response = (
                client.chat.completions.create(
                    model=AIPIPE_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )
            )

        except Exception as error:

            write_log(
                {
                    "event": "model_request_error",
                    "chat_id": chat_id,
                    "step": step_number,
                    "error": str(error),
                }
            )

            continue

        assistant_message = (
            response
            .choices[0]
            .message
        )

        assistant_dict = (
            assistant_message.model_dump(
                exclude_none=True
            )
        )

        messages.append(
            assistant_dict
        )

        tool_calls = (
            assistant_message.tool_calls
            or []
        )

        if not tool_calls:
            final_text = (
                assistant_message.content
                or ""
            )
            break

        # Execute only the first tool call from each model step.
        # This prevents one model response from launching several
        # potentially slow tool calls.
        tool_call = tool_calls[0]

        remaining = (
            deadline
            - time.monotonic()
        )

        if remaining <= 40:
            continue

        tool_code = ""
        tool_output = ""

        if (
            tool_call
            .function
            .name
            != "run_python"
        ):
            tool_output = (
                "Unknown tool requested."
            )

        else:
            try:
                arguments = json.loads(
                    tool_call
                    .function
                    .arguments
                )

                tool_code = str(
                    arguments.get(
                        "code",
                        "",
                    )
                )

            except json.JSONDecodeError:
                tool_output = (
                    "Invalid tool arguments."
                )

            if tool_code:
                tool_output = (
                    run_python(
                        tool_code
                    )
                )

        write_log(
            {
                "event": "tool_call",
                "chat_id": chat_id,
                "step": step_number,
                "tool": (
                    tool_call
                    .function
                    .name
                ),
                "code": tool_code,
                "output": tool_output,
            }
        )

        messages.append(
            {
                "role": "tool",
                "tool_call_id": (
                    tool_call.id
                ),
                "content": tool_output,
            }
        )

    if not final_text:

        messages.append(
            {
                "role": "system",
                "content": (
                    "Return the best final answer immediately "
                    "as exactly one JSON object. "
                    "Do not call any tools."
                ),
            }
        )

        try:
            response = (
                client.chat.completions.create(
                    model=AIPIPE_MODEL,
                    messages=messages,
                )
            )

            final_text = (
                response
                .choices[0]
                .message
                .content
                or ""
            )

        except Exception as error:

            write_log(
                {
                    "event": "emergency_final_error",
                    "chat_id": chat_id,
                    "error": str(error),
                }
            )

            final_text = json.dumps(
                {
                    "answer": (
                        "Unable to complete analysis "
                        "within the time limit"
                    )
                }
            )

    final_reply = (
        normalise_model_reply(
            final_text
        )
    )

    elapsed = round(
        time.monotonic()
        - started_at,
        3,
    )

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

        history = (
            chat_histories[
                chat_id
            ]
        )

        history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    final_reply,
                    ensure_ascii=False,
                ),
            }
        )

        chat_histories[
            chat_id
        ] = history[
            -MAX_HISTORY_MESSAGES:
        ]

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
        raise RuntimeError(
            f"Telegram API error: {data}"
        )

    return data


def send_telegram_message(
    chat_id: int,
    reply: dict[str, Any],
) -> None:

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


def handle_message(
    chat_id: int,
    text: str,
) -> None:

    try:
        reply = solve_question(
            chat_id,
            text,
        )

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
        send_telegram_message(
            chat_id,
            reply,
        )

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

    write_log(
        {
            "event": "polling_started"
        }
    )

    while True:

        try:
            payload: dict[
                str,
                Any
            ] = {
                "timeout": 50,
                "allowed_updates": [
                    "message"
                ],
            }

            if offset is not None:
                payload["offset"] = offset

            response = telegram_request(
                "getUpdates",
                payload,
                timeout=60,
            )

            for update in response.get(
                "result",
                [],
            ):

                offset = (
                    int(
                        update[
                            "update_id"
                        ]
                    )
                    + 1
                )

                message = (
                    update.get(
                        "message"
                    )
                    or {}
                )

                text = (
                    message.get(
                        "text"
                    )
                )

                chat = (
                    message.get(
                        "chat"
                    )
                    or {}
                )

                chat_id = (
                    chat.get(
                        "id"
                    )
                )

                if (
                    not isinstance(
                        text,
                        str,
                    )
                    or
                    chat_id is None
                ):
                    continue

                worker = threading.Thread(
                    target=handle_message,
                    args=(
                        int(chat_id),
                        text,
                    ),
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

        if not BASE_URL.startswith(
            "http"
        ):
            continue

        try:
            requests.get(
                f"{BASE_URL}/health",
                timeout=15,
            )

        except Exception as error:
            write_log(
                {
                    "event": "keep_awake_error",
                    "error": str(error),
                }
            )


# -------------------------------------------------------------------
# Startup
# -------------------------------------------------------------------

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
