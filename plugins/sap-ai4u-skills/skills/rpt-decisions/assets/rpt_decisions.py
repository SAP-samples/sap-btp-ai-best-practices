"""Answer typed questions ("decisions") about an input with SAP-RPT on SAP AI Core.

A decision request describes a ``state`` (a text, a JSON object or a chat
log), declares up to 10 typed ``questions`` and gives labeled ``examples``.
The answer is a probability distribution per question.

SAP AI Core exposes RPT as a table model (``/predict``), so this module
translates the decision request into an RPT table:

* one row per example (context) plus one row per state to decide (query);
* the state becomes one or more feature columns;
* every question becomes a classification target column, requested with
  ``top_k`` equal to its number of options so the full distribution returns;
* the RPT response is mapped back to one answer per question.

The request format follows the decision format of SAP's RPT Playground so
requests can be reused; the table translation is our own design.

Question types:
    noul    yes/no            -> probability of true
    choice  multi-class       -> winning option + probabilities
    score   ordinal 0..n-1    -> expected level + probabilities

Example:
    from rpt_client_api import RPTClientAPI
    from rpt_decisions import decide

    client = RPTClientAPI(model_name="sap-rpt-1.6")
    result = decide(client, request)
    print(result["answers"][0]["tone"])
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from rpt_client_api import RPTClientAPI

PLACEHOLDER = "[PREDICT]"
INDEX_COLUMN = "row_id"
MAX_COLUMN_NAME = 100  # RPT column-name length limit
ENCODINGS = ("described", "plain")


def _options(question: Mapping[str, Any]) -> list[str]:
    """Return the answer keys a question can take, in declaration order.

    Args:
        question: One question definition (``type`` plus ``criteria``).

    Returns:
        ``["true", "false"]`` for noul, option names for choice, level indices
        as strings (``"0"``, ``"1"``, ...) for score.
    """
    qtype = question["type"]
    if qtype == "noul":
        return ["true", "false"]
    if qtype == "choice":
        return list(question["criteria"])
    if qtype == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    raise ValueError(f"Unsupported question type {qtype!r}.")


def _answer_key(question: Mapping[str, Any], answer: Any) -> str:
    """Normalize one example answer to its answer key.

    Args:
        question: Question definition the answer belongs to.
        answer: Example answer (bool for noul, option name for choice,
            level index as string or int for score).

    Returns:
        The normalized answer key as a string (see ``_options``).
    """
    if question["type"] == "noul":
        return str(answer).lower()
    return str(answer)


def _encode(question: Mapping[str, Any], key: str, encoding: str) -> str:
    """Turn an answer key into the cell value written into the RPT table.

    ``described`` puts the human-readable criteria into the value so RPT can use
    their meaning (live tests on the triage example gave better tone/priority
    answers this way); ``plain`` writes the raw answer keys.

    Args:
        question: Question definition.
        key: Answer key from ``_options``.
        encoding: ``"described"`` or ``"plain"``.

    Returns:
        Cell value for the target column.
    """
    if encoding == "plain":
        return key
    qtype = question["type"]
    criteria = question.get("criteria") or {}
    if qtype == "noul":
        label = "yes" if key == "true" else "no"
        return f"{label}: {criteria[key]}" if key in criteria else label
    if qtype == "choice":
        return f"{key}: {criteria[key]}"
    return str(question["criteria"][int(key)])  # score: level label text


def _target_columns(questions: Mapping[str, Any], encoding: str) -> dict[str, str]:
    """Choose one RPT column name per question.

    ``described`` uses the question ``instructions`` (truncated to the RPT limit)
    because RPT reads column names semantically; it falls back to the question
    key when instructions are missing or would collide. ``plain`` uses the key.

    Args:
        questions: Question map from the decision request.
        encoding: ``"described"`` or ``"plain"``.

    Returns:
        Mapping of question key to column name.
    """
    names: dict[str, str] = {}
    for key, question in questions.items():
        name = key
        if encoding == "described" and question.get("instructions"):
            name = str(question["instructions"])[:MAX_COLUMN_NAME]
        if name in names.values():
            name = key
        names[key] = name
    return names


def _state_columns(state: Any) -> dict[str, Any]:
    """Convert one state into feature columns.

    Args:
        state: A string, a JSON object (dict) or a JSON array (e.g. chat log).

    Returns:
        ``{"state": text}`` for strings, one column per key for objects, and
        the JSON-serialized list in ``state`` for arrays.
    """
    if isinstance(state, Mapping):
        return dict(state)
    if isinstance(state, (list, tuple)):
        return {"state": json.dumps(state, ensure_ascii=False)}
    return {"state": state}


def validate_decision_request(request: Mapping[str, Any]) -> None:
    """Check a decision request against the decision format's rules.

    Args:
        request: Decision request with ``questions`` and ``examples``.

    Raises:
        ValueError: With a readable message on the first rule violated:
            1-10 questions, 2-2,048 examples, 2-255 options per choice/score,
            every example answers every question with a declared option, and
            every declared option appears in at least one example (RPT can only
            predict labels it has seen in context).
    """
    questions = request.get("questions") or {}
    examples = request.get("examples") or []
    if not 1 <= len(questions) <= 10:
        raise ValueError("A decision request needs 1-10 questions.")
    if not 2 <= len(examples) <= 2048:
        raise ValueError("A decision request needs 2-2,048 examples.")
    for key, question in questions.items():
        if question.get("type") not in {"noul", "choice", "score"}:
            raise ValueError(f"Question {key!r} has unsupported type {question.get('type')!r}.")
        options = _options(question)
        if question["type"] != "noul" and not 2 <= len(options) <= 255:
            raise ValueError(f"Question {key!r} needs 2-255 options.")
        seen = set()
        for position, example in enumerate(examples):
            answers = example.get("answers") or {}
            if key not in answers:
                raise ValueError(f"Example {position} has no answer for {key!r}.")
            answer = _answer_key(question, answers[key])
            if answer not in options:
                raise ValueError(
                    f"Example {position} answers {key!r} with {answers[key]!r}, "
                    f"expected one of {options}."
                )
            seen.add(answer)
        missing = [option for option in options if option not in seen]
        if missing:
            raise ValueError(
                f"Question {key!r}: options {missing} never appear in the examples; "
                "RPT can only predict labels it has seen in context."
            )


def build_decision_payload(
    request: Mapping[str, Any],
    states: Sequence[Any] | None = None,
    *,
    encoding: str = "described",
) -> dict[str, Any]:
    """Translate a decision request into an RPT ``/predict`` payload.

    Args:
        request: Decision request (``state``, ``questions``, ``examples``).
        states: States to decide. Defaults to ``[request["state"]]``. Up to 512
            (the RPT query-row limit) are answered in one call.
        encoding: ``"described"`` (default) or ``"plain"``; see ``_encode``.

    Returns:
        RPT request dictionary with ``rows``, ``index_column`` and a
        ``prediction_config`` holding one classification target per question.

    Raises:
        ValueError: If the request is invalid or the states collide with the
            reserved column names.
    """
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {ENCODINGS}.")
    validate_decision_request(request)
    questions = request["questions"]
    states = list(states) if states is not None else [request["state"]]
    if not 1 <= len(states) <= 512:
        raise ValueError("Provide 1-512 states per call.")
    columns = _target_columns(questions, encoding)

    def state_row(row_id: str, state: Any) -> dict[str, Any]:
        """Build the feature part of one row and guard against name clashes."""
        features = _state_columns(state)
        clash = set(features) & (set(columns.values()) | {INDEX_COLUMN})
        if clash:
            raise ValueError(f"State fields {sorted(clash)} clash with reserved columns.")
        return {INDEX_COLUMN: row_id, **features}

    rows = []
    for position, example in enumerate(request["examples"]):
        row = state_row(f"example-{position}", example["state"])
        for key, question in questions.items():
            answer = _answer_key(question, example["answers"][key])
            row[columns[key]] = _encode(question, answer, encoding)
        rows.append(row)
    for position, state in enumerate(states):
        row = state_row(f"query-{position}", state)
        row.update({columns[key]: PLACEHOLDER for key in questions})
        rows.append(row)

    # RPT expects every row to carry the same columns; fill absent state keys.
    all_columns = list(dict.fromkeys(name for row in rows for name in row))
    rows = [{name: row.get(name) for name in all_columns} for row in rows]

    target_columns = [
        {
            "name": columns[key],
            "prediction_placeholder": PLACEHOLDER,
            "task_type": "classification",
            "top_k": len(_options(question)),  # full distribution
        }
        for key, question in questions.items()
    ]
    return {
        "index_column": INDEX_COLUMN,
        "prediction_config": {"target_columns": target_columns},
        "rows": rows,
    }


def parse_decision_answers(
    response: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    encoding: str = "described",
) -> list[dict[str, Any]]:
    """Map an RPT response back to one answer per question.

    Args:
        response: RPT ``/predict`` response for a payload from
            ``build_decision_payload`` with the same ``encoding``.
        request: The original decision request.
        encoding: Encoding used to build the payload.

    Returns:
        One answer map per query state, in request order, for example::

            {"tone": {"type": "choice", "choice": "angry",
                      "probabilities": {...}, "confidence": 0.46}}

        Options RPT did not return get probability 0.
    """
    questions = request["questions"]
    columns = _target_columns(questions, encoding)
    decoded = {
        key: {_encode(q, option, encoding): option for option in _options(q)}
        for key, q in questions.items()
    }
    query_rows = [
        row
        for row in response.get("predictions") or []
        if str(row.get(INDEX_COLUMN, "")).startswith("query-")
    ]
    query_rows.sort(key=lambda row: int(str(row[INDEX_COLUMN]).split("-")[1]))

    results = []
    for row in query_rows:
        answers: dict[str, Any] = {}
        for key, question in questions.items():
            probabilities = {option: 0.0 for option in _options(question)}
            for candidate in row.get(columns[key]) or []:
                option = decoded[key].get(str(candidate.get("prediction")))
                if option is not None:
                    probabilities[option] = float(candidate.get("confidence") or 0.0)
            best = max(probabilities, key=probabilities.get)
            qtype = question["type"]
            if qtype == "noul":
                answers[key] = {"type": "noul", "noul": probabilities["true"]}
            elif qtype == "choice":
                answers[key] = {
                    "type": "choice",
                    "choice": best,
                    "probabilities": probabilities,
                    "confidence": probabilities[best],
                }
            else:
                answers[key] = {
                    "type": "score",
                    "score": round(sum(int(k) * p for k, p in probabilities.items()), 4),
                    "legend": {str(i): label for i, label in enumerate(question["criteria"])},
                    "probabilities": probabilities,
                    "confidence": probabilities[best],
                }
        results.append(answers)
    return results


def decide(
    client: RPTClientAPI,
    request: Mapping[str, Any],
    states: Sequence[Any] | None = None,
    *,
    encoding: str = "described",
) -> dict[str, Any]:
    """Answer a decision request with one joint RPT ``/predict`` call.

    Args:
        client: Client bound to a tabular RPT deployment (``sap-rpt-1.6``
            is the recommended default).
        request: Decision request.
        states: Optional batch of states; defaults to ``request["state"]``.
        encoding: ``"described"`` (default) or ``"plain"``.

    Returns:
        ``{"model", "answers": [answer map per state], "metadata"}``.

    Raises:
        ValueError: If the request breaks a decision rule.
        RPTAPIError: If the RPT call fails.
    """
    payload = build_decision_payload(request, states, encoding=encoding)
    response = client.predict_json(payload)
    return {
        "model": client.model_name,
        "answers": parse_decision_answers(response, request, encoding=encoding),
        "metadata": response.get("metadata"),
    }
