import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from llm_validation import check_and_raise_if_truncated, check_schema_step1_flat, TruncationError
from tool import filter_stopwords
from refine import (
    _is_openai_model,
    _use_plain_text_first_for_schema,
    OpenAI,
    build_topic_pruning_prompt_plain_text,
    parse_step2_plain_text,
    build_schema_aware_refine_prompt_plain_text,
    parse_step3_plain_text,
    call_llm,
)

MAX_LLM_NEW_TOKENS = 4096
MISC_SCHEMA = "Misc"
STEP3_TRUNCATION_RETRY_CHUNK_SIZE = 20  # 토픽 수: truncation 재시도 시 청크당 최대
STEP3_TRUNCATION_RETRY_MAX_WORDS = 5   # truncation 재시도 시 토픽당 단어 수

# ---------------------------------------------------------------------------
# Topic word validation (hallucination / noise prevention)
# LLM이 놓친 노이즈·할루시네이션을 사후 검증하여 제거. 2글자 이하 → 전부 삭제.
# ---------------------------------------------------------------------------
_MIN_WORD_LEN = 3  # 2글자 이하 단어 전부 제거


def filter_noise_words(words: List[str]) -> List[str]:
    """Topic word validation: 2글자 이하 토큰 전부 제거. LLM 출력 및 fill 시 공통 적용."""
    if not words:
        return []
    return [w for w in words if isinstance(w, str) and len(w.strip()) >= _MIN_WORD_LEN]


def remove_overlapping_words_across_topics(topics: List[Dict[str, Any]], min_words_after: int = 2) -> None:
    """토픽 간 공통 단어 제거. 2개 이상 토픽에 나온 단어는 삭제 (in-place).
    단, 제거 후 단어가 min_words_after 미만이 되는 토픽에서는 해당 단어를 유지 (빈 토픽 방지)."""
    if not topics:
        return
    word_to_topic_indices: Dict[str, set] = {}
    for i, t in enumerate(topics):
        if not isinstance(t, dict):
            continue
        words = t.get("words", [])
        if not isinstance(words, list):
            continue
        seen_in_this_topic = set()
        for w in words:
            w = str(w).strip()
            if not w:
                continue
            wl = w.lower()
            if wl in seen_in_this_topic:
                continue
            seen_in_this_topic.add(wl)
            if wl not in word_to_topic_indices:
                word_to_topic_indices[wl] = set()
            word_to_topic_indices[wl].add(i)
    overlapping = {w for w, indices in word_to_topic_indices.items() if len(indices) >= 2}
    for t in topics:
        if not isinstance(t, dict):
            continue
        words = t.get("words", [])
        if not isinstance(words, list):
            continue
        filtered = [w for w in words if str(w).strip().lower() not in overlapping]
        if len(filtered) < min_words_after and len(words) > len(filtered):
            t["words"] = words
        else:
            t["words"] = filtered


def load_topic_words_from_file(path: str) -> List[List[str]]:
    topic_words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("topic ") and ":" in line:
                line = line.split(":", 1)[1].strip()
            topic_words.append(line.split())
    return topic_words


def format_topics(topic_words: List[List[str]], max_words: int = 20) -> str:
    return "\n".join(
        [f"Topic {i}: {', '.join(words[:max_words])}" for i, words in enumerate(topic_words)]
    )


def extract_assistant_new_text(tokenizer, output_ids, model_inputs):
    input_len = model_inputs["input_ids"].shape[1]
    new_tokens = output_ids[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _strip_trailing_and_clean_json(text: str) -> str:
    """Strip LLM trailing text and fix common invalid JSON patterns."""
    text = text.strip()
    for sep in (
        "\nNote:", "\nNote ", "\n\nNote:",
        "\nLet me know", "\n\nLet me know",
        "\nI removed", "\nI've followed", "\nIf you need",
        "\nNote: I have removed", "\nI have removed",
    ):
        if sep.lower() in text.lower():
            idx = text.lower().find(sep.lower())
            text = text[:idx].strip()
    text = re.sub(r',\s*//[^\n]*', '', text)
    text = re.sub(r',\s*\.\.\.\s*\]', ']', text)
    return text.strip()


def try_parse_json(text: str):
    text = _strip_trailing_and_clean_json(text)
    try:
        return json.loads(text)
    except Exception:
        pass

    if "```json" in text:
        try:
            text2 = text.split("```json", 1)[1].split("```", 1)[0].strip()
            return json.loads(text2)
        except Exception:
            pass
    elif "```" in text:
        try:
            text2 = text.split("```", 1)[1].split("```", 1)[0].strip()
            return json.loads(text2)
        except Exception:
            pass
    try:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and start < end:
            candidate = _strip_trailing_and_clean_json(text[start : end + 1])
            return json.loads(candidate)
    except Exception:
        pass

    repaired = try_repair_json(text)
    if repaired is not None:
        return repaired
    return None


def _remove_trailing_commas(text: str) -> str:
    cleaned = []
    in_string = False
    escape = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            cleaned.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            cleaned.append(ch)
            i += 1
            continue

        if ch == ",":
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j] in "]}":
                i += 1
                continue

        cleaned.append(ch)
        i += 1
    return "".join(cleaned)


def _append_missing_json_closers(text: str) -> str:
    stack = []
    in_string = False
    escape = False

    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch in "[{":
            stack.append(ch)
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()

    closing_map = {"[": "]", "{": "}"}
    closers = "".join(closing_map[ch] for ch in reversed(stack))
    return text + closers


def try_repair_json(text: str):
    text = text.strip()
    if not text:
        return None

    candidates = [text]

    if "```json" in text:
        candidates.append(text.split("```json", 1)[1].split("```", 1)[0].strip())
    elif "```" in text:
        candidates.append(text.split("```", 1)[1].split("```", 1)[0].strip())

    start_brace = text.find("{")
    start_bracket = text.find("[")
    starts = [idx for idx in (start_brace, start_bracket) if idx != -1]
    if starts:
        candidates.append(text[min(starts):].strip())

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)

        cleaned = _remove_trailing_commas(candidate)
        for variant in (candidate, cleaned):
            variant = variant.strip()
            if not variant:
                continue
            try:
                return json.loads(variant)
            except Exception:
                pass

            repaired = _append_missing_json_closers(variant)
            repaired = _remove_trailing_commas(repaired).strip()
            try:
                return json.loads(repaired)
            except Exception:
                pass

    return None


def parse_schema_labels(schema_text: str) -> List[str]:
    labels = []
    seen = set()
    in_schema = False

    for line in schema_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper() in ("SCHEMA:", "CATEGORY:"):
            in_schema = True
            continue
        if line.upper() == "CRITERION:":
            in_schema = False
            continue
        if in_schema and line.startswith("- "):
            value = line[2:].strip()
            key = value.lower()
            if value and key not in seen:
                seen.add(key)
                labels.append(value)

    return labels


def truncate_schema_step1_repetition(schema_text: str, max_consecutive_dupes: int = 3) -> str:
    """LLM 반복 루프 감지: 동일 라인이 max_consecutive_dupes회 이상 연속이면 그 이전까지만 사용."""
    lines = schema_text.splitlines()
    if len(lines) < max_consecutive_dupes:
        return schema_text
    result = []
    dup_count = 0
    last_line = None
    for line in lines:
        stripped = line.strip()
        if stripped and last_line is not None and stripped == last_line:
            dup_count += 1
            if dup_count >= max_consecutive_dupes:
                print(f"[Step 1] Repetition detected (same line {max_consecutive_dupes}+ times), truncating.")
                break
        else:
            dup_count = 0
            last_line = stripped if stripped else last_line
        result.append(line)
    return "\n".join(result)


def flatten_schema_text(schema_text: str) -> str:
    """Remove hierarchy: keep only top-level labels. One label per line. No indented sub-items."""
    lines = schema_text.splitlines()
    result = []
    in_schema = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append(line)
            continue
        if stripped.upper() in ("SCHEMA:", "CATEGORY:"):
            in_schema = True
            result.append(line)
            continue
        if stripped.upper() == "CRITERION:":
            in_schema = False
            result.append(line)
            continue
        if in_schema:
            # Top-level only: 0~2 spaces before "- ". Skip indented sub-items (4+ spaces).
            indent = len(line) - len(line.lstrip())
            if indent <= 2 and stripped.startswith("- "):
                label = stripped[2:].strip()
                # "Parent: Child" -> keep "Parent" only
                if ":" in label:
                    label = label.split(":")[0].strip()
                if label:
                    result.append("- " + label)
            continue
        result.append(line)
    return "\n".join(result)


def split_keep_and_misc(
    topic_words: List[List[str]],
    scores_json: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split step2 verdict into: (topics for step3, misc topics).
    - keep -> surviving_topics for step3
    - delete -> each kept as separate topic with schema=Misc, topic_name=misc1, misc2, ...
    """
    if not isinstance(scores_json, list):
        return (
            [{"topic_id": i, "words": words} for i, words in enumerate(topic_words)],
            [],
        )

    deleted_ids = set()
    kept_topic_names = {}
    for item in scores_json:
        if not isinstance(item, dict):
            continue
        tid = item.get("topic_id", None)
        decision = (item.get("decision") or "").strip().lower()
        topic_name = str(item.get("topic_name", "")).strip()
        if tid is None:
            continue
        try:
            tid_int = int(tid)
        except (TypeError, ValueError):
            continue
        if tid_int < 0 or tid_int >= len(topic_words):
            continue
        if decision == "delete":
            deleted_ids.add(tid_int)
            continue
        if decision == "keep" and topic_name:
            kept_topic_names[tid_int] = topic_name

    surviving = []
    misc_topics = []
    misc_counter = 1

    for i, words in enumerate(topic_words):
        if i in deleted_ids:
            misc_topics.append({
                "topic_id": i,
                "topic_name": f"misc{misc_counter}",
                "words": words,
                "schema": MISC_SCHEMA,
            })
            misc_counter += 1
            continue
        item = {"topic_id": i, "words": words}
        if i in kept_topic_names:
            item["topic_name"] = kept_topic_names[i]
        surviving.append(item)

    return surviving, misc_topics


def format_surviving_topics(surviving_topics: List[Dict[str, Any]], max_words: int = 20) -> str:
    lines = []
    for item in surviving_topics:
        words = item.get("words", [])[:max_words]
        suggested_name = str(item.get("topic_name", "")).strip()
        if suggested_name:
            lines.append(f"Topic {item['topic_id']} (suggested name: {suggested_name}): {', '.join(words)}")
        else:
            lines.append(f"Topic {item['topic_id']}: {', '.join(words)}")
    return "\n".join(lines)


def build_schema_prompt(topic_words: List[List[str]]) -> List[Dict[str, str]]:
    topics_text = format_topics(topic_words)

    system = (
        "You are an expert in topic modeling. "
        "Your job is to set one global criterion and a list of categories that form the upper layer of the topics—categories must group the topics well (no hierarchy within the list)."
    )

    user = f"""
Topics:
{topics_text}

Task:
1. Look at the topic words above and form a global view of the set.
2. Set one global criterion for how to partition the topics (e.g. by domain, theme, or semantic type). This criterion applies to the whole set.
3. List category labels that are the upper layer of the topics: each category should group several topics well. The list is a flat set of high-level labels—not a long enumeration of fine-grained terms. Aim for a small number (e.g. 5–20) that covers the full set.

Rules:
- Criterion: one sentence only. It states the single partitioning criterion for the topic set.
- Categories: they are the upper layer of topics; they must group the topics well. Keep them broad—one word per label; no slashes, no multi-word phrases; one category per line; no sub-categories, no indentation, no "Label: description". Do not list hundreds of fine-grained labels; stop when you have a compact set that groups the topics.
- No repetition: output each category label at most once; do not repeat the same or similar block of labels. When the list is complete, stop.

Output:
CRITERION:
- <one sentence>

CATEGORY:
- <label 1>
- <label 2>
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def build_topic_pruning_prompt(
    topic_words: List[List[str]],
    schema_text: str,
) -> List[Dict[str, str]]:
    topics_text = format_topics(topic_words)
    n_topics = len(topic_words)

    system = (
        "Topic modeling expert. For each topic: decide keep/delete and give a 1–2 word topic_name. "
        "Use only the given schema and topic words; do not invent meanings."
    )

    user = f"""
Schema:
{schema_text}

Topics:
{topics_text}

Task: For each topic (0 to {n_topics - 1}), decide keep or delete and, if keep, a short topic_name (1–2 words). Keep if interpretable and nameable; delete if unclear, generic, mixed, noisy, or stopword-heavy. When in doubt, keep. topic_name must describe that topic's words only; avoid placeholders like "Thing", "Topic".

Return: Exactly one JSON array of {n_topics} objects. Each object: "topic_id" (int 0..{n_topics - 1}), "decision" ("keep"|"delete"), "topic_name" (string or null). No text before or after the array. No ellipsis or omission—all {n_topics} objects required (parsed by code).

Example shape (one object per topic 0..{n_topics - 1}, no abbreviation): [{{"topic_id":0,"decision":"keep","topic_name":"Name"}},{{"topic_id":1,"decision":"delete","topic_name":null}}]
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _ensure_misc_in_schema(schema_text: str) -> str:
    """Ensure 'misc' is in the schema for step3 (weak topics)."""
    if MISC_SCHEMA.lower() in schema_text.lower():
        return schema_text
    return schema_text.rstrip() + f"\n- {MISC_SCHEMA}"


def build_schema_aware_refine_prompt(
    surviving_topics: List[Dict[str, Any]],
    schema_text: str,
    surviving_topics_n: int,
    step2_misc_topics: Optional[List[Dict[str, Any]]] = None,
    max_words_per_topic: int = 8,
    output_topic_ids: Optional[List[int]] = None,
) -> List[Dict[str, str]]:
    schema_text = _ensure_misc_in_schema(schema_text)
    topics_text = format_surviving_topics(surviving_topics, max_words=max_words_per_topic)
    step2_misc_topics = step2_misc_topics or []
    misc_text = format_surviving_topics(step2_misc_topics, max_words=max_words_per_topic) if step2_misc_topics else ""
    total_n = surviving_topics_n + len(step2_misc_topics)
    ids_for_prompt = output_topic_ids if output_topic_ids is not None else list(range(total_n))
    ids_str = ", ".join(str(i) for i in ids_for_prompt)

    system = (
        "You are an expert in topic modeling. "
        "For each topic, do two things only: "
        "(1) prune unnecessary words based on topic_name, "
        "(2) assign the best schema label. "
        "Use topic_name when it matches the words; replace it if it contradicts. "
        "Be conservative: when in doubt, keep the word. "
        "Do not invent unsupported meanings. "
        "CRITICAL: Output every single topic. Do NOT omit, skip, or abbreviate any topic. "
        "Do NOT use ellipsis (...), 'etc', or 'and so on'. One full JSON object per topic."
    )

    misc_section = ""
    if misc_text:
        misc_section = f"""
Misc topics (word elimination only; assign schema '{MISC_SCHEMA}' for all of these):
{misc_text}

"""

    user = f"""
Schema:
{schema_text}

Surviving topics (word elimination + schema assignment):
{topics_text}
{misc_section}
Task:
For each topic:

1) Word elimination
- Use topic_name only when it matches the words; if it contradicts the words, replace topic_name.
- Keep words that form a tight semantic cluster. Remove only words that dilute the topic's focus or feel unrelated.
- Remove stopwords that harm the topic's meaning.
- Remove only words that are clearly generic, meaningless, filler-like, or inconsistent with the topic.
- Remove 1-2 character tokens only if they are not meaningful abbreviations or domain terms.
- If a word overlaps with another topic, remove it only when it is not important for this topic.
- Keep {max_words_per_topic} most representative words per topic. Do NOT exceed {max_words_per_topic} words per topic.

2) Schema assignment
- Assign the single best schema label using topic_name and the remaining words.
- If the topic is still too weak or unclear, mark it as delete and set schema to null.

Output rules:
- Return exactly one JSON array.
- MANDATORY: Output exactly {total_n} topic objects. One object for each topic_id: {ids_str}. Do NOT omit any topic.
- Each object must contain: topic_id, topic_name, words, schema.
- Replace topic_name if it contradicts the words.
- Use exactly one schema if kept, or null if deleted.
- No explanation before or after the JSON.

CRITICAL - NO TRUNCATION:
- NEVER use ellipsis (...), "etc", or "and so on" in the JSON array.
- Output must contain exactly {total_n} complete objects. No omission.

Output format: A single JSON array of exactly {total_n} objects. Each object has "topic_id", "topic_name", "words" (array of strings, up to {max_words_per_topic} per topic), "schema" (string or null). One object per line inside the array. No explanation before or after.

You must write all {total_n} objects in full. Do not stop early. Do not write "..." or "truncated" or omit any topic—otherwise the response is invalid.
"""
    
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_llm(
    model=None,
    tokenizer=None,
    client=None,
    model_name: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    max_new_tokens: int = 4096,
    device: str = "cuda",
    llm_call_count: Optional[List[int]] = None,
) -> str:
    """LLM 호출. OpenAI API (client, model_name) 또는 HuggingFace (model, tokenizer) 지원."""
    requested_max_new_tokens = int(max_new_tokens)
    clamped_max_new_tokens = min(requested_max_new_tokens, MAX_LLM_NEW_TOKENS)
    if clamped_max_new_tokens != requested_max_new_tokens:
        print(
            "Clamping max_new_tokens from {} to {}.".format(
                requested_max_new_tokens,
                clamped_max_new_tokens,
            )
        )

    if client is not None and model_name:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=clamped_max_new_tokens,
            temperature=0,
        )
        if llm_call_count is not None:
            llm_call_count[0] += 1
        content = response.choices[0].message.content if response.choices else ""
        return (content or "").strip()

    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    try:
        if hasattr(model, "generation_config") and getattr(model.generation_config, "max_length", None) is not None:
            model.generation_config.max_length = None
    except Exception:
        pass
    try:
        if hasattr(model, "config") and getattr(model.config, "max_length", None) is not None:
            model.config.max_length = None
    except Exception:
        pass

    outputs = model.generate(
        **model_inputs,
        max_new_tokens=clamped_max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.3,
    )
    if llm_call_count is not None:
        llm_call_count[0] += 1
    return extract_assistant_new_text(tokenizer, outputs, model_inputs)


def call_llm_until_valid_json(
    model=None,
    tokenizer=None,
    client=None,
    model_name: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    max_new_tokens: int = 4096,
    *,
    step_name: str,
    json_retry_attempts: int = 0,
    device: str = "cuda",
    llm_call_count: Optional[List[int]] = None,
):
    last_text = None
    last_json = None
    total_attempts = max(0, int(json_retry_attempts)) + 1

    for attempt_idx in range(total_attempts):
        attempt_num = attempt_idx + 1
        attempt_max_new_tokens = min(max_new_tokens * attempt_num, MAX_LLM_NEW_TOKENS)
        if attempt_num > 1:
            print(
                "[{}] Retrying malformed JSON ({}/{}) with max_new_tokens={}.".format(
                    step_name,
                    attempt_num,
                    total_attempts,
                    attempt_max_new_tokens,
                )
            )

        text = call_llm(
            model=model,
            tokenizer=tokenizer,
            client=client,
            model_name=model_name,
            messages=messages,
            max_new_tokens=attempt_max_new_tokens,
            device=device,
            llm_call_count=llm_call_count,
        )
        parsed_json = try_parse_json(text)
        if parsed_json is not None:
            if attempt_num > 1:
                print("[{}] Recovered valid JSON on retry {}.".format(step_name, attempt_num))
            return text, parsed_json

        print(
            "[{}] Invalid JSON response on attempt {}/{}.".format(
                step_name,
                attempt_num,
                total_attempts,
            )
        )
        last_text = text
        last_json = None

    return last_text, last_json


def _assign_misc_topic_names(groups: List[Dict[str, Any]]) -> None:
    """Assign misc1, misc2, ... to all topics in Misc schema group."""
    for g in groups:
        if not isinstance(g, dict):
            continue
        if str(g.get("label", "")).strip().lower() != MISC_SCHEMA.lower():
            continue
        topics = g.get("topics", [])
        if not isinstance(topics, list):
            continue
        for i, t in enumerate(topics):
            if isinstance(t, dict):
                t["topic_name"] = f"misc{i + 1}"


def _schema_topics_from_refined_list(refined_topics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """refined_topics (flat) -> schema_topics (grouped by schema). 50개 보장."""
    grouped_map = {}
    schema_order = []
    for t in refined_topics:
        if not isinstance(t, dict):
            continue
        schema = str(t.get("schema", MISC_SCHEMA)).strip() or MISC_SCHEMA
        topic_id = t.get("topic_id")
        topic_name = str(t.get("topic_name", "")).strip() or schema
        words = t.get("words", [])
        if topic_id is None or not isinstance(words, list):
            continue
        if schema not in grouped_map:
            grouped_map[schema] = []
            schema_order.append(schema)
        grouped_map[schema].append({"topic_id": topic_id, "topic_name": topic_name, "words": words})
    groups = []
    for schema in schema_order:
        topics = sorted(grouped_map[schema], key=lambda x: x["topic_id"])
        groups.append({"label": schema, "topics": topics})
    return {"schema": groups}


def _keep_mode_merge_misc_and_clear_rest(
    refined_topics: List[Dict[str, Any]],
    deleted_ids_step2: List[int],
    step2_misc_topics: List[Dict[str, Any]],
    max_merged_words: int = 30,
) -> List[Dict[str, Any]]:
    """
    Keep 모드 개선: delete였던 토픽을 하나로 묶고, 나머지 delete 슬롯은 앵커 미지정(빈 단어).
    SCHEMATOPIC_KEEP_MERGE_MISC=1 일 때만 적용.
    """
    if not deleted_ids_step2 or not step2_misc_topics:
        return refined_topics
    if os.environ.get("SCHEMATOPIC_KEEP_MERGE_MISC", "").strip() not in ("1", "true", "yes"):
        return refined_topics
    print("[Keep mode] SCHEMATOPIC_KEEP_MERGE_MISC=1: merging", len(step2_misc_topics), "misc into one; other delete slots -> anchor-unspecified.")

    merged_words = []
    seen = set()
    for m in step2_misc_topics:
        if not isinstance(m, dict):
            continue
        for w in (m.get("words") or []):
            w = (w or "").strip()
            if len(w) >= _MIN_WORD_LEN and w not in seen:
                seen.add(w)
                merged_words.append(w)
                if len(merged_words) >= max_merged_words:
                    break
        if len(merged_words) >= max_merged_words:
            break
    merged_words = filter_stopwords(filter_noise_words(merged_words))

    chosen_id = min(deleted_ids_step2)
    by_id = {t["topic_id"]: t for t in refined_topics if isinstance(t, dict) and t.get("topic_id") is not None}
    for tid in deleted_ids_step2:
        if tid not in by_id:
            continue
        if tid == chosen_id:
            by_id[tid]["words"] = merged_words
            by_id[tid]["topic_name"] = "misc"
            by_id[tid]["schema"] = MISC_SCHEMA
        else:
            by_id[tid]["words"] = []
            by_id[tid]["topic_name"] = ""
            by_id[tid]["schema"] = MISC_SCHEMA
    return refined_topics


def _fill_missing_topics_for_keep_mode(
    refined_topics: List[Dict[str, Any]],
    all_sources: List[Dict[str, Any]],
    expected_count: int,
) -> List[Dict[str, Any]]:
    """keep 모드: expected_count만큼 토픽이 있어야 함. 누락된 topic_id를 all_sources에서 복구."""
    existing_ids = {t["topic_id"] for t in refined_topics if isinstance(t, dict) and t.get("topic_id") is not None}
    if len(existing_ids) >= expected_count:
        return refined_topics
    source_by_id = {}
    for s in all_sources:
        if isinstance(s, dict) and s.get("topic_id") is not None:
            source_by_id[s["topic_id"]] = s
    filled = list(refined_topics)
    for tid in range(expected_count):
        if tid in existing_ids:
            continue
        if tid not in source_by_id:
            continue
        src = source_by_id[tid]
        raw_words = list(src.get("words", []))[:20]
        filled.append({
            "topic_id": tid,
            "topic_name": src.get("topic_name", MISC_SCHEMA),
            "words": filter_stopwords(filter_noise_words(raw_words)),  # validation: fill 시에도 동일 규칙
            "schema": src.get("schema", MISC_SCHEMA),
        })
    filled.sort(key=lambda x: x.get("topic_id", 0))
    return filled


def merge_step2_misc_into_schema_topics(
    schema_topics: Dict[str, Any],
    step2_misc_topics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Add step2 misc topics to schema_topics. Creates or appends to Misc group. Assigns misc1, misc2, ...
    Only adds topics not already present (e.g. step3 may have already refined and included them)."""
    if not step2_misc_topics:
        return schema_topics

    groups = schema_topics.get("schema", [])
    if not isinstance(groups, list):
        groups = []

    existing_topic_ids = set()
    for g in groups:
        if isinstance(g, dict):
            for t in g.get("topics", []):
                if isinstance(t, dict) and t.get("topic_id") is not None:
                    existing_topic_ids.add(t["topic_id"])

    misc_group = None
    for g in groups:
        if isinstance(g, dict) and str(g.get("label", "")).strip().lower() == MISC_SCHEMA.lower():
            misc_group = g
            break

    misc_topics = [
        t
        for t in step2_misc_topics
        if isinstance(t, dict) and t.get("words") and t.get("topic_id") not in existing_topic_ids
    ]
    if not misc_topics:
        _assign_misc_topic_names(groups)
        return {"schema": groups}

    if misc_group is None:
        groups.append({
            "label": MISC_SCHEMA,
            "topics": misc_topics,
        })
    else:
        misc_group["topics"] = list(misc_group.get("topics", [])) + misc_topics
        misc_group["topics"].sort(key=lambda x: x.get("topic_id", 0))

    _assign_misc_topic_names(groups)
    return {"schema": groups}


def postprocess_final_topics(final_topics):
    grouped = []
    invalid_schema_values = {"", "none", "null"}

    if isinstance(final_topics, dict):
        groups = final_topics.get("schema", [])
        if not isinstance(groups, list):
            return {"schema": []}
        for group in groups:
            if not isinstance(group, dict):
                continue
            schema_raw = group.get("label", "")
            schema = str(schema_raw).strip()
            topics = group.get("topics", [])
            if schema_raw is None or schema.lower() in invalid_schema_values:
                schema = MISC_SCHEMA
            if not isinstance(topics, list):
                continue

            cleaned_topics = []
            seen_topic_ids = set()
            for item in topics:
                if not isinstance(item, dict):
                    continue
                topic_id = item.get("topic_id", None)
                topic_name = str(item.get("topic_name", "")).strip()
                if schema.lower() == MISC_SCHEMA.lower():
                    topic_name = topic_name or MISC_SCHEMA
                words = item.get("words", [])
                if topic_id is None or not topic_name or not isinstance(words, list):
                    continue
                if topic_id in seen_topic_ids:
                    continue

                cleaned_words = []
                seen_words = set()
                for w in words:
                    w = str(w).strip()
                    if not w:
                        continue
                    wl = w.lower()
                    if wl in seen_words:
                        continue
                    seen_words.add(wl)
                    cleaned_words.append(w)

                if len(cleaned_words) < 1:
                    continue

                cleaned_topics.append(
                    {
                        "topic_id": topic_id,
                        "topic_name": topic_name,
                        "words": cleaned_words,
                    }
                )
                seen_topic_ids.add(topic_id)

            cleaned_topics.sort(key=lambda x: x["topic_id"])
            if cleaned_topics:
                grouped.append({"label": schema, "topics": cleaned_topics})
    elif isinstance(final_topics, list):
        grouped_map = {}
        schema_order = []
        for item in final_topics:
            if not isinstance(item, dict):
                continue

            topic_id = item.get("topic_id", None)
            topic_name = str(item.get("topic_name", "")).strip()
            words = item.get("words", [])
            schema_raw = item.get("schema", "")
            schema = str(schema_raw).strip()
            if schema_raw is None or schema.lower() in invalid_schema_values:
                schema = MISC_SCHEMA
            if schema.lower() == MISC_SCHEMA.lower():
                topic_name = topic_name or MISC_SCHEMA
            if topic_id is None or not topic_name or not isinstance(words, list):
                continue

            cleaned_words = []
            seen_words = set()
            for w in words:
                w = str(w).strip()
                if not w:
                    continue
                wl = w.lower()
                if wl in seen_words:
                    continue
                seen_words.add(wl)
                cleaned_words.append(w)

            if len(cleaned_words) < 1:
                continue

            if schema not in grouped_map:
                grouped_map[schema] = []
                schema_order.append(schema)
            grouped_map[schema].append(
                {
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "words": cleaned_words,
                }
            )

        for schema in schema_order:
            topics = sorted(grouped_map[schema], key=lambda x: x["topic_id"])
            if topics:
                grouped.append({"label": schema, "topics": topics})
    else:
        return {"schema": []}

    return {"schema": grouped}


def flatten_schema_topics(schema_topics: Any) -> List[Dict[str, Any]]:
    if not isinstance(schema_topics, dict):
        return []

    groups = schema_topics.get("schema", [])
    if not isinstance(groups, list):
        return []

    invalid_schema_values = {"", "none", "null"}
    flat_topics = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        schema_raw = group.get("label", "")
        schema = str(schema_raw).strip()
        topics = group.get("topics", [])
        if schema_raw is None or schema.lower() in invalid_schema_values:
            schema = MISC_SCHEMA
        if not isinstance(topics, list):
            continue
        for topic in topics:
            if not isinstance(topic, dict):
                continue
            topic_id = topic.get("topic_id", None)
            topic_name = str(topic.get("topic_name", "")).strip() or schema
            words = topic.get("words", [])
            if topic_id is None or not topic_name or not isinstance(words, list):
                continue
            flat_topics.append(
                {
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "words": words,
                    "schema": schema,
                }
            )

    flat_topics.sort(key=lambda x: x["topic_id"])
    return flat_topics


def build_schema_topic_words(schema_topics: Any) -> List[Dict[str, Any]]:
    if not isinstance(schema_topics, dict):
        return []

    groups = schema_topics.get("schema", [])
    if not isinstance(groups, list):
        return []

    invalid_schema_values = {"", "none", "null"}
    schema_word_groups = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        schema_raw = group.get("label", "")
        schema = str(schema_raw).strip()
        topics = group.get("topics", [])
        if schema_raw is None or schema.lower() in invalid_schema_values:
            schema = MISC_SCHEMA
        if not isinstance(topics, list):
            continue

        merged_words = []
        seen_words = set()
        for topic in topics:
            if not isinstance(topic, dict):
                continue
            words = topic.get("words", [])
            if not isinstance(words, list):
                continue
            for w in words:
                w = str(w).strip()
                if not w:
                    continue
                wl = w.lower()
                if wl in seen_words:
                    continue
                seen_words.add(wl)
                merged_words.append(w)

        if merged_words:
            schema_word_groups.append({"schema": schema, "words": merged_words})

    return schema_word_groups


def _run_step3_chunked_retry(
    model=None,
    tokenizer=None,
    client=None,
    model_name: Optional[str] = None,
    use_openai: bool = False,
    surviving_topics: Optional[List[Dict[str, Any]]] = None,
    step2_misc_topics: Optional[List[Dict[str, Any]]] = None,
    schema_text: str = "",
    max_new_tokens: int = 4096,
    json_retry_attempts: int = 0,
    device: str = "cuda",
    llm_call_count: Optional[List[int]] = None,
) -> tuple:
    """
    Truncation 발생 시 재시도: 토픽당 5단어, 청크별 분할 호출.
    Returns (step3_text, step3_json).
    """
    all_topics = sorted(surviving_topics + step2_misc_topics, key=lambda t: t.get("topic_id", 0))
    chunks = []
    for i in range(0, len(all_topics), STEP3_TRUNCATION_RETRY_CHUNK_SIZE):
        chunks.append(all_topics[i : i + STEP3_TRUNCATION_RETRY_CHUNK_SIZE])

    merged_json = []
    text_parts = []

    for chunk_idx, chunk in enumerate(chunks):
        chunk_topic_ids = [t.get("topic_id") for t in chunk if isinstance(t, dict) and t.get("topic_id") is not None]
        step3_messages = build_schema_aware_refine_prompt(
            surviving_topics=chunk,
            schema_text=schema_text,
            surviving_topics_n=len(chunk),
            step2_misc_topics=[],
            max_words_per_topic=STEP3_TRUNCATION_RETRY_MAX_WORDS,
            output_topic_ids=chunk_topic_ids,
        )
        chunk_text, chunk_json = call_llm_until_valid_json(
            model=model,
            tokenizer=tokenizer,
            client=client,
            model_name=model_name if use_openai else None,
            messages=step3_messages,
            max_new_tokens=max_new_tokens,
            step_name=f"Step 3 (chunk {chunk_idx + 1}/{len(chunks)})",
            json_retry_attempts=json_retry_attempts,
            device=device,
            llm_call_count=llm_call_count,
        )
        check_and_raise_if_truncated(
            chunk_text or "",
            step_name=f"Step 3 chunk {chunk_idx + 1}",
            expected_count=len(chunk),
            parsed_json=chunk_json,
        )
        if isinstance(chunk_json, list):
            merged_json.extend(chunk_json)
        text_parts.append(chunk_text or "")

    full_text = "\n\n".join(f"[Chunk {i+1}]\n{t}" for i, t in enumerate(text_parts))
    return full_text, merged_json


def run_llm_four_step_schema_pipeline(
    topic_words: List[List[str]],
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    max_new_tokens_step1: int = 4096,
    max_new_tokens_step2: int = 4096,
    max_new_tokens_step3: int = 4096,
    json_retry_attempts: int = 0,
    out_dir: str = "results",
    run_name: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:

    t0 = time.perf_counter()
    llm_call_count = [0]
    use_openai = _is_openai_model(model_name)
    use_plain_first = _use_plain_text_first_for_schema(model_name)
    if use_plain_first:
        print("[Schema] Using plain-text format first for Step 2/3 (Llama).")

    if use_openai:
        if OpenAI is None:
            raise ImportError("OpenAI API 사용 시 'pip install openai' 필요")
        print("Using OpenAI API model:", model_name)
        client = OpenAI()
        model, tokenizer = None, None
    else:
        print("Loading LLM (HuggingFace):", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device)
        model.eval()
        client = None
        print("LLM loaded.")

    step1_messages = build_schema_prompt(topic_words)
    step1_text = call_llm(
        model=model,
        tokenizer=tokenizer,
        client=client,
        model_name=model_name if use_openai else None,
        messages=step1_messages,
        max_new_tokens=max_new_tokens_step1,
        device=device,
        llm_call_count=llm_call_count,
    )
    step1_text = truncate_schema_step1_repetition(step1_text)
    step1_text = flatten_schema_text(step1_text)
    check_schema_step1_flat(step1_text)
    schema_labels_step1 = parse_schema_labels(step1_text)

    print("\n===== STEP 1: SCHEMA =====\n")
    print(step1_text)
    initial_topic_ids = list(range(len(topic_words)))
    print(f"[Topic Count] Initial topics: {len(initial_topic_ids)}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    step1_path = os.path.join(out_dir, "step1.txt")
    with open(step1_path, "w", encoding="utf-8") as f:
        f.write(step1_text)

    step2_path = os.path.join(out_dir, "step2.txt")
    if use_plain_first:
        step2_messages = build_topic_pruning_prompt_plain_text(topic_words, schema_text=step1_text)
        step2_text = call_llm(
            model=model,
            tokenizer=tokenizer,
            client=client,
            model_name=model_name if use_openai else None,
            messages=step2_messages,
            max_new_tokens=max_new_tokens_step2,
            device=device,
            llm_call_count=llm_call_count,
        )
        step2_json = parse_step2_plain_text(step2_text or "", len(topic_words))
        if step2_json is None:
            raise TruncationError(
                step_name="Step 2",
                message="plain-text parse failed",
                text_preview=(step2_text or "")[:500],
            )
    else:
        step2_messages = build_topic_pruning_prompt(topic_words, schema_text=step1_text)
        step2_text, step2_json = call_llm_until_valid_json(
            model=model,
            tokenizer=tokenizer,
            client=client,
            model_name=model_name if use_openai else None,
            messages=step2_messages,
            max_new_tokens=max_new_tokens_step2,
            step_name="Step 2",
            json_retry_attempts=json_retry_attempts,
            device=device,
            llm_call_count=llm_call_count,
        )
        try:
            check_and_raise_if_truncated(
                step2_text or "",
                step_name="Step 2",
                expected_count=len(topic_words),
                parsed_json=step2_json,
            )
        except TruncationError:
            print("[Step 2] Truncation/invalid JSON detected. Retrying with plain-text format...")
            step2_plain_messages = build_topic_pruning_prompt_plain_text(topic_words, schema_text=step1_text)
            step2_text = call_llm(
                model=model,
                tokenizer=tokenizer,
                client=client,
                model_name=model_name if use_openai else None,
                messages=step2_plain_messages,
                max_new_tokens=max_new_tokens_step2,
                device=device,
                llm_call_count=llm_call_count,
            )
            step2_json = parse_step2_plain_text(step2_text or "", len(topic_words))
            if step2_json is None:
                raise TruncationError(
                    step_name="Step 2",
                    message="plain-text fallback parse failed",
                    text_preview=(step2_text or "")[:500],
                )
            print("[Step 2] Plain-text fallback succeeded.")

    with open(step2_path, "w", encoding="utf-8") as f:
        f.write(step2_text or "")

    print("\n===== STEP 2: SCORE + PRUNE =====\n")
    print(step2_text)

    surviving_topics, step2_misc_topics = split_keep_and_misc(topic_words, step2_json)
    surviving_ids_step2 = sorted(t["topic_id"] for t in surviving_topics)
    deleted_ids_step2 = sorted(set(initial_topic_ids) - set(surviving_ids_step2))
    if deleted_ids_step2:
        print(f"[Step 2] Moved to misc {len(deleted_ids_step2)} topics:", deleted_ids_step2)
    else:
        print("[Step 2] Moved to misc 0 topics: []")
    print(f"[Step 2] Topics for step3: {len(surviving_ids_step2)}")

    step3_expected_count = len(surviving_topics) + len(step2_misc_topics)
    step3_text = None
    step3_json = None
    step3_used_retry = False
    all_topics_for_step3 = surviving_topics + step2_misc_topics

    if use_plain_first:
        step3_messages = build_schema_aware_refine_prompt_plain_text(
            surviving_topics=all_topics_for_step3,
            schema_text=step1_text,
            surviving_topics_n=len(all_topics_for_step3),
        )
        step3_text = call_llm(
            model=model,
            tokenizer=tokenizer,
            client=client,
            model_name=model_name if use_openai else None,
            messages=step3_messages,
            max_new_tokens=max_new_tokens_step3,
            device=device,
            llm_call_count=llm_call_count,
        )
        step3_json = parse_step3_plain_text(step3_text or "", len(all_topics_for_step3))
        if step3_json is None:
            raise TruncationError(
                step_name="Step 3",
                message="plain-text parse failed",
                text_preview=(step3_text or "")[:500],
            )
    else:
        try:
            step3_messages = build_schema_aware_refine_prompt(
                surviving_topics=surviving_topics,
                schema_text=step1_text,
                surviving_topics_n=len(surviving_topics),
                step2_misc_topics=step2_misc_topics,
            )
            step3_text, step3_json = call_llm_until_valid_json(
                model=model,
                tokenizer=tokenizer,
                client=client,
                model_name=model_name if use_openai else None,
                messages=step3_messages,
                max_new_tokens=max_new_tokens_step3,
                step_name="Step 3",
                json_retry_attempts=json_retry_attempts,
                device=device,
                llm_call_count=llm_call_count,
            )
            check_and_raise_if_truncated(
                step3_text or "",
                step_name="Step 3",
                expected_count=step3_expected_count,
                parsed_json=step3_json,
            )
        except TruncationError:
            step3_plain_messages = build_schema_aware_refine_prompt_plain_text(
                surviving_topics=all_topics_for_step3,
                schema_text=step1_text,
                surviving_topics_n=len(all_topics_for_step3),
            )
            print("\n[Step 3] Truncation detected. Retrying with plain-text format...")
            step3_text = call_llm(
                model=model,
                tokenizer=tokenizer,
                client=client,
                model_name=model_name if use_openai else None,
                messages=step3_plain_messages,
                max_new_tokens=max_new_tokens_step3,
                device=device,
                llm_call_count=llm_call_count,
            )
            step3_json = parse_step3_plain_text(step3_text or "", len(all_topics_for_step3))
            if step3_json is not None:
                step3_used_retry = True
                print("[Step 3] Plain-text fallback succeeded.")
            else:
                print("\n[Step 3] Plain-text fallback parse failed. Retrying with 5 words per topic in chunks...")
                step3_text, step3_json = _run_step3_chunked_retry(
                    model=model,
                    tokenizer=tokenizer,
                    client=client,
                    model_name=model_name if use_openai else None,
                    use_openai=use_openai,
                    surviving_topics=surviving_topics,
                    step2_misc_topics=step2_misc_topics,
                    schema_text=step1_text,
                    max_new_tokens=max_new_tokens_step3,
                    json_retry_attempts=json_retry_attempts,
                    device=device,
                    llm_call_count=llm_call_count,
                )
                step3_used_retry = True
                print("[Step 3] Retry succeeded (chunked + 5 words per topic).")

    step3_path = os.path.join(out_dir, "step3.txt")
    with open(step3_path, "w", encoding="utf-8") as f:
        f.write(step3_text or "")
    schema_topics = postprocess_final_topics(step3_json or {})
    # step3에 misc 포함됐으면 이미 반영. 누락된 misc가 있으면 merge로 보완
    schema_topics = merge_step2_misc_into_schema_topics(schema_topics, step2_misc_topics)
    refined_topics = flatten_schema_topics(schema_topics)
    # keep 모드: 50개 유지. LLM이 누락한 topic_id가 있으면 surviving/misc에서 복구
    all_sources = surviving_topics + step2_misc_topics
    refined_topics = _fill_missing_topics_for_keep_mode(
        refined_topics, all_sources, expected_count=len(topic_words)
    )
    refined_topics = _keep_mode_merge_misc_and_clear_rest(
        refined_topics, deleted_ids_step2, step2_misc_topics
    )
    # Topic word validation (hallucination prevention): LLM 출력 사후 검증
    for t in refined_topics:
        if isinstance(t, dict) and t.get("words"):
            t["words"] = filter_stopwords(filter_noise_words(t["words"]))
    remove_overlapping_words_across_topics(refined_topics, min_words_after=1)
    schema_topics = _schema_topics_from_refined_list(refined_topics)
    schema_topic_words = build_schema_topic_words(schema_topics)
    final_topic_ids = sorted(topic["topic_id"] for topic in refined_topics)
    deleted_ids_step3 = sorted(set(surviving_ids_step2) - set(final_topic_ids))

    print("\n===== STEP 3: SCHEMA-AWARE REFINE + ASSIGN =====\n")
    print(step3_text)
    if deleted_ids_step3:
        print(f"[Step 3] Deleted {len(deleted_ids_step3)} topics:", deleted_ids_step3)
    else:
        print("[Step 3] Deleted 0 topics: []")
    print(f"[Final] Remaining topics: {len(final_topic_ids)}")
    print(
        "[Summary] Initial:",
        len(initial_topic_ids),
        "| Step 2 deleted:",
        len(deleted_ids_step2),
        "| Step 3 deleted:",
        len(deleted_ids_step3),
        "| Final remaining:",
        len(final_topic_ids),
    )

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    step1_path = os.path.join(out_dir, "step1.txt")
    step2_path = os.path.join(out_dir, "step2.txt")
    step3_path = os.path.join(out_dir, "step3.txt")
    schema_topics_json_path = os.path.join(out_dir, "schema_topics.json")
    topic_words_path = os.path.join(out_dir, "topic_words.txt")
    schema_topic_words_path = os.path.join(out_dir, "schema_topic_words.txt")

    # step1에 Misc 반영: step2 delete -> Misc인 경우 스키마에 Misc 추가
    step1_to_save = step1_text
    if step2_misc_topics and MISC_SCHEMA.lower() not in step1_text.lower():
        step1_to_save = step1_text.rstrip() + f"\n- {MISC_SCHEMA}\n"
    with open(step1_path, "w", encoding="utf-8") as f:
        f.write(step1_to_save)

    with open(step2_path, "w", encoding="utf-8") as f:
        f.write(step2_text)

    with open(step3_path, "w", encoding="utf-8") as f:
        f.write(step3_text)

    with open(schema_topics_json_path, "w", encoding="utf-8") as f:
        json.dump(schema_topics, f, ensure_ascii=False, indent=2)

    with open(topic_words_path, "w", encoding="utf-8") as f:
        all_topics_for_txt = list(refined_topics) if isinstance(refined_topics, list) else []
        existing_ids = {t["topic_id"] for t in all_topics_for_txt}
        for m in step2_misc_topics:
            if isinstance(m, dict) and m.get("topic_id") not in existing_ids and m.get("words"):
                all_topics_for_txt.append({
                    "topic_id": m["topic_id"],
                    "words": filter_stopwords(filter_noise_words(m.get("words", []))),
                })
                existing_ids.add(m["topic_id"])
        all_topics_for_txt.sort(key=lambda x: x["topic_id"])
        total_k = len(topic_words)
        # Keep 모드: 0..K-1 순서로 정확히 K줄 출력 (빈 슬롯은 앵커 미지정으로 로더가 [] 인식)
        if total_k > 0 and len(all_topics_for_txt) == total_k:
            words_by_id = {t["topic_id"]: t.get("words", []) for t in all_topics_for_txt}
            for i in range(total_k):
                words = words_by_id.get(i, [])
                f.write(f"Topic {i}: {' '.join(words)}\n")
        else:
            for topic in all_topics_for_txt:
                words = topic.get("words", [])
                f.write(f"Topic {topic['topic_id']}: {' '.join(words)}\n")

    with open(schema_topic_words_path, "w", encoding="utf-8") as f:
        for item in schema_topic_words:
            f.write(f"{item['schema']}: {' '.join(item['words'])}\n")

    wall_clock_seconds = time.perf_counter() - t0
    schema_meta_path = os.path.join(out_dir, "schema_meta.json")
    with open(schema_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "wall_clock_seconds": round(wall_clock_seconds, 2),
                "llm_call_count": llm_call_count[0],
            },
            f,
            indent=2,
        )

    return {
        "step1_text": step1_text,
        "schema_labels": schema_labels_step1,
        "initial_topic_ids": initial_topic_ids,
        "step2_text": step2_text,
        "step2_json": step2_json,
        "surviving_topics": surviving_topics,
        "deleted_ids_step2": deleted_ids_step2,
        "step3_text": step3_text,
        "step3_json": step3_json,
        "schema_topics": schema_topics,
        "schema_topic_words": schema_topic_words,
        "refined_topics": refined_topics,
        "deleted_ids_step3": deleted_ids_step3,
        "final_topic_ids": final_topic_ids,
        "step1_path": step1_path,
        "step2_path": step2_path,
        "step3_path": step3_path,
        "schema_topics_json_path": schema_topics_json_path,
        "topic_words_path": topic_words_path,
        "schema_topic_words_path": schema_topic_words_path,
        "schema_meta_path": schema_meta_path,
    }


def run_refine_from_file(
    topic_words_file: str,
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    max_new_tokens_step1: int = 4096,
    max_new_tokens_step2: int = 4096,
    max_new_tokens_step3: int = 4096,
    json_retry_attempts: int = 0,
    out_dir: str = "results",
    run_name: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    topic_words = load_topic_words_from_file(topic_words_file)
    result = run_llm_four_step_schema_pipeline(
        topic_words=topic_words,
        model_name=model_name,
        max_new_tokens_step1=max_new_tokens_step1,
        max_new_tokens_step2=max_new_tokens_step2,
        max_new_tokens_step3=max_new_tokens_step3,
        json_retry_attempts=json_retry_attempts,
        out_dir=out_dir,
        run_name=run_name,
        device=device,
    )
    print("\nSaved:")
    print("step1 (schema):", result["step1_path"])
    print("step2 (scores):", result["step2_path"])
    print("step3 (refine):", result["step3_path"])
    print("schema-topic json:", result["schema_topics_json_path"])
    print("topic-word:", result["topic_words_path"])
    print("schema-topic-word:", result["schema_topic_words_path"])
    return result
