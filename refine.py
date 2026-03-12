import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


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


def format_topics(topic_words: List[List[str]]) -> str:
    return "\n".join(
        [f"Topic {i}: {', '.join(words)}" for i, words in enumerate(topic_words)]
    )


def extract_assistant_new_text(tokenizer, output_ids, model_inputs):
    input_len = model_inputs["input_ids"].shape[1]
    new_tokens = output_ids[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def try_parse_json(text: str):
    text = text.strip()
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
            candidate = text[start : end + 1].strip()
            return json.loads(candidate)
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
        if line.upper() == "SCHEMA:":
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


def filter_surviving_topics_by_verdict(
    topic_words: List[List[str]],
    scores_json: Any,
) -> List[Dict[str, Any]]:
    if not isinstance(scores_json, list):
        return [
            {"topic_id": i, "words": words}
            for i, words in enumerate(topic_words)
        ]

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
    for i, words in enumerate(topic_words):
        if i in deleted_ids:
            continue
        item = {"topic_id": i, "words": words}
        if i in kept_topic_names:
            item["topic_name"] = kept_topic_names[i]
        surviving.append(item)
    return surviving


def format_surviving_topics(surviving_topics: List[Dict[str, Any]]) -> str:
    lines = []
    for item in surviving_topics:
        suggested_name = str(item.get("topic_name", "")).strip()
        if suggested_name:
            lines.append(
                f"Topic {item['topic_id']} (suggested name: {suggested_name}): {', '.join(item['words'])}"
            )
        else:
            lines.append(f"Topic {item['topic_id']}: {', '.join(item['words'])}")
    return "\n".join(lines)


def build_schema_prompt(topic_words: List[List[str]]) -> List[Dict[str, str]]:
    topics_text = format_topics(topic_words)

    system = (
        "You are an expert in topic modeling. "
        "Think step by step internally, but do not reveal chain-of-thought. "
        "Use only the given topic words. "
        "Do not invent unsupported meanings."
    )

    user = f"""
Topics:
{topics_text}

Task:
First think of a partitioning criterion for grouping the topics.
Then create a compact schema list based on that criterion.
Return the criterion explanation together with the schema.

Rules:
- The criterion must be supported by the topic words.
- The schema labels must be broad and non-redundant.
- Keep the explanation short.
- No extra text.

Return plain text only in this exact format:

CRITERION:
- <short explanation>

SCHEMA:
- Label1
- Label2
- Label3
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
        "You are an expert in topic modeling. "
        "Use only the given schema and topic words. "
        "For each topic, assess interpretability and specificity, try to form a good short topic name, "
        "and then make one pruning decision. "
        "Do not invent unsupported meanings."
    )

    user = f"""
Schema:
{schema_text}

Topics:
{topics_text}

Task:
For each topic, follow this order:
1) assess interpretability: can it be clearly understood and fit the schema?
2) assess specificity: does it represent a specific semantic theme rather than generic, vague, or mixed words?
3) try to form a semantically natural topic name in one or two words
4) then make one decision: "keep" or "delete"

Output only:
- topic_id
- decision: "keep" or "delete"
- topic_name: one or two words if "keep", otherwise null
- reason: short justification reflecting interpretability, specificity, and naming quality

Decision rule:
- keep only if the topic is clearly interpretable, clearly specific, and can be given a good semantic topic_name
- delete if it is unclear, generic, mixed, noisy, or cannot be named well

Naming rule:
- topic_name must be natural, concise, semantic, and grounded in the topic words
- use one or two words only
- do not use vague or placeholder-like names such as "Thing", "Way", "Point", "Line", "Time", "Topic", "People Talk", "Like Thing", "Time Make"
- if no good topic_name is possible, choose "delete"

Rules:
- Do not decide "keep" first and then invent a weak name.
- The topic_name is a test of interpretability, not a post-hoc decoration.
- If you are not clearly confident, choose "delete".
- There are exactly {n_topics} topics. Output exactly {n_topics} JSON objects with topic_id 0 to {n_topics - 1}.
- No extra text before or after the JSON.

JSON format:
[
  {{ "topic_id": 0, "decision": "keep", "topic_name": "Computer Hardware", "reason": "clear, specific, and naturally named as a hardware topic" }},
  {{ "topic_id": 1, "decision": "delete", "topic_name": null, "reason": "too generic and cannot be given a good semantic name" }}
]
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_schema_aware_refine_prompt(
    surviving_topics: List[Dict[str, Any]],
    schema_text: str,
    surviving_topics_n: int,
) -> List[Dict[str, str]]:
    topics_text = format_surviving_topics(surviving_topics)

    system = (
        "You are an expert in topic modeling. "
        "Your role is limited to two actions: "
        "(1) eliminate generic, discourse-level, and metadata-like words from each topic that do not contribute clearly to the topic's semantic core, and"
        "(2) assign the topic to the most relevant schema label. "
        "If a topic remains too weak or semantically unclear after word elimination, you may mark it as delete instead of assigning a schema. "
        "Use the provided schema and surviving topics, but you may revise a better-fitting schema label when necessary. "
        "Do not invent unsupported meanings."
    )

    user = f"""
Schema:
{schema_text}

Surviving topics:
{topics_text}

Task:
For each topic, do only these two things in order:

1) Word elimination
- Remove words that are weak, generic, conversational, redundant, filler-like, or not clearly relevant to the topic's semantic core.
- Keep only informative and representative words.
- Retain words that help a human understand the topic's main meaning.
- Prefer original words only. Do not add new unsupported words.

2) Schema assignment
- Using the remaining words and the given topic_name, assign the topic to the single best schema label from the provided schema.
- If a topic is still too weak or unclear after word elimination, mark it as delete and do not assign schema.
- Use the given schema labels as the default, but permit only minimal deletion, addition, or modification when necessary to improve semantic accuracy.

Output:
Return only one JSON array.
Each object must contain:
- topic_id
- topic_name
- words
- schema

Rules:
- Exactly {surviving_topics_n} topic objects.
- Do not add, remove, split, or merge topics.
- Keep the given topic_name unless it is clearly inconsistent with the remaining words.
- Assign exactly one schema if kept; use null if deleted.
- No explanation before or after the JSON.

JSON format:
[
  {{
    "topic_id": 0,
    "topic_name": "Computer Hardware",
    "words": ["clipper", "irq", "isa", "centris", "ati"],
    "schema": "Technology"
  }},
  {{
    "topic_id": 1,
    "topic_name": "Time Make",
    "words": ["time", "make"],
    "schema": null
  }}
]
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_llm(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    device: str = "cuda",
) -> str:
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
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return extract_assistant_new_text(tokenizer, outputs, model_inputs)


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
            if (
                schema_raw is None
                or schema.lower() in invalid_schema_values
                or not isinstance(topics, list)
            ):
                continue

            cleaned_topics = []
            seen_topic_ids = set()
            for item in topics:
                if not isinstance(item, dict):
                    continue
                topic_id = item.get("topic_id", None)
                topic_name = str(item.get("topic_name", "")).strip()
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

                if len(cleaned_words) < 3:
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
            if (
                topic_id is None
                or not topic_name
                or not isinstance(words, list)
                or schema_raw is None
                or schema.lower() in invalid_schema_values
            ):
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

            if len(cleaned_words) < 3:
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
        if (
            schema_raw is None
            or schema.lower() in invalid_schema_values
            or not isinstance(topics, list)
        ):
            continue
        for topic in topics:
            if not isinstance(topic, dict):
                continue
            topic_id = topic.get("topic_id", None)
            topic_name = str(topic.get("topic_name", "")).strip()
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
        if (
            schema_raw is None
            or schema.lower() in invalid_schema_values
            or not isinstance(topics, list)
        ):
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


def run_llm_four_step_schema_pipeline(
    topic_words: List[List[str]],
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    max_new_tokens_step1: int = 1200,
    max_new_tokens_step2: int = 1500,
    max_new_tokens_step3: int = 4096,
    out_dir: str = "results",
    run_name: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:

    print("Loading LLM:", model_name)
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
    print("LLM loaded.")

    step1_messages = build_schema_prompt(topic_words)
    step1_text = call_llm(
        model=model,
        tokenizer=tokenizer,
        messages=step1_messages,
        max_new_tokens=max_new_tokens_step1,
        device=device,
    )
    schema_labels_step1 = parse_schema_labels(step1_text)

    print("\n===== STEP 1: SCHEMA =====\n")
    print(step1_text)
    initial_topic_ids = list(range(len(topic_words)))
    print(f"[Topic Count] Initial topics: {len(initial_topic_ids)}")

    step2_messages = build_topic_pruning_prompt(topic_words, schema_text=step1_text)
    step2_text = call_llm(
        model=model,
        tokenizer=tokenizer,
        messages=step2_messages,
        max_new_tokens=max_new_tokens_step2,
        device=device,
    )

    print("\n===== STEP 2: SCORE + PRUNE =====\n")
    print(step2_text)

    step2_json = try_parse_json(step2_text)
    surviving_topics = filter_surviving_topics_by_verdict(topic_words, step2_json)
    surviving_ids_step2 = sorted(t["topic_id"] for t in surviving_topics)
    deleted_ids_step2 = sorted(set(initial_topic_ids) - set(surviving_ids_step2))
    if deleted_ids_step2:
        print(f"[Step 2] Deleted {len(deleted_ids_step2)} topics:", deleted_ids_step2)
    else:
        print("[Step 2] Deleted 0 topics: []")
    print(f"[Step 2] Remaining topics: {len(surviving_ids_step2)}")

    step3_messages = build_schema_aware_refine_prompt(
        surviving_topics=surviving_topics,
        schema_text=step1_text,
        surviving_topics_n=len(surviving_topics),
    )
    step3_text = call_llm(
        model=model,
        tokenizer=tokenizer,
        messages=step3_messages,
        max_new_tokens=max_new_tokens_step3,
        device=device,
    )
    step3_json = try_parse_json(step3_text)
    schema_topics = postprocess_final_topics(step3_json or {})
    refined_topics = flatten_schema_topics(schema_topics)
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

    with open(step1_path, "w", encoding="utf-8") as f:
        f.write(step1_text)

    with open(step2_path, "w", encoding="utf-8") as f:
        f.write(step2_text)

    with open(step3_path, "w", encoding="utf-8") as f:
        f.write(step3_text)

    with open(schema_topics_json_path, "w", encoding="utf-8") as f:
        json.dump(schema_topics, f, ensure_ascii=False, indent=2)

    with open(topic_words_path, "w", encoding="utf-8") as f:
        if isinstance(refined_topics, list):
            for topic in refined_topics:
                words = topic.get("words", [])
                f.write(f"Topic {topic['topic_id']}: {' '.join(words)}\n")

    with open(schema_topic_words_path, "w", encoding="utf-8") as f:
        for item in schema_topic_words:
            f.write(f"{item['schema']}: {' '.join(item['words'])}\n")

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
    }


def run_refine_from_file(
    topic_words_file: str,
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    max_new_tokens_step1: int = 4096,
    max_new_tokens_step2: int = 4096,
    max_new_tokens_step3: int = 4096,
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
