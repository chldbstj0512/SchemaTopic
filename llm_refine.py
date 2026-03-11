import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

DEFAULT_REFINED_WORDS_N = 10


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


def count_deleted_topics(delete_text: str) -> int:
    deleted = set()
    for line in delete_text.splitlines():
        line = line.strip()
        if line.startswith("- Topic "):
            try:
                idx = int(line.split("- Topic ", 1)[1].split()[0])
                deleted.add(idx)
            except Exception:
                pass
    return len(deleted)


def postprocess_final_topics(final_topics, refined_words_n: int):
    if not isinstance(final_topics, list):
        return final_topics

    cleaned = []

    for item in final_topics:
        if not isinstance(item, dict):
            continue

        topic_id = item.get("topic_id", None)
        topic_name = str(item.get("topic_name", "")).strip()
        words = item.get("words", [])
        changes = str(item.get("changes", "")).strip()

        if topic_id is None:
            continue
        if not topic_name:
            continue
        if not isinstance(words, list):
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

        cleaned_words = cleaned_words[:refined_words_n]
        if len(cleaned_words) < 3:
            continue

        cleaned.append(
            {
                "topic_id": topic_id,
                "topic_name": topic_name,
                "words": cleaned_words,
                "changes": changes,
            }
        )

    cleaned.sort(key=lambda x: x["topic_id"])
    return cleaned


def postprocess_schema_json(schema_json, schema_labels: List[str], refined_words_n: int):
    if not isinstance(schema_json, list):
        return schema_json

    valid_schema = set([x.lower() for x in schema_labels])
    cleaned = []

    for item in schema_json:
        if not isinstance(item, dict):
            continue

        schema = str(item.get("schema", "")).strip()
        topic_name = str(item.get("topic_name", "")).strip()
        words = item.get("words", [])

        if not schema or not topic_name or not isinstance(words, list):
            continue
        if schema.lower() not in valid_schema:
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

        cleaned_words = cleaned_words[:refined_words_n]
        if len(cleaned_words) < 3:
            continue

        cleaned.append(
            {
                "schema": schema,
                "topic_name": topic_name,
                "words": cleaned_words,
            }
        )

    return cleaned


def parse_deleted_topic_ids(delete_text: str) -> set:
    deleted = set()

    for line in delete_text.splitlines():
        line = line.strip()
        if not line.startswith("- Topic "):
            continue
        try:
            idx = int(line.split("- Topic ", 1)[1].split()[0])
            deleted.add(idx)
        except Exception:
            pass

    return deleted


def filter_surviving_topics(topic_words: List[List[str]], delete_text: str) -> List[Dict[str, Any]]:
    deleted_ids = parse_deleted_topic_ids(delete_text)

    surviving = []
    for i, words in enumerate(topic_words):
        if i in deleted_ids:
            continue
        surviving.append({
            "topic_id": i,
            "words": words,
        })
    return surviving


def format_surviving_topics(surviving_topics: List[Dict[str, Any]]) -> str:
    lines = []
    for item in surviving_topics:
        lines.append(f"Topic {item['topic_id']}: {', '.join(item['words'])}")
    return "\n".join(lines)
# ---------------- STEP 1 ----------------
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
- Return 1 to 12 schema labels.
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

# ---------------- STEP 2 ----------------
def build_delete_prompt(
    topic_words: List[List[str]],
) -> List[Dict[str, str]]:
    topics_text = format_topics(topic_words)

    system = (
        "You are an expert in topic modeling. "
        "Use only the given topic words and previous result. "
        "Do not invent unsupported meanings."
    )

    user = f"""
Topics:
{topics_text}

Task:
Delete unnecessary topics.

Delete a topic only if:
- it is mostly junk,
- it is too generic,
- it has no clear semantic meaning,
- it is a weak near-duplicate of another topic.

Return the delete result and short explanation.

Rules:
- Be conservative.
- Keep interpretable topics.
- No extra text.

Return plain text only in this exact format:

DELETE:
- Topic k
- Topic m
or
- None

EXPLANATION:
- <short explanation>
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

# ---------------- STEP 3 ----------------
def build_refine_and_assign_prompt(
    surviving_topics: List[Dict[str, Any]],
    surviving_topics_n: int,
    schema_text: str,
    refined_words_n: int,
) -> List[Dict[str, str]]:
    topics_text = format_surviving_topics(surviving_topics)

    system = (
        "You are an expert in topic modeling. "
        "Use only the given topic words and schema labels as evidence. "
        "Do not invent unsupported meanings."
    )

    user = f"""
Schema:
{schema_text}

Topics:
{topics_text}

Task:
For each topic:
1. assign a short semantic topic name (1-2 words),
2. remove unrelated or weak words,
3. keep exactly one refined topic per input topic,
4. assign the topic to the most appropriate schema,
5. return the refinement result and short change note.

Rules:
- The number of topics is exactly {surviving_topics_n}; do not add or remove any topics.
- Do not merge topics.
- Prefer original words.
- Do not add unsupported new words.
- Use at most {refined_words_n} words per topic.
- For schema assignment, you may use an existing schema label from the Schema section, or create a new schema label if none fits well.
- Unused schema labels should be omitted naturally.
- Each topic must have exactly one schema label.
- Output one JSON object for every input topic.
- Do not write any explanation before or after the JSON.

JSON format:
[
  {{
    "topic_id": 0,
    "topic_name": "short topic name",
    "words": ["w1", "w2", "w3", "w4", "w5"],
    "schema": "Schema Label",
    "changes": "short explanation"
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

    outputs = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return extract_assistant_new_text(tokenizer, outputs, model_inputs)

def parse_deleted_topic_ids(delete_text: str) -> set:
    deleted = set()
    for line in delete_text.splitlines():
        line = line.strip()
        if not line.startswith("- Topic "):
            continue
        try:
            idx = int(line.split("- Topic ", 1)[1].split()[0])
            deleted.add(idx)
        except Exception:
            pass
    return deleted

def filter_surviving_topics(topic_words: List[List[str]], delete_text: str) -> List[Dict[str, Any]]:
    deleted_ids = parse_deleted_topic_ids(delete_text)
    surviving = []
    for i, words in enumerate(topic_words):
        if i in deleted_ids:
            continue
        surviving.append({"topic_id": i, "words": words})
    return surviving

def format_surviving_topics(surviving_topics: List[Dict[str, Any]]) -> str:
    lines = []
    for item in surviving_topics:
        lines.append(f"Topic {item['topic_id']}: {', '.join(item['words'])}")
    return "\n".join(lines)

def postprocess_final_topics(final_topics, refined_words_n: int):
    if not isinstance(final_topics, list):
        return final_topics

    cleaned = []

    for item in final_topics:
        if not isinstance(item, dict):
            continue

        topic_id = item.get("topic_id", None)
        topic_name = str(item.get("topic_name", "")).strip()
        words = item.get("words", [])
        schema = str(item.get("schema", "")).strip()
        changes = str(item.get("changes", "")).strip()

        if topic_id is None or not topic_name or not isinstance(words, list) or not schema:
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

        cleaned_words = cleaned_words[:refined_words_n]
        if len(cleaned_words) < 3:
            continue

        cleaned.append(
            {
                "topic_id": topic_id,
                "topic_name": topic_name,
                "words": cleaned_words,
                "schema": schema,
                "changes": changes,
            }
        )

    cleaned.sort(key=lambda x: x["topic_id"])
    return cleaned

def postprocess_schema_json(schema_json, schema_labels: List[str], refined_words_n: int):
    if not isinstance(schema_json, list):
        return schema_json

    valid_schema = {x.lower() for x in schema_labels}
    cleaned = []

    for item in schema_json:
        if not isinstance(item, dict):
            continue

        schema = str(item.get("schema", "")).strip()
        topics = item.get("topics", [])

        if not schema or schema.lower() not in valid_schema or not isinstance(topics, list):
            continue

        cleaned_topics = []
        seen_topic_names = set()

        for topic in topics:
            if not isinstance(topic, dict):
                continue

            topic_name = str(topic.get("topic_name", "")).strip()
            words = topic.get("words", [])

            if not topic_name or not isinstance(words, list):
                continue
            if topic_name.lower() in seen_topic_names:
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

            cleaned_words = cleaned_words[:refined_words_n]
            if len(cleaned_words) < 3:
                continue

            cleaned_topics.append(
                {
                    "topic_name": topic_name,
                    "words": cleaned_words,
                }
            )
            seen_topic_names.add(topic_name.lower())

        if cleaned_topics:
            cleaned.append(
                {
                    "schema": schema,
                    "topics": cleaned_topics,
                }
            )

    return cleaned

def run_llm_four_step_schema_pipeline(
    topic_words: List[List[str]],
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    refined_words_n: int = DEFAULT_REFINED_WORDS_N,
    max_new_tokens_step1: int = 1200,
    max_new_tokens_step2: int = 1500,
    max_new_tokens_step3: int = 4096,
    out_dir: str = "results/llm_refine",
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

    # STEP 1
    step1_messages = build_schema_prompt(topic_words)
    step1_text = call_llm(
        model=model,
        tokenizer=tokenizer,
        messages=step1_messages,
        max_new_tokens=max_new_tokens_step1,
        device=device,
    )
    schema_labels = parse_schema_labels(step1_text)

    print("\n===== STEP 1: SCHEMA =====\n")
    print(step1_text)

    # STEP 2
    step2_messages = build_delete_prompt(topic_words)
    step2_text = call_llm(
        model=model,
        tokenizer=tokenizer,
        messages=step2_messages,
        max_new_tokens=max_new_tokens_step2,
        device=device,
    )

    print("\n===== STEP 2: DELETE =====\n")
    print(step2_text)

    # STEP 3
    surviving_topics = filter_surviving_topics(topic_words, step2_text)

    step3_messages = build_refine_and_assign_prompt(
        surviving_topics=surviving_topics,
        surviving_topics_n = len(surviving_topics),
        schema_text=step1_text,
        refined_words_n=refined_words_n,
    )

    step3_text = call_llm(
        model=model,
        tokenizer=tokenizer,
        messages=step3_messages,
        max_new_tokens=max_new_tokens_step3,
        device=device,
    )
    step3_json = try_parse_json(step3_text)
    step3_json = postprocess_final_topics(step3_json, refined_words_n)

    print("\n===== STEP 3: REFINE + ASSIGN =====\n")
    print(step3_text)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    suffix = run_name or "three_step_schema"

    step1_path = os.path.join(out_dir, f"llm_{suffix}_step1_schema.txt")
    step2_path = os.path.join(out_dir, f"llm_{suffix}_step2_delete.txt")
    step3_raw_path = os.path.join(out_dir, f"llm_{suffix}_step3_refine_assign_raw.txt")
    step3_json_path = os.path.join(out_dir, f"llm_{suffix}_step3_refine_assign.json")
    words_path = os.path.join(out_dir, f"llm_{suffix}_words.txt")

    with open(step1_path, "w", encoding="utf-8") as f:
        f.write(step1_text)

    with open(step2_path, "w", encoding="utf-8") as f:
        f.write(step2_text)

    with open(step3_raw_path, "w", encoding="utf-8") as f:
        f.write(step3_text)

    with open(step3_json_path, "w", encoding="utf-8") as f:
        json.dump(step3_json, f, ensure_ascii=False, indent=2)

    with open(words_path, "w", encoding="utf-8") as f:
        if isinstance(step3_json, list):
            for topic in step3_json:
                words = topic.get("words", [])
                f.write(" ".join(words) + "\n")

    return {
        "step1_text": step1_text,
        "schema_labels": schema_labels,
        "step2_text": step2_text,
        "surviving_topics": surviving_topics,
        "step3_text": step3_text,
        "step3_json": step3_json,
        "step1_path": step1_path,
        "step2_path": step2_path,
        "step3_raw_path": step3_raw_path,
        "step3_json_path": step3_json_path,
        "words_path": words_path,
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple 4-step LLM topic schema pipeline"
    )
    parser.add_argument("--topic_words_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--refined_words_n", type=int, default=DEFAULT_REFINED_WORDS_N)
    parser.add_argument("--max_new_tokens_step1", type=int, default=1200)
    parser.add_argument("--max_new_tokens_step2", type=int, default=1500)
    parser.add_argument("--max_new_tokens_step3", type=int, default=4096)
    parser.add_argument("--out_dir", type=str, default="results/llm_refine")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    topic_words = load_topic_words_from_file(args.topic_words_file)

    result = run_llm_four_step_schema_pipeline(
        topic_words=topic_words,
        model_name=args.model_name,
        refined_words_n=args.refined_words_n,
        max_new_tokens_step1=args.max_new_tokens_step1,
        max_new_tokens_step2=args.max_new_tokens_step2,
        max_new_tokens_step3=args.max_new_tokens_step3,
        out_dir=args.out_dir,
        run_name=args.run_name,
        device=args.device,
    )

    print("\nSaved:")
    print("step1:", result["step1_path"])
    print("step2:", result["step2_path"])
    print("step3 raw:", result["step3_raw_path"])
    print("step3 json:", result["step3_json_path"])
    print("words:", result["words_path"])