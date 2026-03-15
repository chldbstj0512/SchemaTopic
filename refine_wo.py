"""
Schema refinement pipeline with optional step ablation (without step 1, 2, or 3).
Used for experiment4-style ablation: skip_step1 / skip_step2 / skip_step3.
Flow and prompts match refine.py (auto/delete mode); only which steps run is conditional.
"""
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from llm_validation import check_and_raise_if_truncated, check_schema_step1_flat, TruncationError
from tool import filter_stopwords

from refine import (
    _is_openai_model,
    OpenAI,
    MISC_SCHEMA,
    load_topic_words_from_file,
    format_topics,
    format_surviving_topics,
    parse_schema_labels,
    flatten_schema_text,
    filter_surviving_topics_by_verdict,
    filter_noise_words,
    remove_overlapping_words_across_topics,
    build_schema_prompt,
    build_topic_pruning_prompt,
    build_schema_aware_refine_prompt,
    postprocess_final_topics,
    flatten_schema_topics,
    build_schema_topic_words,
    call_llm,
    call_llm_until_valid_json,
)
from refine import _refined_list_to_schema

# Default schema when step 1 is skipped (no LLM schema induction)
DEFAULT_SCHEMA_TEXT = """CRITERION:
- General topics.

SCHEMA:
- {}
""".format(
    MISC_SCHEMA
).strip()


def run_llm_schema_pipeline_wo(
    topic_words: List[List[str]],
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    skip_step1: bool = False,
    skip_step2: bool = False,
    skip_step3: bool = False,
    max_new_tokens_step1: int = 4096,
    max_new_tokens_step2: int = 4096,
    max_new_tokens_step3: int = 4096,
    json_retry_attempts: int = 0,
    out_dir: str = "results",
    run_name: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run 3-step schema pipeline with optional step ablation.
    - skip_step1: do not call LLM for schema; use DEFAULT_SCHEMA_TEXT.
    - skip_step2: do not call LLM for prune; treat all topics as keep.
    - skip_step3: do not call LLM for refine; build schema from surviving_topics only.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    t0 = time.perf_counter()
    llm_call_count = [0]
    use_openai = _is_openai_model(model_name)

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

    initial_topic_ids = list(range(len(topic_words)))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ----- Step 1 -----
    if skip_step1:
        step1_text = DEFAULT_SCHEMA_TEXT
        schema_labels_step1 = [MISC_SCHEMA]
        print("\n===== STEP 1: SCHEMA (skipped, using default) =====\n")
    else:
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
        step1_text = flatten_schema_text(step1_text)
        check_schema_step1_flat(step1_text)
        schema_labels_step1 = parse_schema_labels(step1_text)
        print("\n===== STEP 1: SCHEMA =====\n")
    print(step1_text)
    print(f"[Topic Count] Initial topics: {len(initial_topic_ids)}")

    step1_path = os.path.join(out_dir, "step1.txt")
    with open(step1_path, "w", encoding="utf-8") as f:
        f.write(step1_text)

    # ----- Step 2 -----
    if skip_step2:
        n = len(topic_words)
        step2_json = [
            {"topic_id": i, "decision": "keep", "topic_name": f"Topic {i}"}
            for i in range(n)
        ]
        step2_text = json.dumps(step2_json)
        print("\n===== STEP 2: SCORE + PRUNE (skipped, all kept) =====\n")
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
        check_and_raise_if_truncated(
            step2_text or "",
            step_name="Step 2",
            expected_count=len(topic_words),
            parsed_json=step2_json,
        )
        print("\n===== STEP 2: SCORE + PRUNE =====\n")
    print(step2_text or "")

    surviving_topics = filter_surviving_topics_by_verdict(topic_words, step2_json)
    surviving_ids_step2 = sorted(t["topic_id"] for t in surviving_topics)
    deleted_ids_step2 = sorted(set(initial_topic_ids) - set(surviving_ids_step2))
    if deleted_ids_step2:
        print(f"[Step 2] Deleted {len(deleted_ids_step2)} topics:", deleted_ids_step2)
    else:
        print("[Step 2] Deleted 0 topics: []")
    print(f"[Step 2] Remaining topics: {len(surviving_ids_step2)}")

    step2_path = os.path.join(out_dir, "step2.txt")
    with open(step2_path, "w", encoding="utf-8") as f:
        f.write(step2_text if step2_text else "")

    # ----- Step 3 -----
    if skip_step3:
        # No LLM: build refined_topics from surviving_topics, all under MISC_SCHEMA
        refined_topics = []
        for t in surviving_topics:
            if not isinstance(t, dict):
                continue
            refined_topics.append({
                "topic_id": t.get("topic_id"),
                "topic_name": (t.get("topic_name") or MISC_SCHEMA).strip() or MISC_SCHEMA,
                "words": list(t.get("words", []))[:20],
                "schema": MISC_SCHEMA,
            })
        refined_topics.sort(key=lambda x: x.get("topic_id", 0))
        schema_topics = _refined_list_to_schema(refined_topics)
        step3_text = "(Step 3 skipped: no LLM refine; schema built from step 2 surviving topics only.)"
        step3_json = None
        print("\n===== STEP 3: SCHEMA-AWARE REFINE (skipped) =====\n")
    else:
        step3_messages = build_schema_aware_refine_prompt(
            surviving_topics=surviving_topics,
            schema_text=step1_text,
            surviving_topics_n=len(surviving_topics),
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
            expected_count=len(surviving_topics),
            parsed_json=step3_json,
        )
        schema_topics = postprocess_final_topics(step3_json or {})
        refined_topics = flatten_schema_topics(schema_topics)
        if len(refined_topics) < len(surviving_topics):
            existing_ids = {t["topic_id"] for t in refined_topics if isinstance(t, dict)}
            for src in surviving_topics:
                if not isinstance(src, dict):
                    continue
                tid = src.get("topic_id")
                if tid in existing_ids:
                    continue
                refined_topics.append({
                    "topic_id": tid,
                    "topic_name": src.get("topic_name") or MISC_SCHEMA,
                    "words": list(src.get("words", []))[:20],
                    "schema": MISC_SCHEMA,
                })
            refined_topics.sort(key=lambda x: x.get("topic_id", 0))
            schema_topics = _refined_list_to_schema(refined_topics)
        print("\n===== STEP 3: SCHEMA-AWARE REFINE + ASSIGN =====\n")

    # Topic word validation
    for t in refined_topics:
        if isinstance(t, dict) and t.get("words"):
            t["words"] = filter_stopwords(filter_noise_words(t["words"]))
    remove_overlapping_words_across_topics(refined_topics, min_words_after=1)
    for g in schema_topics.get("schema", []):
        if isinstance(g, dict):
            for t in g.get("topics", []):
                if isinstance(t, dict) and t.get("words"):
                    t["words"] = filter_stopwords(filter_noise_words(t["words"]))

    schema_topic_words = build_schema_topic_words(schema_topics)
    final_topic_ids = sorted(t["topic_id"] for t in refined_topics)
    deleted_ids_step3 = sorted(set(surviving_ids_step2) - set(final_topic_ids))

    print(step3_text if step3_text else "")
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

    step3_path = os.path.join(out_dir, "step3.txt")
    schema_topics_json_path = os.path.join(out_dir, "schema_topics.json")
    topic_words_path = os.path.join(out_dir, "topic_words.txt")
    schema_topic_words_path = os.path.join(out_dir, "schema_topic_words.txt")

    with open(step3_path, "w", encoding="utf-8") as f:
        f.write(step3_text if step3_text else "")

    with open(schema_topics_json_path, "w", encoding="utf-8") as f:
        json.dump(schema_topics, f, ensure_ascii=False, indent=2)

    with open(topic_words_path, "w", encoding="utf-8") as f:
        for topic in refined_topics:
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
                "skip_step1": skip_step1,
                "skip_step2": skip_step2,
                "skip_step3": skip_step3,
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


def run_refine_from_file_wo(
    topic_words_file: str,
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    skip_step1: bool = False,
    skip_step2: bool = False,
    skip_step3: bool = False,
    max_new_tokens_step1: int = 4096,
    max_new_tokens_step2: int = 4096,
    max_new_tokens_step3: int = 4096,
    json_retry_attempts: int = 0,
    out_dir: str = "results",
    run_name: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Entry point: load topic words from file and run pipeline with step ablation."""
    topic_words = load_topic_words_from_file(topic_words_file)
    result = run_llm_schema_pipeline_wo(
        topic_words=topic_words,
        model_name=model_name,
        skip_step1=skip_step1,
        skip_step2=skip_step2,
        skip_step3=skip_step3,
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
