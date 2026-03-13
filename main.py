import argparse
import os
import shutil
import tempfile
from argparse import Namespace
from pathlib import Path

from dataset import infer_dataset_name, list_available_datasets
from topic_models import list_supported_topic_models
from refine import run_refine_from_file
from refine_k import run_refine_from_file as run_refine_from_file_keep
from train import run_train, run_eval_from_checkpoint


def add_common_training_arguments(parser):
    supported_models = ", ".join(list_supported_topic_models())
    available_datasets = list_available_datasets()
    dataset_help = "dataset name under datasets/"
    if available_datasets:
        dataset_help = "{} (available: {})".format(
            dataset_help,
            ", ".join(available_datasets),
        )
    parser.add_argument("--dataset", type=str, default="20News", help=dataset_help)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="explicit dataset directory path; overrides --dataset when set",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="etm",
        help="topic model backend (supported: {})".format(supported_models),
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--num_topics", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wdecay", type=float, default=1.2e-6)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--theta_act", type=str, default="relu")
    parser.add_argument("--t_hidden_size", type=int, default=500)
    parser.add_argument("--enc_drop", type=float, default=0.0)
    parser.add_argument("--bow_norm", type=int, default=1)
    parser.add_argument("--topk_words", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1)
    return parser


def add_vanilla_arguments(parser):
    add_common_training_arguments(parser)
    parser.add_argument("--out_dir", type=str, default=None)
    return parser


def add_schema_arguments(parser):
    parser.add_argument("--topic_words_file", type=str, required=True)
    parser.add_argument(
        "--keep",
        action="store_true",
        default=False,
        help="use k-keep mode: retain all topics (no delete) in LLM refinement",
    )
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--max_new_tokens_step1", type=int, default=4096)
    parser.add_argument("--max_new_tokens_step2", type=int, default=4096)
    parser.add_argument("--max_new_tokens_step3", type=int, default=4096)
    parser.add_argument(
        "--json_retry_attempts",
        type=int,
        default=2,
        help="retry malformed JSON responses for step2/step3 this many times",
    )
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def add_anchor_source_arguments(parser):
    parser.add_argument(
        "--schema_dir",
        type=str,
        default=None,
        help="schema output directory containing topic_words.txt or schema_topics.json",
    )
    parser.add_argument(
        "--anchor_words_file",
        type=str,
        default=None,
        help="topic-wise anchor words file; overrides --schema_dir auto-detection",
    )
    parser.add_argument(
        "--anchor_topics_json",
        type=str,
        default=None,
        help="JSON anchor topics file; used when --anchor_words_file is not provided",
    )
    return parser


def add_anchor_arguments(parser):
    add_common_training_arguments(parser)
    add_anchor_source_arguments(parser)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--lambda_anchor", type=float, default=1.0)
    return parser


def add_pipeline_arguments(parser):
    add_common_training_arguments(parser)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--lambda_anchor", type=float, default=1.0)

    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--max_new_tokens_step1", type=int, default=4096)
    parser.add_argument("--max_new_tokens_step2", type=int, default=4096)
    parser.add_argument("--max_new_tokens_step3", type=int, default=4096)
    parser.add_argument(
        "--json_retry_attempts",
        type=int,
        default=2,
        help="retry malformed JSON responses for step2/step3 this many times",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--keep",
        action="store_true",
        default=False,
        help="use k-keep mode: retain all topics (no delete) in LLM refinement",
    )
    return parser


def ensure_results_root():
    results_root = Path("results")
    results_root.mkdir(parents=True, exist_ok=True)
    return results_root


def resolve_dataset_settings(args):
    if getattr(args, "data_dir", None):
        data_dir = args.data_dir
    else:
        data_dir = str(Path("datasets") / args.dataset)
    dataset_name = infer_dataset_name(data_dir)
    return data_dir, dataset_name


def default_vanilla_dir(dataset_name, model, num_topics):
    return ensure_results_root() / "vanilla_{}_{}_{}".format(dataset_name, model, num_topics)


def default_pipeline_dir(dataset_name, model, num_topics, keep=False):
    base = "pipeline_{}_{}_{}".format(dataset_name, model, num_topics)
    if keep:
        base += "_keep"
    return ensure_results_root() / base


def default_anchor_dir(dataset_name, model, num_topics):
    return ensure_results_root() / "anchor_{}_{}_{}".format(dataset_name, model, num_topics)


def default_step2_dir(final_topic_count, parent_dir=None):
    if parent_dir is None:
        parent = ensure_results_root()
    else:
        parent = Path(parent_dir)
        parent.mkdir(parents=True, exist_ok=True)
    return parent / "schema_{}".format(final_topic_count)


def finalize_refine_result(result, final_dir):
    final_dir = Path(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(result["step1_path"]).parent
    updated_result = dict(result)

    for key, value in result.items():
        if not key.endswith("_path"):
            continue
        src = Path(value)
        dst = final_dir / src.name
        if src.exists():
            os.replace(str(src), str(dst))
        updated_result[key] = str(dst)

    try:
        temp_dir.rmdir()
    except OSError:
        shutil.rmtree(str(temp_dir), ignore_errors=True)

    return updated_result


def run_schema(args, output_parent_dir=None, folder_name=None):
    run_refine = run_refine_from_file_keep if getattr(args, "keep", False) else run_refine_from_file
    if args.out_dir is not None:
        return run_refine(
            topic_words_file=args.topic_words_file,
            model_name=args.model_name,
            max_new_tokens_step1=args.max_new_tokens_step1,
            max_new_tokens_step2=args.max_new_tokens_step2,
            max_new_tokens_step3=args.max_new_tokens_step3,
            json_retry_attempts=args.json_retry_attempts,
            out_dir=args.out_dir,
            run_name=args.run_name,
            device=args.device,
        )

    if output_parent_dir is None:
        output_parent_dir = ensure_results_root()
    else:
        output_parent_dir = Path(output_parent_dir)
        output_parent_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=".step2_tmp_",
            dir=str(output_parent_dir),
        )
    )
    result = run_refine(
        topic_words_file=args.topic_words_file,
        model_name=args.model_name,
        max_new_tokens_step1=args.max_new_tokens_step1,
        max_new_tokens_step2=args.max_new_tokens_step2,
        max_new_tokens_step3=args.max_new_tokens_step3,
        json_retry_attempts=args.json_retry_attempts,
        out_dir=str(temp_dir),
        run_name=args.run_name,
        device=args.device,
    )

    final_topic_count = len(result.get("final_topic_ids", []))
    if folder_name is None:
        final_dir = default_step2_dir(final_topic_count, parent_dir=output_parent_dir)
    else:
        final_dir = Path(output_parent_dir) / folder_name.format(
            final_topic_count=final_topic_count
        )

    result = finalize_refine_result(result, final_dir)
    print("Final schema output dir:", final_dir)
    return result


def resolve_anchor_inputs(args):
    anchor_words_file = getattr(args, "anchor_words_file", None)
    anchor_topics_json = getattr(args, "anchor_topics_json", None)
    schema_dir = getattr(args, "schema_dir", None)

    if schema_dir is not None:
        schema_dir = Path(schema_dir)
        if anchor_words_file is None:
            topic_words_path = schema_dir / "topic_words.txt"
            if topic_words_path.exists():
                anchor_words_file = str(topic_words_path)
        if anchor_topics_json is None:
            schema_topics_path = schema_dir / "schema_topics.json"
            if schema_topics_path.exists():
                anchor_topics_json = str(schema_topics_path)

    if anchor_words_file is None and anchor_topics_json is None:
        raise ValueError(
            "Anchor mode requires one of --schema_dir, --anchor_words_file, or --anchor_topics_json."
        )

    return anchor_words_file, anchor_topics_json


def build_train_namespace(
    args,
    *,
    out_dir,
    anchor_words_file=None,
    anchor_topics_json=None,
    lambda_anchor=0.0,
):
    return Namespace(
        dataset=args.dataset_name,
        data_dir=args.data_dir,
        model=args.model,
        epochs=args.epochs,
        num_topics=args.num_topics,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        wdecay=args.wdecay,
        optimizer=args.optimizer,
        clip=args.clip,
        log_interval=args.log_interval,
        theta_act=args.theta_act,
        t_hidden_size=args.t_hidden_size,
        enc_drop=args.enc_drop,
        bow_norm=args.bow_norm,
        topk_words=args.topk_words,
        out_dir=str(out_dir),
        seed=args.seed,
        output_suffix=None,
        anchor_words_file=anchor_words_file,
        anchor_topics_json=anchor_topics_json,
        lambda_anchor=lambda_anchor,
    )


def run_anchor(args):
    anchor_words_file, anchor_topics_json = resolve_anchor_inputs(args)
    out_dir = args.out_dir or str(default_anchor_dir(args.dataset_name, args.model, args.num_topics))
    anchor_args = build_train_namespace(
        args,
        out_dir=out_dir,
        anchor_words_file=anchor_words_file,
        anchor_topics_json=anchor_topics_json,
        lambda_anchor=args.lambda_anchor,
    )
    return run_train(anchor_args)


def build_parser():
    parser = argparse.ArgumentParser(description="SchemaTopic main entrypoint.")
    subparsers = parser.add_subparsers(dest="command")

    vanilla_parser = subparsers.add_parser(
        "vanilla",
        aliases=["train"],
        help="Run vanilla neural topic model training.",
    )
    add_vanilla_arguments(vanilla_parser)

    schema_parser = subparsers.add_parser(
        "schema",
        aliases=["step2", "refine"],
        help="Run LLM schema induction/refinement on a specified topic-words file.",
    )
    add_schema_arguments(schema_parser)

    anchor_parser = subparsers.add_parser(
        "anchor",
        help="Run anchor-guided topic model training from schema outputs.",
    )
    add_anchor_arguments(anchor_parser)

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run vanilla -> schema -> anchor end-to-end.",
    )
    add_pipeline_arguments(pipeline_parser)

    eval_parser = subparsers.add_parser(
        "eval",
        help="Load a saved checkpoint and run evaluation (TC, TD, Purity, NMI).",
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model.pt or directory containing model.pt (e.g., results/pipeline_X/anchor)",
    )
    eval_parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Dataset path (default: from checkpoint training_args)",
    )

    return parser


def run_pipeline(args):
    pipeline_root = (
        Path(args.out_dir)
        if args.out_dir
        else default_pipeline_dir(
            args.dataset_name,
            args.model,
            args.num_topics,
            keep=getattr(args, "keep", False),
        )
    )
    pipeline_root.mkdir(parents=True, exist_ok=True)

    print("\n=== Stage 1/3: Train vanilla topic model ===")
    stage1_args = build_train_namespace(
        args,
        out_dir=pipeline_root / "vanilla",
    )
    stage1_result = run_train(stage1_args)

    print("\n=== Stage 2/3: Build schema with LLM ===")
    schema_args = Namespace(
        topic_words_file=stage1_result["top_words_path"],
        model_name=args.model_name,
        max_new_tokens_step1=args.max_new_tokens_step1,
        max_new_tokens_step2=args.max_new_tokens_step2,
        max_new_tokens_step3=args.max_new_tokens_step3,
        json_retry_attempts=args.json_retry_attempts,
        out_dir=None,
        run_name=args.run_name,
        device=args.device,
        keep=getattr(args, "keep", False),
    )
    schema_result = run_schema(
        schema_args,
        output_parent_dir=pipeline_root,
        folder_name="schema_{final_topic_count}",
    )

    print("\n=== Stage 3/3: Train anchor-guided topic model ===")
    anchor_arg_values = dict(vars(args))
    anchor_arg_values.update(
        {
            "out_dir": str(pipeline_root / "anchor"),
            "schema_dir": None,
            "anchor_words_file": schema_result["topic_words_path"],
            "anchor_topics_json": schema_result["schema_topics_json_path"],
        }
    )
    anchor_args = Namespace(**anchor_arg_values)
    anchor_result = run_anchor(anchor_args)

    print("\n=== Pipeline complete ===")
    print("Stage 1 top words:", stage1_result["top_words_path"])
    print("Schema topic words:", schema_result["topic_words_path"])
    print("Anchor metrics:", anchor_result["metrics_path"])
    return {
        "stage1": stage1_result,
        "schema": schema_result,
        "anchor": anchor_result,
    }


def main():
    parser = build_parser()
    args = parser.parse_args()

    if getattr(args, "command", None) is None:
        parser.print_help()
        raise SystemExit(1)

    if args.command in ("vanilla", "train", "anchor", "pipeline"):
        args.data_dir, args.dataset_name = resolve_dataset_settings(args)

    if args.command in ("vanilla", "train"):
        vanilla_args = build_train_namespace(
            args,
            out_dir=args.out_dir or default_vanilla_dir(args.dataset_name, args.model, args.num_topics),
        )
        run_train(vanilla_args)
        return

    if args.command in ("schema", "step2", "refine"):
        run_schema(args)
        return

    if args.command == "anchor":
        run_anchor(args)
        return

    if args.command == "pipeline":
        run_pipeline(args)
        return

    if args.command == "eval":
        run_eval_from_checkpoint(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
        )
        return

    parser.print_help()
    raise SystemExit(1)


if __name__ == "__main__":
    main()
