import argparse
import os
import shutil
import tempfile
from argparse import Namespace
from pathlib import Path

from topic_models import list_supported_topic_models
from refine import run_refine_from_file
from train import run_train


def add_common_training_arguments(parser):
    supported_models = ", ".join(list_supported_topic_models())
    parser.add_argument("--data_dir", type=str, default="datasets/20News")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    return parser


def add_vanilla_arguments(parser):
    add_common_training_arguments(parser)
    parser.add_argument("--out_dir", type=str, default=None)
    return parser


def add_step2_arguments(parser):
    parser.add_argument("--topic_words_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--max_new_tokens_step1", type=int, default=4096)
    parser.add_argument("--max_new_tokens_step2", type=int, default=4096)
    parser.add_argument("--max_new_tokens_step3", type=int, default=4096)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def add_pipeline_arguments(parser):
    add_common_training_arguments(parser)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--lambda_anchor", type=float, default=1.0)

    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--max_new_tokens_step1", type=int, default=4096)
    parser.add_argument("--max_new_tokens_step2", type=int, default=4096)
    parser.add_argument("--max_new_tokens_step3", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def ensure_results_root():
    results_root = Path("results")
    results_root.mkdir(parents=True, exist_ok=True)
    return results_root


def default_vanilla_dir(model, num_topics):
    return ensure_results_root() / "vanilla_{}_{}".format(model, num_topics)


def default_pipeline_dir(model, num_topics):
    return ensure_results_root() / "pipeline_{}_{}".format(model, num_topics)


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


def run_step2(args, output_parent_dir=None, folder_name=None):
    if args.out_dir is not None:
        return run_refine_from_file(
            topic_words_file=args.topic_words_file,
            model_name=args.model_name,
            max_new_tokens_step1=args.max_new_tokens_step1,
            max_new_tokens_step2=args.max_new_tokens_step2,
            max_new_tokens_step3=args.max_new_tokens_step3,
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
    result = run_refine_from_file(
        topic_words_file=args.topic_words_file,
        model_name=args.model_name,
        max_new_tokens_step1=args.max_new_tokens_step1,
        max_new_tokens_step2=args.max_new_tokens_step2,
        max_new_tokens_step3=args.max_new_tokens_step3,
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
    print("Final step2 output dir:", final_dir)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="SchemaTopic main entrypoint.")
    subparsers = parser.add_subparsers(dest="command")

    vanilla_parser = subparsers.add_parser(
        "vanilla",
        aliases=["train"],
        help="Run vanilla neural topic model training.",
    )
    add_vanilla_arguments(vanilla_parser)

    step2_parser = subparsers.add_parser(
        "step2",
        aliases=["refine"],
        help="Run LLM refinement on a specified topic-words file.",
    )
    add_step2_arguments(step2_parser)

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run vanilla -> step2 -> guided train end-to-end.",
    )
    add_pipeline_arguments(pipeline_parser)

    return parser


def run_pipeline(args):
    pipeline_root = Path(args.out_dir) if args.out_dir else default_pipeline_dir(args.model, args.num_topics)
    pipeline_root.mkdir(parents=True, exist_ok=True)

    print("\n=== Stage 1/3: Train vanilla topic model ===")
    stage1_args = Namespace(
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
        out_dir=str(pipeline_root / "vanilla"),
        seed=args.seed,
        patience=args.patience,
        output_suffix=None,
        anchor_words_file=None,
        anchor_topics_json=None,
        lambda_anchor=0.0,
    )
    stage1_result = run_train(stage1_args)

    print("\n=== Stage 2/3: Refine topics with LLM ===")
    refine_args = Namespace(
        topic_words_file=stage1_result["top_words_path"],
        model_name=args.model_name,
        max_new_tokens_step1=args.max_new_tokens_step1,
        max_new_tokens_step2=args.max_new_tokens_step2,
        max_new_tokens_step3=args.max_new_tokens_step3,
        out_dir=None,
        run_name=args.run_name,
        device=args.device,
    )
    refine_result = run_step2(
        refine_args,
        output_parent_dir=pipeline_root,
        folder_name="schema_{final_topic_count}",
    )

    print("\n=== Stage 3/3: Train anchor-guided topic model ===")
    guided_args = Namespace(
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
        out_dir=str(pipeline_root / "guided"),
        seed=args.seed,
        patience=args.patience,
        output_suffix=None,
        anchor_words_file=refine_result["topic_words_path"],
        anchor_topics_json=None,
        lambda_anchor=args.lambda_anchor,
    )
    guided_result = run_train(guided_args)

    print("\n=== Pipeline complete ===")
    print("Stage 1 top words:", stage1_result["top_words_path"])
    print("Refine topic words:", refine_result["topic_words_path"])
    print("Guided metrics:", guided_result["metrics_path"])
    return {
        "stage1": stage1_result,
        "refine": refine_result,
        "guided": guided_result,
    }


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command in ("vanilla", "train"):
        vanilla_args = Namespace(
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
            out_dir=args.out_dir or str(default_vanilla_dir(args.model, args.num_topics)),
            seed=args.seed,
            patience=args.patience,
            output_suffix=None,
            anchor_words_file=None,
            anchor_topics_json=None,
            lambda_anchor=0.0,
        )
        run_train(vanilla_args)
        return

    if args.command in ("step2", "refine"):
        run_step2(args)
        return

    if args.command == "pipeline":
        run_pipeline(args)
        return

    parser.print_help()
    raise SystemExit(1)


if __name__ == "__main__":
    main()
