# SchemaTopic

## Environment

```bash
conda activate llm_itl
```

For schema stages using OpenAI models (gpt-4o, gpt-5.2, opus), set `OPENAI_API_KEY` and install: `pip install openai`.

---

## 레포 구성

- **진입점**: `main.py` (모든 명령은 `python main.py <command> ...`)
- **데이터**: `datasets/` (20News, AGNews, DBpedia, R8 등 전처리된 데이터)
- **출력**: 기본값은 모두 `results/` 아래 (`results/vanilla_*`, `results/schema_*`, `results/pipeline_*`, `results/anchor_*`)
- **스크립트**: `scripts/run_schema_gpt_example.sh` (schema + GPT 예시 실행)

---

## Quick run (복사해서 실행)

아래는 모두 **프로젝트 루트(SchemaTopic)** 에서 실행한다고 가정합니다.

```bash
conda activate llm_itl
cd /path/to/SchemaTopic   # 실제 레포 경로로
```

**1) Schema만 (이미 있는 top_words 사용, GPT)**

```bash
# .env 에 OPENAI_API_KEY 있으면 로드
set -a && [ -f .env ] && source .env && set +a

python main.py schema \
  --topic_words_file results/vanilla_20News_etm_50/top_words.txt \
  --model_name gpt-4o \
  --out_dir results/schema_gpt_example
```

`results/vanilla_*` 가 없으면 먼저 vanilla를 돌리거나, 아래 스크립트는 `experiments/4/full/vanilla_20News_etm_50/top_words.txt` 를 기본으로 사용합니다:

```bash
bash scripts/run_schema_gpt_example.sh
# 인자로 경로 지정 가능: bash scripts/run_schema_gpt_example.sh <top_words.txt> <out_dir>
```

**2) Vanilla → Schema (처음부터)**

```bash
set -a && [ -f .env ] && source .env && set +a

python main.py vanilla --model etm --dataset 20News --num_topics 50
# 출력: results/vanilla_20News_etm_50/top_words.txt, metrics.json

python main.py schema \
  --topic_words_file results/vanilla_20News_etm_50/top_words.txt \
  --model_name gpt-4o
# 출력: results/schema_<N>/ (step1~3, schema_topics.json, topic_words.txt 등)
```

**3) Pipeline 한 번에 (vanilla → schema → anchor)**

```bash
python main.py pipeline --model etm --dataset 20News --num_topics 50
# 출력: results/pipeline_20News_etm_50/vanilla/ , schema_<N>/ , anchor/
```

**4) Pipeline에서 vanilla 건너뛰고 기존 top_words 로 schema부터**

```bash
python main.py pipeline --model etm --dataset 20News --num_topics 50 \
  --topic_words_file results/pipeline_20News_etm_50/vanilla/top_words.txt
```

**5) 4번 실험 (Step ablation)**

Vanilla 한 번 돌린 뒤, full / no_step1 / no_step2 / no_step3 네 가지 schema 구성을 각각 pipeline으로 돌리는 예시. (OpenAI 쓰면 `set -a && [ -f .env ] && source .env && set +a` 먼저 실행.)

```bash
# 1) Vanilla 1회 (공통 입력)
python main.py vanilla --model etm --dataset 20News --num_topics 50
TOPIC_WORDS=results/vanilla_20News_etm_50/top_words.txt

# 2) Full (step 1,2,3 모두 사용)
python main.py pipeline --model etm --dataset 20News --num_topics 50 \
  --topic_words_file "$TOPIC_WORDS" --out_dir results/exp4_etm_full

# 3) Without step 1
python main.py pipeline --model etm --dataset 20News --num_topics 50 \
  --topic_words_file "$TOPIC_WORDS" --skip_step1 --out_dir results/exp4_etm_no_step1

# 4) Without step 2
python main.py pipeline --model etm --dataset 20News --num_topics 50 \
  --topic_words_file "$TOPIC_WORDS" --skip_step2 --out_dir results/exp4_etm_no_step2

# 5) Without step 3
python main.py pipeline --model etm --dataset 20News --num_topics 50 \
  --topic_words_file "$TOPIC_WORDS" --skip_step3 --out_dir results/exp4_etm_no_step3
```

출력 디렉터리: `results/exp4_etm_full/`, `results/exp4_etm_no_step1/`, `results/exp4_etm_no_step2/`, `results/exp4_etm_no_step3/` (각각 vanilla/, schema_*/, anchor/ 포함).

한 번에 돌리려면:

```bash
bash scripts/run_experiment4_example.sh
```

---

## Commands Overview

| Command | Aliases | Description |
|--------|---------|-------------|
| `vanilla` | `train` | Train base neural topic model only |
| `schema` | `step2`, `refine` | LLM schema induction/refinement on a topic-words file |
| `anchor` | — | Anchor-guided training from schema outputs |
| `pipeline` | — | End-to-end: vanilla → schema → anchor |
| `eval` | — | Evaluate a saved checkpoint (TC, TD, Purity, NMI) |
| `hierarchy` | — | TraCo hierarchy metrics from schema_topics.json |

---

## Common Options (vanilla / anchor / pipeline)

**Topic model & data**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `etm` | Backend: `etm`, `ecrtm`, `nvdm`, `plda`, `nstm`, `scholar` |
| `--dataset` | `20News` | Dataset name under `datasets/` (e.g. 20News, AGNews, DBpedia, R8) |
| `--data_dir` | — | Custom dataset path; overrides `--dataset` |
| `--num_topics` | `50` | Number of topics |

**Training (defaults)**

- `--epochs` 250, `--lr` 0.001, `--batch_size` 500, `--eval_batch_size` 256  
- `--optimizer` adam, `--clip` 1.0, `--log_interval` 10  
- `--theta_act` relu, `--t_hidden_size` 500, `--enc_drop` 0.0  
- `--bow_norm` 1, `--topk_words` 15, `--seed` 1, `--wdecay` 1.2e-6  

---

## 1. Vanilla

Train the base neural topic model only. (프로젝트 루트에서 실행)

```bash
conda activate llm_itl
python main.py vanilla --model etm --dataset 20News --num_topics 50
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--out_dir` | — | Output directory (default: `results/vanilla_{dataset}_{model}_{num_topics}`) |

**Outputs**

- `results/vanilla_20News_etm_50/top_words.txt`
- `results/vanilla_20News_etm_50/metrics.json`

---

## 2. Schema

Run LLM schema induction/refinement on an existing topic-words file. 입력은 vanilla 출력 `results/vanilla_<dataset>_<model>_<K>/top_words.txt` 또는 그 경로.

```bash
python main.py schema --topic_words_file results/vanilla_20News_etm_50/top_words.txt
```

**K-keep** (토픽 수 유지, delete → Misc):

```bash
python main.py schema --topic_words_file results/vanilla_20News_etm_50/top_words.txt --keep
```

**OpenAI** (gpt-4o, gpt-5.2, opus). `OPENAI_API_KEY` 필요:

```bash
python main.py schema --topic_words_file results/vanilla_20News_etm_50/top_words.txt --model_name gpt-4o --out_dir results/schema_gpt
```

**Step ablation** (step 1/2/3 중 일부 생략):

```bash
python main.py schema --topic_words_file results/vanilla_20News_etm_50/top_words.txt --skip_step1 --out_dir results/schema_wo1
python main.py schema --topic_words_file results/vanilla_20News_etm_50/top_words.txt --skip_step2 --skip_step3 --out_dir results/schema_wo23
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--topic_words_file` | *(required)* | Path to topic-words file (e.g. vanilla `top_words.txt`) |
| `--keep` | False | K-keep mode: retain all topics, no delete |
| `--model_name` | `meta-llama/Meta-Llama-3-8B-Instruct` | LLM: HuggingFace id or OpenAI (gpt-4o, gpt-5.2, opus) |
| `--max_new_tokens_step1` | 4096 | Max new tokens for schema step 1 |
| `--max_new_tokens_step2` | 4096 | Max new tokens for schema step 2 |
| `--max_new_tokens_step3` | 4096 | Max new tokens for schema step 3 |
| `--json_retry_attempts` | 2 | Retries for malformed JSON in step2/step3 |
| `--out_dir` | — | Override output directory |
| `--run_name` | — | Optional run name |
| `--device` | cuda | Device for HuggingFace model |
| `--skip_step1` | False | Ablation: skip schema induction (use default schema) |
| `--skip_step2` | False | Ablation: skip score+prune (keep all topics) |
| `--skip_step3` | False | Ablation: skip schema-aware refine (build from step2 only) |

**Outputs** (under `results/schema_<final_topic_count>/` or `--out_dir`)

- `step1.txt`, `step2.txt`, `step3.txt`
- `schema_topics.json`, `topic_words.txt`, `schema_topic_words.txt`

---

## 3. Anchor

Run anchor-guided topic model training from a schema result. `--schema_dir` 는 schema 단계 출력 디렉터리(`topic_words.txt`, `schema_topics.json` 포함).

```bash
python main.py anchor --model etm --dataset 20News --schema_dir results/schema_24 --num_topics 50
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--schema_dir` | — | Schema output dir (auto-loads `topic_words.txt` / `schema_topics.json`) |
| `--anchor_words_file` | — | Explicit topic-words file; overrides `--schema_dir` detection |
| `--anchor_topics_json` | — | Explicit schema JSON; used when anchor words not from file |
| `--out_dir` | — | Output directory (default: `results/anchor_{dataset}_{model}_{num_topics}`) |
| `--run_name` | — | Optional run name |
| `--lambda_anchor` | 1.0 | Anchor loss weight |

One of `--schema_dir`, `--anchor_words_file`, or `--anchor_topics_json` is required.

**Outputs**

- `results/anchor_20News_etm_50/` (model.pt, metrics.json, top_words.txt 등)

---

## 4. Pipeline

Full flow: vanilla → schema → anchor. 출력은 `results/pipeline_<dataset>_<model>_<num_topics>/` 아래에 vanilla/, schema_<N>/, anchor/ 로 쌓임.

```bash
python main.py pipeline --model etm --dataset 20News --num_topics 50
```

**Vanilla 건너뛰기** (이미 있는 top_words 로 schema부터):

```bash
python main.py pipeline --model etm --dataset 20News --num_topics 50 \
  --topic_words_file results/pipeline_20News_etm_50/vanilla/top_words.txt
```

**K-keep + OpenAI:**

```bash
python main.py pipeline --model etm --dataset 20News --num_topics 50 --keep --model_name gpt-4o
```

| Argument | Default | Description |
|----------|---------|-------------|
| *(all common training args)* | — | Same as vanilla/anchor |
| `--out_dir` | — | Pipeline root (default: `results/pipeline_{dataset}_{model}_{num_topics}` or `_keep` suffix) |
| `--run_name` | — | Optional run name |
| `--lambda_anchor` | 1.0 | Anchor loss weight (stage 3) |
| `--model_name` | `meta-llama/Meta-Llama-3-8B-Instruct` | LLM for schema stage |
| `--max_new_tokens_step1/2/3` | 4096 | Schema step token limits |
| `--json_retry_attempts` | 2 | JSON retries for schema |
| `--device` | cuda | Device for schema LLM |
| `--keep` | False | K-keep mode in schema stage |
| `--topic_words_file` | — | If set: skip vanilla, use this file for schema |
| `--skip_step1` | False | Ablation: skip schema step in schema stage |
| `--skip_step2` | False | Ablation: skip prune step in schema stage |
| `--skip_step3` | False | Ablation: skip refine step in schema stage |

**Outputs**

- `results/pipeline_20News_etm_50/vanilla/` (top_words.txt, metrics.json 등)
- `results/pipeline_20News_etm_50/schema_<N>/` (step1~3, schema_topics.json, topic_words.txt 등)
- `results/pipeline_20News_etm_50/anchor/` (model.pt, metrics.json 등)

---

## 5. Eval

Evaluate a saved checkpoint (model.pt 가 있는 디렉터리 또는 파일).

```bash
python main.py eval --checkpoint results/pipeline_20News_etm_50/anchor --data_dir datasets/20News
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | *(required)* | Path to `model.pt` or directory containing it |
| `--data_dir` | — | Dataset path (default: from checkpoint training_args) |

---

## 6. Hierarchy

Compute TraCo-style hierarchy metrics from `schema_topics.json`.

```bash
python main.py hierarchy --schema results/schema_24/schema_topics.json --data_dir datasets/20News
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--schema` | *(required)* | Path to `schema_topics.json` |
| `--data_dir` | *(required)* | Dataset path (e.g. `datasets/20News`) |
| `--num_top_words` | 15 | Number of top words per topic for metrics |

---

## Refine modules: refine vs refine_k vs refine_wo

| Module | Mode | Step 2 semantics | Step 3 input | Use case |
|--------|------|------------------|--------------|----------|
| **refine.py** | Auto (delete) | `decision`: keep/delete. Deleted topics are **dropped**; only kept topics go to step 3. | Surviving topics only | Default: schema induction + prune + refine; final topic count can decrease. |
| **refine_k.py** | K-keep | `decision`: keep/delete. Deleted topics go to **Misc**; step 3 gets both surviving + misc (all original topics retained). | Surviving + misc topics | When you must keep the same number of topics (e.g. fixed K); delete → Misc, then step 3 assigns/refines all. |
| **refine_wo.py** | Ablation | Same as refine (delete). **Optional skip** of step 1, 2, or 3 via `--skip_step1` / `--skip_step2` / `--skip_step3`. | Same as refine | Experiment4-style ablation: without step 1 (no schema LLM), without step 2 (no prune), or without step 3 (no refine LLM). |

**Prompt / flow differences**

- **refine** and **refine_wo** use the same step 1/2/3 prompts (schema induction → score+prune → schema-aware refine). Step 2 output is “keep” or “delete”; deleted topics are not passed to step 3. Final output has fewer topics than input when some are deleted.
- **refine_k** uses the same step 1 and step 2 prompts, but step 2 is interpreted as keep vs **misc** (not drop). All topics (surviving + misc) are passed to step 3; the step 3 prompt includes a “Misc” block and asks the LLM to refine and assign schema for every topic. So the final topic count equals the initial count. refine_k also adds step1 repetition truncation and step3 chunked retry for long runs.

**When each is used**

- `schema` (no `--keep`, no `--skip_step*`) → **refine**
- `schema --keep` → **refine_k**
- `schema --skip_step1` or `--skip_step2` or `--skip_step3` → **refine_wo**

---

## About

SchemaTopic: neural topic modeling with LLM-driven schema induction and anchor-guided refinement.
