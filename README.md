# SchemaTopic

## Environment

Use the project conda environment before running commands:

```bash
conda activate llm_itl
```

## Run Commands

Supported `--model` values:
- `etm`
- `ecrtm`
- `nvdm`
- `plda`
- `nstm`
- `scholar`

Supported `--dataset` values in this repo:
- `20News`
- `AGNews`
- `DBpedia`
- `R8`

Training defaults:
- `epochs=250`, `lr=0.001`, and `batch_size=500` are used by default
- `--dataset 20News` is the default; you can also use `--data_dir` for a custom preprocessed dataset path

### 1. Vanilla
Train the base neural topic model only.

```bash
python main.py vanilla --model etm --dataset AGNews --num_topics 50
```

Outputs:
- `results/vanilla_AGNews_etm_50/top_words.txt`
- `results/vanilla_AGNews_etm_50/metrics.json`

### 2. Schema
Run LLM schema induction/refinement on an existing topic-words file.

```bash
python main.py schema --topic_words_file results/vanilla_AGNews_etm_50/top_words.txt
```

K-keep mode (retain all topics; delete → Misc):
```bash
python main.py schema --topic_words_file results/vanilla_AGNews_etm_50/top_words.txt --keep
```

Outputs:
- `results/schema_<final_topic_count>/`

Saved files inside the folder:
- `step1.txt`
- `step2.txt`
- `step3.txt`
- `schema_topics.json`
- `topic_words.txt`
- `schema_topic_words.txt`

### 3. Anchor
Run anchor-guided topic model training from a schema result.

```bash
python main.py anchor --model etm --dataset AGNews --schema_dir results/run_all_schema_AGNews_k50/etm --num_topics 50
```

Outputs:
- `results/anchor_AGNews_etm_50/`

Anchor inputs:
- `--schema_dir` to auto-load `topic_words.txt` or `schema_topics.json`
- or `--anchor_words_file` / `--anchor_topics_json` for explicit sources

### 4. Pipeline
Run the full end-to-end flow: vanilla training -> schema induction -> anchor-guided training.

```bash
python main.py pipeline --model etm --dataset DBpedia --num_topics 50
```

K-keep mode (schema stage retains all topics):
```bash
python main.py pipeline --model etm --dataset DBpedia --num_topics 50 --keep
```

Outputs:
- `results/pipeline_DBpedia_etm_50/vanilla/`
- `results/pipeline_DBpedia_etm_50/schema_<final_topic_count>/`
- `results/pipeline_DBpedia_etm_50/anchor/`

Default flow:
1. Train vanilla topic model
2. Build a schema from the stage-1 `top_words.txt`
3. Train anchor-guided topic model using the refined topic words
