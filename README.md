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
- `wete`

Training defaults:
- `epochs=250`, `lr=0.001`, and `batch_size=500` are used by default
- `patience` uses the CLI default unless you override it manually

### 1. Vanilla
Train the base neural topic model only.

```bash
python main.py vanilla --model etm --num_topics 50
```

Outputs:
- `results/vanilla_etm_50/top_words.txt`
- `results/vanilla_etm_50/metrics.json`

### 2. Step2
Run LLM refinement on an existing topic-words file.

```bash
python main.py step2 --topic_words_file results/vanilla_etm_50/top_words.txt
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

### 3. Pipeline
Run the full end-to-end flow: vanilla training -> step2 refinement -> guided training.

```bash
python main.py pipeline --model etm --num_topics 50
```

Outputs:
- `results/pipeline_etm_50/vanilla/`
- `results/pipeline_etm_50/schema_<final_topic_count>/`
- `results/pipeline_etm_50/guided/`

Default flow:
1. Train vanilla topic model
2. Refine `results/vanilla_etm_50/top_words.txt`
3. Train anchor-guided topic model using the refined topic words
