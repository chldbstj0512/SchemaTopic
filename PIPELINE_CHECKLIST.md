# Pipeline 실행 전 체크리스트

`run_pipeline_auto.sh` / `run_pipeline_keep.sh` 실행 전 점검용

---

## 1. 환경

| 항목 | 상태 | 비고 |
|------|------|------|
| **Python** | ✅ | 스크립트가 conda ys0660 경로 자동 사용 (conda 미활성화 시에도 동작) |
| **torch + CUDA** | ✅ | ys0660 env: torch 2.1.2, cuda 사용 가능 |
| **transformers** | ✅ | Llama 로드 가능 |
| **Java** | ✅ | OpenJDK 11 (Palmetto용) |

---

## 2. 데이터셋

| 데이터셋 | train.pkl | test.pkl | voc.txt | word_embeddings.npy |
|----------|-----------|----------|---------|---------------------|
| 20News | ✅ | ✅ | ✅ | ✅ |
| AGNews | ✅ | ✅ | ✅ | ✅ |
| DBpedia | ✅ | ✅ | ✅ | ✅ |
| R8 | ✅ | ✅ | ✅ | ✅ |

---

## 3. NTM 모델

| 모델 | 상태 |
|------|------|
| ecrtm | ✅ |
| etm | ✅ |
| nstm | ✅ |
| nvdm | ✅ |
| plda | ✅ |

---

## 4. 평가 (Palmetto C_V)

| 항목 | 상태 |
|------|------|
| palmetto-0.1.5-exec.jar | ✅ 프로젝트 루트 |
| wikipedia_bd/ | ✅ 프로젝트 루트 |

---

## 5. LLM (HuggingFace)

| 항목 | 상태 |
|------|------|
| meta-llama/Meta-Llama-3-8B-Instruct | ✅ 캐시/접근 가능 |

---

## 6. 스크립트

| 항목 | 상태 |
|------|------|
| run_pipeline_auto.sh | ✅ 실행 권한 |
| run_pipeline_keep.sh | ✅ 실행 권한 |
| retry_failed.sh | ✅ (실패 시 재실행용) |

---

## 7. 디스크

| 항목 | 상태 |
|------|------|
| **여유 공간** | ⚠️ **약 55GB** (97% 사용 중) |

- 200개 실험 × ~50–100MB ≈ 10–20GB 예상
- 로그 추가 용량 고려 시 **여유 공간 부족 가능성 있음**
- `results/` 또는 `results/experiment_logs/` 정리 권장

---

## 8. 실행 전 필수 확인

```bash
# 1. dry-run으로 명령 확인
./run_pipeline_auto.sh --dry-run | head -30
./run_pipeline_keep.sh --dry-run | head -30
```

---

## 9. 실행 예시

```bash
# 터미널 1
CUDA_VISIBLE_DEVICES=0 ./run_pipeline_auto.sh

# 터미널 2
CUDA_VISIBLE_DEVICES=1 ./run_pipeline_keep.sh
```

---

## 10. 주의사항

- **`python`**: 스크립트가 ys0660 conda env 경로를 자동 사용. (conda activate 없이도 동작)
- **디스크**: 여유 공간이 적으면 실험 중 디스크 풀 에러 가능
- **실패 시**: `results/experiment_logs/failed_auto.txt`, `failed_keep.txt` 확인 후 `./retry_failed.sh auto` / `./retry_failed.sh keep`로 재실행
