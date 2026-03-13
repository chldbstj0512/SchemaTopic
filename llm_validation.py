"""
LLM 응답 검증: 생략(truncation) 감지.
생략된 응답은 metric에서 "truncated"로 표현됨.
"""
import re
from typing import Optional, Tuple, Any

# 생략 표시: JSON 배열/객체 내 ... (가장 흔한 패턴)
_ELLIPSIS_PATTERN = re.compile(r"\.\.\.\s*[\]}]", re.IGNORECASE)  # ... ] 또는 ... }


class TruncationError(Exception):
    """LLM 응답이 생략(truncation)되어 불완전함."""
    def __init__(self, step_name: str, message: str, text_preview: str = ""):
        self.step_name = step_name
        self.message = message
        self.text_preview = text_preview[:500] if text_preview else ""
        super().__init__(f"[{step_name}] {message}")


def validate_llm_response_no_truncation(
    text: str,
    step_name: str,
    expected_count: Optional[int] = None,
    parsed_json: Optional[Any] = None,
) -> Tuple[bool, Optional[str]]:
    """
    LLM 응답에 생략(truncation)이 없는지 검증.
    (1) ... 이 JSON 내에 있으면 생략으로 간주
    (2) 토픽 개수가 expected_count 미만이면 생략

    Returns:
        (is_valid, error_message): 유효하면 (True, None), 생략 감지 시 (False, "truncated")
    """
    if not text or not isinstance(text, str):
        return False, "truncated"

    # 1) ... 패턴 검사
    if _ELLIPSIS_PATTERN.search(text):
        return False, "truncated"

    # 2) 토픽 개수 검사
    if parsed_json is not None and expected_count is not None:
        actual = 0
        if isinstance(parsed_json, list):
            actual = len(parsed_json)
        elif isinstance(parsed_json, dict):
            groups = parsed_json.get("schema", [])
            if isinstance(groups, list):
                for g in groups:
                    if isinstance(g, dict):
                        topics = g.get("topics", [])
                        actual += len(topics) if isinstance(topics, list) else 0
        if actual < expected_count:
            return False, "truncated"

    return True, None


def check_and_raise_if_truncated(
    text: str,
    step_name: str,
    expected_count: Optional[int] = None,
    parsed_json: Optional[Any] = None,
) -> None:
    """
    생략 감지 시 TruncationError 발생.
    """
    is_valid, err = validate_llm_response_no_truncation(
        text, step_name, expected_count, parsed_json
    )
    if not is_valid:
        raise TruncationError(
            step_name=step_name,
            message=err or "truncated",
            text_preview=text[:800] if text else "",
        )


# Step 1 스키마: 1줄 1라벨, 계층 구조 없음 검증
# 들여쓰기된 하위 항목: 4칸 이상 공백 후 "- " 또는 "*", "+", "•"
_SCHEMA_INDENTED_SUBITEM = re.compile(r"^\s{4,}\s*[-*+•]\s+", re.MULTILINE)
# 라벨 내부에 콜론으로 하위 항목: "Parent: Child" 형태
_SCHEMA_COLON_SUBITEM = re.compile(r"^\s*-\s+[^:]+:\s*[^\s]", re.MULTILINE)


def validate_schema_step1_flat(schema_text: str) -> Tuple[bool, Optional[str]]:
    """
    Step 1 스키마 출력이 flat(1줄 1라벨, 계층 없음)인지 검증.

    Returns:
        (is_valid, error_message): 유효하면 (True, None), 계층 감지 시 (False, "schema_hierarchy")
    """
    if not schema_text or not isinstance(schema_text, str):
        return True, None  # 빈 입력은 검증 생략

    in_schema = False
    for line in schema_text.splitlines():
        stripped = line.strip()
        if stripped.upper() == "SCHEMA:":
            in_schema = True
            continue
        if stripped.upper() == "CRITERION:":
            in_schema = False
            continue
        if not in_schema:
            continue

        # SCHEMA 섹션 내 검사
        # 1) 들여쓰기된 하위 항목 (4칸 이상)
        if _SCHEMA_INDENTED_SUBITEM.match(line):
            return False, "schema_hierarchy"

        # 2) "*", "+", "•"로 시작하는 하위 항목 (들여쓰기 있음)
        lead = line[: len(line) - len(line.lstrip())]
        if len(lead) >= 4 and stripped and stripped[0] in "*+•":
            return False, "schema_hierarchy"

        # 3) "Parent: Child" 형태 (라벨 내 콜론 하위)
        if re.match(r"^\s*-\s+[^:]+:\s*\S", line):
            return False, "schema_hierarchy"

    return True, None


def check_schema_step1_flat(schema_text: str) -> None:
    """스키마가 flat이 아니면 SchemaHierarchyError 발생."""
    is_valid, err = validate_schema_step1_flat(schema_text)
    if not is_valid:
        raise ValueError(
            f"[Step 1] Schema must be flat (one label per line). Detected hierarchy: {err}"
        )
