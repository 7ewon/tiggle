"""한국어 명령 정규화. 같은 뜻의 다른 표기를 한 가지로 모아 규칙과 캐시가 쉽게 맞도록 한다.

- 숫자: '세 대' -> '3대', '삼십 센티' -> '30cm', '반 바퀴' -> '180도'
- 단위: 센티/센치/cm -> cm, 미터 -> m, 밀리 -> mm
- 로봇 지칭: '1번 로봇이랑 3번' -> '1번+3번', '1번부터 4번까지' -> '1~4번'
- 좌표: '(1, 0.5)' -> '(1 0.5)'  (쉼표는 절 구분자로 쓰이므로 치운다)
"""

from __future__ import annotations

import re
import unicodedata

_NATIVE = {
    "한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "네": 4, "넷": 4,
    "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9,
}
_SINO_DIGIT = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}

_NATIVE_WORD = r"열\s*(?:하나|한|둘|두|셋|세|넷|네|다섯)?|하나|한|둘|두|셋|세|넷|네|다섯|여섯|일곱|여덟|아홉"
_NATIVE_RE = re.compile(rf"(?<![가-힣])({_NATIVE_WORD})\s*(대|명|바퀴|줄|칸|개)")
_SINO_RE = re.compile(
    r"(?<![가-힣])([일이삼사오육칠팔구십백]+)\s*(센티미터|센티|센치|미터|밀리미터|밀리|도(?=$|[^가-힣]|로|만큼|씩|정도|를))"
)


def _native_to_int(word: str) -> int:
    word = word.replace(" ", "")
    if word.startswith("열"):
        return 10 + _NATIVE.get(word[1:], 0)
    return _NATIVE[word]


def _sino_to_int(word: str) -> int | None:
    total, cur = 0, 0
    for ch in word:
        if ch in _SINO_DIGIT:
            if cur:
                return None  # '삼사'처럼 자릿수 없이 이어진 숫자
            cur = _SINO_DIGIT[ch]
        elif ch == "십":
            total += (cur or 1) * 10
            cur = 0
        elif ch == "백":
            total += (cur or 1) * 100
            cur = 0
    return total + cur


def _sub_sino(m: re.Match) -> str:
    n = _sino_to_int(m.group(1))
    return m.group(0) if n is None else f"{n}{m.group(2)}"


def normalize(text: str) -> str:
    s = unicodedata.normalize("NFKC", text).lower().strip()
    s = s.replace("〜", "~").replace("–", "-").replace("−", "-")
    s = re.sub(r"[!?\"'“”‘’]", " ", s)

    # 숫자 말 -> 아라비아 숫자
    s = _NATIVE_RE.sub(lambda m: f"{_native_to_int(m.group(1))}{m.group(2)}", s)
    s = _SINO_RE.sub(_sub_sino, s)
    s = re.sub(r"반\s*바퀴", "180도", s)
    s = re.sub(r"(\d+)\s*바퀴", lambda m: f"{int(m.group(1)) * 360}도", s)
    s = re.sub(r"직각", "90도", s)
    s = re.sub(r"(?<![가-힣])반\s*(?=미터|m(?![a-z]))", "0.5", s)

    # 단위
    s = re.sub(r"(\d)\s*(?:센티미터|센티|센치|cm)", r"\1cm", s)
    s = re.sub(r"(\d)\s*(?:밀리미터|밀리|mm)", r"\1mm", s)
    s = re.sub(r"(\d)\s*(?:미터|m(?![a-z]))", r"\1m", s)

    # 로봇 지칭
    s = re.sub(r"로봇\s*(\d+)\s*번", r"\1번", s)
    s = re.sub(r"(\d+)\s*번\s*로봇", r"\1번", s)
    s = re.sub(r"(\d+)\s*번?\s*(?:부터|에서)\s*(\d+)\s*번(?:\s*까지)?", r"\1~\2번", s)
    s = re.sub(r"(\d+)\s*[~-]\s*(\d+)\s*번(?:\s*까지)?", r"\1~\2번", s)
    conj = re.compile(r"(\d+(?:~\d+)?번)\s*(?:하고|이랑|랑|과|와|및|,|\s)\s*(?=\d+(?:~\d+)?번)")
    while conj.search(s):
        s = conj.sub(r"\1+", s)

    # 좌표 (x, y) -> (x y)
    s = re.sub(
        r"\(\s*(-?\d+(?:\.\d+)?)\s*m?\s*,\s*(-?\d+(?:\.\d+)?)\s*m?\s*\)", r"(\1 \2)", s
    )
    return re.sub(r"\s+", " ", s).strip()
