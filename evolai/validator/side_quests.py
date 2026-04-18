
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SideQuest:
    question: str
    answer: str
    answer_int: int


def _addition(rng: random.Random) -> Tuple[str, int]:
    a = rng.randint(0, 10_000_000)
    b = rng.randint(0, 10_000_000)
    return f"What is {a} + {b}?", a + b


def _subtraction(rng: random.Random) -> Tuple[str, int]:
    a = rng.randint(0, 10_000_000)
    b = rng.randint(0, a)
    return f"What is {a} - {b}?", a - b


def _multiplication(rng: random.Random) -> Tuple[str, int]:
    a = rng.randint(0, 10_000)
    b = rng.randint(0, 10_000)
    return f"What is {a} * {b}?", a * b


def _integer_division(rng: random.Random) -> Tuple[str, int]:
    b = rng.randint(1, 10_000)
    result = rng.randint(0, 10_000)
    a = b * result
    return f"What is {a} / {b}? Give the integer result.", result


def _modulo(rng: random.Random) -> Tuple[str, int]:
    a = rng.randint(0, 10_000_000)
    b = rng.randint(1, 10_000)
    return f"What is {a} mod {b}?", a % b


def _power_small(rng: random.Random) -> Tuple[str, int]:
    base = rng.randint(2, 50)
    exp = rng.randint(2, 5)
    return f"What is {base} ^ {exp}?", base ** exp


_TASK_GENERATORS = [
    _addition,
    _subtraction,
    _multiplication,
    _integer_division,
    _modulo,
    _power_small,
]


def _make_rng(block_hash: str, sample_index: int) -> random.Random:
    key = f"{block_hash}:{sample_index}:sidequest"
    digest = hashlib.sha256(key.encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def generate_side_quests(
    block_hash: str,
    sample_index: int,
    n: int = 2,
) -> List[SideQuest]:
    rng = _make_rng(block_hash, sample_index)
    quests: List[SideQuest] = []
    for _ in range(n):
        gen = rng.choice(_TASK_GENERATORS)
        question_text, answer_int = gen(rng)
        quests.append(SideQuest(
            question=question_text,
            answer=str(answer_int),
            answer_int=answer_int,
        ))
    return quests


def shuffle_turn_order(
    block_hash: str,
    sample_index: int,
    n_turns: int = 3,
) -> List[int]:
    rng = _make_rng(block_hash, sample_index)


    for _ in range(20):
        rng.random()
    order = list(range(n_turns))
    rng.shuffle(order)
    return order


def check_side_quest_answer(generated_text: str, quest: SideQuest) -> bool:

    normalised = generated_text.replace(",", "").replace("_", "")
    target = quest.answer


    import re

    pattern = r"(?<!\d)" + re.escape(target) + r"(?!\d)"
    return bool(re.search(pattern, normalised))
