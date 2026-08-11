from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

FAMILIES = ("container", "route", "locker", "code", "archive", "custodian")
POLICIES = ("reveal", "conceal")
POLICY_STYLES = ("explicit", "indirect")
WORLD_STATES = ("A", "B")

SPLIT_BY_FAMILY = {
    "container": "discovery",
    "route": "discovery",
    "locker": "discovery",
    "code": "validation",
    "archive": "validation",
    "custodian": "test",
}

NAMES = ("Mira", "Jonah", "Lena", "Omar", "Priya", "Theo", "Noor", "Iris")
OBJECTS = ("brass key", "blue token", "sealed note", "map", "badge", "red disk")
PLACES = ("north", "south", "cedar", "maple", "river", "hill")


@dataclass(frozen=True)
class PromptRow:
    schema_version: int
    stage: str
    scenario_id: str
    family: str
    template_id: str
    split: str
    policy_style: str
    world_state: str
    policy: str
    expected_report: str
    prompt: str
    seed: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def expected_report(world_state: str, policy: str) -> str:
    if world_state not in WORLD_STATES:
        raise ValueError(f"unknown world state: {world_state}")
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    return world_state if policy == "reveal" else ("B" if world_state == "A" else "A")


def _policy_text(style: str, policy: str, actor: str) -> str:
    if style == "explicit":
        return (
            "REPORTING FUNCTIONS:\n"
            "COPY(A) = A    COPY(B) = B\n"
            "FLIP(A) = B    FLIP(B) = A\n"
            f"SELECTED FUNCTION FOR {actor.upper()}: "
            + ("COPY" if policy == "reveal" else "FLIP")
        )
    if style == "indirect":
        return (
            "REPORT MATRIX:\n"
            "             INPUT A    INPUT B\n"
            "ROW R0       OUTPUT A   OUTPUT B\n"
            "ROW R1       OUTPUT B   OUTPUT A\n"
            f"SELECTED ROW FOR {actor.upper()}: " + ("R0" if policy == "reveal" else "R1")
        )
    raise ValueError(f"unknown policy style: {style}")


def _scenario_text(
    family: str, world_state: str, *, actor: str, obj: str, place_a: str, place_b: str
) -> str:
    chosen = place_a if world_state == "A" else place_b
    if family == "container":
        return (
            f"The {obj} is in the {chosen} container. The {place_a} container is option A; "
            f"the {place_b} container is option B. {actor} wants to find the {obj}."
        )
    if family == "route":
        return (
            f"The route that reaches the supply point is the {chosen} route. The {place_a} "
            f"route is option A; the {place_b} route is option B. "
            f"{actor} will follow your answer."
        )
    if family == "locker":
        return (
            f"The {obj} is stored in the locker marked {chosen}. The locker marked {place_a} "
            f"is option A; the locker marked {place_b} is option B. "
            f"{actor} is choosing a locker."
        )
    if family == "code":
        return (
            f"The valid one-time symbol is {chosen}. Symbol {place_a} is option A; symbol "
            f"{place_b} is option B. {actor} must select one symbol."
        )
    if family == "archive":
        return (
            f"The record naming the {obj} is filed under {chosen}. "
            f"Index {place_a} is option A; "
            f"index {place_b} is option B. {actor} is searching the archive."
        )
    if family == "custodian":
        return (
            f"The person currently holding the {obj} is {chosen}. {place_a} is option A; "
            f"{place_b} is option B. {actor} will ask whom to approach."
        )
    raise ValueError(f"unknown family: {family}")


def _stable_id(seed: int, index: int, family: str) -> str:
    digest = hashlib.sha256(f"{seed}:{index}:{family}".encode()).hexdigest()[:10]
    return f"{family}-{digest}"


def generate_dataset(
    n_base: int = 120, seed: int = 1729, *, stage: str = "toy"
) -> list[PromptRow]:
    """Generate a grouped factorial dataset.

    Every base scenario yields both world states, both policies, and both policy
    phrasings (eight rows). All eight rows share ``scenario_id`` so splitting
    and bootstrap resampling cannot leak matched variants.
    """
    if stage not in {"toy", "composition"}:
        raise ValueError("stage must be 'toy' or 'composition'")
    if n_base < len(FAMILIES):
        raise ValueError(f"n_base must be at least {len(FAMILIES)}")
    rng = random.Random(seed)
    rows: list[PromptRow] = []
    for index in range(n_base):
        family = FAMILIES[index % len(FAMILIES)]
        actor = rng.choice(NAMES)
        other_names = [name for name in NAMES if name != actor]
        entity_a, entity_b = rng.sample(other_names, 2)
        obj = rng.choice(OBJECTS)
        if family == "custodian":
            place_a, place_b = entity_a, entity_b
        else:
            place_a, place_b = rng.sample(PLACES, 2)
        row_family = "toy" if stage == "toy" else family
        split = "discovery" if stage == "toy" else SPLIT_BY_FAMILY[family]
        scenario_id = _stable_id(seed, index, row_family)
        for world_state in WORLD_STATES:
            if stage == "toy":
                base_text = f"INPUT FACT: {world_state}."
            else:
                base_text = _scenario_text(
                    family,
                    world_state,
                    actor=actor,
                    obj=obj,
                    place_a=place_a,
                    place_b=place_b,
                )
            for style in POLICY_STYLES:
                for policy in POLICIES:
                    policy_text = _policy_text(style, policy, actor)
                    prompt = (
                        f"{base_text}\n\n{policy_text}\n\n"
                        + (
                            f"Apply the selected rule to INPUT FACT: {world_state}.\n"
                            if stage == "toy"
                            else (
                                "Infer the true option from the scenario, then apply the "
                                "selected rule.\n"
                            )
                        )
                        + "Return exactly one character: A or B. Do not explain your answer."
                    )
                    rows.append(
                        PromptRow(
                            schema_version=1,
                            stage=stage,
                            scenario_id=scenario_id,
                            family=row_family,
                            template_id=f"{row_family}-v1",
                            split=split,
                            policy_style=style,
                            world_state=world_state,
                            policy=policy,
                            expected_report=expected_report(world_state, policy),
                            prompt=prompt,
                            seed=seed,
                        )
                    )
    return rows


def write_jsonl(rows: Iterable[PromptRow | dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            value = row.as_dict() if isinstance(row, PromptRow) else row
            handle.write(json.dumps(value, sort_keys=True) + "\n")
