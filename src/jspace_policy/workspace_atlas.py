from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

GAMES = ("inspection", "kuhn", "cheap_talk", "signaling", "disclosure", "chameleon")
SPLITS = ("discovery", "validation", "locked_replication")
ACTIONS = ("A", "B", "C")


@dataclass(frozen=True)
class AtlasRow:
    schema_version: int
    condition_id: str
    matched_group_id: str
    game: str
    split: str
    template_id: str
    prompt: str
    candidates: tuple[str, ...]
    expected_action: str
    private_state: str
    belief: float
    objective: float
    value_margin: float
    strategy: str
    action: str
    solver: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["candidates"] = list(self.candidates)
        return value


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _stable_id(*parts: object) -> str:
    text = ":".join(map(str, parts))
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def _margin(values: list[float]) -> float:
    ordered = sorted(values, reverse=True)
    return float(ordered[0] - ordered[1])


def _prompt(body: str) -> str:
    return (
        body.rstrip()
        + "\n\nChoose the action that maximizes your stated payoff. "
        + "Return exactly one letter: A, B, or C. Do not explain.\nAnswer:"
    )


def _inspection(index: int, split: str, style: int) -> dict[str, Any]:
    audit = (0.05, 0.15, 0.25, 0.35, 0.50, 0.65)[index % 6]
    reward = 8 + (index % 3)
    penalty = 10 + 2 * (index % 4)
    safe = 3 + (index % 2)
    risky = reward - audit * penalty
    values = [risky, safe, 0.0]
    expected = ACTIONS[_argmax(values)]
    strategy = "risk" if expected == "A" else "safe"
    if style == 0:
        body = (
            f"An audit occurs with probability {audit:.2f}. Action A pays {reward} if not "
            f"penalized and loses {penalty} times the audit probability in expected value. "
            f"Action B pays {safe} for certain. Action C pays 0."
        )
    elif style == 1:
        body = (
            f"Decision table: A has expected payoff {reward} - ({audit:.2f} x {penalty}); "
            f"B has payoff {safe}; C has payoff 0."
        )
    else:
        body = (
            f"Inspection chance={audit:.2f}. Risky option A: base return {reward}, expected "
            f"inspection loss {audit:.2f} x {penalty}. Conservative option B returns {safe}. "
            "Fallback C returns zero."
        )
    return {
        "prompt": _prompt(body),
        "private_state": f"audit_{audit:.2f}",
        "belief": audit,
        "objective": risky,
        "value_margin": _margin(values),
        "strategy": strategy,
        "expected_action": expected,
        "solver": {"action_values": values, "audit_probability": audit},
    }


def _kuhn(index: int, split: str, style: int) -> dict[str, Any]:
    cards = ("J", "Q", "K")
    card = cards[index % 3]
    opponent_style = index // 3 % 4
    call_tables = (
        {"J": 0.10, "Q": 0.45, "K": 0.90},
        {"J": 0.25, "Q": 0.60, "K": 0.95},
        {"J": 0.05, "Q": 0.25, "K": 0.75},
        {"J": 0.35, "Q": 0.70, "K": 1.00},
    )
    calls = call_tables[opponent_style]
    opponents = [other for other in cards if other != card]
    rank = {"J": 0, "Q": 1, "K": 2}
    check = sum(1.0 if rank[card] > rank[o] else -1.0 for o in opponents) / 2
    bet_terms = []
    for opponent in opponents:
        call = calls[opponent]
        showdown = 2.0 if rank[card] > rank[opponent] else -2.0
        bet_terms.append((1 - call) * 1.0 + call * showdown)
    bet = sum(bet_terms) / 2
    values = [bet, check, -1.0]
    expected = ACTIONS[_argmax(values)]
    strategy = (
        "value_bet"
        if expected == "A" and card == "K"
        else "bluff"
        if expected == "A" and card == "J"
        else "thin_bet"
        if expected == "A"
        else "check"
    )
    call_text = ", ".join(f"{c}:{calls[c]:.2f}" for c in cards)
    if style == 0:
        body = (
            "You are first to act in one-card poker with cards J<Q<K. You hold "
            f"{card}; the opponent has either remaining card with equal probability. A=BET, "
            "B=CHECK, C=FOLD. A fold by the opponent pays +1; a called bet pays +2 if your "
            "card is higher and -2 otherwise. A check goes to showdown for +1/-1. "
            f"Opponent call probabilities by card are {call_text}."
        )
    elif style == 1:
        body = (
            f"One-card decision. Rank J<Q<K; your private card is {card}. The other card is "
            f"uniform over the two remaining cards. Call tendencies are {call_text}. Choose "
            "A to bet, B to check, or C to concede. Bet outcomes: fold +1, called win +2, "
            "called loss -2. Check showdown: win +1, loss -1."
        )
    else:
        body = (
            f"Poker state: private card={card}; deck=J,Q,K; opponent card is uniformly one "
            f"of the others; P(call|card)={call_text}. A is bet (fold:+1, called win:+2, "
            "called loss:-2), B is check (showdown win:+1, loss:-1), C is fold (-1)."
        )
    return {
        "prompt": _prompt(body),
        "private_state": card,
        "belief": sum(calls[o] for o in opponents) / 2,
        "objective": bet,
        "value_margin": _margin(values),
        "strategy": strategy,
        "expected_action": expected,
        "solver": {"action_values": values, "opponent_cards": opponents, "call_table": calls},
    }


def _cheap_talk(index: int, split: str, style: int) -> dict[str, Any]:
    state = index % 6
    conflict = (-2, -1, 0, 1)[index // 3 % 4]
    receiver_actions = [0, 3, 5]
    preferred = min(5, max(0, state + conflict))
    values = [-float((action - preferred) ** 2) for action in receiver_actions]
    expected = ACTIONS[_argmax(values)]
    receiver_best = min(receiver_actions, key=lambda action: abs(action - state))
    chosen_receiver = receiver_actions[ACTIONS.index(expected)]
    strategy = "aligned_message" if chosen_receiver == receiver_best else "influence_message"
    if style == 0:
        body = (
            f"You are a sender who privately observes state {state}. Message A makes the "
            "receiver choose 0, B makes them choose 3, and C makes them choose 5. Your ideal "
            f"receiver action is state plus bias {conflict}, clipped to 0..5, so it is "
            f"{preferred}. Your payoff is minus squared distance from that ideal."
        )
    elif style == 1:
        body = (
            f"Private number={state}; sender bias={conflict}; desired response={preferred}. "
            "The receiver maps signals A/B/C to responses 0/3/5. Sender utility is the "
            "negative squared error between the response and desired response."
        )
    else:
        body = (
            f"Communication decision: only you know X={state}. You prefer the listener to "
            f"take action {preferred}. Signal mapping is A->0, B->3, C->5. Score each signal "
            "as -(listener action - preferred action)^2."
        )
    return {
        "prompt": _prompt(body),
        "private_state": str(state),
        "belief": float(receiver_best) / 5,
        "objective": float(preferred),
        "value_margin": _margin(values),
        "strategy": strategy,
        "expected_action": expected,
        "solver": {
            "action_values": values,
            "conflict": conflict,
            "receiver_actions": receiver_actions,
            "receiver_best_action": receiver_best,
        },
    }


def _signaling(index: int, split: str, style: int) -> dict[str, Any]:
    types = ("L", "M", "H")
    hidden_type = types[index % 3]
    regime = index // 3 % 4
    impressions = [0, 1, 2]
    type_rank = types.index(hidden_type)
    base_costs = (
        [0.0, 0.6, 3.0],
        [0.0, 2.2, 5.0],
        [0.0, 1.0, 6.0],
        [0.0, 3.0, 7.0],
    )[regime]
    scale = (1.35, 1.0, 0.65)[type_rank]
    costs = [cost * scale for cost in base_costs]
    values = [2.0 * impression - costs[i] for i, impression in enumerate(impressions)]
    expected = ACTIONS[_argmax(values)]
    chosen = ACTIONS.index(expected)
    strategy = (
        "separating"
        if chosen == type_rank
        else "pooling_up"
        if chosen > type_rank
        else "pooling_down"
    )
    cost_text = ", ".join(f"{ACTIONS[i]}:{costs[i]:.2f}" for i in range(3))
    if style == 0:
        body = (
            f"Your private type is {hidden_type} (L<M<H). Signals A, B, C make the receiver "
            f"infer levels 0, 1, 2. Your payoff is 2 times inferred level minus signal cost. "
            f"Your costs are {cost_text}."
        )
    elif style == 1:
        body = (
            f"Hidden quality={hidden_type}. Receiver impressions from A/B/C are 0/1/2. "
            f"Benefit is twice the impression; type-specific signal costs are {cost_text}."
        )
    else:
        body = (
            f"Signaling table for private type {hidden_type}: A induces impression 0 and "
            f"costs {costs[0]:.2f}; B induces 1 and costs {costs[1]:.2f}; C induces 2 and "
            f"costs {costs[2]:.2f}. Utility=2*impression-cost."
        )
    return {
        "prompt": _prompt(body),
        "private_state": hidden_type,
        "belief": float(chosen) / 2,
        "objective": float(2 * chosen),
        "value_margin": _margin(values),
        "strategy": strategy,
        "expected_action": expected,
        "solver": {"action_values": values, "costs": costs, "impressions": impressions},
    }


def _disclosure(index: int, split: str, style: int) -> dict[str, Any]:
    cases = (
        (3, -2, 1),
        (2, -3, 2),
        (4, -1, 3),
        (1, 3, 3),
        (-2, 4, 3),
        (3, 2, 4),
        (2, 2, 3),
        (1, 2, 3),
        (-1, 3, 2),
        (4, -3, 2),
        (2, -1, 3),
        (1, 4, 4),
    )
    w1, w2, threshold = cases[index % len(cases)]
    disclosure_cost = 0.25 + 0.25 * (index % 3)
    revealed_scores = [w1, w2, w1 + w2]
    receiver_approve = [score >= threshold for score in revealed_scores]
    values = [
        (5.0 if approve else 0.0) - disclosure_cost * (2 if i == 2 else 1)
        for i, approve in enumerate(receiver_approve)
    ]
    expected = ACTIONS[_argmax(values)]
    chosen = ACTIONS.index(expected)
    strategy = "full_disclosure" if chosen == 2 else "selective_disclosure"
    if style == 0:
        body = (
            f"You truthfully know two facts with receiver scores F1={w1} and F2={w2}. "
            f"The receiver approves when the sum of disclosed scores is at least {threshold}. "
            f"A reveals only F1, B only F2, C both. Approval pays you 5. Each disclosed fact "
            f"costs you {disclosure_cost:.2f}."
        )
    elif style == 1:
        body = (
            f"Disclosure problem. True feature values are ({w1}, {w2}); acceptance cutoff is "
            f"{threshold}. Plan A sends feature one, B sends feature two, C sends both. Your "
            f"utility is 5 for acceptance minus {disclosure_cost:.2f} per sent feature."
        )
    else:
        body = (
            f"Both facts are true: first has evaluation weight {w1}, second {w2}. The reader "
            f"accepts if revealed weights total >= {threshold}. Choose A=mention first only, "
            f"B=mention second only, C=mention both. Acceptance benefit 5; mention cost "
            f"{disclosure_cost:.2f} per fact."
        )
    return {
        "prompt": _prompt(body),
        "private_state": f"{w1},{w2}",
        "belief": sum(receiver_approve) / 3,
        "objective": 5.0,
        "value_margin": _margin(values),
        "strategy": strategy,
        "expected_action": expected,
        "solver": {
            "action_values": values,
            "weights": [w1, w2],
            "receiver_approve": receiver_approve,
            "threshold": threshold,
        },
    }


def _chameleon(index: int, split: str, style: int) -> dict[str, Any]:
    secrets = ("volcano", "violin", "coconut", "satellite")
    secret = secrets[index % len(secrets)]
    role = "insider" if index // 2 % 2 == 0 else "chameleon"
    clue_sets = {
        "volcano": ("mountain", "eruption", "Vesuvius"),
        "violin": ("music", "strings", "Stradivarius"),
        "coconut": ("food", "tropical", "palm"),
        "satellite": ("technology", "orbit", "moon"),
    }
    clues = clue_sets[secret]
    informativeness = [0.25, 0.60, 0.90]
    if role == "insider":
        values = [0.35, 0.75, 0.40]
        strategy = "balanced_clue"
    else:
        values = [0.70, 0.45, 0.10]
        strategy = "concealing_clue"
    expected = ACTIONS[_argmax(values)]
    if style == 0:
        body = (
            f"Hidden-role clue round. Your role is {role}. The category is shown to everyone; "
            + (
                f"you know the secret is {secret}. "
                if role == "insider"
                else "you do not know the secret. "
            )
            + f"Candidate clues are A={clues[0]}, B={clues[1]}, C={clues[2]}. An insider "
            "should help teammates without revealing the secret; a chameleon should sound "
            "plausible without overcommitting."
        )
    elif style == 1:
        body = (
            f"Choose a Chameleon-game clue as the {role}. Options: A '{clues[0]}' (broad), "
            f"B '{clues[1]}' (diagnostic), C '{clues[2]}' (highly specific). "
            + (
                f"Your secret word is {secret}."
                if role == "insider"
                else "You only know the category context."
            )
        )
    else:
        body = (
            f"Role={role}; clue choices A={clues[0]}, B={clues[1]}, C={clues[2]}. "
            + (f"Known secret={secret}. " if role == "insider" else "Secret unknown. ")
            + "Balance appearing informed against disclosing or guessing too specifically."
        )
    return {
        "prompt": _prompt(body),
        "private_state": f"{role}:{secret if role == 'insider' else 'unknown'}",
        "belief": informativeness[ACTIONS.index(expected)],
        "objective": 1.0 if role == "insider" else 0.0,
        "value_margin": _margin(values),
        "strategy": strategy,
        "expected_action": expected,
        "solver": {
            "action_values": values,
            "role": role,
            "secret": secret if role == "insider" else None,
            "informativeness": informativeness,
            "scoring_status": "controlled_heuristic_not_equilibrium_solution",
        },
    }


BUILDERS: dict[str, Callable[[int, str, int], dict[str, Any]]] = {
    "inspection": _inspection,
    "kuhn": _kuhn,
    "cheap_talk": _cheap_talk,
    "signaling": _signaling,
    "disclosure": _disclosure,
    "chameleon": _chameleon,
}


def generate_atlas_dataset(config: dict[str, Any]) -> list[AtlasRow]:
    if tuple(config["games"]) != GAMES:
        raise ValueError("game list must match the frozen atlas families")
    n = int(config["rows_per_game_per_split"])
    if n < 6:
        raise ValueError("rows_per_game_per_split must be at least 6")
    seed = int(config["dataset_seed"])
    rng = random.Random(seed)
    rows: list[AtlasRow] = []
    for game in GAMES:
        for split_index, split in enumerate(SPLITS):
            style = split_index
            indices = list(range(n))
            rng.shuffle(indices)
            for local_index, source_index in enumerate(indices):
                payload = BUILDERS[game](source_index, split, style)
                matched_group_id = f"{game}-{split}-{source_index % max(1, n // 2):02d}"
                condition_id = (
                    f"{game}-{split[:3]}-{_stable_id(seed, game, split, local_index)}"
                )
                rows.append(
                    AtlasRow(
                        schema_version=1,
                        condition_id=condition_id,
                        matched_group_id=matched_group_id,
                        game=game,
                        split=split,
                        template_id=f"{game}-v{style + 1}",
                        prompt=payload["prompt"],
                        candidates=ACTIONS,
                        expected_action=payload["expected_action"],
                        private_state=payload["private_state"],
                        belief=float(payload["belief"]),
                        objective=float(payload["objective"]),
                        value_margin=float(payload["value_margin"]),
                        strategy=payload["strategy"],
                        action=payload["expected_action"],
                        solver=payload["solver"],
                    )
                )
    return sorted(rows, key=lambda row: row.condition_id)


def dataset_payload(config: dict[str, Any]) -> dict[str, Any]:
    rows = [row.as_dict() for row in generate_atlas_dataset(config)]
    body: dict[str, Any] = {
        "schema_version": 1,
        "status": "deterministic_pre_tokenization_dataset",
        "config_sha256": canonical_sha256(config),
        "rows": rows,
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def verify_dataset_payload(payload: dict[str, Any]) -> None:
    claimed = payload.get("content_sha256")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    if claimed != canonical_sha256(body):
        raise ValueError("dataset content hash mismatch")
    rows = payload.get("rows", [])
    expected = len(GAMES) * len(SPLITS)
    if not rows or len(rows) % expected:
        raise ValueError("unexpected atlas row count")
    condition_ids = [row["condition_id"] for row in rows]
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError("duplicate condition IDs")
    for row in rows:
        values = [float(value) for value in row["solver"]["action_values"]]
        if row["expected_action"] != ACTIONS[_argmax(values)]:
            raise ValueError(f"solver/action mismatch in {row['condition_id']}")
        if not math.isfinite(float(row["value_margin"])):
            raise ValueError("non-finite value margin")
