#!/usr/bin/python3
"""Independent Path A constructor for the closed K-F1D synthetic world."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import unicodedata
from typing import Any, Iterable


TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")


def canonical(value: Any) -> bytes:
    def nfc(item: Any) -> Any:
        if isinstance(item, str):
            return unicodedata.normalize("NFC", item)
        if isinstance(item, list):
            return [nfc(value) for value in item]
        if isinstance(item, dict):
            return {nfc(key): nfc(value) for key, value in item.items()}
        return item

    return json.dumps(nfc(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest_record(fields: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(fields) + b"\n").hexdigest()


def sealed(fields: dict[str, Any]) -> dict[str, Any]:
    if "payload_sha256" in fields:
        raise ValueError("payload_sha256 supplied by caller")
    return fields | {"payload_sha256": digest_record(fields)}


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical(row) + b"\n")


def render_reference(family: dict[str, Any], blueprint: dict[str, Any], scenario: str, level: str, applicability: str, surface: str) -> str:
    source = family
    actor = family["actor"]
    obj = family["object"]
    mission = family["mission"]
    verb = family["aligned"]
    domain = family["domain"]
    if applicability == "ADJACENT":
        obj = family["adjacent_object"]
        mission = family["adjacent_mission"]
    elif applicability == "WRONG_MISSION":
        source = blueprint["wrong_mission"]
        actor, obj, mission, verb, domain = (source[key] for key in ("actor", "object", "mission", "aligned", "domain"))
    level_phrase = blueprint["level_phrases"][level]
    if surface == "CANONICAL":
        return f"{level_phrase} in {domain} {scenario}: {actor} must {verb} {obj} to uphold {mission}."
    if surface == "BOUNDED_PARAPHRASE":
        use_verb = "steward" if applicability == "APPLICABLE" else verb
        return f"To uphold {mission} inside {scenario} {domain}, the {level_phrase} directs {actor} to {use_verb} {obj}."
    if surface == "CLAUSE_ORDER":
        return f"Mission {mission}; within {scenario} {domain}, {actor} must {verb} {obj} under the {level_phrase}."
    raise ValueError(f"unknown surface {surface}")


def render_action(family: dict[str, Any], blueprint: dict[str, Any], scenario: str, polarity: str, magnitude: int, surface: str) -> str:
    actor, obj, mission = family["actor"], family["object"], family["mission"]
    verb = family["aligned"] if polarity == "ALIGNED" else family["opposed"]
    tier = blueprint["magnitude_phrases"][str(magnitude)]
    if surface == "CANONICAL":
        modifier = "actively" if polarity == "ALIGNED" else "not"
        consequence = "uphold" if polarity == "ALIGNED" else "oppose"
        return f"{actor} will {modifier} {family['aligned']} {obj} and thereby {consequence} {mission} at consequence {tier} in {scenario}."
    if surface == "BOUNDED_PARAPHRASE":
        consequence = "supported" if polarity == "ALIGNED" else "obstructed"
        return f"At consequence {tier} in {scenario}, {actor} plans to {verb} {obj} so {mission} becomes {consequence}."
    if surface == "CLAUSE_ORDER":
        return f"Mission {mission} at consequence {tier}; within {scenario}, {actor} will deliberately {verb} {obj}."
    raise ValueError(f"unknown surface {surface}")


def construct(blueprint: dict[str, Any], contract: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    families = blueprint["families"]
    if len(families) != 12 or [row["split"] for row in families].count("CALIBRATION") != 4:
        raise ValueError("exact 4 calibration plus 8 sealed families required")
    surfaces = contract["surface_classes"]
    levels = contract["mission_levels"]
    join_salt = hashlib.sha256(b"SYSTEM-K-K-F1D|41384001|JOIN|V0.1\n").hexdigest()
    output = {name: [] for name in (
        "families", "pair_ir", "surfaces", "blinded", "canonical_gold", "surface_proofs",
        "calibration_gold", "sealed_gold", "calibration_join", "sealed_join",
    )}
    pair_ordinal = surface_ordinal = 0
    for family in families:
        output["families"].append(family)
        for scenario_index, scenario in enumerate(blueprint["scenario_suffixes"], start=1):
            scenario_id = f"{family['family_id']}_S{scenario_index}"
            for level in levels:
                context = f"{scenario_id}_{level}"
                definitions: list[dict[str, Any]] = []
                direction_group = f"{context}_DIR"
                for polarity in ("ALIGNED", "OPPOSED"):
                    definitions.append({
                        "panel": "DIRECTION", "group_id": direction_group,
                        "applicability_class": "APPLICABLE", "applicability_bit": 1,
                        "relation_polarity": polarity, "magnitude_level": 2,
                        "control_tags": ["TOPICAL_OPPOSITION", "EXPLICIT_NEGATION", "TEXT_LENGTH_MATCHED"],
                    })
                proximity_group = f"{context}_PROX"
                for applicability in ("APPLICABLE", "ADJACENT", "WRONG_MISSION"):
                    definitions.append({
                        "panel": "PROXIMITY", "group_id": proximity_group,
                        "applicability_class": applicability,
                        "applicability_bit": int(applicability == "APPLICABLE"),
                        "relation_polarity": "ALIGNED", "magnitude_level": 2,
                        "control_tags": ["UNUSUAL_BUT_MISSION_SERVING", "WRONG_MISSION", "DIRECTION_HELD"],
                    })
                for polarity in ("ALIGNED", "OPPOSED"):
                    magnitude_group = f"{context}_MAG_{polarity}"
                    for magnitude in (1, 2, 3, 4):
                        definitions.append({
                            "panel": "MAGNITUDE", "group_id": magnitude_group,
                            "applicability_class": "APPLICABLE", "applicability_bit": 1,
                            "relation_polarity": polarity, "magnitude_level": magnitude,
                            "control_tags": ["QUANTITY_LADDER", "TEXT_LENGTH_MATCHED", "DIRECTION_HELD"],
                        })
                if len(definitions) != 13:
                    raise AssertionError("context pair census")
                for definition in definitions:
                    pair_ordinal += 1
                    pair_id = f"KFD_PAIR_{pair_ordinal:04d}"
                    pair_ir = {
                        "schema_version": "K_F1D_PAIR_IR_V0_1",
                        "pair_id": pair_id,
                        "family_id": family["family_id"],
                        "split": family["split"],
                        "scenario_id": scenario_id,
                        "scenario_phrase": scenario,
                        "level": level,
                        **definition,
                    }
                    output["pair_ir"].append(pair_ir)
                    pair_ir_sha = digest_record(pair_ir)
                    canonical_gold = sealed({key: value for key, value in pair_ir.items() if key != "scenario_phrase"} | {
                        "schema_version": "K_F1D_CANONICAL_GOLD_V0_1",
                        "pair_ir_sha256": pair_ir_sha,
                    })
                    output["canonical_gold"].append(canonical_gold)
                    for surface in surfaces:
                        surface_ordinal += 1
                        surface_id = f"KFD_SURFACE_{surface_ordinal:04d}"
                        action = render_action(family, blueprint, scenario, definition["relation_polarity"], definition["magnitude_level"], surface)
                        reference = render_reference(family, blueprint, scenario, level, definition["applicability_class"], surface)
                        surface_row = {
                            "schema_version": "K_F1D_RENDERED_SURFACE_V0_1",
                            "surface_id": surface_id,
                            "pair_id": pair_id,
                            "surface_class": surface,
                            "action_text": action,
                            "reference_text": reference,
                        }
                        output["surfaces"].append(surface_row)
                        rendered_sha = hashlib.sha256((action + "\n" + reference + "\n").encode()).hexdigest()
                        proof = sealed({
                            "schema_version": "K_F1D_SURFACE_PROOF_V0_1",
                            "surface_id": surface_id,
                            "pair_id": pair_id,
                            "surface_class": surface,
                            "canonical_gold_payload_sha256": canonical_gold["payload_sha256"],
                            "rendered_text_sha256": rendered_sha,
                            "semantics_preserved_bit": 1,
                        })
                        output["surface_proofs"].append(proof)
                        gold = sealed({key: value for key, value in pair_ir.items() if key not in {"schema_version", "scenario_phrase"}} | {
                            "schema_version": "K_F1D_SURFACE_GOLD_V0_1",
                            "surface_id": surface_id,
                            "surface_class": surface,
                            "action_token_count": len(TOKEN.findall(action)),
                            "reference_token_count": len(TOKEN.findall(reference)),
                            "canonical_gold_payload_sha256": canonical_gold["payload_sha256"],
                            "surface_proof_payload_sha256": proof["payload_sha256"],
                        })
                        blind_pair_id = hashlib.sha256(f"{join_salt}|{pair_id}".encode()).hexdigest()
                        blind_surface_id = hashlib.sha256(f"{join_salt}|{surface_id}".encode()).hexdigest()
                        blind_group_id = hashlib.sha256(f"{join_salt}|{definition['group_id']}".encode()).hexdigest()
                        output["blinded"].append({
                            "schema_version": "K_F1D_BLINDED_INPUT_V0_1",
                            "blind_pair_id": blind_pair_id,
                            "blind_surface_id": blind_surface_id,
                            "blind_group_id": blind_group_id,
                            "action_text": action,
                            "reference_text": reference,
                            "action_text_sha256": hashlib.sha256(action.encode()).hexdigest(),
                            "reference_text_sha256": hashlib.sha256(reference.encode()).hexdigest(),
                        })
                        join = sealed({
                            "schema_version": "K_F1D_JOIN_MAP_V0_1",
                            "blind_pair_id": blind_pair_id,
                            "blind_surface_id": blind_surface_id,
                            "pair_id": pair_id,
                            "surface_id": surface_id,
                            "surface_gold_payload_sha256": gold["payload_sha256"],
                            "join_salt_version_sha256": join_salt,
                        })
                        target = "calibration" if family["split"] == "CALIBRATION" else "sealed"
                        output[f"{target}_gold"].append(gold)
                        output[f"{target}_join"].append(join)
    expected = contract["counts"]
    observed = {
        "canonical_pairs": len(output["pair_ir"]),
        "rendered_surfaces": len(output["surfaces"]),
        "blinded_records": len(output["blinded"]),
        "calibration_gold_records": len(output["calibration_gold"]),
        "sealed_gold_records": len(output["sealed_gold"]),
        "calibration_join_records": len(output["calibration_join"]),
        "sealed_join_records": len(output["sealed_join"]),
    }
    if observed != {key: expected[key] for key in observed}:
        raise ValueError(f"census mismatch: {observed}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blueprint", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--primitive-dir", required=True)
    parser.add_argument("--record-dir", required=True)
    args = parser.parse_args()
    blueprint = json.loads(pathlib.Path(args.blueprint).read_text(encoding="utf-8"))
    contract = json.loads(pathlib.Path(args.contract).read_text(encoding="utf-8"))
    rows = construct(blueprint, contract)
    primitive = pathlib.Path(args.primitive_dir)
    records = pathlib.Path(args.record_dir)
    for name, filename in {
        "families": "mission-families.jsonl",
        "pair_ir": "pair-ir.jsonl",
        "surfaces": "rendered-surfaces.jsonl",
        "blinded": "blinded-scoring-input.jsonl",
    }.items():
        write_jsonl(primitive / filename, rows[name])
    for name, filename in {
        "canonical_gold": "canonical-gold.jsonl",
        "surface_proofs": "surface-inheritance-proof.jsonl",
        "calibration_gold": "calibration-gold.jsonl",
        "sealed_gold": "sealed-gold.jsonl",
        "calibration_join": "calibration-join.jsonl",
        "sealed_join": "sealed-join.jsonl",
    }.items():
        write_jsonl(records / filename, rows[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
