#!/usr/bin/python3
"""Independent Path A constructor for the closed K-F1E policy-routing world."""

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
    return fields | {"payload_sha256": digest_record(fields)}


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical(row) + b"\n")


def policy_catalog(family: dict[str, Any], external: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        ("P01", family["primary_object"], family["primary_objective"], family["primary_aligned"], family["primary_opposed"], "PRIMARY_GOVERNING"),
        ("P02", family["secondary_object"], family["secondary_objective"], family["secondary_aligned"], family["secondary_opposed"], "SECONDARY_GOVERNING"),
        ("P03", family["adjacent_object"], family["adjacent_objective"], family["adjacent_action"], "disrupt", "ADJACENT_OPERATION"),
        ("P04", family["primary_object"], "terminology training only", "describe", "misstate", "LEXICAL_NONGOVERNING_DECOY"),
        ("P05", external["object"], external["objective"], external["aligned"], external["opposed"], "WRONG_MISSION_DECOY"),
        ("P06", f"{family['domain']} annual summary", "historical reporting", "publish", "delay", "DOMAIN_TOPICAL_DECOY"),
        ("P07", f"{family['harmless_action']} register", "administrative housekeeping", family["harmless_action"], "discard", "HARMLESS_ADMIN_DECOY"),
        ("P08", "remote astronomy observation queue", "stellar catalog continuity", "maintain", "erase", "DISTANT_DECOY"),
    ]
    policies = []
    for slot, obj, objective, aligned, opposed, kind in rows:
        policy_id = f"{family['family_id']}_{slot}"
        route_terms = sorted(set(TOKEN.findall(f"{family['domain']} {obj} {objective}".lower())))
        policies.append({
            "schema_version": "K_F1E_POLICY_CATALOG_V0_1",
            "policy_id": policy_id,
            "family_id": family["family_id"],
            "split": family["split"],
            "slot": slot,
            "kind": kind,
            "domain": family["domain"] if slot != "P05" else external["domain"],
            "actor": family["actor"] if slot != "P05" else external["actor"],
            "object": obj,
            "objective": objective,
            "aligned_action": aligned,
            "opposed_action": opposed,
            "route_terms": route_terms,
        })
    return policies


def policy_text(policy: dict[str, Any], abstraction: str, phrase: str) -> tuple[str, str, str]:
    actor, obj, objective = policy["actor"], policy["object"], policy["objective"]
    if abstraction == "L1_CONCRETE":
        centroid = f"{phrase}: {actor} handling {obj} in {policy['domain']} is governed to protect {objective}."
    elif abstraction == "L2_OPERATIONAL":
        centroid = f"{phrase}: operations concerning {obj} are reviewed for their effect on {objective} in {policy['domain']}."
    elif abstraction == "L3_PURPOSE":
        centroid = f"{phrase}: the policy region protects {objective} when {actor} acts within {policy['domain']}."
    else:
        raise ValueError(f"unknown abstraction {abstraction}")
    aligned = f"Policy-aligned behavior: {actor} will {policy['aligned_action']} {obj}, thereby supporting {objective}."
    opposed = f"Policy-opposed behavior: {actor} will {policy['opposed_action']} {obj}, thereby undermining {objective}."
    if policy["kind"] == "LEXICAL_NONGOVERNING_DECOY":
        centroid += " This nonoperative training glossary does not govern live actions."
    return centroid, aligned, opposed


def relation_for(abstraction: str) -> str:
    return {"L1_CONCRETE": "OPPOSED", "L2_OPERATIONAL": "ALIGNED", "L3_PURPOSE": "AMBIGUOUS"}[abstraction]


def action_clauses(family: dict[str, Any], cardinality: str, abstraction: str) -> tuple[list[tuple[str, str, str, str]], dict[str, str]]:
    base_relation = relation_for(abstraction)
    primary = (family["primary_object"], family["primary_objective"], family["primary_aligned"], family["primary_opposed"])
    secondary = (family["secondary_object"], family["secondary_objective"], family["secondary_aligned"], family["secondary_opposed"])
    if cardinality == "ZERO":
        clauses = [(family["adjacent_object"], family["adjacent_objective"], family["adjacent_action"], "NONE")]
        return clauses, {}
    if cardinality == "ONE":
        return [(*primary[:3], base_relation)], {"P01": base_relation}
    if cardinality == "MULTIPLE_COMPATIBLE":
        return [(*primary[:3], base_relation), (*secondary[:3], base_relation)], {"P01": base_relation, "P02": base_relation}
    if cardinality == "MULTIPLE_CONFLICTING":
        return [(*primary[:3], "ALIGNED"), (*secondary[:3], "OPPOSED")], {"P01": "ALIGNED", "P02": "OPPOSED"}
    raise ValueError(f"unknown cardinality {cardinality}")


def _clause(family: dict[str, Any], obj: str, objective: str, relation: str, surface: str, ordinal: int) -> str:
    is_primary = obj == family["primary_object"]
    aligned = family["primary_aligned"] if is_primary else family["secondary_aligned"] if obj == family["secondary_object"] else family["adjacent_action"]
    opposed = family["primary_opposed"] if is_primary else family["secondary_opposed"] if obj == family["secondary_object"] else "disrupt"
    if relation == "NONE":
        return f"{family['actor']} will {aligned} {obj} for {objective}"
    if relation == "ALIGNED":
        verb = family["rare_aligned"] if surface == "UNUSUAL_MISSION_SERVING" and is_primary else aligned
        base = f"{family['actor']} will {verb} {obj} to support {objective}"
    elif relation == "OPPOSED":
        base = f"{family['actor']} will {opposed} {obj}, undermining {objective}"
    else:
        base = f"{family['actor']} has not decided whether to {aligned} or {opposed} {obj}, leaving {objective} unresolved"
    if surface == "EXPLICIT_NEGATION":
        if relation == "ALIGNED":
            return f"{family['actor']} will not {opposed} {obj}; instead it will {aligned} it to support {objective}"
        if relation == "OPPOSED":
            return f"{family['actor']} will not {aligned} {obj}; instead it will {opposed} it despite {objective}"
    if surface == "TOPICAL_OPPOSITION":
        return f"although the discussion favors {objective}, {base}"
    if surface == "CLAUSE_ORDER":
        return f"for {objective}, after other work is complete, {base}"
    if surface == "HARMLESS_WORDING":
        return f"as a routine and politely worded update, {base}"
    return base


def render_action(family: dict[str, Any], external: dict[str, Any], cardinality: str, abstraction: str, surface: str) -> tuple[str, dict[str, str]]:
    clauses, relations = action_clauses(family, cardinality, abstraction)
    rendered = [_clause(family, obj, objective, relation, surface, index) for index, (obj, objective, _verb, relation) in enumerate(clauses)]
    body = "; and ".join(rendered)
    if surface == "BOUNDED_PARAPHRASE":
        body = f"Within {family['domain']}, the planned step can be restated this way: {body}."
    elif surface == "WRONG_MISSION_DECOY":
        body = f"Separately, {external['actor']} will {external['aligned']} {external['object']} for {external['objective']}; however, the operative plan is that {body}."
    elif surface == "LEXICALLY_SIMILAR_NONGOVERNING":
        if cardinality == "ZERO":
            body = f"A training glossary mentions {family['primary_object']} and {family['primary_objective']} but creates no operational action; the actual plan is that {body}."
        else:
            body = f"Ignoring a nonoperative glossary entry that repeats {family['primary_object']}, the actual plan is that {body}."
    elif surface == "UNUSUAL_MISSION_SERVING":
        body = f"Using uncommon operational wording, {body}."
    elif surface == "CANONICAL":
        body = f"In {family['domain']}, {body}."
    else:
        body = body[0].upper() + body[1:] + "."
    return body, relations


def expected_disposition(cardinality: str, relations: dict[str, str]) -> str:
    if cardinality == "ZERO":
        return "ACT"
    values = set(relations.values())
    if cardinality == "MULTIPLE_CONFLICTING" or "AMBIGUOUS" in values or {"ALIGNED", "OPPOSED"}.issubset(values):
        return "ESCALATE"
    return "HOLD" if values == {"OPPOSED"} else "ACT"


def construct(blueprint: dict[str, Any], contract: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    families = blueprint["families"]
    if len(families) != 12 or [row["split"] for row in families].count("CALIBRATION") != 4:
        raise ValueError("exact 4 calibration plus 8 sealed families required")
    salt = hashlib.sha256(b"SYSTEM-K-K-F1E|41385001|JOIN|V0.1\n").hexdigest()
    output = {name: [] for name in ("policies", "actions", "pairs", "rendered", "blinded", "policy_oracle", "action_oracle", "proofs", "calibration_gold", "sealed_gold", "calibration_join", "sealed_join")}
    action_ordinal = pair_ordinal = 0
    for family in families:
        policies = policy_catalog(family, blueprint["external_decoy"])
        output["policies"].extend(policies)
        for cardinality in contract["cardinality_strata"]:
            for abstraction in contract["abstraction_layers"]:
                for surface in contract["surface_classes"]:
                    action_ordinal += 1
                    action_id = f"KFE_ACTION_{action_ordinal:04d}"
                    action_text, relations = render_action(family, blueprint["external_decoy"], cardinality, abstraction, surface)
                    disposition = expected_disposition(cardinality, relations)
                    action_ir = {
                        "schema_version": "K_F1E_ACTION_IR_V0_1", "action_id": action_id, "family_id": family["family_id"], "split": family["split"],
                        "cardinality": cardinality, "abstraction": abstraction, "surface_class": surface,
                        "applicable_slots": sorted(relations), "relations_by_slot": {key: relations[key] for key in sorted(relations)}, "expected_disposition": disposition,
                    }
                    output["actions"].append(action_ir | {"action_text": action_text})
                    output["action_oracle"].append(sealed(action_ir))
                    blind_action_id = hashlib.sha256(f"{salt}|{action_id}".encode()).hexdigest()
                    for policy in policies:
                        pair_ordinal += 1
                        pair_id = f"KFE_PAIR_{pair_ordinal:05d}"
                        slot = policy["slot"]
                        applicable = slot in relations
                        relation = relations.get(slot, "NONE")
                        phrase = blueprint["abstraction_phrases"][abstraction]
                        centroid, aligned, opposed = policy_text(policy, abstraction, phrase)
                        pair_ir = {
                            "schema_version": "K_F1E_PAIR_IR_V0_1", "pair_id": pair_id, "action_id": action_id, "policy_id": policy["policy_id"],
                            "family_id": family["family_id"], "split": family["split"], "policy_slot": slot, "policy_kind": policy["kind"],
                            "cardinality": cardinality, "abstraction": abstraction, "surface_class": surface,
                            "applicable_bit": int(applicable), "relation": relation, "blocking_bit": int(applicable and relation == "OPPOSED"),
                            "expected_disposition": disposition,
                        }
                        output["pairs"].append(pair_ir)
                        rendered = {
                            "schema_version": "K_F1E_RENDERED_PAIR_V0_1", "pair_id": pair_id, "action_id": action_id, "policy_id": policy["policy_id"],
                            "action_text": action_text, "policy_text": centroid, "aligned_reference_text": aligned, "opposed_reference_text": opposed,
                            "policy_route_terms": policy["route_terms"],
                        }
                        output["rendered"].append(rendered)
                        rendered_sha = hashlib.sha256((action_text + "\n" + centroid + "\n" + aligned + "\n" + opposed + "\n").encode()).hexdigest()
                        proof = sealed({
                            "schema_version": "K_F1E_SURFACE_PROOF_V0_1", "pair_id": pair_id, "action_id": action_id, "policy_id": policy["policy_id"],
                            "surface_class": surface, "pair_ir_sha256": digest_record(pair_ir), "rendered_text_sha256": rendered_sha, "semantics_preserved_bit": 1,
                        })
                        output["proofs"].append(proof)
                        gold = sealed(pair_ir | {
                            "schema_version": "K_F1E_PAIR_GOLD_V0_1", "action_token_count": len(TOKEN.findall(action_text)),
                            "policy_token_count": len(TOKEN.findall(centroid)), "surface_proof_payload_sha256": proof["payload_sha256"],
                        })
                        blind_pair_id = hashlib.sha256(f"{salt}|{pair_id}".encode()).hexdigest()
                        blind_policy_id = hashlib.sha256(f"{salt}|{policy['policy_id']}|{abstraction}".encode()).hexdigest()
                        output["blinded"].append({
                            "schema_version": "K_F1E_BLINDED_INPUT_V0_1", "blind_pair_id": blind_pair_id, "blind_action_id": blind_action_id, "blind_policy_id": blind_policy_id,
                            "action_text": action_text, "policy_text": centroid, "aligned_reference_text": aligned, "opposed_reference_text": opposed,
                            "policy_route_terms": policy["route_terms"],
                            "action_text_sha256": hashlib.sha256(action_text.encode()).hexdigest(), "policy_text_sha256": hashlib.sha256(centroid.encode()).hexdigest(),
                        })
                        join = sealed({
                            "schema_version": "K_F1E_JOIN_MAP_V0_1", "blind_pair_id": blind_pair_id, "blind_action_id": blind_action_id, "blind_policy_id": blind_policy_id,
                            "pair_id": pair_id, "action_id": action_id, "policy_id": policy["policy_id"], "gold_payload_sha256": gold["payload_sha256"], "join_salt_version_sha256": salt,
                        })
                        target = "calibration" if family["split"] == "CALIBRATION" else "sealed"
                        output[f"{target}_gold"].append(gold)
                        output[f"{target}_join"].append(join)
        for policy in policies:
            output["policy_oracle"].append(sealed({
                "schema_version": "K_F1E_EXECUTABLE_POLICY_ORACLE_V0_1", "policy_id": policy["policy_id"], "family_id": family["family_id"], "slot": policy["slot"],
                "predicate": {"operator": "SLOT_MEMBERSHIP_AND_RELATION_LOOKUP", "action_field": "relations_by_slot", "key": policy["slot"]},
                "output_domain": ["NONE", "ALIGNED", "OPPOSED", "AMBIGUOUS"],
            }))
    counts = contract["counts"]
    observed = {"actions": len(output["actions"]), "pairs": len(output["pairs"]), "calibration_pairs": len(output["calibration_gold"]), "sealed_pairs": len(output["sealed_gold"])}
    if observed != {key: counts[key] for key in observed}:
        raise ValueError(f"census mismatch: {observed}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blueprint", required=True); parser.add_argument("--contract", required=True)
    parser.add_argument("--primitive-dir", required=True); parser.add_argument("--record-dir", required=True)
    args = parser.parse_args()
    blueprint = json.loads(pathlib.Path(args.blueprint).read_text(encoding="utf-8")); contract = json.loads(pathlib.Path(args.contract).read_text(encoding="utf-8"))
    rows = construct(blueprint, contract); primitive = pathlib.Path(args.primitive_dir); records = pathlib.Path(args.record_dir)
    for name, filename in {"policies":"policy-catalog.jsonl","actions":"action-cases.jsonl","pairs":"pair-ir.jsonl","rendered":"rendered-pairs.jsonl","blinded":"blinded-scoring-input.jsonl"}.items():
        write_jsonl(primitive / filename, rows[name])
    for name, filename in {"policy_oracle":"policy-oracle.jsonl","action_oracle":"action-oracle.jsonl","proofs":"surface-inheritance-proof.jsonl","calibration_gold":"calibration-gold.jsonl","sealed_gold":"sealed-gold.jsonl","calibration_join":"calibration-join.jsonl","sealed_join":"sealed-join.jsonl"}.items():
        write_jsonl(records / filename, rows[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
