"""TP-399: swap Gemini 2.5 -> 3.5 (same base) in AIFL agent config tables for ONE env.

Backs up every scanned table to JSON first. Dry run by default.
Usage: python tp399_dynamo_swap.py --env staging [--flash-thinking-budget N] [--apply]
"""
import argparse, json, os, datetime, boto3
from decimal import Decimal

MAP = {"gemini-2.5-flash-lite": "gemini-3.5-flash-lite", "gemini-2.5-flash": "gemini-3.5-flash"}
TABLES = [("truss-agent-configs-{env}", "config_id"), ("truss-annotation-classifier-configs-{env}", None)]

ap = argparse.ArgumentParser()
ap.add_argument("--env", required=True, choices=["dev", "staging", "prod"])
ap.add_argument("--flash-thinking-budget", type=int, default=None)
ap.add_argument("--apply", action="store_true")
ap.add_argument("--backup-dir", default=os.path.expanduser("~/Documents/TRUSS/backups/tp399"))
a = ap.parse_args()

ddb = boto3.resource("dynamodb", region_name="eu-west-2")
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def dflt(o):
    return float(o) if isinstance(o, Decimal) else str(o)

for tmpl, _ in TABLES:
    name = tmpl.format(env=a.env)
    t = ddb.Table(name)
    keys = [k["AttributeName"] for k in t.key_schema]
    resp = t.scan(); items = resp["Items"]
    while "LastEvaluatedKey" in resp:
        resp = t.scan(ExclusiveStartKey=resp["LastEvaluatedKey"]); items += resp["Items"]
    bk = os.path.join(a.backup_dir, f"backup_{name}_{ts}.json")
    json.dump(items, open(bk, "w"), default=dflt, indent=1)
    print(f"== {name}: {len(items)} items, backup {bk}")
    n = 0
    for it in items:
        # new table: model under model_config; legacy table: model at top level or under model_config
        for path in (("model_config", "model"), ("model",)):
            cur = it
            for p in path[:-1]:
                cur = cur.get(p) if isinstance(cur, dict) else None
            if not isinstance(cur, dict) or cur.get(path[-1]) not in MAP:
                continue
            old = cur[path[-1]]; new = MAP[old]
            key = {k: it[k] for k in keys}
            expr_names = {f"#p{i}": p for i, p in enumerate(path)}
            target = ".".join(f"#p{i}" for i in range(len(path)))
            upd = f"SET {target} = :m"
            vals = {":m": new}
            if new == "gemini-3.5-flash" and a.flash_thinking_budget is not None and path[0] == "model_config":
                upd += ", #p0.#tb = :tb"; expr_names["#tb"] = "thinking_budget"; vals[":tb"] = a.flash_thinking_budget
            print(f"  {key} {'.'.join(path)}: {old} -> {new}" + (f" (+thinking_budget={a.flash_thinking_budget})" if ":tb" in vals else ""))
            n += 1
            if a.apply:
                t.update_item(Key=key, UpdateExpression=upd, ExpressionAttributeNames=expr_names,
                              ExpressionAttributeValues={**vals, ":old": old}, ConditionExpression=f"{target} = :old")
    print(f"  {n} updates {'APPLIED' if a.apply else '(dry run)'}")
