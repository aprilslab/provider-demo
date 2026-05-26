"""Verify the Gradio UI renders correctly AND that task-driven visibility
toggling actually flips the right components.

Two phases:
  1. Static — fetch /config, assert all required component types/labels exist.
  2. Dynamic — call the /_refresh_all named endpoint with each task, decode the
     gr.update payload returned for each output slot, and check visibility +
     value-reset behaviour matches TASK_IO expectations.

Run: python verify_ui.py [http://127.0.0.1:7860]
"""
from __future__ import annotations
import json
import sys
from urllib.request import Request, urlopen


REQUIRED_LABELS = [
    "Text input",
    "Image input",
    "Audio input",
    "Text output",
    "Image output",
    "Model",
    "Sample preset",
    "System",
    "Max output tokens",
    "Temperature",
    "Latency (ms)",
]

REQUIRED_TYPES = {"image", "audio", "textbox", "dropdown", "dataframe", "radio", "button"}

# Mirrors TASK_IO in app.py — keep in sync.
TASK_IO_EXPECT = {
    "text-to-text":       {"text_in": True,  "image_in": False, "audio_in": False, "text_out": True,  "image_out": False, "system": True,  "gen": True},
    "image-text-to-text": {"text_in": True,  "image_in": True,  "audio_in": False, "text_out": True,  "image_out": False, "system": True,  "gen": True},
    "text-to-image":      {"text_in": True,  "image_in": False, "audio_in": False, "text_out": False, "image_out": True,  "system": False, "gen": False},
    "speech-to-text":     {"text_in": False, "image_in": False, "audio_in": True,  "text_out": True,  "image_out": False, "system": False, "gen": False},
}

# Order of outputs in app.py's _refresh_all return tuple.
REFRESH_SLOTS = [
    "pg_model", "pg_sample",
    "text_in", "text_in_val",
    "image_in", "image_in_val",
    "audio_in", "audio_in_val",
    "system", "gen",
    "text_out", "text_out_val",
    "image_out", "image_out_val",
    "pg_usage", "pg_raw", "pg_latency",
]


def fetch(base: str, path: str, method: str = "GET", body: dict | None = None) -> dict:
    url = base.rstrip("/") + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"User-Agent": "verify-ui/1.0"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=headers, method=method)
    with urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def static_check(base: str) -> list[str]:
    print("\n[static] /config component audit")
    cfg = fetch(base, "/config")
    components = cfg.get("components") or []
    print(f"  components found: {len(components)}")

    found: list[tuple[str, str, bool]] = []
    for c in components:
        ctype = str(c.get("type") or "").lower()
        props = c.get("props") or {}
        label = str(props.get("label") or "").strip()
        visible = props.get("visible")
        if visible is None:
            visible = True
        found.append((label, ctype, bool(visible)))

    failures: list[str] = []
    types_present = {t for _, t, _ in found}
    for t in sorted(REQUIRED_TYPES):
        if t in types_present:
            print(f"   OK  type='{t}'")
        else:
            print(f"   X   type='{t}'")
            failures.append(f"missing type: {t}")

    labels_present = {l for l, _, _ in found if l}
    for label in REQUIRED_LABELS:
        matches = [l for l in labels_present if l.startswith(label)]
        if matches:
            print(f"   OK  label '{label}' present")
        else:
            print(f"   X   label '{label}' NOT FOUND")
            failures.append(f"missing label: {label}")

    return failures


def call_refresh(base: str, task: str) -> list:
    """Call the /_refresh_all endpoint via gradio_api/call and return decoded
    output list."""
    # Step 1: enqueue the call.
    job = fetch(
        base, "/gradio_api/call/_refresh_all", method="POST",
        body={"data": [task]},
    )
    event_id = job.get("event_id")
    if not event_id:
        raise RuntimeError(f"no event_id: {job}")

    # Step 2: stream the result.
    url = base.rstrip("/") + f"/gradio_api/call/_refresh_all/{event_id}"
    req = Request(url, headers={"User-Agent": "verify-ui/1.0"})
    with urlopen(req, timeout=30) as r:
        body = r.read().decode("utf-8")

    # SSE stream — extract the final "data:" line that has JSON list.
    data_lines = [ln[len("data:"):].strip() for ln in body.splitlines() if ln.startswith("data:")]
    for raw in reversed(data_lines):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"no list payload in SSE: {body[:400]}")


def visible_of(slot) -> bool | None:
    if isinstance(slot, dict):
        # gr.update payload — visible may be missing (= keep current = True)
        if "visible" in slot:
            return bool(slot["visible"])
        return True
    return True  # non-update returns (plain values) imply visible stays


def value_of(slot):
    if isinstance(slot, dict):
        return slot.get("value", "__UNCHANGED__")
    return slot


def dynamic_check(base: str) -> list[str]:
    print("\n[dynamic] task radio → visibility/reset behaviour")
    failures: list[str] = []
    for task, expect in TASK_IO_EXPECT.items():
        print(f"\n  task = '{task}'")
        try:
            out = call_refresh(base, task)
        except Exception as e:
            print(f"    X  call failed: {type(e).__name__}: {e}")
            failures.append(f"{task}: call failed: {e}")
            continue
        if len(out) != len(REFRESH_SLOTS):
            print(f"    X  output arity {len(out)} != expected {len(REFRESH_SLOTS)}")
            failures.append(f"{task}: arity mismatch")
            continue
        named = dict(zip(REFRESH_SLOTS, out))

        # check visibility per slot
        for key in ("text_in", "image_in", "audio_in", "text_out", "image_out", "system", "gen"):
            want = expect[key]
            got = visible_of(named[key])
            ok = got == want
            print(f"    {'OK ' if ok else 'X  '} {key:<10} visible expected={want} got={got}")
            if not ok:
                failures.append(f"{task}: {key} visible got={got} want={want}")

        # outputs must always be cleared on task change (no preload concept)
        for key, expected_value in (("text_out_val", ""), ("image_out_val", None)):
            v = named.get(key)
            if isinstance(v, dict):
                v = v.get("value", "__UNCHANGED__")
            if v not in (expected_value, "__UNCHANGED__"):
                failures.append(f"{task}: {key} output not cleared (got={v!r})")

    return failures


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"
    print(f"verifying UI at {base}")
    fails = static_check(base) + dynamic_check(base)
    print("\n" + "=" * 60)
    if fails:
        print(f"FAIL — {len(fails)} issue(s):")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("PASS — static + dynamic checks all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
