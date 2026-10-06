#!/usr/bin/env python3
"""Compare two Gear JSON exports and write a readable HTML report.

    python scripts/gear_diff.py OLD.json NEW.json [-o diff.html] [--text changelog.txt] [--open]

The exports come from the Gear panel's "Export JSON" button or from
`quicktag.exe -v goliath <packages> --export-gear-json <file>`.

What the report takes care of:
  * Items are matched by `internal_hash` (the stable investment identity), so a
    package rebuild that renumbers every tag does not turn into "everything
    changed". Display hash and name are fallbacks only.
  * Changes are sorted into three kinds: real data, recomputed per-weapon mod
    numbers (which cascade whenever a weapon changes), and bare tag references.
    Only the first kind is shown by default.
  * Lists are compared by identity (rating id, effect name, weapon, slot), not
    by position, and float noise is ignored.
  * A field one export does not have at all (an older schema) is reported once
    instead of on every item.
  * Perk data (`perks`: scalars, labels, hop-on stat modifiers, constants) is
    compared by perk name, hop-on order, modifier stat index and constant
    position, so a retuned value reads as "Range +25 → +40", not as a new perk.
    Perk tags, effect indices and byte offsets count as tag references.
"""

import argparse
import difflib
import html
import json
import os
import re
import sys
import time
import webbrowser
from collections import Counter, defaultdict

TAG_KEYS = {
    "hash", "definition_tag", "icon_texture", "model_tag",
    # Perk data: tags, table positions and byte offsets renumber between builds.
    "perk_tag", "pattern", "component", "effect_index", "offset",
}
TAG_LISTS = {"detail_textures"}
RARITY_ORDER = ["Contraband", "Unique", "Prestige", "Superior", "Deluxe", "Enhanced", "Standard", "Quest", "Dynamic"]
RARITY_COLOR = {
    "Standard": "#9ea5ae", "Enhanced": "#52bf69", "Deluxe": "#4891e0", "Superior": "#a65cd6",
    "Prestige": "#e8b848", "Contraband": "#da4848", "Dynamic": "#48cdc2", "Quest": "#eccd5b",
    "Unique": "#a8d400",
}
STATUS_ORDER = ["added", "removed", "changed", "derived", "tags"]
STATUS_LABEL = {
    "added": "Added", "removed": "Removed", "changed": "Changed",
    "derived": "Recomputed only", "tags": "Tag refs only", "same": "Unchanged",
}
LABELS = {
    "rpm": "RPM", "ads": "ADS", "id": "ID", "html": "HTML",
}
PERK_KIND_SUFFIX = {"rating": " rating", "multiplier": "", "raw": ""}
UNITS = {"_seconds": "s", "_percent": "%", "_degrees": "°", "_metres": "m", "_rpm": "RPM"}
# Restated by another field of the same record.
REDUNDANT_KEYS = {"rarity_code"}
SECTION_ORDER = [
    "name", "rarity", "rarity_code", "item_type", "subcategory", "price", "buying_price",
    "description_text", "description", "weapon_stats", "base_ratings", "preset_mods",
    "compatible_mods", "mod_stats", "effects", "perks", "implant_stats", "implant_slot",
]


# ---------------------------------------------------------------- comparing

def empty(value):
    return value is None or value == [] or value == {} or value == ""


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def same_number(a, b):
    return abs(a - b) <= 1e-5 * max(1.0, abs(a), abs(b))


def freeze(value):
    """A hashable stand-in for a list element; plain values pass through."""
    return value if isinstance(value, (str, int, float, bool, type(None))) else json.dumps(value, sort_keys=True)


def humanize(key):
    key = str(key)
    if key == "description_text":
        return "Description"
    unit = ""
    for suffix, symbol in UNITS.items():
        if key.endswith(suffix):
            key, unit = key[: -len(suffix)], f" ({symbol})"
    words = [LABELS.get(word, word) for word in key.replace("_", " ").split()]
    text = " ".join(words) + unit
    return text[:1].upper() + text[1:]


def element_identity(list_key, element, position):
    """(matching key, label) of one element of a list of objects."""
    get = element.get
    if list_key == "effects":
        return get("name"), get("name") or "?"
    if list_key == "implant_stats":
        return get("raw_name_hash") or get("name"), get("name") or "?"
    if list_key in ("base_ratings", "ratings"):
        return get("rating_id"), get("name") or f"Rating {get('rating_id')}"
    if list_key == "preset_mods":
        if get("name"):
            label = f"{get('name')} ({get('rarity')})" if get("rarity") else get("name")
            return ("named", get("name"), get("rarity")), label
        return ("intrinsic",), "Intrinsic plug"
    if list_key == "compatible_mods":
        return get("slot"), get("slot") or "?"
    if list_key == "mods":
        label = f"{get('name')} ({get('rarity')})" if get("rarity") else str(get("name"))
        return (get("name"), get("rarity")), label
    if list_key == "weapons":
        return get("weapon"), get("weapon") or "?"
    if list_key == "changes":
        return get("name"), get("name") or "?"
    if list_key == "curves":
        key = (get("semantic"), get("occurrence"))
        return key, f"semantic {get('semantic')} / {get('occurrence')}"
    # Perk data. A perk is named by its authoring folder; its hop-ons have no
    # stable name, so they pair by order and are labelled by what they touch.
    if list_key == "perks":
        name = get("name") or ""
        # A perk without an authoring path is exported under its tag, which
        # renumbers between builds; such perks pair by order instead.
        if re.fullmatch(r"[0-9A-Fa-f]{8}", name):
            return ("unnamed",), "Unnamed perk"
        return name, name or "?"
    if list_key == "hop_ons":
        stats = [m.get("stat") or "?" for m in get("modifiers") or []]
        summary = ", ".join(stats[:3]) + (f" +{len(stats) - 3}" if len(stats) > 3 else "")
        return position, f"Hop-on {position + 1}" + (f" ({summary})" if summary else "")
    if list_key == "modifiers":
        return (get("index"), get("mode")), get("stat") or f"index {get('index')}"
    if list_key == "constants":
        return (get("after"), get("rel")), f"{get('after')} +{get('rel'):#x}" if get("rel") is not None else "?"
    stripped = {k: v for k, v in element.items() if k not in TAG_KEYS}
    return json.dumps(stripped, sort_keys=True), f"#{position + 1}"


def keyed(list_key, elements):
    """Elements by identity; repeated identities get a running number."""
    seen = Counter()
    out = {}
    for position, element in enumerate(elements):
        key, label = element_identity(list_key, element, position)
        seen[key] += 1
        if seen[key] > 1:
            label = f"{label} #{seen[key]}"
        out[(key, seen[key])] = (label, element)
    return out


def describe(value, limit=140):
    """One-line summary of a value that was added or removed whole."""
    if isinstance(value, dict) and {"stat", "kind", "value"} <= value.keys():
        return modifier_text(value)
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if key in TAG_KEYS or empty(item):
                continue
            if isinstance(item, list):
                parts.append(f"{humanize(key).lower()}: {len(item)}")
            elif isinstance(item, dict):
                continue
            else:
                parts.append(f"{humanize(key).lower()} {format_value(item)}")
        text = ", ".join(parts)
    else:
        text = format_value(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_number(value):
    if isinstance(value, int):
        return f"{value:,}"
    text = f"{value:,.4f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def format_value(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if is_number(value):
        return format_number(value)
    if isinstance(value, list):
        return "[" + ", ".join(format_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return describe(value)
    return str(value)


def category_of(schema_path):
    if schema_path[:2] == ("mod_stats", "weapons"):
        return "derived"
    if schema_path[-1] in TAG_KEYS or any(part in TAG_LISTS for part in schema_path):
        return "tags"
    return "data"


class Differ:
    def __init__(self, old_schema, new_schema):
        self.old_schema = old_schema
        self.new_schema = new_schema
        self.ignored = Counter()

    def diff(self, old, new):
        changes = []
        self._dict(old, new, (), [], changes)
        return changes

    def _emit(self, changes, schema_path, labels, kind, old=None, new=None, **extra):
        changes.append({
            "cat": category_of(schema_path),
            "top": schema_path[0],
            "leaf": schema_path[-1],
            "label": " › ".join(labels[1:-1] if len(labels) > 2 and schema_path[-1] == "value" else labels[1:]),
            "kind": kind, "old": old, "new": new, **extra,
        })

    def _dict(self, old, new, schema_path, labels, changes):
        old, new = old or {}, new or {}
        for key in list(old) + [key for key in new if key not in old]:
            path = schema_path + (key,)
            if path not in self.old_schema or path not in self.new_schema:
                self.ignored[(path, "old" if path not in self.old_schema else "new")] += 1
                continue
            if key in REDUNDANT_KEYS:
                continue
            a, b = old.get(key), new.get(key)
            # The markup field repeats the plain text; show it only when the
            # colouring alone changed.
            if key == "description" and old.get("description_text") != new.get("description_text"):
                if "description_text" in old or "description_text" in new:
                    continue
            # A perk's containers add nothing to "Perks › name › Hop-on 1 › Range".
            step = [] if schema_path[:1] == ("perks",) and key in ("hop_ons", "modifiers") else [humanize(key)]
            self._value(a, b, path, labels + step, changes)

    def _value(self, a, b, schema_path, labels, changes):
        if empty(a) and empty(b):
            return
        if isinstance(a, dict) or isinstance(b, dict):
            if (a is None or isinstance(a, dict)) and (b is None or isinstance(b, dict)):
                return self._dict(a, b, schema_path, labels, changes)
        if isinstance(a, list) or isinstance(b, list):
            if (a is None or isinstance(a, list)) and (b is None or isinstance(b, list)):
                return self._list(a or [], b or [], schema_path, labels, changes)
        if is_number(a) and is_number(b):
            if not same_number(a, b):
                self._emit(changes, schema_path, labels, "num", a, b)
            return
        if a != b:
            kind = "text" if isinstance(a, str) and isinstance(b, str) else "value"
            self._emit(changes, schema_path, labels, kind, a, b)

    def _list(self, a, b, schema_path, labels, changes):
        sample = (a + b)[0]
        if isinstance(sample, dict):
            old, new = keyed(schema_path[-1], a), keyed(schema_path[-1], b)
            for key, (label, element) in old.items():
                if key in new:
                    self._dict(element, new[key][1], schema_path, labels + [label], changes)
                else:
                    self._emit(changes, schema_path, labels + [label], "gone", element)
            for key, (label, element) in new.items():
                if key not in old:
                    self._emit(changes, schema_path, labels + [label], "new", None, element)
        elif all(is_number(item) for item in a + b):
            if len(a) != len(b) or not all(same_number(x, y) for x, y in zip(a, b)):
                self._emit(changes, schema_path, labels, "value", a, b)
        else:
            # By count, so a list that repeats an entry (perk labels do) still
            # shows one of the repeats going away.
            removed = list((Counter(map(freeze, a)) - Counter(map(freeze, b))).elements())
            added = list((Counter(map(freeze, b)) - Counter(map(freeze, a))).elements())
            if removed or added:
                self._emit(changes, schema_path, labels, "set", removed, added)


def collect_schema(records):
    paths = set()

    def walk(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                paths.add(path + (key,))
                walk(item, path + (key,))
        elif isinstance(value, list):
            for item in value:
                walk(item, path)

    for record in records:
        walk(record, ())
    return paths


def match_records(old, new):
    """Pairs of (old index, new index, how), plus the unmatched of each side."""
    left, right = set(range(len(old))), set(range(len(new)))
    pairs = []

    def run(how, key, allowed=lambda a, b: True):
        a_keys, b_keys = defaultdict(list), defaultdict(list)
        for index in left:
            if (k := key(old[index])) is not None:
                a_keys[k].append(index)
        for index in right:
            if (k := key(new[index])) is not None:
                b_keys[k].append(index)
        for k, indices in a_keys.items():
            others = b_keys.get(k, [])
            if len(indices) == 1 and len(others) == 1 and allowed(old[indices[0]], new[others[0]]):
                pairs.append((indices[0], others[0], how))
                left.discard(indices[0])
                right.discard(others[0])

    def no_conflicting_identity(a, b):
        ia, ib = a.get("internal_hash"), b.get("internal_hash")
        return (ia is None or ib is None or ia == ib) and a.get("item_type") == b.get("item_type")

    run("internal hash", lambda r: r.get("internal_hash"))
    run("internal hash + rarity", lambda r: r.get("internal_hash") and (r["internal_hash"], r.get("rarity")))
    run("display hash", lambda r: r.get("hash"), no_conflicting_identity)
    run("name", lambda r: (r.get("name"), r.get("rarity"), r.get("item_type"), r.get("subcategory")),
        no_conflicting_identity)
    return pairs, sorted(left), sorted(right)


# ---------------------------------------------------------------- rendering

def esc(value):
    return html.escape(str(value), quote=True)


def word_diff(old, new):
    tokens = lambda text: re.findall(r"\s+|\w+|[^\w\s]", text)
    a, b = tokens(old), tokens(new)
    out = []
    for op, a0, a1, b0, b1 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(esc("".join(a[a0:a1])))
            continue
        if a1 > a0:
            out.append(f"<del>{esc(''.join(a[a0:a1]))}</del>")
        if b1 > b0:
            out.append(f"<ins>{esc(''.join(b[b0:b1]))}</ins>")
    return "".join(out).replace("\n", "<br>")


def delta_text(old, new):
    delta = new - old
    text = ("+" if delta > 0 else "−") + format_number(abs(delta))
    if abs(old) > 1e-9:
        percent = delta / abs(old) * 100
        text += f", {'+' if percent > 0 else '−'}{abs(percent):.1f}%".replace(".0%", "%")
    return text


def render_change(change):
    kind, old, new = change["kind"], change["old"], change["new"]
    label = esc(change["label"])
    if kind == "num":
        up = new > old
        return (f'<div class="row"><span class="k">{label}</span>'
                f'<span class="v"><span class="old">{esc(format_number(old))}</span><span class="arrow">→</span>'
                f'<span class="new">{esc(format_number(new))}</span></span>'
                f'<span class="d {"up" if up else "down"}">{"▲" if up else "▼"} {esc(delta_text(old, new))}</span></div>')
    if kind == "text":
        long = len(old) + len(new) > 60 and (" " in old or " " in new)
        if long:
            return f'<div class="row wide"><span class="k">{label}</span><span class="v prose">{word_diff(old, new)}</span></div>'
    if kind in ("text", "value"):
        return (f'<div class="row"><span class="k">{label}</span>'
                f'<span class="v"><span class="old">{esc(format_value(old))}</span><span class="arrow">→</span>'
                f'<span class="new">{esc(format_value(new))}</span></span><span class="d"></span></div>')
    if kind == "set":
        chips = "".join(f'<span class="chip minus">− {esc(format_value(item))}</span>' for item in old)
        chips += "".join(f'<span class="chip plus">+ {esc(format_value(item))}</span>' for item in new)
        return f'<div class="row wide"><span class="k">{label}</span><span class="v chips">{chips}</span></div>'
    sign, css, value = ("+", "plus", new) if kind == "new" else ("−", "minus", old)
    return (f'<div class="row wide"><span class="k"><span class="chip {css}">{sign} {label}</span></span>'
            f'<span class="v dim">{esc(describe(value))}</span></div>')


def text_change(change):
    kind, old, new = change["kind"], change["old"], change["new"]
    name = humanize(change["top"]) + (f" › {change['label']}" if change["label"] else "")
    if kind == "num":
        return f"{name}: {format_number(old)} → {format_number(new)} ({delta_text(old, new)})"
    if kind in ("text", "value"):
        return f"{name}: {format_value(old)} → {format_value(new)}"
    if kind == "set":
        parts = [f"-{format_value(item)}" for item in old] + [f"+{format_value(item)}" for item in new]
        return f"{name}: " + ", ".join(parts)
    return f"{name}: {'added' if kind == 'new' else 'removed'} ({describe(new if kind == 'new' else old)})"


def section_key(top):
    return (SECTION_ORDER.index(top) if top in SECTION_ORDER else len(SECTION_ORDER), top)


def render_groups(changes):
    groups = defaultdict(list)
    for change in changes:
        groups[change["top"]].append(change)
    out = []
    for top in sorted(groups, key=section_key):
        rows = groups[top]
        if len(rows) == 1 and not rows[0]["label"]:
            row = dict(rows[0], label=humanize(top))
            out.append(f'<div class="group solo">{render_change(row)}</div>')
        else:
            body = "".join(render_change(dict(row, label=row["label"] or humanize(top))) for row in rows)
            out.append(f'<div class="group"><div class="gh">{esc(humanize(top))}'
                       f'<span class="count">{len(rows)}</span></div>{body}</div>')
    return "".join(out)


def render_record(value, depth=0):
    """Full nested view of an added or removed record."""
    if isinstance(value, dict):
        rows = []
        for key, item in value.items():
            if empty(item):
                continue
            if key == "description" and value.get("description_text"):
                continue
            if key == "weapons" and depth and isinstance(item, list):
                rows.append(f'<div class="row"><span class="k">{esc(humanize(key))}</span>'
                            f'<span class="v dim">{len(item)} weapons with recomputed changes</span></div>')
                continue
            if isinstance(item, (dict, list)) and not all(not isinstance(i, (dict, list)) for i in (item if isinstance(item, list) else [item])):
                rows.append(f'<div class="nest"><div class="gh">{esc(humanize(key))}</div>{render_record(item, depth + 1)}</div>')
            elif isinstance(item, list):
                chips = "".join(f'<span class="chip">{esc(format_value(i))}</span>' for i in item)
                rows.append(f'<div class="row wide"><span class="k">{esc(humanize(key))}</span><span class="v chips">{chips}</span></div>')
            else:
                css = "v prose" if isinstance(item, str) and len(item) > 60 else "v"
                rows.append(f'<div class="row wide"><span class="k">{esc(humanize(key))}</span>'
                            f'<span class="{css}">{esc(format_value(item))}</span></div>')
        return "".join(rows)
    if isinstance(value, list):
        if all(isinstance(item, dict) and all(not isinstance(v, (dict, list)) for v in item.values()) for item in value):
            return "".join(f'<div class="row wide"><span class="v">{esc(describe(item, 400))}</span></div>' for item in value)
        return "".join(f'<div class="nest">{render_record(item, depth + 1)}</div>' for item in value)
    return esc(format_value(value))


def modifier_text(modifier):
    """`Range +90 rating` or `Damage ×1.15`, as the Gear panel prints it."""
    value, kind = modifier.get("value"), modifier.get("kind")
    if not is_number(value):
        return str(modifier.get("stat"))
    if kind == "multiplier":
        amount = "×" + format_number(value)
    elif kind == "rating":
        amount = ("+" if value >= 0 else "−") + format_number(abs(value)) + PERK_KIND_SUFFIX[kind]
    else:
        amount = format_number(value)
    return f"{modifier.get('stat')} {amount}"


def perk_facts(record, limit=8):
    """Exact perk values of a record: scalars first, then hop-on modifiers."""
    facts = []
    for perk in record.get("perks") or []:
        scalars = perk.get("scalars") or []
        if scalars:
            facts.append(f"{perk.get('name')} " + " / ".join("×" + format_number(s) for s in scalars))
        for hop_on in perk.get("hop_ons") or []:
            facts += [modifier_text(modifier) for modifier in hop_on.get("modifiers") or []]
    if len(facts) > limit:
        facts = facts[:limit] + [f"+{len(facts) - limit} more perk values"]
    return facts


def headline(record):
    """The facts a dataminer wants from a new or removed item at a glance."""
    facts = []
    if record.get("price") is not None:
        facts.append(f"price {format_number(record['price'])}")
    stats = record.get("weapon_stats") or {}
    for key, label in (("damage", "dmg"), ("rate_of_fire_rpm", "RPM"), ("magazine", "mag"), ("range_metres", "m range")):
        if stats.get(key) is not None:
            facts.append(f"{format_number(round(stats[key], 2))} {label}")
    ratings = (record.get("mod_stats") or {}).get("ratings") or []
    facts += [f"{r['name']} {r['value']:+g}" for r in ratings]
    facts += [f"{s['name']} {s['value']:+d}" for s in record.get("implant_stats") or []]
    facts += perk_facts(record)
    if record.get("compatible_weapons"):
        facts.append("for " + ", ".join(record["compatible_weapons"]))
    return " · ".join(facts)


def build_items(old, new):
    differ = Differ(collect_schema(old), collect_schema(new))
    pairs, removed, added = match_records(old, new)
    items = []
    for a, b, how in pairs:
        changes = differ.diff(old[a], new[b])
        data = [c for c in changes if c["cat"] == "data"]
        derived = [c for c in changes if c["cat"] == "derived"]
        tags = [c for c in changes if c["cat"] == "tags"]
        status = "changed" if data else "derived" if derived else "tags" if tags else "same"
        items.append({"status": status, "old": old[a], "new": new[b], "how": how,
                      "data": data, "derived": derived, "tags": tags})
    items += [{"status": "removed", "old": old[i], "new": None, "data": [], "derived": [], "tags": []} for i in removed]
    items += [{"status": "added", "old": None, "new": new[i], "data": [], "derived": [], "tags": []} for i in added]
    return items, differ.ignored


def item_text(item):
    record = item["new"] or item["old"]
    kind = " ".join(str(part) for part in (record.get("rarity"), record.get("item_type")) if part)
    title = record.get("name") or "?"
    if item["status"] == "changed" and item["old"].get("name") != item["new"].get("name"):
        title = f"{item['old'].get('name')} → {item['new'].get('name')}"
    sign = {"added": "+", "removed": "-"}.get(item["status"], "~")
    lines = [f"{sign} {title} [{kind}]"]
    if item["status"] in ("added", "removed"):
        if headline(record):
            lines.append("    " + headline(record))
        if record.get("description_text"):
            lines.append("    " + record["description_text"].replace("\n", " "))
    lines += ["    " + text_change(change) for change in item["data"]]
    return "\n".join(lines)


def render_item(item, number):
    record = item["new"] or item["old"]
    status = item["status"]
    rarity = record.get("rarity") or ""
    item_type = record.get("item_type") or "Uncategorized"
    name = esc(record.get("name") or "?")
    renamed = status not in ("added", "removed") and item["old"].get("name") != item["new"].get("name")
    if renamed:
        name = f'<span class="old">{esc(item["old"].get("name"))}</span><span class="arrow">→</span>{name}'

    sub = " · ".join(esc(part) for part in (record.get("subcategory"), record.get("applies_to")) if part)
    ids = " ".join(esc(part) for part in (record.get("internal_hash"), record.get("hash")) if part)
    fields = sorted({change["top"] for change in item["data"]}, key=section_key)
    badges = "".join(f'<span class="tag">{esc(humanize(field))}</span>' for field in fields[:6])
    if len(fields) > 6:
        badges += f'<span class="tag">+{len(fields) - 6}</span>'
    if renamed:
        badges = '<span class="tag warn">renamed</span>' + badges
    if item.get("how") == "name":
        badges += '<span class="tag warn" title="No stable hash in common; paired by name, rarity and type">matched by name</span>'

    body = []
    if status in ("added", "removed"):
        facts = headline(record)
        if facts:
            body.append(f'<div class="facts">{esc(facts)}</div>')
        if record.get("description_text"):
            body.append(f'<div class="prose desc">{esc(record["description_text"])}</div>')
        body.append(f'<details class="more"><summary>Full record</summary>{render_record(record)}</details>')
    else:
        body.append(render_groups(item["data"]))
        if item["derived"]:
            weapons = len({change["label"].split(" › ")[0] for change in item["derived"]})
            body.append(f'<details class="more"{" open" if status == "derived" and len(item["derived"]) < 40 else ""}>'
                        f'<summary>Recomputed per-weapon values · {len(item["derived"])} across {weapons} weapons</summary>'
                        f'{"".join(render_change(c) for c in item["derived"])}</details>')
        if item["tags"]:
            rows = "".join(render_change(dict(c, label=(humanize(c["top"]) + (" › " + c["label"] if c["label"] else ""))))
                           for c in item["tags"])
            body.append(f'<details class="more"><summary>Tag references · {len(item["tags"])}</summary>{rows}</details>')

    search = " ".join(str(part) for part in (
        record.get("name"), item["old"].get("name") if item["old"] else "", rarity, item_type, record.get("subcategory"),
        record.get("internal_type"), record.get("internal_hash"), record.get("hash"), record.get("definition_tag"),
        record.get("description_text"), " ".join(humanize(f) for f in fields),
        " ".join(c["label"] for c in item["data"])) if part).lower()
    color = RARITY_COLOR.get(rarity, "var(--line-strong)")
    open_attr = " open" if status in ("changed", "derived") else ""
    return (
        f'<details class="item s-{status}" id="i{number}" data-status="{status}" data-type="{esc(item_type)}" '
        f'data-rarity="{esc(rarity)}" data-fields="|{esc("|".join(fields))}|" data-search="{esc(search)}" '
        f'data-text="{esc(item_text(item)) if status in ("added", "removed", "changed") else ""}" style="--rarity:{color}"{open_attr}>'
        f'<summary><span class="mark">{ {"added": "+", "removed": "−"}.get(status, "~") }</span>'
        f'<span class="name">{name}</span>'
        f'{f"""<span class="pill">{esc(rarity)}</span>""" if rarity else ""}'
        f'<span class="sub">{sub}</span><span class="badges">{badges}</span>'
        f'<span class="ids">{ids}</span></summary>'
        f'<div class="body">{"".join(body)}</div></details>'
    )


def sort_key(item):
    record = item["new"] or item["old"]
    rarity = record.get("rarity")
    return (
        STATUS_ORDER.index(item["status"]),
        RARITY_ORDER.index(rarity) if rarity in RARITY_ORDER else len(RARITY_ORDER),
        (record.get("name") or "").lower(),
    )


def render_page(items, ignored, old_meta, new_meta):
    counts = Counter(item["status"] for item in items)
    shown = [item for item in items if item["status"] != "same"]
    by_type = defaultdict(list)
    for item in shown:
        by_type[(item["new"] or item["old"]).get("item_type") or "Uncategorized"].append(item)

    sections, number = [], 0
    for item_type in sorted(by_type, key=str.lower):
        cards = []
        for item in sorted(by_type[item_type], key=sort_key):
            number += 1
            cards.append(render_item(item, number))
        sections.append(f'<section data-type="{esc(item_type)}"><h2>{esc(item_type)}<span class="count"></span></h2>'
                        f'{"".join(cards)}</section>')

    tiles = "".join(
        f'<label class="tile s-{status}"><input type="checkbox" data-status="{status}"'
        f'{" checked" if status in ("added", "removed", "changed") else ""}{" disabled" if not counts[status] else ""}>'
        f'<b>{counts[status]:,}</b><span>{STATUS_LABEL[status]}</span></label>'
        for status in STATUS_ORDER)
    tiles += f'<div class="tile s-same"><b>{counts["same"]:,}</b><span>Unchanged</span></div>'

    field_counts = Counter(field for item in shown for field in {c["top"] for c in item["data"]})
    field_chips = "".join(
        f'<button class="fchip" data-field="{esc(field)}">{esc(humanize(field))}<span>{count}</span></button>'
        for field, count in sorted(field_counts.items(), key=lambda pair: (-pair[1], pair[0])))

    matrix_rows = "".join(
        f'<tr data-type="{esc(item_type)}"><th>{esc(item_type)}</th>'
        + "".join(f'<td class="s-{status}">{sum(i["status"] == status for i in by_type[item_type]) or ""}</td>'
                  for status in STATUS_ORDER) + "</tr>"
        for item_type in sorted(by_type, key=lambda t: (-sum(i["status"] in ("added", "removed", "changed") for i in by_type[t]), t.lower())))
    matrix = (f'<table class="matrix"><thead><tr><th></th>'
              + "".join(f"<th>{STATUS_LABEL[s]}</th>" for s in STATUS_ORDER)
              + f"</tr></thead><tbody>{matrix_rows}</tbody></table>") if by_type else ""

    type_options = "".join(f'<option value="{esc(t)}">{esc(t)} ({len(by_type[t])})</option>' for t in sorted(by_type, key=str.lower))
    rarities = Counter((i["new"] or i["old"]).get("rarity") or "" for i in shown)
    rarity_options = "".join(f'<option value="{esc(r)}">{esc(r)} ({rarities[r]})</option>' for r in RARITY_ORDER if rarities[r])

    schema_note = ""
    if ignored:
        only = defaultdict(list)
        for (path, missing), _ in sorted(ignored.items()):
            only["new" if missing == "old" else "old"].append(".".join(path))
        parts = [f'<b>only in {side}:</b> ' + ", ".join(f"<code>{esc(p)}</code>" for p in paths) for side, paths in only.items()]
        schema_note = ('<div class="note"><b>Schema differs.</b> These fields exist in one export only, so they are '
                       'left out of the comparison instead of being flagged on every item. ' + " &nbsp; ".join(parts) + "</div>")

    return PAGE.format(
        old_name=esc(old_meta["name"]), new_name=esc(new_meta["name"]),
        old_info=esc(f'{old_meta["count"]:,} records · {old_meta["time"]}'),
        new_info=esc(f'{new_meta["count"]:,} records · {new_meta["time"]}'),
        tiles=tiles, matrix=matrix, field_chips=field_chips, type_options=type_options,
        rarity_options=rarity_options, schema_note=schema_note, sections="".join(sections), css=CSS, js=JS)


CSS = """
:root{--bg:#0f1216;--panel:#171b21;--panel2:#1d222a;--text:#e6e9ee;--dim:#98a2b0;--line:#272d37;--line-strong:#3a4250;
--add:#4cc38a;--add-bg:rgba(76,195,138,.14);--del:#f2707a;--del-bg:rgba(242,112,122,.14);--chg:#e5b454;--chg-bg:rgba(229,180,84,.14);
--info:#6aa9ff;--accent:#6aa9ff;--mono:ui-monospace,"Cascadia Mono","JetBrains Mono",Consolas,monospace}
@media (prefers-color-scheme:light){:root{--bg:#f5f6f8;--panel:#fff;--panel2:#f0f2f5;--text:#1b1f26;--dim:#5d6877;--line:#e1e5eb;--line-strong:#c5ccd6;
--add:#12804a;--add-bg:rgba(18,128,74,.11);--del:#c4313d;--del-bg:rgba(196,49,61,.10);--chg:#96690a;--chg-bg:rgba(150,105,10,.12);--info:#1f62c9;--accent:#1f62c9}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 "Segoe UI",system-ui,-apple-system,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:0 20px 80px}
header{padding:26px 0 8px}
h1{font-size:22px;margin:0 0 10px;letter-spacing:-.01em}
.files{display:flex;flex-wrap:wrap;gap:10px;align-items:stretch}
.file{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 12px;min-width:0}
.file small{display:block;color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.file b{font-family:var(--mono);font-size:13px;word-break:break-all}.file span{display:block;color:var(--dim);font-size:12px}
.files .arrow{align-self:center;color:var(--dim);font-size:18px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:16px 0}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px;cursor:pointer;position:relative;user-select:none;border-top:3px solid var(--c,var(--line-strong))}
.tile b{display:block;font-size:24px;line-height:1.1;font-variant-numeric:tabular-nums}.tile span{color:var(--dim);font-size:12px}
.tile input{position:absolute;top:10px;right:10px;accent-color:var(--accent)}
.tile:has(input:not(:checked)){opacity:.55}.tile.s-same{cursor:default;opacity:.7}
.s-added{--c:var(--add)}.s-removed{--c:var(--del)}.s-changed{--c:var(--chg)}.s-derived{--c:var(--info)}.s-tags,.s-same{--c:var(--line-strong)}
.note{background:var(--chg-bg);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:0 0 16px;font-size:13px}
code{font-family:var(--mono);font-size:12px}
details.overview{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin-bottom:16px}
details.overview>summary{padding:9px 12px;cursor:pointer;color:var(--dim);font-size:13px}
.matrix-wrap{overflow-x:auto;padding:0 12px 12px}
.matrix{border-collapse:collapse;font-size:13px;min-width:520px}
.matrix th,.matrix td{padding:4px 12px;text-align:right;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
.matrix thead th{color:var(--dim);font-weight:500;font-size:12px}.matrix tbody th{text-align:left;font-weight:500}
.matrix td{color:var(--c);font-weight:600}.matrix tbody tr{cursor:pointer}.matrix tbody tr:hover{background:var(--panel2)}
.bar{position:sticky;top:0;z-index:5;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--line);margin-bottom:6px}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
input[type=search],select,button{font:inherit;color:var(--text);background:var(--panel);border:1px solid var(--line-strong);border-radius:6px;padding:6px 10px}
input[type=search]{flex:1 1 260px;min-width:180px}button{cursor:pointer}button:hover{border-color:var(--accent)}
.spacer{flex:1}.visible{color:var(--dim);font-size:12px;font-variant-numeric:tabular-nums}
.fields{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.fchip{padding:2px 9px;border-radius:999px;font-size:12px;background:var(--panel)}
.fchip span{color:var(--dim);margin-left:6px;font-variant-numeric:tabular-nums}
.fchip.on{background:var(--accent);border-color:var(--accent);color:#fff}.fchip.on span{color:#fff}
h2{font-size:15px;margin:26px 0 8px;display:flex;align-items:baseline;gap:8px}
.count{color:var(--dim);font-weight:400;font-size:12px;margin-left:6px;font-variant-numeric:tabular-nums}
.item{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--c);border-radius:8px;margin:6px 0;overflow:hidden}
.item>summary{display:flex;flex-wrap:wrap;align-items:center;gap:4px 10px;padding:8px 12px;cursor:pointer;list-style:none}
.item>summary::-webkit-details-marker{display:none}
.mark{width:20px;height:20px;border-radius:5px;display:inline-grid;place-items:center;font-weight:700;font-family:var(--mono);color:var(--c);background:color-mix(in srgb,var(--c) 16%,transparent);flex:none}
.name{font-weight:600;font-size:15px}.name .old{color:var(--dim);text-decoration:line-through;font-weight:400}
.pill{font-size:11px;font-weight:600;padding:1px 8px;border-radius:999px;border:1px solid var(--rarity);background:color-mix(in srgb,var(--rarity) 18%,transparent)}
.sub{color:var(--dim);font-size:13px}.badges{display:flex;flex-wrap:wrap;gap:4px}
.tag{font-size:11px;color:var(--dim);background:var(--panel2);border-radius:4px;padding:1px 6px}.tag.warn{color:var(--chg);background:var(--chg-bg)}
.ids{margin-left:auto;font-family:var(--mono);font-size:11px;color:var(--dim)}
.body{padding:2px 12px 12px 42px;border-top:1px solid var(--line)}
.group{margin-top:10px}.gh{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--dim);margin-bottom:2px}
.row{display:grid;grid-template-columns:minmax(150px,260px) minmax(0,1fr) minmax(0,auto);gap:4px 16px;align-items:baseline;padding:3px 0;border-bottom:1px dashed var(--line)}
.row:last-child{border-bottom:0}.row.wide{grid-template-columns:minmax(150px,260px) minmax(0,1fr)}
.k{color:var(--dim);overflow-wrap:anywhere}
.v{font-family:var(--mono);font-size:13px;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.v.prose,.prose{font-family:inherit;font-size:14px;line-height:1.55}.v.dim{color:var(--dim);font-family:inherit}
.old{color:var(--del)}.new{color:var(--add);font-weight:600}.arrow{color:var(--dim);margin:0 8px}
.d{font-family:var(--mono);font-size:12px;white-space:nowrap;text-align:right}.d.up{color:var(--add)}.d.down{color:var(--del)}
del{background:var(--del-bg);color:var(--del);text-decoration:line-through;border-radius:3px;padding:0 2px}
ins{background:var(--add-bg);color:var(--add);text-decoration:none;border-radius:3px;padding:0 2px}
.chips{display:flex;flex-wrap:wrap;gap:4px}
.chip{font-family:var(--mono);font-size:12px;background:var(--panel2);border-radius:4px;padding:0 6px;overflow-wrap:anywhere}
.chip.plus{background:var(--add-bg);color:var(--add)}.chip.minus{background:var(--del-bg);color:var(--del)}
.facts{margin-top:10px;font-family:var(--mono);font-size:13px}.desc{margin-top:6px;color:var(--dim);white-space:pre-line}
.more{margin-top:10px;border:1px solid var(--line);border-radius:6px;background:var(--panel2)}
.more>summary{padding:6px 10px;cursor:pointer;color:var(--dim);font-size:12px}.more>.row,.more>.nest{margin:0 10px}
.nest{padding:4px 0 4px 12px;border-left:2px solid var(--line);margin-top:6px}
.empty{display:none;text-align:center;color:var(--dim);padding:60px 0}
.toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:var(--text);color:var(--bg);padding:8px 14px;border-radius:6px;font-size:13px;opacity:0;transition:opacity .2s;pointer-events:none}
.toast.on{opacity:1}
@media (max-width:640px){.wrap{padding:0 16px 60px}.body{padding-left:12px}.row,.row.wide{grid-template-columns:1fr}.d{text-align:left}.ids{margin-left:0;width:100%}}
"""

JS = """
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const items=$$('.item'),search=$('#q'),typeSel=$('#type'),raritySel=$('#rarity');
let field='';
function apply(){
  const on=new Set($$('.tile input:checked').map(i=>i.dataset.status));
  const words=search.value.toLowerCase().split(/\\s+/).filter(Boolean);
  let shown=0;
  for(const el of items){
    const ok=on.has(el.dataset.status)&&(!typeSel.value||el.dataset.type===typeSel.value)
      &&(!raritySel.value||el.dataset.rarity===raritySel.value)
      &&(!field||el.dataset.fields.includes('|'+field+'|'))
      &&words.every(w=>el.dataset.search.includes(w));
    el.hidden=!ok;if(ok)shown++;
  }
  for(const s of $$('section')){
    const n=$$('.item:not([hidden])',s).length;s.hidden=!n;$('.count',s).textContent=n;
  }
  $('#visible').textContent=shown+' of '+items.length+' shown';
  $('.empty').style.display=shown?'none':'block';
}
function visible(){return items.filter(e=>!e.hidden)}
search.addEventListener('input',apply);typeSel.addEventListener('change',apply);raritySel.addEventListener('change',apply);
$$('.tile input').forEach(i=>i.addEventListener('change',apply));
$$('.fchip').forEach(b=>b.addEventListener('click',()=>{
  field=field===b.dataset.field?'':b.dataset.field;
  $$('.fchip').forEach(x=>x.classList.toggle('on',x.dataset.field===field));apply();}));
$$('.matrix tbody tr').forEach(r=>r.addEventListener('click',()=>{
  typeSel.value=typeSel.value===r.dataset.type?'':r.dataset.type;apply();
  $('.bar').scrollIntoView({behavior:'smooth'});}));
$('#expand').addEventListener('click',()=>visible().forEach(e=>e.open=true));
$('#collapse').addEventListener('click',()=>visible().forEach(e=>e.open=false));
$('#copy').addEventListener('click',async()=>{
  const text=visible().map(e=>e.dataset.text).join('\\n');
  try{await navigator.clipboard.writeText(text)}catch(_){
    const t=document.createElement('textarea');t.value=text;document.body.append(t);t.select();document.execCommand('copy');t.remove();}
  const toast=$('.toast');toast.textContent='Copied '+visible().length+' items as text';
  toast.classList.add('on');setTimeout(()=>toast.classList.remove('on'),1600);});
document.addEventListener('keydown',e=>{if(e.key==='/'&&document.activeElement!==search){e.preventDefault();search.focus()}});
apply();
"""

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gear diff</title><style>{css}</style></head><body><div class="wrap">
<header><h1>Gear diff</h1>
<div class="files"><div class="file"><small>Old</small><b>{old_name}</b><span>{old_info}</span></div>
<span class="arrow">→</span><div class="file"><small>New</small><b>{new_name}</b><span>{new_info}</span></div></div></header>
<div class="tiles">{tiles}</div>
{schema_note}
<details class="overview"><summary>Breakdown by item type (click a row to filter)</summary><div class="matrix-wrap">{matrix}</div></details>
<div class="bar"><div class="controls">
<input type="search" id="q" placeholder="Search name, hash, internal type, description, changed field…  ( / )">
<select id="type"><option value="">All types</option>{type_options}</select>
<select id="rarity"><option value="">All rarities</option>{rarity_options}</select>
<button id="expand">Expand</button><button id="collapse">Collapse</button><button id="copy">Copy as text</button>
<span class="visible" id="visible"></span></div>
<div class="fields">{field_chips}</div></div>
{sections}
<div class="empty">Nothing matches the current filters.</div>
</div><div class="toast"></div><script>{js}</script></body></html>
"""


def load(path):
    with open(path, encoding="utf-8-sig") as file:
        records = json.load(file)
    if not isinstance(records, list):
        sys.exit(f"{path}: expected a JSON array of gear records")
    meta = {"name": os.path.basename(path), "count": len(records),
            "time": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))}
    return records, meta


def main():
    parser = argparse.ArgumentParser(description="Compare two Gear JSON exports.")
    parser.add_argument("old")
    parser.add_argument("new")
    parser.add_argument("-o", "--output", default="gear-diff.html", help="HTML report (default: gear-diff.html)")
    parser.add_argument("--text", help="also write a plain-text changelog to this file")
    parser.add_argument("--open", action="store_true", help="open the report in the browser")
    args = parser.parse_args()

    old, old_meta = load(args.old)
    new, new_meta = load(args.new)
    items, ignored = build_items(old, new)

    with open(args.output, "w", encoding="utf-8") as file:
        file.write(render_page(items, ignored, old_meta, new_meta))
    if args.text:
        listed = [item for item in items if item["status"] in ("added", "removed", "changed")]
        with open(args.text, "w", encoding="utf-8") as file:
            file.write("\n".join(item_text(item) for item in sorted(listed, key=sort_key)) + "\n")

    counts = Counter(item["status"] for item in items)
    print(", ".join(f"{counts[s]} {STATUS_LABEL[s].lower()}" for s in STATUS_ORDER + ["same"]))
    print(f"Wrote {args.output}")
    if args.open:
        webbrowser.open("file://" + os.path.abspath(args.output))


if __name__ == "__main__":
    main()
