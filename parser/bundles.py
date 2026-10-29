"""Parse a libtpu `*-final_bundles.txt` dump into a structured program.

Grammar observed in libtpu 0.0.50.1 dumps (see HARDWARE_FACTS.md / parser/MNEMONICS.md):

    <addr> [LABEL]: [>...] { op ;; op ;; ... } [/* trailing comment */]

addr is decimal for 0 and hex (0x..) after; LABEL is one of LH LB LE PB PF CT; each
'>' is one level of loop nesting. An op is `%dest = mnemonic args`, optionally
predicated by `(%pN)` or `(!%pN)` among its args. A conditional move
`smov (%pN, %sOld), new` yields new when pN holds and sOld otherwise (Op.cond). Comments `/* ... */` may span
lines and carry DMA descriptors, operand labels (`/*hbm=*/`), and region markers
(`Start region N`, `End region N`, `Start/End empty region N`) that resolve
`sbr.rel ... target = $regionN`.
"""

import re
from dataclasses import dataclass, field

BUNDLE_START = re.compile(r"^\s*(0x[0-9a-f]+|\d+)\s+([A-Z]{2})?:\s*(>*)\s*\{", re.M)
DEST = re.compile(r"^%(\S+?)\s*=\s*")
PHYS = re.compile(r"_((?:vm|s|p|v)\d+)$")
PRED = re.compile(r"\((!?)%(\w+)\)")
COND = re.compile(r"^\((!?)%(\w+),\s*(%\w+)\)$")
ATTR = re.compile(r"\[(\w+):\$([^\]\s]+)\]")
LABEL = re.compile(r"^/\*(\w+)=\*/$")
REGION_START = re.compile(r"Start(?:/End empty)? region (\d+)")
REGION_END = re.compile(r"(?:End|Start/End empty) region (\d+)")


@dataclass
class Op:
    mnemonic: str
    dest: str | None
    args: list[str]
    labels: dict[str, str]
    attrs: dict[str, str]
    pred: tuple[bool, str] | None
    cond: tuple[bool, str, str] | None
    comment: str
    text: str

    @property
    def dest_reg(self):
        """Physical register written (s9, p3, v12, vm0), or None."""
        return phys(self.dest)

    @property
    def unit(self):
        return self.mnemonic.split(".", 1)[0]


@dataclass
class Bundle:
    index: int
    addr: int
    label: str | None
    depth: int
    ops: list[Op]
    trailing: str
    region_starts: list[int] = field(default_factory=list)
    region_ends: list[int] = field(default_factory=list)


@dataclass
class Program:
    bundles: list[Bundle]
    region_start: dict[int, int]

    def target(self, region):
        # A region with no bundles (e.g. the exit a kernel jumps to when its work range is empty)
        # has no marker in the dump; branching there ends the program.
        return self.region_start.get(region, len(self.bundles))


def phys(name):
    if name is None:
        return None
    m = PHYS.search(name)
    return m.group(1) if m else None


def _split_top(text, sep):
    """Split on sep outside /* */ comments, [] and ()."""
    parts, depth, i, start = [], 0, 0, 0
    while i < len(text):
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = len(text) if j < 0 else j + 2
            continue
        c = text[i]
        if c in "[(":
            depth += 1
        elif c in "])":
            depth -= 1
        elif depth == 0 and text.startswith(sep, i):
            parts.append(text[start:i])
            i += len(sep)
            start = i
            continue
        i += 1
    parts.append(text[start:])
    return parts


def _body_end(chunk, open_at):
    """Index of the '}' closing the bundle body that opens at chunk[open_at]."""
    depth, i = 0, open_at
    while i < len(chunk):
        if chunk.startswith("/*", i):
            j = chunk.find("*/", i + 2)
            i = len(chunk) if j < 0 else j + 2
            continue
        if chunk[i] == "{":
            depth += 1
        elif chunk[i] == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError(f"unterminated bundle: {chunk[:120]!r}")


def _split_mem_operand(tok):
    """`[smem:[...]] %s5` (store address then value, no comma) -> two args."""
    if not tok.startswith("["):
        return [tok]
    depth = 0
    for i, c in enumerate(tok):
        depth += (c == "[") - (c == "]")
        if depth == 0:
            rest = tok[i + 1:].strip()
            return [tok[: i + 1]] + ([rest] if rest else [])
    return [tok]


def _comments(text):
    return [m.group(1).strip() for m in re.finditer(r"/\*(.*?)\*/", text, re.S)]


def parse_op(raw):
    raw = raw.strip()
    comment_parts = [c for c in _comments(raw) if not re.fullmatch(r"\w+=", c)]
    m = DEST.match(raw)
    dest = m.group(1) if m else None
    rest = raw[m.end():] if m else raw
    mm = re.match(r"([A-Za-z_][\w.]*)", rest)
    mnemonic = mm.group(1)
    rest = rest[mm.end():]
    args, labels, attrs, pred, cond, pending = [], {}, {}, None, None, None
    for tok in _split_top(rest, ","):
        tok = tok.strip()
        lab = re.search(r"/\*(\w+)=\*/\s*", tok)
        if lab:
            pending, tok = lab.group(1), tok[: lab.start()] + " " + tok[lab.end():]
        tok = re.sub(r"/\*.*?\*/", "", tok, flags=re.S).strip()
        cm = COND.match(tok)
        if cm:
            cond = (cm.group(1) == "!", cm.group(2), cm.group(3))
            continue
        pm = PRED.search(tok)
        if pm and pred is None:
            pred = (pm.group(1) == "!", pm.group(2))
            tok = (tok[: pm.start()] + " " + tok[pm.end():]).strip()
        for am in list(ATTR.finditer(tok)):
            attrs[am.group(1)] = am.group(2)
        tok = ATTR.sub("", tok).strip()
        if not tok:
            continue
        tok = " ".join(tok.split())
        if pending:
            labels[pending] = tok
            pending = None
        args.extend(_split_mem_operand(tok))
    text = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    return Op(mnemonic, dest, args, labels, attrs, pred, cond, "\n".join(comment_parts), " ".join(text.split()))


def parse(text):
    starts = list(BUNDLE_START.finditer(text))
    bundles, region_start = [], {}
    for n, m in enumerate(starts):
        end = starts[n + 1].start() if n + 1 < len(starts) else len(text)
        chunk = text[m.start():end]
        open_at = m.end() - m.start() - 1
        close = _body_end(chunk, open_at)
        body, trailing = chunk[open_at + 1:close], chunk[close + 1:]
        ops = [parse_op(o) for o in _split_top(body, ";;") if o.strip()]
        trail = " ".join(_comments(trailing))
        b = Bundle(
            index=n,
            addr=int(m.group(1), 0),
            label=m.group(2),
            depth=len(m.group(3)),
            ops=ops,
            trailing=trail,
            region_starts=[int(x) for x in REGION_START.findall(trail)],
            region_ends=[int(x) for x in REGION_END.findall(trail)],
        )
        for r in b.region_starts:
            region_start.setdefault(r, n)
        bundles.append(b)
    for i, b in enumerate(bundles):
        if b.addr != i:
            raise ValueError(f"bundle {i} has address {b.addr:#x}; expected dense addresses")
    return Program(bundles, region_start)


def parse_file(path):
    with open(path) as f:
        return parse(f.read())
