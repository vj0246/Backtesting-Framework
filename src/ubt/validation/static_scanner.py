"""Static AST scanner for known leakage anti-patterns.

Fast, deterministic — and only as good as the patterns below. Zero false
negatives on code shaped exactly like these three, blind to anything shaped
differently, and blind to leakage that lives in data content rather than
code (e.g. restated fundamentals presented as if point-in-time). A fourth
check — forward-index access like df.iloc[t+1] — was scoped out for now:
highest false-positive risk of the four originally planned, better to ship
three checks that mean something than four where one is mostly noise.

Read each _flag_* docstring for exactly what it does and doesn't catch.
"Clean" means "nothing these three patterns look for was found," not "safe."
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Callable

from ubt.core.interfaces import DataFeed, Strategy
from ubt.core.results import LeakageFlag
from ubt.validation.interfaces import LeakageCheck


class StaticScanner(LeakageCheck):
    def run(self, strategy: Strategy, feed: DataFeed) -> list[LeakageFlag]:
        flags: list[LeakageFlag] = []
        for target in strategy.scan_targets():
            flags.extend(self._scan_callable(target))
        return flags

    def _scan_callable(self, fn: Callable) -> list[LeakageFlag]:
        name = getattr(fn, "__name__", repr(fn))
        try:
            source = textwrap.dedent(inspect.getsource(fn))
            filename = inspect.getsourcefile(fn) or name
            _, start_line = inspect.getsourcelines(fn)
        except (OSError, TypeError) as exc:
            return [LeakageFlag(
                source="static", severity="info",
                message=f"couldn't read source for {name!r} ({exc}); skipped",
            )]

        tree = ast.parse(source)
        flags: list[LeakageFlag] = []
        flags += self._flag_negative_shift(tree, filename, start_line)
        flags += self._flag_centered_rolling(tree, filename, start_line)
        flags += self._flag_fit_before_split(tree, filename, start_line)
        return flags

    def _flag_negative_shift(self, tree: ast.AST, filename: str, offset: int) -> list[LeakageFlag]:
        """Flags `.shift(-N)` / `.shift(periods=-N)` — shifts a series
        backward in time, so row t ends up holding what was originally at
        row t+N. Only sees literal negative constants — misses shifts done
        through a variable that happens to be negative at runtime."""
        flags = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "shift"):
                continue
            candidates = list(node.args) + [kw.value for kw in node.keywords if kw.arg == "periods"]
            for arg in candidates:
                if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub) and isinstance(arg.operand, ast.Constant):
                    flags.append(LeakageFlag(
                        source="static", severity="high",
                        message=f".shift(-{arg.operand.value}) shifts data backward in time — row t sees row t+{arg.operand.value}",
                        file=filename, line=offset + node.lineno - 1,
                    ))
        return flags

    def _flag_centered_rolling(self, tree: ast.AST, filename: str, offset: int) -> list[LeakageFlag]:
        """Flags `.rolling(..., center=True)` — a centered window at row t
        pulls in rows after t, not just before it."""
        flags = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "rolling"):
                continue
            for kw in node.keywords:
                if kw.arg == "center" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    flags.append(LeakageFlag(
                        source="static", severity="high",
                        message="rolling(center=True) — window at row t includes rows after t",
                        file=filename, line=offset + node.lineno - 1,
                    ))
        return flags

    def _flag_fit_before_split(self, tree: ast.AST, filename: str, offset: int) -> list[LeakageFlag]:
        """Heuristic, not proof: flags a `.fit(...)` call at an earlier line
        than the first call with 'split' in its name, anywhere in the same
        snippet. Statement order, not real data-flow analysis — misses
        fit/split split across function boundaries, false-positives on code
        with multiple fit/split calls or branches. Also misses splits done
        by plain slicing (`data[:80], data[80:]`) instead of a named split
        call — confirmed while writing the test for this, not theoretical.
        A prompt to look closer, not a verdict."""
        fit_lines, split_lines = [], []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "fit":
                        fit_lines.append(node.lineno)
                    if "split" in node.func.attr.lower():
                        split_lines.append(node.lineno)
                elif isinstance(node.func, ast.Name) and "split" in node.func.id.lower():
                    split_lines.append(node.lineno)

        if not fit_lines or not split_lines:
            return []
        first_fit, first_split = min(fit_lines), min(split_lines)
        if first_fit < first_split:
            return [LeakageFlag(
                source="static", severity="warn",
                message=(
                    f".fit() on line {offset + first_fit - 1} appears before a split "
                    f"call on line {offset + first_split - 1} — check it isn't fit on the full dataset"
                ),
                file=filename, line=offset + first_fit - 1,
            )]
        return []
