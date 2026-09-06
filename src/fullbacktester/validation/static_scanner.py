"""Static AST scanner for known look-ahead anti-patterns in strategy code.

Fast and deterministic, and only as good as its patterns. It catches code
shaped like the four checks below and nothing else: not leakage through
variables that happen to be negative at runtime, not leakage in data content
(restated fundamentals, survivorship-biased universes), not a strategy that
reads a global DataFrame it was never handed. "Clean" means "none of these
four shapes were found", and the report says so.

Checks:
    ``.shift(-N)``                         HIGH  row t receives row t+N
    ``.rolling(..., center=True)``          HIGH  window at t includes rows after t
    ``.bfill()`` / ``fillna(method="bfill")`` WARN gaps filled with later values
    ``.fit()`` before the first ``split``   WARN  statement-order heuristic only
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from typing import TYPE_CHECKING

from fullbacktester.flags import Flag, Severity

if TYPE_CHECKING:
    from fullbacktester.strategy.base import Strategy

_BACKFILL_NAMES = frozenset({"bfill", "backfill"})


class StaticScanner:
    source = "static"

    def scan(self, strategy: Strategy) -> list[Flag]:
        flags: list[Flag] = []
        for target in strategy.scan_targets():
            flags.extend(self.scan_callable(target))
        return flags

    def scan_callable(self, fn: Callable[..., object]) -> list[Flag]:
        name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)
        try:
            source = textwrap.dedent(inspect.getsource(fn))
            filename = inspect.getsourcefile(fn) or name
            _, start_line = inspect.getsourcelines(fn)
        except (OSError, TypeError) as exc:
            return [
                Flag(
                    source=self.source,
                    severity=Severity.INFO,
                    message=f"could not read source for {name!r} ({exc}); not scanned",
                )
            ]
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return [
                Flag(
                    source=self.source,
                    severity=Severity.INFO,
                    message=f"could not parse source for {name!r} (inline lambda?); not scanned",
                    file=filename,
                    line=start_line,
                )
            ]
        offset = start_line - 1
        flags: list[Flag] = []
        flags += self._flag_negative_shift(tree, filename, offset)
        flags += self._flag_centered_rolling(tree, filename, offset)
        flags += self._flag_backward_fill(tree, filename, offset)
        flags += self._flag_fit_before_split(tree, filename, offset)
        return flags

    def _flag_negative_shift(self, tree: ast.AST, filename: str, offset: int) -> list[Flag]:
        """Literal negative constants only; a variable that is negative at runtime is invisible."""
        flags = []
        for node in ast.walk(tree):
            if not _is_method_call(node, "shift"):
                continue
            assert isinstance(node, ast.Call)
            candidates = list(node.args) + [kw.value for kw in node.keywords if kw.arg == "periods"]
            for arg in candidates:
                if (
                    isinstance(arg, ast.UnaryOp)
                    and isinstance(arg.op, ast.USub)
                    and isinstance(arg.operand, ast.Constant)
                    and isinstance(arg.operand.value, int | float)
                ):
                    n = arg.operand.value
                    flags.append(
                        Flag(
                            source=self.source,
                            severity=Severity.HIGH,
                            message=(
                                f".shift(-{n}) moves data backward in time: row t sees row t+{n}"
                            ),
                            file=filename,
                            line=offset + node.lineno,
                        )
                    )
        return flags

    def _flag_centered_rolling(self, tree: ast.AST, filename: str, offset: int) -> list[Flag]:
        flags = []
        for node in ast.walk(tree):
            if not _is_method_call(node, "rolling"):
                continue
            assert isinstance(node, ast.Call)
            for kw in node.keywords:
                if (
                    kw.arg == "center"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                ):
                    flags.append(
                        Flag(
                            source=self.source,
                            severity=Severity.HIGH,
                            message=(
                                "rolling(center=True): the window at row t includes rows after t"
                            ),
                            file=filename,
                            line=offset + node.lineno,
                        )
                    )
        return flags

    def _flag_backward_fill(self, tree: ast.AST, filename: str, offset: int) -> list[Flag]:
        flags = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            attr = node.func.attr
            hit = attr in _BACKFILL_NAMES
            if attr == "fillna":
                for kw in node.keywords:
                    if (
                        kw.arg == "method"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value in _BACKFILL_NAMES
                    ):
                        hit = True
            if hit:
                flags.append(
                    Flag(
                        source=self.source,
                        severity=Severity.WARN,
                        message="backward fill replaces a gap at row t with a value from after t",
                        file=filename,
                        line=offset + node.lineno,
                    )
                )
        return flags

    def _flag_fit_before_split(self, tree: ast.AST, filename: str, offset: int) -> list[Flag]:
        """Heuristic: a ``.fit`` call on an earlier line than the first ``*split*`` call.

        Statement order, not data flow. Misses fit/split across function
        boundaries and splits done by slicing; false-positives on code with
        several fit/split pairs. A prompt to look, not a verdict.
        """
        fit_lines: list[int] = []
        split_lines: list[int] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
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
        if first_fit >= first_split:
            return []
        return [
            Flag(
                source=self.source,
                severity=Severity.WARN,
                message=(
                    f".fit() on line {offset + first_fit} precedes the first split call on line "
                    f"{offset + first_split}; check the model is not fit on the full sample"
                ),
                file=filename,
                line=offset + first_fit,
            )
        ]


def _is_method_call(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    )
