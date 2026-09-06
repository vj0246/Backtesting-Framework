"""PerturbationTester is declared but not implemented. This pins that down
explicitly: it's expected to start FAILING the moment real behavioral
detection replaces `raise NotImplementedError` — update it then, don't just
delete it. StaticScanner moved to test_static_scanner.py once it stopped
being a stub.
"""

import pytest

from ubt.core.interfaces import RuleBasedStrategy
from ubt.validation.interfaces import PerturbationTester


def test_perturbation_tester_is_not_implemented_yet():
    checker = PerturbationTester()
    strategy = RuleBasedStrategy(lambda df: {})
    with pytest.raises(NotImplementedError):
        checker.run(strategy=strategy, feed=None)
