"""Make historical print-and-count assertions effective under pytest too."""
import pytest


@pytest.fixture(autouse=True)
def enforce_historical_checks(request):
    yield
    module = request.module
    failures = [label for label, ok, *_ in getattr(module, 'results', []) if not ok]
    failures.extend(getattr(module, '_FAILURES', []))
    assert not failures, f'Historical checks failed: {failures}'
