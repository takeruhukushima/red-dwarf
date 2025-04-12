import pytest

def test_polis_import_deprecation_warning():
    with pytest.warns(DeprecationWarning, match=r"'reddwarf.polis' is deprecated"):
        from reddwarf.polis import PolisClient