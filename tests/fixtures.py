import pytest
import json
import re

REPORT_ID_RE = r"^r[a-z0-9]{20}$"
CONVO_ID_RE = r"^\d[a-z0-9]{4,9}$"

@pytest.fixture
def polis_convo_data(request):
    """
    Allows parametrized testing against curated local datasets, and remote
    datasets.

    - If the parameter does match a dataset descriptor, it's selected for use.
    - If the parameter doesn't match, but it matches the format of either a (1)
    report ID or (2) conversation ID on the Polis platform, this fixture will
    try to download that conversation's data for testing.
    """
    path = None

    if request.param in ["small", "small-no-meta", "small-with-mod-out"]:
        # See: https://pol.is/4cvkai2ctw
        # See: https://pol.is/report/r6bpmcmizi2kyvhzkhfr7
        # 23 ptpts, 2 groups, 0/9 meta, strict=yes
        path = "tests/fixtures/below-100-ptpts"
    elif request.param == "small-no-meta-bad":
        # BUG: DATA INTEGRITY ISSUES.
        # Missing votes via API and so doesn't match user_vote_count
        # See: https://pol.is/6ukkcvfbre
        # See: https://pol.is/report/r64ajcsmp9butzxhzj44c
        # 51 ptpts, 3 groups, 0/29 meta, strict=yes
        path = "tests/fixtures/50ptpt-3gp-strict-no-meta"
    elif request.param == "small-with-meta":
        # See: https://pol.is/2dhnep37ie
        # See: https://pol.is/report/r6ipxzfudddppwesbmtmn
        # 27 ptpts, 3 groups, 4/57 meta, strict=no
        path = "tests/fixtures/25ptpt-3gp-no-strict-with-meta"
    elif request.param in ["medium", "medium-with-meta"]:
        # See: https://pol.is/3ntrtcehas
        # See: https://pol.is/report/r68fknmmmyhdpi3sh4ctc
        # 234 ptpts, 4 groups, 2/53 meta, strict=no
        path = "tests/fixtures/above-100-ptpts"
    elif request.param in ["medium-no-meta", "medium-no-mod-out"]:
        # See: https://pol.is/4asymkcrjf
        # See: https://pol.is/report/r4zdxrdscmukmkakmbz3k
        # 160 ptpts, 3 groups, 0/118 meta, strict=no
        path = "tests/fixtures/150ptpt-3gp-no-strict-no-meta"

    if path:
        filename = "math-pca2.json"
        with open(f"{path}/{filename}", 'r') as f:
            data = json.load(f)

        yield data, path, filename
        return

    # If no match of specific local fixture, see if this looks like remote Polis data.
    if re.match(REPORT_ID_RE, request.param) or re.match(CONVO_ID_RE, request.param):
        from reddwarf.data_loader import Loader
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = tmpdirname
            filename = "math-pca2.json"
            # Download remote data based on Polis report or conversation ID.
            loader = Loader(polis_id=request.param, output_dir=path)
            data = loader.math_data
            yield data, path, filename
            # Tmpdir cleanup will happen after yield completes
            return

    raise ValueError("No fixture matching local or remote fixture data found")
