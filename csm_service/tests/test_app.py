import os

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("CSM_TEST") != "1",
    reason="CSM microservice tests require CSM_TEST=1 and a GPU-capable environment",
)


def test_placeholder():
    # Intentional: the real microservice tests are only run in GPU CI.
    assert True

