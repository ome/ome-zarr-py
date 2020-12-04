import pytest

from napari.conftest import make_test_viewer

@pytest.fixture(autouse=True, scope="session")
def load_napari_conftest(pytestconfig):
    from napari import conftest

    pytestconfig.pluginmanager.register(conftest, "napari-conftest")
