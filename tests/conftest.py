import pytest


@pytest.fixture(autouse=True, scope="session")
def load_napari_conftest(pytestconfig):
    from napari import conftest

    pytestconfig.pluginmanager.register(conftest, "napari-conftest")
