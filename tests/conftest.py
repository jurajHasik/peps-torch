import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption("--device", type=str, help='cpu or cuda', default='cpu', action='store')
    parser.addoption("--omp_cores", type=int, help='Number of cores to use', action='store', default=1)

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

@pytest.fixture
def config_kwargs(request):
    return {'omp_cores': request.config.getoption("--omp_cores"),
            'device': request.config.getoption("--device"),}
