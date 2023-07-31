import pytest 

def pytest_addoption(parser):
    parser.addoption(
            "--save", action="store_true", 
            help="option to save calculated lineloas")

@pytest.fixture
def save(request):
    return request.config.getoption("--save")
