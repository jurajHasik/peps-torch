import pytest
import context
import config as cfg

def pytest_addoption(parser):
    parser.addoption(
        "--device",
        default="cpu",
        help="choose device",
    )

def pytest_configure(config):
    device= config.getoption("--device")
    cfg.global_args.device= device
