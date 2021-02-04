def pytest_addoption(parser):
    """
    add `--show-viewer` as a valid command line flag
    """
    parser.addoption(
        "--show-viewer",
        action="store_true",
        default=False,
        help="don't show viewer during tests",
    )
