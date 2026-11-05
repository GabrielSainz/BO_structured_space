"""Compatibility entry point for the original integration-test path."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("tests.test_main_entry_point", run_name="__main__")
else:
    from tests.test_main_entry_point import *  # noqa: F403
