"""Repo-wide pytest options.

``--require-cuda`` turns the CUDA smoke suite's environment skips into
hard failures (Task 3, Gate B: the acceptance invocation must fail if
CUDA is required but absent, rather than quietly skipping).  Without the
flag every suite keeps its current skip behavior, so plain ``pytest``
on a host without torch or a GPU is unchanged.
"""

from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption(
        "--require-cuda", action="store_true", default=False,
        help="fail (instead of skip) tests that need a working torch CUDA "
             "device; the Orin acceptance contract from "
             "docs/plans/jetson-orin-nano.md Task 3")
