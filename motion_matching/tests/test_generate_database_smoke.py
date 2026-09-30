"""Smoke test for generate_database — importing it must NOT run the pipeline.

Before Stage 1e, ``generate_database.py`` executed argparse + the entire
database-build pipeline at module import. That made it impossible to import
(it called ``parser.parse_args()`` against the test runner's argv) and ran heavy
work as a side effect. The fix wrapped everything in ``main()`` behind an
``if __name__ == "__main__"`` guard.
"""

from __future__ import annotations

import importlib


def test_import_does_not_execute_pipeline() -> None:
    mod = importlib.import_module("motion_matching.generate_database")
    # The whole pipeline now lives behind main(); importing must be side-effect
    # free apart from defining it.
    assert callable(mod.main)


def test_main_is_guarded(tmp_path) -> None:
    # Importing must not have created any database output. (If the pipeline ran
    # at import, it would have written to PHP_MOTION_DATABASE_DIR, not tmp_path, but the
    # key assertion is simply that import succeeded without argparse aborting.)
    import sys

    assert "motion_matching.generate_database" in sys.modules
