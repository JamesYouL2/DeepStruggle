from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterator

import pytest

from tools.lib.corpus_paths import corpus_dir, missing_corpus_reason
from tools.lib.engine_fingerprint import staleness_reason
from web.server.replay import REPLAYS_DIR_ENV

if TYPE_CHECKING:
    from tests.replay_corpus import ReplayCorpus


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-fuzz",
        action="store_true",
        default=False,
        help="Run differential fuzzing tests against external engines",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-fuzz"):
        return
    expr = getattr(config.option, "markexpr", "") or ""
    if "differential_fuzz" in expr:
        return
    for arg in config.args:
        if "test_differential_fuzzing" in arg:
            return

    skip_marker = pytest.mark.skip(
        reason="Differential fuzzers against external engines are not run by default. Use --run-fuzz, -m differential_fuzz, or specify tests/differential/test_differential_fuzzing.py"
    )
    for item in items:
        if "differential_fuzz" in item.keywords:
            item.add_marker(skip_marker)

    # Everything under tests/replayer needs the corpus. Requiring it centrally keeps 59 test
    # files from each carrying (and drifting on) their own guard.
    #
    # Appending to `fixturenames` rather than adding a `usefixtures` marker: by the time this
    # hook runs the fixture closure for each item has already been computed, so a marker added
    # here is silently ignored and the tests fail later with a bare FileNotFoundError instead
    # of the message telling you to run the downloader.
    for item in items:
        if not isinstance(item, pytest.Function):
            continue
        if "replayer" in item.path.parts and "require_corpus" not in item.fixturenames:
            item.fixturenames.insert(0, "require_corpus")


def pytest_ignore_collect(collection_path, config: pytest.Config) -> bool | None:
    """Keep tests/differential out of collection unless it is asked for.

    Marking those tests skipped is not enough: the skip is applied after collection, and the
    modules fail at *import*, so a plain `pytest tests/` aborts with a collection error instead
    of running the suite. Ignoring the directory here rather than through an `--ignore` in
    pytest.ini keeps the escape hatches working -- `--run-fuzz`, `-m differential_fuzz`, or
    naming a path under tests/differential -- since those are all visible on `config`.
    """
    if "differential" not in collection_path.parts:
        return None
    if config.getoption("--run-fuzz"):
        return None
    if "differential_fuzz" in (getattr(config.option, "markexpr", "") or ""):
        return None
    if any("differential" in str(arg) for arg in config.args):
        return None
    return True


@pytest.fixture(scope="session")
def generated_replay_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A directory holding one freshly generated replay, and the server pointed at it.

    `data/replays/` is git-ignored, so nothing in the suite may assume it holds anything. The
    tempting alternative -- skip when it is empty -- is the failure mode this repository keeps
    rediscovering: a check that quietly measures nothing while passing everywhere. It has
    happened repeatedly -- pyrefly was found exiting 0 having examined zero files, and the test
    written to catch a stale engine spent two commits silently skipping after a directory move.

    Generating instead also tests more than a committed file could. Invariant 6 requires every
    replay to be written by `generate_self_play_replay`, so a schema assertion over generated
    output covers the writer as it is now; a frozen fixture can drift from what the writer emits
    and the test would still pass. No checkpoint is needed -- the generator falls back to an
    untrained network when none is on disk -- and one game takes about a second. A single replay
    is also ~3.7 MB, which is reason enough not to commit one.
    """
    target = tmp_path_factory.mktemp("replays")
    out = str(target / "fixture_selfplay.tslog.json")

    previous = os.environ.get(REPLAYS_DIR_ENV)
    os.environ[REPLAYS_DIR_ENV] = str(target)
    try:
        from ai.models.coldwar_net import create_coldwar_net
        from tools.lib.self_play import generate_self_play_replay

        _, saved = generate_self_play_replay(
            model=create_coldwar_net("cpu"),
            seed=2026,
            game_id="fixture_selfplay",
            output_path=out,
            device="cpu",
            verbose=False,
        )
        # A generation failure must surface as an error, never as an empty directory that the
        # dependent tests then skip over.
        assert os.path.exists(saved) and os.path.getsize(saved) > 0, (
            f"replay fixture was not written to {saved}"
        )
        yield str(target)
    finally:
        if previous is None:
            os.environ.pop(REPLAYS_DIR_ENV, None)
        else:
            os.environ[REPLAYS_DIR_ENV] = previous


# ---------------------------------------------------------------------------------------------
# ts-replayer corpus
# ---------------------------------------------------------------------------------------------

def _verify_engine_is_not_stale() -> None:
    """Abort the run when the imported ts_engine was not built from these sources.

    Key invariant 10 -- never measure against a stale engine -- was enforced only for commands
    launched through `tools/scripts/check_engine_fresh.sh`. pytest is the one entry point that
    never goes through it, and the one where the mistake is easiest to make: `pytest.ini` sets

        pythonpath = build/release .

    and an ini `pythonpath` is prepended *ahead of* the environment's PYTHONPATH. So a stale
    extension left in build/release wins over the build the caller passed in, and nothing says
    which one was loaded. A checkout carrying a months-old one there produced 175 failures and
    222 errors that had nothing to do with the code under test. The green version of the same
    mistake is worse, because nothing looks wrong at all.

    The check is against the directory `ts_engine` was actually imported from, not against the
    build directory named in a config file -- importing one build while believing you configured
    another is precisely the failure.

    No engine at all is left alone: an ImportError here would say nothing useful, and the tests
    that need it fail on their own import with a clearer message.
    """
    try:
        import ts_engine
    except Exception:
        return

    module_file = getattr(ts_engine, "__file__", None)
    if not module_file:
        return

    reason = staleness_reason(module_file)
    if reason is not None:
        raise pytest.UsageError("stale ts_engine\n\n" + reason)


def _check_every_cached_mask() -> None:
    """Every legal mask the search tree and the batch runner read from their cache is compared with
    a freshly generated one, and the process aborts if they differ (ts::CachedState). The cache is
    correct by construction -- only the slot's own calls can change its state, and each one
    invalidates -- and this is what checks the construction, on everything the suite plays."""
    try:
        import ts_engine
    except Exception:
        return
    ts_engine.set_mask_cache_checks(True)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "corpus_full: converts the entire ts-replayer corpus; minutes, not seconds. "
        "Deselected by default -- run with -m corpus_full before merging.",
    )
    _verify_engine_is_not_stale()
    _check_every_cached_mask()


@pytest.fixture(scope="session")
def corpus() -> ReplayCorpus:
    """The ts-replayer corpus (tests/replay_corpus.py), failing with instructions when absent.

    Deliberately a failure and not a skip. Two thirds of tests/replayer used to sit behind
    `skipif(corpus not downloaded)` pointed at a hardcoded absolute path, so on any machine but
    the one that path was written for, they skipped and the suite reported green; ~30 more files
    each carried a `_game()` helper ending in `pytest.skip("corpus not downloaded")`.

    Session-scoped, so a conversion is done once per replay id per worker; the conversions it
    returns are SHARED and read-only. Imported here rather than at the top because it imports the
    engine, and tests/lint and tests/docs run without a build.
    """
    from tests.replay_corpus import ReplayCorpus

    reason = missing_corpus_reason()
    if reason is not None:
        pytest.fail(reason, pytrace=False)
    return ReplayCorpus(corpus_dir())


@pytest.fixture(scope="session")
def require_corpus() -> None:
    """Fail -- not skip -- when the ts-replayer corpus is missing.

    Applied automatically to everything under tests/replayer (see
    pytest_collection_modifyitems), so no test file has to remember it.
    """
    reason = missing_corpus_reason()
    if reason is not None:
        pytest.fail(reason, pytrace=False)
