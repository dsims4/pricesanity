"""Regression coverage for lock ownership, paid attempts, and terminal diagnostics."""

import fcntl
import io
import json
import warnings

import pytest
from databento.common.error import BentoError, BentoWarning

from pricesanity.data import databento_fetch as fetch
from pricesanity.data import managed_download as managed
from pricesanity.data.download_progress import DownloadProgress, component_progress
from test_managed_download import FakeClient, plan, repair_plan, request


def snapshot(directory):
    """Capture file contents and modification times to detect unauthorized writes."""

    return {
        str(file_path.relative_to(directory)): (
            file_path.stat().st_mtime_ns,
            file_path.read_bytes() if file_path.is_file() else None,
        )
        for file_path in [directory, *directory.rglob("*")]
    }


@pytest.mark.parametrize("mode", ["download", "repair", "repair-estimate"])
@pytest.mark.parametrize("existing_log", [False, True])
def test_lock_loser_cli_leaves_all_managed_state_unchanged(
    tmp_path, monkeypatch, capsys, mode, existing_log
):
    """Verify lock loser CLI leaves all managed state unchanged."""

    download_plan = repair_plan(tmp_path) if mode.startswith("repair") else plan(tmp_path)
    download_plan.directory.mkdir(exist_ok=True)
    download_plan.save()

    # Exercise both preserving an existing log and refusing to create a new one after lock
    # contention.
    if existing_log:
        download_plan.log_path.write_text("existing diagnostics\n")

    client = FakeClient()
    monkeypatch.setattr(fetch, "create_historical_client", lambda: client)
    monkeypatch.setattr(
        fetch,
        "build_raw_data_paths",
        lambda *args: fetch.DatabentoRawDataPaths(
            download_plan.directory / "candlesticks.csv",
            download_plan.directory / "status.csv",
            download_plan.directory / "condition.json",
        ),
    )

    # Hold a real OS lock while invoking the CLI so the loser must leave the entire corpus
    # untouched.
    with (download_plan.directory / ".download.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = snapshot(download_plan.directory)
        args = [
            "--start",
            "2020-06-06",
            "--end",
            "2022-09-12",
            "--max-cost-usd",
            "6",
            "--status-chunk-years",
            "1",
            "--raw-data-directory",
            str(tmp_path),
        ]

        # Repair estimation also adopts source ranges and must obey the same writer ownership
        # rules.
        if mode.endswith("estimate"):
            args += ["--estimate-only"]

        assert (fetch.repair_main if mode.startswith("repair") else fetch.download_main)(args) == 1
        assert snapshot(download_plan.directory) == before

    assert not client.calls and not client.counts and not client.estimates
    assert "Download busy:" in capsys.readouterr().err
    assert not (tmp_path / "download-errors.log").exists()


def test_direct_repair_adoption_holds_os_lock(tmp_path, monkeypatch):
    """Verify direct repair adoption holds OS lock."""

    download_plan = repair_plan(tmp_path)
    client = FakeClient()
    original = client.get_record_count

    def count(**params):
        """Check real writer ownership while forwarding the metadata completeness probe."""

        # Use a distinct file descriptor to test actual lock ownership during the metadata
        # probe.
        with (download_plan.directory / ".download.lock").open("a") as contender:
            # Adoption must still own the lock when vendor metadata is queried, not only while
            # files are copied.
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)

        return original(**params)

    monkeypatch.setattr(client, "get_record_count", count)
    download_plan.prepare_repair(client)

    assert download_plan.manifest_path.exists()


def test_read_only_estimate_does_not_require_writer_lock(tmp_path, monkeypatch):
    """Verify read only estimate does not require writer lock."""

    download_plan = plan(tmp_path)
    download_plan.directory.mkdir()
    download_plan.save()
    client = FakeClient()
    monkeypatch.setattr(fetch, "create_historical_client", lambda: client)
    monkeypatch.setattr(
        fetch,
        "build_raw_data_paths",
        lambda *args: fetch.DatabentoRawDataPaths(
            download_plan.directory / "candlesticks.csv",
            download_plan.directory / "status.csv",
            download_plan.directory / "condition.json",
        ),
    )

    # An ordinary estimate is read-only and must succeed while another process owns the writer
    # lock.
    with (download_plan.directory / ".download.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = snapshot(download_plan.directory)

        assert (
            fetch.download_main(
                [
                    "--start",
                    "2020-06-06",
                    "--end",
                    "2022-09-12",
                    "--max-cost-usd",
                    "0",
                    "--estimate-only",
                    "--status-chunk-years",
                    "1",
                ]
            )
            == 0
        )
        assert snapshot(download_plan.directory) == before


def test_stale_plan_cannot_overwrite_new_checkpoint(tmp_path):
    """Verify stale plan cannot overwrite new checkpoint."""

    download_plan = plan(tmp_path)
    download_plan.directory.mkdir()
    download_plan.save()
    stale = plan(tmp_path)
    download_plan.manifest["marker"] = "other writer"
    download_plan.save()

    # A plan read before another checkpoint was saved must be rejected rather than overwrite
    # newer state.
    with pytest.raises(ValueError, match="state changed"):
        stale.run(FakeClient(), max_cost_usd=6)

    assert json.loads(download_plan.manifest_path.read_text())["marker"] == "other writer"
    assert not download_plan.log_path.exists()


def test_metadata_retries_do_not_reserve_paid_attempts(tmp_path, monkeypatch):
    """Verify metadata retries do not reserve paid attempts."""

    client = FakeClient()
    original = client.get_record_count
    failures = [True, True]

    def count(**params):
        """Fail two metadata probes before allowing the first paid request to proceed."""

        # Fail only metadata probes so the eventual first paid attempt still fits the original
        # exact ceiling.
        if failures:
            failures.pop()
            raise BentoError("Response ended prematurely")

        return original(**params)

    monkeypatch.setattr(client, "get_record_count", count)
    download_plan = plan(tmp_path)
    download_plan.run(client, max_cost_usd=6, sleep=lambda _: None)

    assert len(client.calls) == 6
    assert download_plan.manifest["diagnostics"]["errors"] == 2


@pytest.mark.parametrize("ceiling,paid_calls,success", [(6, 1, False), (7, 7, True)])
def test_paid_retry_charged_only_when_started(tmp_path, ceiling, paid_calls, success):
    """Verify paid retry charged only when started."""

    client = FakeClient()
    client.failures[("ohlcv-1m", request().start_date)] = [
        BentoError("Response ended prematurely")
    ]
    download_plan = plan(tmp_path)

    # The larger ceiling permits one extra paid attempt; the exact original ceiling must stop
    # before it.
    if success:
        download_plan.run(client, max_cost_usd=ceiling, sleep=lambda _: None)
    else:
        # Reject the extra paid request before calling the vendor when its estimate would exceed
        # the ceiling.
        with pytest.raises(managed.CostLimitError):
            download_plan.run(client, max_cost_usd=ceiling, sleep=lambda _: None)

    assert len(client.calls) == paid_calls


def test_final_error_logged_once(tmp_path):
    """Verify final error logged once."""

    client = FakeClient()
    client.failures[("ohlcv-1m", request().start_date)] = [ValueError("permanent failure")]
    download_plan = plan(tmp_path)

    # A permanent failure should escape once and be recorded only by the outer operation.
    with pytest.raises(ValueError):
        download_plan.run(client, max_cost_usd=6)

    entries = [json.loads(line) for line in download_plan.log_path.read_text().splitlines()]

    assert len(entries) == 1
    assert entries[0]["category"] == "ValueError"
    assert "Traceback" in entries[0]["traceback"]


def test_warning_details_persist_but_terminal_only_counts(tmp_path, monkeypatch, capsys):
    """Verify warning details persist but terminal only counts."""

    client = FakeClient()
    original = client.get_range
    secret = "private-test-key"
    monkeypatch.setenv("DATABENTO_API_KEY", secret)

    def get_range(**params):
        """Record the paid attempt and return fixture data or the injected failure."""

        warnings.warn(f"verbose degraded warning {secret}", BentoWarning)

        return original(**params)

    monkeypatch.setattr(client, "get_range", get_range)
    download_plan = plan(tmp_path)
    download_plan.run(client, max_cost_usd=6)
    terminal = capsys.readouterr()

    assert "Warnings: 6 | Errors: 0" in terminal.out
    assert "verbose degraded" not in terminal.out + terminal.err
    entries = [json.loads(line) for line in download_plan.log_path.read_text().splitlines()]

    assert len(entries) == 6
    assert all(
        diagnostic_entry["category"] == "BentoWarning"
        and diagnostic_entry["context"]
        and diagnostic_entry["time"]
        for diagnostic_entry in entries
    )
    assert all(
        "verbose degraded warning" in diagnostic_entry["message"] for diagnostic_entry in entries
    )
    assert secret not in download_plan.log_path.read_text()


def test_resume_progress_counts_later_completed_chunks(tmp_path, monkeypatch):
    """Verify resume progress counts later completed chunks."""

    download_plan = plan(tmp_path)
    download_plan.run(FakeClient(), max_cost_usd=6)
    download_plan.chunk_path(
        "candlesticks", download_plan.manifest["components"]["candlesticks"]["chunks"][0]
    ).write_text("corrupt")
    resumed = plan(tmp_path)
    observed = []
    original = DownloadProgress.render

    def render(self, manifest, context, *args, **kwargs):
        """Update the cumulative display without flooding redirected output."""

        # Capture the first resumed candle update to ensure later completed chunks contribute to
        # progress.
        if context.startswith("candlesticks "):
            observed.append(component_progress(manifest["components"]["candlesticks"]))

        return original(self, manifest, context, *args, **kwargs)

    monkeypatch.setattr(DownloadProgress, "render", render)
    resumed.run(FakeClient(), max_cost_usd=1)

    assert observed[0]["completed_chunks"] == 2
    assert observed[0]["total_chunks"] == 3
    assert observed[0]["percent"] == pytest.approx(200 / 3)


@pytest.mark.parametrize("tty", [False, True])
def test_dashboard_redraw_and_compact_final(tmp_path, monkeypatch, tty):
    """Verify dashboard redraw and compact final."""

    class Stream(io.StringIO):
        def isatty(self):
            """Expose the terminal capability required by this rendering scenario."""

            return tty

    monkeypatch.setenv("TERM", "xterm")
    stream = Stream()
    display = DownloadProgress(stream)
    download_plan = plan(tmp_path)
    display.render(download_plan.manifest, "status 2020-06-06 to 2021-01-01", 3, 1)
    display.render(download_plan.manifest, "status 2021-01-01 to 2022-01-01", 4, 1)
    display.render(download_plan.manifest, "status stopped", 4, 1, final=True)
    output = stream.getvalue()

    assert ("\x1b[" in output) == tty
    assert "Warnings: 4 | Errors: 1" in output
    assert "Missing [start,end):" in output
    assert "completed [start,end) ranges" not in output


def test_defaults_match_cli_helper_and_class(tmp_path, monkeypatch):
    """Verify defaults match CLI helper and class."""

    direct = managed.ManagedDownload(request(), tmp_path / "direct")
    calls = []
    monkeypatch.setattr(
        managed.ManagedDownload,
        "run",
        lambda self, *arguments, **keyword_arguments: calls.append(self.manifest["chunk_config"]),
    )
    monkeypatch.setattr(fetch, "create_historical_client", FakeClient)
    paths = fetch.build_raw_data_paths(tmp_path / "helper", request())
    fetch.download_raw_data(FakeClient(), request(), paths, max_cost_usd=13)

    assert (
        fetch.download_main(
            [
                "--start",
                "2020-06-06",
                "--end",
                "2022-09-12",
                "--max-cost-usd",
                "13",
                "--raw-data-directory",
                str(tmp_path / "cli"),
            ]
        )
        == 0
    )
    assert calls == [direct.manifest["chunk_config"]] * 2
    assert direct.manifest["chunk_config"]["status_chunk_months"] == 3


def test_conditions_partition_is_independent(tmp_path):
    """Verify conditions partition is independent."""

    plans = [
        managed.ManagedDownload(
            request(), tmp_path / str(chunk_months), status_chunk_months=chunk_months
        )
        for chunk_months in (1, 3, 6)
    ]

    assert all(
        download_plan.manifest["components"]["conditions"]["chunks"]
        == plans[0].manifest["components"]["conditions"]["chunks"]
        for download_plan in plans
    )
    assert len(plans[0].manifest["components"]["conditions"]["chunks"]) == 3


def test_status_validation_uses_recv_clock(tmp_path):
    """Verify status validation uses recv clock."""

    from pricesanity.data.download_files import validate_csv

    path = tmp_path / "status.csv"
    path.write_text(
        "ts_recv,ts_event,reason,trading_event,is_trading\n"
        "2020-06-06T01:00:00.000000001Z,2020-06-05T23:59:00Z,scheduled,none,Y\n"
        "2020-06-06T01:00:00.000000002Z,2020-06-05T23:58:00Z,scheduled,none,N\n"
    )

    assert validate_csv(path, "status", request().start_date, request().end_date, 2)["rows"] == 2


def test_existing_quarterly_conditions_resume_without_repartitioning(tmp_path, monkeypatch):
    """Verify existing quarterly conditions resume without repartitioning."""

    from copy import deepcopy

    download_plan = managed.ManagedDownload(request(), tmp_path / "old")
    download_plan.manifest.pop("condition_chunk_years")
    download_plan.manifest["components"]["conditions"] = deepcopy(
        download_plan.manifest["components"]["status"]
    )
    download_plan.run(FakeClient(), max_cost_usd=13)
    client = FakeClient()

    # Make every vendor endpoint fail if a completed request attempts unnecessary network work.
    for method in ("get_cost", "get_range", "get_record_count", "get_dataset_condition"):
        monkeypatch.setattr(
            client, method, lambda **kwargs: pytest.fail("Completed corpus contacted vendor")
        )

    resumed = managed.ManagedDownload(request(), download_plan.directory)

    assert len(resumed.manifest["components"]["conditions"]["chunks"]) == 10
    resumed.run(client, max_cost_usd=0)


def test_readonly_warning_logged_outside_locked_corpus(tmp_path, monkeypatch, capsys):
    """Verify readonly warning logged outside locked corpus."""

    download_plan = plan(tmp_path)
    download_plan.directory.mkdir()
    download_plan.save()
    client = FakeClient()

    def estimate(**params):
        """Emit a metadata warning during a read-only cost estimate."""

        warnings.warn("free metadata warning", BentoWarning)

        return 1.0

    monkeypatch.setattr(client, "get_cost", estimate)

    # Hold writer ownership during estimation to prove that read-only diagnostics
    # stay outside the corpus.
    with (download_plan.directory / ".download.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = snapshot(download_plan.directory)

        assert download_plan.estimate(client) == 6
        download_plan.summary(0, 6)

        assert snapshot(download_plan.directory) == before

    terminal = capsys.readouterr()

    assert "free metadata warning" not in terminal.out + terminal.err
    assert "Warnings: 6" in terminal.out
    assert len((tmp_path / "download-errors.log").read_text().splitlines()) == 6


def test_logging_failure_does_not_mask_original_error(tmp_path, monkeypatch):
    """Verify logging failure does not mask original error."""

    download_plan = plan(tmp_path)
    client = FakeClient()
    client.failures[("ohlcv-1m", request().start_date)] = [ValueError("original error")]
    monkeypatch.setattr(
        download_plan,
        "log_failure",
        lambda diagnostic_entry: (_ for _ in ()).throw(OSError("disk unavailable")),
    )

    # The original failure must survive even when writing its diagnostic log also raises an
    # error.
    with pytest.raises(ValueError, match="original error"):
        download_plan.run(client, max_cost_usd=6)

    assert not download_plan._locked


def test_dumb_terminal_has_no_ansi(tmp_path, monkeypatch):
    """Verify dumb terminal has no ANSI."""

    class Stream(io.StringIO):
        def isatty(self):
            """Expose the terminal capability required by this rendering scenario."""

            return True

    monkeypatch.setenv("TERM", "dumb")
    stream = Stream()
    display = DownloadProgress(stream)
    display.render(plan(tmp_path).manifest, "planning", 0, 0, final=True)

    assert "\x1b" not in stream.getvalue()


def test_many_chunks_do_not_expand_noninteractive_output(tmp_path):
    """Verify many chunks do not expand noninteractive output."""

    stream = io.StringIO()
    download_plan = managed.ManagedDownload(request(), tmp_path / "many", status_chunk_months=1)
    display = DownloadProgress(stream)

    # Render many status ranges to verify redirected output stays bounded as the request grows.
    for chunk in download_plan.manifest["components"]["status"]["chunks"]:
        display.render(download_plan.manifest, f"status {chunk['start']} to {chunk['end']}", 0, 0)

    display.render(download_plan.manifest, "status stopped", 0, 0, final=True)

    assert len(stream.getvalue().splitlines()) < 20
    assert "(+25 more)" in stream.getvalue()


def test_metadata_failure_between_paid_attempts_does_not_double_charge(tmp_path, monkeypatch):
    """Verify metadata failure between paid attempts does not double charge."""

    download_plan = plan(tmp_path)
    client = FakeClient()
    original_count, original_get = client.get_record_count, client.get_range
    key = ("ohlcv-1m", request().start_date)
    client.failures[key] = [BentoError("Response ended prematurely")]
    missing_probe = []

    def get_range(**params):
        """Record the paid attempt and return fixture data or the injected failure."""

        # Remember that the first paid request failed before injecting a later metadata-only
        # failure.
        try:
            return original_get(**params)

        # Remove the probe checkpoint to force the next attempt through metadata again.
        except BentoError:
            download_plan.manifest["components"]["candlesticks"]["chunks"][0].pop("expected_rows")
            missing_probe.append(True)
            raise

    def count(**params):
        """Fail one replacement metadata probe between two paid attempts."""

        # Fail the repeated metadata probe once; this attempt must not reserve another paid
        # request.
        if missing_probe:
            missing_probe.pop()
            raise BentoError("Response ended prematurely")

        return original_count(**params)

    monkeypatch.setattr(client, "get_range", get_range)
    monkeypatch.setattr(client, "get_record_count", count)
    download_plan.run(client, max_cost_usd=7, sleep=lambda _: None)

    assert len(client.calls) == 7


def test_exhausted_retries_log_exactly_one_entry_per_failure(tmp_path):
    """Verify exhausted retries log exactly one entry per failure."""

    client = FakeClient()
    client.failures[("ohlcv-1m", request().start_date)] = [
        BentoError("Response ended prematurely")
    ] * 3
    download_plan = plan(tmp_path)

    # Exhaust the bounded attempts so diagnostic counts can be checked without an eventual
    # success.
    with pytest.raises(BentoError):
        download_plan.run(client, max_cost_usd=8, sleep=lambda _: None)

    entries = [json.loads(line) for line in download_plan.log_path.read_text().splitlines()]

    assert len(entries) == 3
    assert download_plan.manifest["diagnostics"]["errors"] == 3


def test_condition_policy_keeps_exact_inclusive_end(tmp_path):
    """Verify condition policy keeps exact inclusive end."""

    from datetime import date

    identity = fetch.DatabentoFetchRequest(date(2020, 12, 31), date(2021, 1, 2))
    download_plan = managed.ManagedDownload(identity, tmp_path / "edges")
    client = FakeClient()
    download_plan.run(client, max_cost_usd=4)

    assert client.conditions == [
        (date(2020, 12, 31), date(2020, 12, 31)),
        (date(2021, 1, 1), date(2021, 1, 1)),
    ]


def test_conditions_manifest_gap_is_refused(tmp_path):
    """Verify conditions manifest gap is refused."""

    download_plan = managed.ManagedDownload(request(), tmp_path / "gap")
    download_plan.directory.mkdir()
    download_plan.manifest["components"]["conditions"]["chunks"][1]["start"] = "2021-01-02"
    download_plan.save()

    # Reject the damaged conditions partition instead of silently accepting a gap in coverage.
    with pytest.raises(ValueError, match="chunk ranges"):
        managed.ManagedDownload(request(), download_plan.directory)


def test_final_display_failure_releases_lock_ownership(tmp_path, monkeypatch):
    """Verify final display failure releases lock ownership."""

    download_plan = plan(tmp_path)

    def render(*, final=False, event=False):
        """Update the cumulative display without flooding redirected output."""

        # Fail only final display output to exercise ownership cleanup after the data work has
        # finished.
        if final:
            raise BrokenPipeError("output closed")

    monkeypatch.setattr(download_plan, "update_progress", render)

    # A broken output pipe may propagate, but it must not leave the instance claiming lock
    # ownership.
    with pytest.raises(BrokenPipeError):
        download_plan.run(FakeClient(), max_cost_usd=6)

    assert not download_plan._locked

    # Reacquire the real lock to verify that failed terminal output also released OS-level
    # ownership.
    with (download_plan.directory / ".download.lock").open("a") as contender:
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
