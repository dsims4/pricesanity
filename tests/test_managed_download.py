"""Managed downloads exercise real files with fake metadata and time-series calls."""

from collections import Counter
from datetime import date, timedelta
import json
from pathlib import Path

import pytest
from databento.common.error import BentoError, BentoClientError

from pricesanity.data import databento_fetch as fetch
from pricesanity.data import managed_download as managed
from pricesanity.data.download_files import atomic_json, assemble_files, validate_csv


class FakeClient:
    def __init__(self):
        """Validate the request and initialize its resumable local state."""

        self.metadata = self
        self.timeseries = self
        self.calls = []
        self.estimates = []
        self.counts = []
        self.conditions = []
        self.failures = {}
        self.import_count = 1

    def get_cost(self, **params):
        """Record the metadata request and return the fixed per-chunk test cost."""

        self.estimates.append(params)

        return 1.0

    def get_record_count(self, **params):
        """Return deterministic metadata counts for the requested fake schema."""

        self.counts.append(params)

        return self.import_count if params["schema"] == "ohlcv-1m" else 1

    def get_dataset_condition(self, *, dataset, start_date, end_date):
        """Return one available-condition record per inclusive metadata date."""

        self.conditions.append((start_date, end_date))

        return [
            {
                "date": (start_date + timedelta(days=day_offset)).isoformat(),
                "condition": "available",
            }
            for day_offset in range((end_date - start_date).days + 1)
        ]

    def get_range(self, **params):
        """Record the paid attempt and return fixture data or the injected failure."""

        key = (params["schema"], params["start"])
        self.calls.append(key)
        failures = self.failures.get(key, [])

        # Consume injected failures in order so retry tests control which attempts reach a
        # successful response.
        if failures:
            raise failures.pop(0)

        stamp = params["start"].isoformat() + "T01:00:00Z"
        content = (
            "ts_event,open,high,low,close\n" + stamp + ",100,101,99,100\n"
            if params["schema"] == "ohlcv-1m"
            else "ts_event,reason,trading_event,is_trading\n" + stamp + ",scheduled,none,Y\n"
        )

        class Store:
            def to_csv(self, path, **kwargs):
                """Write the fake response while preserving exclusive file-creation behavior."""

                assert kwargs["mode"] == "x"

                # Match the SDK's exclusive file creation so stale temporary files cannot be
                # silently overwritten.
                with Path(path).open("x") as stream:
                    stream.write(content)

        return Store()


def request():
    """Use a range with partial edge years to exercise chunk boundaries."""

    return fetch.DatabentoFetchRequest(date(2020, 6, 6), date(2022, 9, 12))


def plan(tmp_path, **kwargs):
    """Build an isolated yearly download plan for deterministic retry and resume tests."""

    # Existing scenarios explicitly exercise yearly status; defaults have separate coverage.
    # Keep these small fixtures yearly unless a test explicitly exercises the quarterly default.
    if "status_chunk_years" not in kwargs and "status_chunk_months" not in kwargs:
        kwargs["status_chunk_years"] = 1

    return managed.ManagedDownload(request(), tmp_path / "corpus", **kwargs)


def test_year_chunks_no_gaps_or_overlaps():
    """Verify year chunks no gaps or overlaps."""

    chunks = managed.yearly_ranges(date(2010, 6, 6), date(2026, 9, 12))

    assert len(chunks) == 17
    assert chunks[0] == (date(2010, 6, 6), date(2011, 1, 1))
    assert chunks[-1] == (date(2026, 1, 1), date(2026, 9, 12))
    assert all(
        previous_range[1] == current_range[0]
        for previous_range, current_range in zip(chunks, chunks[1:])
    )
    assert managed.yearly_ranges(date(2020, 6, 1), date(2022, 1, 1), 2) == [
        (date(2020, 6, 1), date(2022, 1, 1))
    ]


@pytest.mark.parametrize("months", [1, 2, 3, 4, 6, 12])
def test_month_chunks_cover_partial_edges_and_leap_day(months):
    """Verify month chunks cover partial edges and leap day."""

    start, end = date(2020, 2, 29), date(2021, 5, 12)
    chunks = managed.monthly_ranges(start, end, months)

    assert chunks[0][0] == start and chunks[-1][1] == end
    assert all(range_start < range_end for range_start, range_end in chunks)
    assert all(
        previous_range[1] == current_range[0]
        for previous_range, current_range in zip(chunks, chunks[1:])
    )
    assert all(
        range_end.day == 1 and (range_end.month - 1) % months == 0 for _, range_end in chunks[:-1]
    )
    assert (
        sum((range_end - range_start).days for range_start, range_end in chunks)
        == (end - start).days
    )


def test_month_chunks_handle_maximum_year():
    """Verify month chunks handle maximum year."""

    assert managed.monthly_ranges(date(9999, 11, 1), date(9999, 12, 31), 3) == [
        (date(9999, 11, 1), date(9999, 12, 31))
    ]


@pytest.mark.parametrize("months", [0, -1, 5, 13])
def test_invalid_month_chunks_refused(months):
    """Verify invalid month chunks refused."""

    # Reject month sizes that cannot produce the supported stable calendar partition.
    with pytest.raises(ValueError, match="months"):
        managed.monthly_ranges(request().start_date, request().end_date, months)


def test_quarterly_status_resume_preserves_candle_chunking(tmp_path):
    """Verify quarterly status resume preserves candle chunking."""

    client = FakeClient()
    download_plan = plan(tmp_path, status_chunk_months=3)

    assert len(download_plan.manifest["components"]["candlesticks"]["chunks"]) == 3
    assert len(download_plan.manifest["components"]["status"]["chunks"]) == 10
    failed = ("status", date(2021, 4, 1))
    client.failures[failed] = [BentoError("Response ended prematurely")]

    # Interrupt the first invocation so the following invocation must reuse durable completed
    # chunks.
    with pytest.raises(BentoError):
        download_plan.run(client, max_cost_usd=13, retries=0)

    completed = set(client.calls) - {failed}
    resumed = plan(tmp_path, status_chunk_months=3)
    remaining = resumed.estimate(client)
    resumed.run(client, max_cost_usd=remaining)

    assert all(Counter(client.calls)[key] == 1 for key in completed)
    assert all(part["state"] == "complete" for part in resumed.manifest["components"].values())
    status_ranges = [
        (chunk["start"], chunk["end"])
        for chunk in resumed.manifest["components"]["status"]["chunks"]
    ]
    condition_ranges = [
        (range_start.isoformat(), (range_end + timedelta(days=1)).isoformat())
        for range_start, range_end in client.conditions
    ]

    assert condition_ranges == [
        (range_start.isoformat(), range_end.isoformat())
        for range_start, range_end in managed.yearly_ranges(
            request().start_date, request().end_date
        )
    ]

    # A different month partition cannot reinterpret the ranges recorded by an existing
    # checkpoint.
    with pytest.raises(ValueError, match="chunk_config"):
        plan(tmp_path)

    # Changing from months to years also requires a distinct request rather than reusing this
    # manifest.
    with pytest.raises(ValueError, match="chunk_config"):
        plan(tmp_path, status_chunk_months=1)


def test_monthly_status_cli_and_exclusive_options(tmp_path, monkeypatch):
    """Verify monthly status CLI and exclusive options."""

    client = FakeClient()
    monkeypatch.setattr(fetch, "create_historical_client", lambda: client)
    args = [
        "--start",
        "2020-06-06",
        "--end",
        "2022-09-12",
        "--max-cost-usd",
        "31",
        "--status-chunk-months",
        "1",
        "--raw-data-directory",
        str(tmp_path),
    ]

    assert fetch.download_main(args) == 0
    assert (
        len([request_call for request_call in client.calls if request_call[0] == "ohlcv-1m"]) == 3
    )
    assert len(
        [request_call for request_call in client.calls if request_call[0] == "status"]
    ) == 28

    # Argparse must reject conflicting status units before any download plan is constructed.
    with pytest.raises(SystemExit):
        fetch.build_download_argument_parser().parse_args(args + ["--status-chunk-years", "1"])


def test_download_cli_defaults_to_yearly_ohlc_and_quarterly_status(tmp_path, monkeypatch):
    """Verify download CLI defaults to yearly OHLC and quarterly status."""

    client = FakeClient()
    monkeypatch.setattr(fetch, "create_historical_client", lambda: client)
    args = [
        "--start",
        "2020-06-06",
        "--end",
        "2022-09-12",
        "--max-cost-usd",
        "13",
        "--raw-data-directory",
        str(tmp_path),
    ]

    assert fetch.download_main(args) == 0
    assert len([call for call in client.calls if call[0] == "ohlcv-1m"]) == 3
    assert len([call for call in client.calls if call[0] == "status"]) == 10
    manifest = json.loads(next(tmp_path.rglob("manifest.json")).read_text())

    assert manifest["chunk_config"] == {
        "candlestick_chunk_years": 1,
        "status_chunk_years": 1,
        "status_chunk_months": 3,
    }


@pytest.mark.parametrize("component", ["ohlcv-1m", "status"])
def test_failure_and_resume_only_missing_chunks(tmp_path, component):
    """Verify failure and resume only missing chunks."""

    client = FakeClient()
    failed = (component, date(2021, 1, 1))
    client.failures[failed] = [BentoError("Response ended prematurely")]
    initial = plan(tmp_path)

    # Leave an interrupted request to verify that the next invocation estimates only missing
    # ranges.
    with pytest.raises(BentoError):
        initial.run(client, max_cost_usd=6, retries=0)

    completed = set(client.calls) - {failed}
    resumed = plan(tmp_path)
    expected_missing = 6 - len(completed)

    assert resumed.estimate(client) == expected_missing
    assert not any(
        (request_parameters["schema"], request_parameters["start"]) in completed
        for request_parameters in client.estimates[-expected_missing:]
    )
    resumed.run(client, max_cost_usd=expected_missing)

    assert all(Counter(client.calls)[key] == 1 for key in completed)
    assert all(
        component_state["state"] == "complete"
        for component_state in resumed.manifest["components"].values()
    )
    assert client.conditions[-1][1] == date(2022, 9, 11)
    assert (resumed.directory / "candlesticks.csv").read_text().count("ts_event") == 1


def test_transient_retry_and_backoff(tmp_path):
    """Verify transient retry and backoff."""

    client = FakeClient()
    key = ("ohlcv-1m", request().start_date)
    client.failures[key] = [BentoError("Response ended prematurely")]
    sleeps = []
    plan(tmp_path).run(client, max_cost_usd=7, sleep=sleeps.append)

    assert Counter(client.calls)[key] == 2
    assert sleeps == [1.0]


@pytest.mark.parametrize(
    "error,attempts",
    [
        (BentoError("Response ended prematurely"), 3),
        (BentoClientError(401, message="Unauthorized"), 1),
    ],
)
def test_retry_exhaustion_and_permanent_errors(tmp_path, error, attempts):
    """Verify retry exhaustion and permanent errors."""

    client = FakeClient()
    key = ("ohlcv-1m", request().start_date)
    client.failures[key] = [error] * 4
    sleeps = []

    # Permanent vendor failures must propagate without being retried as temporary transport
    # errors.
    with pytest.raises(type(error)):
        plan(tmp_path).run(client, max_cost_usd=10, sleep=sleeps.append)

    assert len(client.calls) == attempts
    assert sleeps == ([1.0, 2.0] if attempts == 3 else [])


def test_retry_cannot_bypass_cost_ceiling(tmp_path):
    """Verify retry cannot bypass cost ceiling."""

    client = FakeClient()
    client.failures[("ohlcv-1m", request().start_date)] = [
        BentoError("Response ended prematurely")
    ]

    # An exact initial ceiling cannot authorize another possibly billable response attempt.
    with pytest.raises(managed.CostLimitError, match="Retry"):
        plan(tmp_path).run(client, max_cost_usd=6, sleep=lambda seconds: None)

    assert len(client.calls) == 1


def test_ceiling_prevents_all_paid_requests(tmp_path):
    """Verify ceiling prevents all paid requests."""

    client = FakeClient()

    # Reject an insufficient initial budget before requesting any paid data.
    with pytest.raises(managed.CostLimitError):
        plan(tmp_path).run(client, max_cost_usd=5)

    assert client.calls == []


def test_partial_files_are_not_complete(tmp_path):
    """Verify partial files are not complete."""

    client = FakeClient()
    client.failures[("ohlcv-1m", request().start_date)] = [ValueError("permanent")]
    download_plan = plan(tmp_path)

    # Invalid estimates cannot establish a safe additional-spending ceiling.
    with pytest.raises(ValueError):
        download_plan.run(client, max_cost_usd=6)

    chunk = download_plan.manifest["components"]["candlesticks"]["chunks"][0]
    download_plan.partial_path(download_plan.chunk_path("candlesticks", chunk)).write_text(
        "partial"
    )

    assert len(plan(tmp_path).missing()) == 9


def test_crash_between_rename_and_checkpoint_recovers(tmp_path, monkeypatch):
    """Verify crash between rename and checkpoint recovers."""

    download_plan = plan(tmp_path)
    original_replace = Path.replace
    triggered = False

    def replace(path, target):
        """Interrupt once at chunk publication while preserving normal replacements afterward."""

        nonlocal triggered
        result = original_replace(path, target)

        # Interrupt once after the temporary CSV is produced to exercise chunk durability
        # boundaries.
        if path.name.endswith(".partial.csv") and not triggered:
            triggered = True
            raise KeyboardInterrupt()

        return result

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(download_plan, "log_failure", lambda error: None)
    client = FakeClient()

    # Keyboard interruption must escape while leaving already committed ranges available to
    # resume.
    with pytest.raises(KeyboardInterrupt):
        download_plan.run(client, max_cost_usd=6)

    monkeypatch.setattr(Path, "replace", original_replace)
    plan(tmp_path).run(client, max_cost_usd=5)

    assert Counter(client.calls)[("ohlcv-1m", request().start_date)] == 1


def test_atomic_checkpoint_keeps_previous_on_failure(tmp_path, monkeypatch):
    """Verify atomic checkpoint keeps previous on failure."""

    path = tmp_path / "manifest.json"
    atomic_json(path, {"state": "old"})
    monkeypatch.setattr(Path, "replace", lambda *args: (_ for _ in ()).throw(OSError("disk")))

    # A failed checkpoint replacement must preserve the previously published manifest.
    with pytest.raises(OSError):
        atomic_json(path, {"state": "new"})

    assert json.loads(path.read_text()) == {"state": "old"}
    assert list(tmp_path.iterdir()) == [path]


def test_conflicting_manifest_and_unmanaged_directory_refused(tmp_path):
    """Verify conflicting manifest and unmanaged directory refused."""

    download_plan = plan(tmp_path)
    download_plan.directory.mkdir()

    # Existing unmanaged files require explicit adoption; directory existence cannot imply a
    # manifest.
    with pytest.raises(ValueError, match="Unmanaged"):
        plan(tmp_path)

    download_plan.save()
    manifest = json.loads(download_plan.manifest_path.read_text())
    manifest["request"]["symbol"] = "OTHER"
    atomic_json(download_plan.manifest_path, manifest)

    # Reject the changed request identity before its checkpoint can be reused for another
    # instrument.
    with pytest.raises(ValueError, match="Conflicting"):
        plan(tmp_path)


def test_local_assembly_retry_has_no_network_calls(tmp_path, monkeypatch):
    """Verify local assembly retry has no network calls."""

    download_plan = plan(tmp_path)
    original_assemble = managed.assemble_files
    monkeypatch.setattr(
        managed, "assemble_files", lambda *args: (_ for _ in ()).throw(OSError("disk"))
    )
    client = FakeClient()

    # Interrupt local assembly after downloading so the next run must reuse completed chunks.
    with pytest.raises(OSError):
        download_plan.run(client, max_cost_usd=6)

    monkeypatch.setattr(managed, "assemble_files", original_assemble)

    # Disallow every vendor endpoint to prove assembly retry is entirely local.
    for method in ("get_range", "get_cost", "get_record_count", "get_dataset_condition"):
        monkeypatch.setattr(
            client, method, lambda **kwargs: pytest.fail("Assembly contacted vendor")
        )

    plan(tmp_path).run(client, max_cost_usd=0)


def test_corrupt_chunk_is_requested_again_but_valid_chunks_are_not(tmp_path):
    """Verify corrupt chunk is requested again but valid chunks are not."""

    client = FakeClient()
    download_plan = plan(tmp_path)
    download_plan.run(client, max_cost_usd=6)
    first = download_plan.manifest["components"]["candlesticks"]["chunks"][0]
    download_plan.chunk_path("candlesticks", first).write_text("corrupt")
    before = len(client.calls)
    plan(tmp_path).run(client, max_cost_usd=1)

    assert len(client.calls) == before + 1


def test_csv_assembly_allows_distinct_simultaneous_statuses(tmp_path):
    """Verify CSV assembly allows distinct simultaneous statuses."""

    path = tmp_path / "part.csv"
    header = "ts_event,reason,trading_event,is_trading\n"
    path.write_text(
        header + "2020-06-06T01:00:00Z,scheduled,none,Y\n2020-06-06T01:00:00Z,scheduled,none,N\n"
    )
    final = tmp_path / "status.csv"
    stats = assemble_files([path], final, "status", date(2020, 6, 6), date(2020, 6, 7), 2)

    assert stats["rows"] == 2

    # Distinct simultaneous statuses are valid, but duplicated chunks must fail final
    # validation.
    with pytest.raises(ValueError, match="chronological|Duplicate"):
        assemble_files([path, path], final, "status", date(2020, 6, 6), date(2020, 6, 7), 4)

    assert validate_csv(final, "status", date(2020, 6, 6), date(2020, 6, 7))["rows"] == 2


def test_csv_assembly_rejects_header_mismatch(tmp_path):
    """Verify CSV assembly rejects header mismatch."""

    first_chunk_path, second_chunk_path = tmp_path / "a.csv", tmp_path / "b.csv"
    first_chunk_path.write_text("ts_event,reason,trading_event,is_trading\n")
    second_chunk_path.write_text("ts_event,is_trading,reason,trading_event\n")

    # Different headers cannot be concatenated because later records would use the wrong column
    # meanings.
    with pytest.raises(ValueError, match="headers"):
        assemble_files(
            [first_chunk_path, second_chunk_path],
            tmp_path / "status.csv",
            "status",
            date(2020, 1, 1),
            date(2021, 1, 1),
            0,
        )


def test_cli_short_error_full_sanitized_log(tmp_path, monkeypatch, capsys):
    """Verify CLI short error full sanitized log."""

    client = FakeClient()
    secret = "test-secret-do-not-log"
    monkeypatch.setenv("DATABENTO_API_KEY", secret)
    client.failures[("ohlcv-1m", request().start_date)] = [ValueError(f"API_KEY={secret} invalid")]
    monkeypatch.setattr(fetch, "create_historical_client", lambda: client)
    status = fetch.download_main(
        [
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
    )
    terminal = capsys.readouterr()

    assert status == 1
    assert "Traceback" not in terminal.err
    assert "Resumable: True" in terminal.err
    log = next(tmp_path.rglob("download.log")).read_text()

    assert "Traceback" in log and "ValueError" in log and "candlesticks" in log
    assert secret not in log + terminal.err + terminal.out


def repair_plan(tmp_path, *, complete=False):
    """Create an interrupted candle source that repair can validate and adopt."""

    directory = tmp_path / "corpus"
    directory.mkdir(exist_ok=True)
    dates = ["2020-06-06"]

    # Vary source completeness so repair exercises both safe adoption and replacement of missing
    # ranges.
    if complete:
        dates.extend(["2021-01-01", "2022-01-01"])

    records = "".join(f"{day}T01:00:00Z,100,101,99,100\n" for day in dates)
    (directory / "candlesticks.csv").write_text("ts_event,open,high,low,close\n" + records)
    (directory / ".DS_Store").write_bytes(b"irrelevant")

    return managed.ManagedDownload(request(), directory, repair=True, status_chunk_years=1)


def test_repair_adopts_complete_ohlc_ranges_without_downloading_them(tmp_path):
    """Verify repair adopts complete OHLC ranges without downloading them."""

    download_plan = repair_plan(tmp_path, complete=True)
    path = download_plan.directory / "candlesticks.csv"
    client = FakeClient()
    download_plan.run(client, max_cost_usd=3)

    assert all(schema == "status" for schema, _ in client.calls)
    assert (
        validate_csv(
            path,
            "candlesticks",
            request().start_date,
            request().end_date,
            3,
        )["rows"] == 3
    )
    assert all(
        part["state"] == "complete" for part in download_plan.manifest["components"].values()
    )
    assert download_plan.manifest["repair"]["sources"]["candlesticks"]["adopted_rows"] == 3
    assert len([count for count in client.counts if count["schema"] == "ohlcv-1m"]) == 3


def test_repair_merges_partial_ohlc_with_only_missing_downloads(tmp_path):
    """Verify repair merges partial OHLC with only missing downloads."""

    download_plan = repair_plan(tmp_path)
    client = FakeClient()
    download_plan.run(client, max_cost_usd=5)
    candle_calls = [call for call in client.calls if call[0] == "ohlcv-1m"]

    assert candle_calls == [
        ("ohlcv-1m", date(2021, 1, 1)),
        ("ohlcv-1m", date(2022, 1, 1)),
    ]
    stats = validate_csv(
        download_plan.directory / "candlesticks.csv",
        "candlesticks",
        request().start_date,
        request().end_date,
        3,
    )

    assert stats["rows"] == 3


def test_repair_rejects_unknown_files(tmp_path):
    """Verify repair rejects unknown files."""

    repair_plan(tmp_path)
    (tmp_path / "corpus" / "unknown.csv").write_text("unknown")

    # Unfamiliar files may belong to another corpus and must prevent automatic repair adoption.
    with pytest.raises(ValueError, match="conflicting"):
        managed.ManagedDownload(request(), tmp_path / "corpus", repair=True)


def test_repair_ceiling_is_enforced_after_adoption(tmp_path):
    """Verify repair ceiling is enforced after adoption."""

    download_plan = repair_plan(tmp_path)
    client = FakeClient()

    # Repair metadata and adopted files do not authorize paid requests beyond the approved
    # remaining estimate.
    with pytest.raises(managed.CostLimitError, match="Remaining estimate"):
        download_plan.run(client, max_cost_usd=4)

    assert client.calls == []


def test_interrupted_command_records_checkpoint_and_returns_130(tmp_path, monkeypatch, capsys):
    """Verify interrupted command records checkpoint and returns 130."""

    client = FakeClient()
    client.failures[("ohlcv-1m", request().start_date)] = [KeyboardInterrupt()]
    monkeypatch.setattr(fetch, "create_historical_client", lambda: client)

    assert (
        fetch.download_main(
            [
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
        )
        == 130
    )
    terminal = capsys.readouterr()

    assert "Traceback" not in terminal.err
    manifest = json.loads(next(tmp_path.rglob("manifest.json")).read_text())

    assert manifest["last_error"]["message"] == "KeyboardInterrupt"
    assert manifest["components"]["candlesticks"]["chunks"][0]["state"] == "partial"


def test_complete_managed_request_needs_no_vendor_calls(tmp_path, monkeypatch):
    """Verify complete managed request needs no vendor calls."""

    client = FakeClient()
    plan(tmp_path).run(client, max_cost_usd=6)

    # Completed repair must use its validated final artifacts without any new vendor requests.
    for method in ("get_range", "get_cost", "get_record_count", "get_dataset_condition"):
        monkeypatch.setattr(
            client, method, lambda **kwargs: pytest.fail("Complete request contacted vendor")
        )

    plan(tmp_path).run(client, max_cost_usd=0)


def test_repair_estimate_only_checkpoints_one_time_probes(tmp_path, monkeypatch, capsys):
    """Verify repair estimate only checkpoints one time probes."""

    directory = tmp_path / "ES-v-0_2020-06-06_2022-09-12"
    directory.mkdir()
    (directory / "candlesticks.csv").write_text(
        "ts_event,open,high,low,close\n2020-06-06T01:00:00Z,100,101,99,100\n"
    )
    client = FakeClient()
    monkeypatch.setattr(fetch, "create_historical_client", lambda: client)
    result = fetch.repair_main(
        [
            "--start",
            "2020-06-06",
            "--end",
            "2022-09-12",
            "--max-cost-usd",
            "0",
            "--status-chunk-years",
            "1",
            "--estimate-only",
            "--raw-data-directory",
            str(tmp_path),
        ]
    )

    assert result == 0
    assert client.calls == [] and client.conditions == []
    assert len([count for count in client.counts if count["schema"] == "ohlcv-1m"]) == 3
    assert (directory / "manifest.json").exists()
    first_probe_count = len(client.counts)
    resumed = managed.ManagedDownload(request(), directory, repair=True, status_chunk_years=1)
    resumed.prepare_repair(client)

    assert len(client.counts) == first_probe_count
    assert "repair probes and adopted ranges were checkpointed" in capsys.readouterr().out


def test_repair_probe_failure_is_logged_and_remains_resumable(tmp_path, monkeypatch, capsys):
    """Verify repair probe failure is logged and remains resumable."""

    directory = tmp_path / "ES-v-0_2020-06-06_2022-09-12"
    directory.mkdir()
    (directory / "candlesticks.csv").write_text(
        "ts_event,open,high,low,close\n" "2020-06-06T01:00:00Z,100,101,99,100\n"
    )
    client = FakeClient()

    def fail_probe(**params):
        """Fail the metadata completeness probe before repair can request paid data."""

        raise BentoError("metadata unavailable")

    monkeypatch.setattr(client, "get_record_count", fail_probe)
    monkeypatch.setattr(fetch, "create_historical_client", lambda: client)
    result = fetch.repair_main(
        [
            "--start",
            "2020-06-06",
            "--end",
            "2022-09-12",
            "--max-cost-usd",
            "0",
            "--estimate-only",
            "--raw-data-directory",
            str(tmp_path),
        ]
    )

    assert result == 1
    terminal = capsys.readouterr()

    assert "Resumable: True" in terminal.err
    assert "Traceback" not in terminal.err
    log = directory / "download.log"

    assert log.exists()
    assert "metadata unavailable" in log.read_text()


def test_download_and_repair_manifests_use_distinct_commands(tmp_path):
    """Verify download and repair manifests use distinct commands."""

    download_plan = repair_plan(tmp_path)
    client = FakeClient()
    download_plan.prepare_repair(client)

    # A repair manifest must resume through repair so its adoption and cleanup policy stays
    # consistent.
    with pytest.raises(ValueError, match="pricesanity-repair-download"):
        managed.ManagedDownload(request(), tmp_path / "corpus", status_chunk_years=1)


@pytest.mark.parametrize(
    "record",
    [
        "not-a-timestamp,100,101,99,100",
        "2020-06-05T23:59:00Z,100,101,99,100",
        "2020-06-06T01:00:00Z,100,nan,99,100",
    ],
)
def test_repair_rejects_malformed_candles(tmp_path, record):
    """Verify repair rejects malformed candles."""

    directory = tmp_path / "corpus"
    directory.mkdir()
    (directory / "candlesticks.csv").write_text("ts_event,open,high,low,close\n" + record + "\n")

    # A failed repair preflight must not fabricate a checkpoint for invalid source data.
    with pytest.raises(ValueError):
        managed.ManagedDownload(request(), directory, repair=True)

    assert not (directory / "manifest.json").exists()


def test_status_nanosecond_order_and_duplicate_records(tmp_path):
    """Verify status nanosecond order and duplicate records."""

    path = tmp_path / "status.csv"
    header = "ts_event,reason,trading_event,is_trading\n"
    path.write_text(
        header
        + "2020-06-06T01:00:00.000000002Z,scheduled,none,Y\n"
        + "2020-06-06T01:00:00.000000001Z,scheduled,none,N\n"
    )

    # Nanosecond ordering must survive parsing even when datetime alone would collapse nearby
    # timestamps.
    with pytest.raises(ValueError, match="chronological"):
        validate_csv(path, "status", date(2020, 6, 6), date(2020, 6, 7))


def test_duplicate_condition_dates_preserve_final_on_assembly_failure(tmp_path):
    """Verify duplicate condition dates preserve final on assembly failure."""

    part = tmp_path / "conditions.json"
    atomic_json(part, [{"date": "2020-06-06", "condition": "available"}])
    final = tmp_path / "condition.json"
    final.write_text("original")

    # Repeated condition dates invalidate assembly and must leave the existing final file
    # untouched.
    with pytest.raises(ValueError, match="unique"):
        assemble_files([part, part], final, "conditions", date(2020, 6, 6), date(2020, 6, 7), 2)

    assert final.read_text() == "original"
    assert part.exists()


def test_repair_resumed_assembly_does_not_contact_vendor(tmp_path, monkeypatch):
    """Verify repair resumed assembly does not contact vendor."""

    download_plan = repair_plan(tmp_path, complete=True)
    client = FakeClient()
    original = managed.assemble_files
    monkeypatch.setattr(
        managed, "assemble_files", lambda *args: (_ for _ in ()).throw(OSError("disk"))
    )

    # Leave repair at a local publication failure so a later invocation can finish without
    # redownloading.
    with pytest.raises(OSError):
        download_plan.run(client, max_cost_usd=3)

    monkeypatch.setattr(managed, "assemble_files", original)

    # Make network work impossible during the resumed repair assembly.
    for method in ("get_range", "get_cost", "get_record_count", "get_dataset_condition"):
        monkeypatch.setattr(
            client, method, lambda **kwargs: pytest.fail("Assembly contacted vendor")
        )

    managed.ManagedDownload(request(), tmp_path / "corpus", repair=True, status_chunk_years=1).run(
        client, max_cost_usd=0
    )
