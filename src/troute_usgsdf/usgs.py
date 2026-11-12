from __future__ import annotations

import logging
from contextlib import contextmanager

import pandas as pd
from hydrotools.waterdata_client import ContinuousClient
from hydrotools.waterdata_client.transformers import NoDataError, to_dataframe

# The API caps each request at 50,000 rows, so request the gages in batches
# small enough that a batch can't be silently truncated.
LIMIT = 50_000
BATCH_SIZE = 40

# cubic feet per second -> cubic meters per second
CFS_TO_CMS = 0.0283168

# hydrotools never raises on a failed HTTP request: it logs the failure on this
# logger and returns None, which it then silently drops. A rate-limited (429)
# batch therefore looks exactly like a gage that has no data. We watch this
# logger so failed requests can be told apart from genuinely empty results.
_HYDROTOOLS_HTTP_LOGGER = "hydrotools.waterdata_client.async_web_client"
API_KEY_ENV_VAR = "HYDROTOOLS_USGS_API_KEY"


class USGSRequestError(RuntimeError):
    """Raised when one or more USGS requests failed, so the data would be incomplete."""


class USGSRateLimitError(USGSRequestError):
    """Raised when the USGS API rate-limited us (HTTP 429)."""


class _FailedRequestCollector(logging.Handler):
    """Collects the HTTP status of every request hydrotools reports as failed."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.statuses: list[int | None] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.msg if isinstance(record.msg, str) else ""
        if msg.startswith("Request to") and "failed with status" in msg:
            # "Request to %s failed with status %s" -> 4xx response
            status = record.args[1] if isinstance(record.args, tuple) and len(record.args) > 1 else None
            self.statuses.append(int(status) if status is not None else None)
        elif msg.startswith("Retries exhausted"):
            # 5xx / timeouts / connection errors after all retries
            self.statuses.append(None)
        elif msg.startswith("Failed to decode JSON"):
            self.statuses.append(None)


@contextmanager
def _collect_failed_requests():
    logger = logging.getLogger(_HYDROTOOLS_HTTP_LOGGER)
    handler = _FailedRequestCollector()
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)


def _raise_if_requests_failed(statuses: list[int | None], batch_start: int, batch_len: int) -> None:
    if not statuses:
        return
    where = f"sites {batch_start + 1}-{batch_start + batch_len}"
    if 429 in statuses:
        raise USGSRateLimitError(
            f"USGS API rate limit exceeded (HTTP 429) while downloading {where}; "
            f"{statuses.count(429)} request(s) were rejected. Stopping rather than "
            "writing missing data as if it were real. Get a free API key at "
            f"https://api.waterdata.usgs.gov/signup and set the {API_KEY_ENV_VAR} "
            "environment variable, or wait and try again."
        )
    shown = ", ".join(sorted({str(s) if s is not None else "retries exhausted/bad response" for s in statuses}))
    raise USGSRequestError(
        f"{len(statuses)} USGS request(s) failed while downloading {where} ({shown}). "
        "Stopping rather than writing missing data as if it were real."
    )


def fetch_usgs_streamflow(sites: list[str], start_time, end_time) -> pd.DataFrame:
    """Fetch 00060 streamflow for many sites at once, batched under the row cap."""
    client = ContinuousClient(transformer=to_dataframe)
    frames = []
    for i in range(0, len(sites), BATCH_SIZE):
        batch = sites[i : i + BATCH_SIZE]
        print(f"Downloading USGS data for {len(batch)} sites ({i + 1}-{i + len(batch)})")
        with _collect_failed_requests() as failed:
            try:
                observations = client.get(
                    monitoring_location_id=batch,
                    limit=LIMIT,
                    parameter_code="00060",  # Volumetric streamflow in ft^3/s
                    datetime=[pd.Timestamp(start_time), pd.Timestamp(end_time)],
                )
            except NoDataError:
                observations = None
        # check before trusting the result: a failed request is dropped by
        # hydrotools and would otherwise look like "no data" or a partial batch
        _raise_if_requests_failed(failed.statuses, i, len(batch))
        if observations is None:
            print(f"No data found for batch {batch}")
            continue
        if len(observations) >= LIMIT:
            print(
                f"WARNING: batch starting at {i} hit the {LIMIT} row cap; "
                "reduce BATCH_SIZE to avoid truncation"
            )
        frames.append(observations)
    if not frames:
        return pd.DataFrame(columns=["usgs_site_code", "value_time", "value"])
    observations = pd.concat(frames, ignore_index=True)
    # cast value_time to datetime, they are strings in UTC -> tz-naive UTC
    observations["value_time"] = pd.to_datetime(
        observations["value_time"], utc=True
    ).dt.tz_localize(None)
    return observations[["usgs_site_code", "value_time", "value"]]


def interpolate_to_grid(observations: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.Series:
    """Interpolate irregular observations onto the target time grid."""
    if observations.empty:
        return pd.Series(index=target_index, dtype="float64")
    series = (
        observations.assign(value=pd.to_numeric(observations["value"], errors="coerce"))
        .set_index("value_time")["value"]
        .sort_index()
    )
    # collapse any duplicate timestamps
    series = series[~series.index.duplicated(keep="first")]
    # linear time interpolation onto the union of observed + target times,
    # then keep only the target grid (no extrapolation beyond observed range)
    union_index = series.index.union(target_index)
    series = series.reindex(union_index).interpolate(method="time")
    return series.reindex(target_index) * CFS_TO_CMS
