from __future__ import annotations

from pathlib import Path

import numpy as np


def _gage_strings(gages: np.ndarray) -> np.ndarray:
    """Turn RouteLink's gage variable into a 1-D array of stripped byte strings.

    RouteLink stores gage ids as a fixed-width (link, IDLength) char array padded
    with spaces. Viewing each row as one fixed-width string avoids a Python-level
    loop over all ~2.7M CONUS links.
    """
    gages = np.asarray(gages)
    if gages.ndim == 2:  # (link, IDLength) of single chars
        gages = np.ascontiguousarray(gages).view(f"S{gages.shape[1]}").ravel()
    elif gages.dtype.kind == "U":  # netCDF4 already joined them into str
        gages = np.char.encode(gages, "utf-8")
    return np.char.strip(gages.astype(bytes))


def read_route_link_gages(route_link_nc_path: Path) -> dict[int, str]:
    """Map NWM link id -> USGS gage id for every RouteLink entry that carries a gage."""
    from netCDF4 import Dataset

    with Dataset(str(route_link_nc_path)) as ds:
        if "link" not in ds.variables or "gages" not in ds.variables:
            raise ValueError(
                f"RouteLink file {route_link_nc_path} must contain 'link' and 'gages' variables"
            )
        ds.set_auto_mask(False)
        links = np.asarray(ds.variables["link"][:])
        gages = _gage_strings(ds.variables["gages"][:])

    has_gage = gages != b""
    return dict(
        zip(
            links[has_gage].astype(np.int64).tolist(),
            np.char.decode(gages[has_gage], "utf-8").tolist(),
        )
    )


def read_link_mask(mask_path: Path) -> set[int]:
    """Read a t-route mask file: one link id per line, trailing commas and blanks ignored.

    Same format t-route reads for `supernetwork_parameters.mask_file_path`.
    """
    with Path(mask_path).open() as f:
        return {int(line.strip(", \n")) for line in f if line.strip(", \n")}


def read_gage_crosswalk_route_link(
    route_link_nc_path: Path, mask_path: Path | None = None
) -> dict[int, str]:
    """Map NWM link id -> USGS gage id from the RouteLink alone, with no geopackage.

    The RouteLink already carries both halves of the crosswalk, so a hydrofabric
    geopackage is only needed when the output should be keyed by flowpath id instead.
    """
    link_to_gage = read_route_link_gages(route_link_nc_path)
    if mask_path is not None:
        mask = read_link_mask(mask_path)
        link_to_gage = {
            link: gage for link, gage in link_to_gage.items() if link in mask
        }
    return link_to_gage 