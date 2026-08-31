"""Deterministic exogenous demand-schedule context for TSC transfer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from cf_h2o.traffic_signal.sumo_static_inputs import (
    demand_route_mixture,
    iter_static_demand_records,
    load_static_route_catalog,
    parse_sumo_time_seconds,
    parse_xml,
    static_demand_input_paths,
    triggered_vehicle_departure_times,
)


DEMAND_SCHEDULE_PROTOCOL = "sumo-static-demand-schedule-context-v1"
DEMAND_SCHEDULE_FEATURE_NAMES = (
    "log_scheduled_vehicle_count",
    "scheduled_departures_per_hour_per_tls",
    "departure_300s_cv",
    "departure_600s_cv",
    "departure_900s_cv",
    "departure_600s_trough_ratio",
    "departure_600s_peak_ratio",
    "origin_hhi",
    "mean_route_edge_count_norm",
)


def _resolve_cfg_paths(sumocfg: Path, tag: str) -> tuple[Path, ...]:
    root = parse_xml(sumocfg).getroot()
    paths: list[Path] = []
    for element in root.findall(f".//{tag}"):
        for value in str(element.attrib.get("value", "")).split(","):
            value = value.strip()
            if value:
                paths.append((sumocfg.parent / value).resolve())
    return tuple(paths)


def demand_input_sha256(sumocfg: Path) -> str:
    """Hash the SUMO config and exact network/route inputs used by the profile."""

    sumocfg = Path(sumocfg).resolve()
    paths = {
        sumocfg,
        *_resolve_cfg_paths(sumocfg, "net-file"),
        *static_demand_input_paths(sumocfg),
    }
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _flow_rate(element: ET.Element, begin: float, end: float) -> float:
    duration = max(float(end - begin), 0.0)
    if duration <= 0.0:
        return 0.0
    if "number" in element.attrib:
        return max(float(element.attrib["number"]), 0.0) / duration
    if "vehsPerHour" in element.attrib:
        return max(float(element.attrib["vehsPerHour"]), 0.0) / 3600.0
    if "period" in element.attrib:
        period = parse_sumo_time_seconds(element.attrib["period"])
        return 1.0 / period if period > 0.0 else 0.0
    if "probability" in element.attrib:
        return max(float(element.attrib["probability"]), 0.0)
    return 1.0 / duration


def _add_uniform_flow(
    bins: np.ndarray,
    *,
    bin_width: float,
    start: float,
    stop: float,
    rate_per_second: float,
    window_start: float,
) -> None:
    if stop <= start or rate_per_second <= 0.0:
        return
    if bin_width <= 0.0:
        raise ValueError("flow schedule bin width must be positive")
    first = max(int(math.floor((start - window_start) / bin_width)), 0)
    last = min(int(math.ceil((stop - window_start) / bin_width)), bins.size)
    for index in range(first, last):
        bin_start = window_start + index * bin_width
        bin_stop = bin_start + bin_width
        overlap = max(min(stop, bin_stop) - max(start, bin_start), 0.0)
        bins[index] += overlap * rate_per_second


def _coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(np.mean(values)) if values.size else 0.0
    return float(np.std(values) / mean) if mean > 0.0 else 0.0


@dataclass(frozen=True)
class DemandScheduleProfile:
    protocol: str
    horizon_sec: int
    tls_count: int
    scheduled_vehicle_count: float
    input_sha256: str
    feature_names: tuple[str, ...]
    features: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            self.protocol != DEMAND_SCHEDULE_PROTOCOL
            or self.horizon_sec <= 0
            or self.tls_count <= 0
            or self.feature_names != DEMAND_SCHEDULE_FEATURE_NAMES
            or len(self.features) != len(self.feature_names)
            or not np.isfinite(np.asarray(self.features, dtype=float)).all()
        ):
            raise ValueError("invalid demand schedule profile")

    def vector(self) -> np.ndarray:
        return np.asarray(self.features, dtype=float)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_demand_schedule_profile(
    sumocfg: Path, *, horizon_sec: int = 3600
) -> DemandScheduleProfile:
    """Build a label-free profile from scheduled SUMO departures.

    Vehicle and trip departures are counted exactly. Flow demand is integrated
    over each time bin, preserving expected volume without sampling.
    """

    sumocfg = Path(sumocfg).resolve()
    if int(horizon_sec) <= 0:
        raise ValueError("demand profile horizon must be positive")
    cfg_root = parse_xml(sumocfg).getroot()
    begin_element = cfg_root.find(".//time/begin")
    window_start = (
        float(begin_element.attrib.get("value", 0.0))
        if begin_element is not None
        else 0.0
    )
    window_stop = window_start + int(horizon_sec)

    net_files = _resolve_cfg_paths(sumocfg, "net-file")
    if len(net_files) != 1:
        raise ValueError("demand profile requires exactly one SUMO net file")
    net_root = parse_xml(net_files[0]).getroot()
    tls_ids = {str(element.attrib["id"]) for element in net_root.findall(".//tlLogic")}
    if not tls_ids:
        tls_ids = {
            str(element.attrib["id"])
            for element in net_root.findall(".//junction")
            if str(element.attrib.get("type", "")).startswith("traffic_light")
        }
    tls_count = len(tls_ids)
    if tls_count <= 0:
        raise ValueError("SUMO network has no controlled traffic signals")

    bin_widths = (300, 600, 900)
    schedules = {
        width: np.zeros(int(math.ceil(horizon_sec / width)), dtype=float)
        for width in bin_widths
    }
    origins: dict[str, float] = {}
    route_edge_total = 0.0
    scheduled_count = 0.0

    demand_paths = static_demand_input_paths(sumocfg)
    catalog = load_static_route_catalog(demand_paths)
    triggered_departures = triggered_vehicle_departure_times(demand_paths)
    for record in iter_static_demand_records(demand_paths):
        mixtures = demand_route_mixture(record, catalog)
        if record.kind == "flow":
            begin = parse_sumo_time_seconds(
                record.attributes.get("begin", window_start)
            )
            end = parse_sumo_time_seconds(record.attributes.get("end", begin))
            clipped_start = max(begin, window_start)
            clipped_stop = min(end, window_stop)
            proxy = ET.Element("flow", attrib=dict(record.attributes))
            rate = _flow_rate(proxy, begin, end)
            count = max(clipped_stop - clipped_start, 0.0) * rate
            for width, bins in schedules.items():
                _add_uniform_flow(
                    bins,
                    bin_width=float(width),
                    start=clipped_start,
                    stop=clipped_stop,
                    rate_per_second=rate,
                    window_start=window_start,
                )
        else:
            depart_value = str(record.attributes.get("depart", window_start))
            if depart_value == "triggered":
                vehicle_id = str(record.attributes.get("id", ""))
                if vehicle_id not in triggered_departures:
                    raise ValueError(
                        "triggered SUMO vehicle has no person-plan reference: "
                        f"{vehicle_id}"
                    )
                depart = float(triggered_departures[vehicle_id])
            else:
                depart = parse_sumo_time_seconds(depart_value)
            if not window_start <= depart < window_stop:
                continue
            count = 1.0
            offset = depart - window_start
            for width, bins in schedules.items():
                index = min(int(offset // width), bins.size - 1)
                bins[index] += 1.0
        if count <= 0.0:
            continue
        scheduled_count += count
        if not mixtures:
            origins["__unknown__"] = origins.get("__unknown__", 0.0) + count
        for edges, probability in mixtures:
            weight = count * probability
            origin = edges[0] if edges else "__unknown__"
            origins[origin] = origins.get(origin, 0.0) + weight
            route_edge_total += weight * len(edges)

    if scheduled_count <= 0.0:
        raise ValueError("SUMO demand schedule contains no departures in the horizon")
    origin_shares = np.asarray(list(origins.values()), dtype=float) / scheduled_count
    bins_300 = schedules[300]
    bins_600 = schedules[600]
    bins_900 = schedules[900]
    mean_600 = float(np.mean(bins_600))
    features = (
        math.log1p(scheduled_count),
        scheduled_count / float(horizon_sec) * 3600.0 / tls_count,
        _coefficient_of_variation(bins_300),
        _coefficient_of_variation(bins_600),
        _coefficient_of_variation(bins_900),
        float(np.min(bins_600) / mean_600) if mean_600 > 0.0 else 0.0,
        float(np.max(bins_600) / mean_600) if mean_600 > 0.0 else 0.0,
        float(np.sum(origin_shares**2)),
        route_edge_total / scheduled_count / 20.0,
    )
    return DemandScheduleProfile(
        protocol=DEMAND_SCHEDULE_PROTOCOL,
        horizon_sec=int(horizon_sec),
        tls_count=tls_count,
        scheduled_vehicle_count=float(scheduled_count),
        input_sha256=demand_input_sha256(sumocfg),
        feature_names=DEMAND_SCHEDULE_FEATURE_NAMES,
        features=tuple(float(value) for value in features),
    )


def augment_with_demand_profile(
    candidate_features: np.ndarray, profile: DemandScheduleProfile
) -> np.ndarray:
    matrix = np.asarray(candidate_features, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("candidate feature matrix must be finite and two-dimensional")
    demand = np.repeat(profile.vector()[None, :], matrix.shape[0], axis=0)
    return np.concatenate([matrix, demand], axis=1)
