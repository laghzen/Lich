from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess
import time


# NVIDIA nvidia-smi/NVML throttle-reason bits. The power-cap bit is
# informational on a 120 W laptop GPU: power-limited operation is normal at
# the configured board limit and is not the same as thermal throttling.
THROTTLE_SW_POWER_CAP = 0x00000004
THROTTLE_HW_SLOWDOWN = 0x00000008
THROTTLE_SW_THERMAL = 0x00000020
THROTTLE_HW_THERMAL = 0x00000040


@dataclass(frozen=True)
class Telemetry:
    temperature_c: float | None
    power_w: float | None
    clock_sm_mhz: float | None
    clock_mem_mhz: float | None
    util_gpu_pct: float | None
    util_mem_pct: float | None
    throttle_reasons: int | None = None

    @property
    def thermal_throttle(self) -> bool:
        bits = self.throttle_reasons or 0
        return bool(bits & (THROTTLE_SW_THERMAL | THROTTLE_HW_THERMAL))

    @property
    def power_limited(self) -> bool:
        bits = self.throttle_reasons or 0
        return bool(bits & THROTTLE_SW_POWER_CAP)

    @property
    def hardware_slowdown(self) -> bool:
        bits = self.throttle_reasons or 0
        return bool(bits & THROTTLE_HW_SLOWDOWN)


def _f(s: str) -> float | None:
    s = s.strip()
    if not s or s.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _i(s: str) -> int | None:
    s = s.strip()
    if not s or s.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def read_telemetry(index: int = 0) -> Telemetry:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return Telemetry(None, None, None, None, None, None, None)
    q = (
        "temperature.gpu,power.draw,clocks.sm,clocks.mem,"
        "utilization.gpu,utilization.memory,clocks_throttle_reasons.active"
    )
    p = subprocess.run(
        [exe, "-i", str(index), f"--query-gpu={q}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if p.returncode != 0 or not p.stdout.strip():
        return Telemetry(None, None, None, None, None, None, None)
    parts = [x.strip() for x in p.stdout.splitlines()[0].split(",")]
    while len(parts) < 7:
        parts.append("")
    vals = [_f(x) for x in parts[:6]]
    return Telemetry(*vals, _i(parts[6]))


class ThermalGuard:
    """Workload-side thermal governor with low-overhead telemetry sampling.

    The hot training loop does not invoke ``nvidia-smi`` for every batch. A
    cached observation is reused for ``telemetry_interval_s``; only cooldown
    paths poll at ``poll_s``. The guard never changes NVIDIA driver power or
    clock settings.
    """

    def __init__(
        self,
        *,
        max_temp_c: float = 76.0,
        resume_temp_c: float | None = None,
        poll_s: float = 0.5,
        telemetry_interval_s: float = 1.0,
        device_index: int = 0,
    ):
        self.max_temp_c = float(max_temp_c)
        self.resume_temp_c = float(resume_temp_c if resume_temp_c is not None else max_temp_c - 4.0)
        self.poll_s = float(poll_s)
        self.telemetry_interval_s = float(telemetry_interval_s)
        self.device_index = int(device_index)
        self._last_query_monotonic = -float("inf")
        self._cached: Telemetry | None = None

    def sample(self, *, force: bool = False) -> Telemetry:
        now = time.monotonic()
        if not force and self._cached is not None and now - self._last_query_monotonic < self.telemetry_interval_s:
            return self._cached
        self._cached = read_telemetry(self.device_index)
        self._last_query_monotonic = now
        return self._cached

    def wait_until_safe(self, *, verbose: bool = False) -> Telemetry:
        last = self.sample()
        while True:
            too_hot = last.temperature_c is not None and last.temperature_c > self.max_temp_c
            thermal_slowdown = last.thermal_throttle or (last.hardware_slowdown and too_hot)
            if not too_hot and not thermal_slowdown:
                return last
            if verbose:
                reasons = []
                if too_hot:
                    reasons.append(f"{last.temperature_c:.1f}C > {self.max_temp_c:.1f}C")
                if last.thermal_throttle:
                    reasons.append("NVIDIA thermal slowdown active")
                if last.hardware_slowdown and too_hot:
                    reasons.append("hardware slowdown while hot")
                print("[thermal-guard] pausing: " + "; ".join(reasons))
            time.sleep(self.poll_s)
            last = self.sample(force=True)

    def check(self, *, force: bool = False) -> Telemetry:
        return self.sample(force=force)

    def assert_no_thermal_throttle(self) -> Telemetry:
        t = self.sample(force=True)
        if t.thermal_throttle:
            raise RuntimeError(
                f"GPU thermal throttle is active: reasons=0x{(t.throttle_reasons or 0):x}, "
                f"temperature={t.temperature_c}"
            )
        return t

    @staticmethod
    def format_status(t: Telemetry) -> str:
        temp = "n/a" if t.temperature_c is None else f"{t.temperature_c:.1f}C"
        power = "n/a" if t.power_w is None else f"{t.power_w:.1f}W"
        sm = "n/a" if t.clock_sm_mhz is None else f"{t.clock_sm_mhz:.0f}MHz"
        reasons = "n/a" if t.throttle_reasons is None else f"0x{t.throttle_reasons:x}"
        flags = []
        if t.power_limited:
            flags.append("power-cap")
        if t.thermal_throttle:
            flags.append("thermal-throttle")
        if t.hardware_slowdown:
            flags.append("hw-slowdown")
        suffix = " [" + ",".join(flags) + "]" if flags else ""
        return f"temp={temp} power={power} sm={sm} reasons={reasons}{suffix}"
