"""Durable, operational-only local wall budgets; no training or remote access.

An open interval is an intent recorded *before* work. After an unclean exit its
whole elapsed wall time, including downtime, is charged conservatively. Unknown
clock histories block further work and have no automatic reset/resolve method.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import math
import numbers
import os
from pathlib import Path
import re
import tempfile
import time


SCHEMA_VERSION = "paper3-operational-wall-budget-v1"
CLOCK_TOLERANCE_SECONDS = 5.0


class WallBudgetError(ValueError):
    pass


class WallBudgetBlocked(WallBudgetError):
    pass


class WallBudgetExhausted(WallBudgetError):
    pass


def _utc_now():
    return datetime.now(timezone.utc)


def _boot_id():
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() or None
    except OSError:
        return None


def _number(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise WallBudgetError("A finite operational wall-time number is required")
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise WallBudgetError("Operational wall-time numbers must be finite and nonnegative")
    return value


def _atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        # Persist the rename too; a successful method is a durable journal step.
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


class PersistentWallBudget:
    """One exclusively locked local JSON journal for one scientific identity.

    ``output_path`` is a journal filename. ``start_interval`` must precede work;
    ``checkpoint`` accepts a *cumulative* checkpoint/progress wall high-water
    mark. ``graceful_close`` ends an interval; ``close`` also releases the lock.
    A changed constructor limit requires a larger limit plus amendment_note.
    Operational limits/notes are deliberately outside identity_sha256.

    For an unclosed interval, same-boot UTC and monotonic elapsed times must be
    nonnegative and differ by at most five seconds. Charge their maximum. Boot
    changes, clock rollback, or larger disagreement permanently close the gate
    in this journal; no constructor flag or budget amendment clears that block.
    """

    def __init__(self, output_path, identity_sha256, max_wall_seconds,
                 checkpoint_wall_seconds=0, *, monotonic=time.monotonic,
                 utc_now=_utc_now, boot_id=None, amendment_note=None,
                 minimum_charged_seconds=None):
        if not isinstance(identity_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", identity_sha256) is None:
            raise WallBudgetError("A concrete scientific identity SHA256 is required")
        maximum = _number(max_wall_seconds, positive=True)
        floor = _number(checkpoint_wall_seconds)
        if minimum_charged_seconds is not None:
            floor = max(floor, _number(minimum_charged_seconds))
        self.path = Path(output_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity_sha256 = identity_sha256
        self._monotonic, self._utc_now = monotonic, utc_now
        self._boot_id = _boot_id if boot_id is None else boot_id
        self._lock = self.path.with_name(self.path.name + ".lock").open("a")
        self._closed = False
        try:
            try:
                fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise WallBudgetError("Another process owns this operational wall-budget journal") from None
            if self.path.exists():
                try:
                    self._state = json.loads(self.path.read_text())
                except (OSError, json.JSONDecodeError):
                    raise WallBudgetError("Operational wall-budget journal cannot be read; it was not reset") from None
                self._validate_state()
                if self._state["identity_sha256"] != identity_sha256:
                    raise WallBudgetError("Operational wall-budget scientific identity changed")
                self._raise_floor(floor)
                # Recovery always precedes a limit amendment. An amendment may
                # authorize more time, never excuse an unknown clock interval.
                if self._state["interval"] is not None and not self.blocked:
                    self._settle_interval(recovery=True)
                self._save()  # A rejected limit change cannot erase checkpoint evidence.
                if maximum != self.max_wall_seconds:
                    self.amend_limit(maximum, amendment_note)
                else:
                    self._save()
            else:
                self._state = {"schema_version": SCHEMA_VERSION, "identity_sha256": identity_sha256,
                    "scope": "operational_only_no_scientific_identity_change",
                    "initial_max_wall_seconds": maximum, "max_wall_seconds": maximum,
                    "durable_wall_seconds": floor, "checkpoint_wall_high_water_seconds": floor,
                    "interval": None, "next_interval_sequence": 1, "clock_block": None,
                    "amendments": [], "recovery_events": [], "closed_intervals": 0}
                self._save()
        except BaseException:
            self._lock.close(); self._closed = True
            raise

    def _validate_state(self):
        s = self._state
        try:
            if (s["schema_version"] != SCHEMA_VERSION or
                    s["scope"] != "operational_only_no_scientific_identity_change" or
                    not isinstance(s["amendments"], list) or not isinstance(s["recovery_events"], list) or
                    type(s["next_interval_sequence"]) is not int or s["next_interval_sequence"] < 1 or
                    type(s["closed_intervals"]) is not int or s["closed_intervals"] < 0 or
                    (s["clock_block"] is not None and not isinstance(s["clock_block"], dict))):
                raise WallBudgetError("Operational wall-budget journal schema is invalid")
            initial = _number(s["initial_max_wall_seconds"], positive=True)
            current = _number(s["max_wall_seconds"], positive=True)
            durable = _number(s["durable_wall_seconds"])
            checkpoint = _number(s["checkpoint_wall_high_water_seconds"])
            if durable < checkpoint:
                raise WallBudgetError("Operational wall-budget high-water mark moved backwards")
            limit = initial
            for amendment in s["amendments"]:
                if (amendment["previous_max_wall_seconds"] != limit or
                        _number(amendment["new_max_wall_seconds"], positive=True) <= limit or
                        not isinstance(amendment["note"], str) or not amendment["note"].strip() or
                        amendment["scientific_identity_changed"] is not False):
                    raise WallBudgetError("Operational wall-budget amendment history is invalid")
                limit = amendment["new_max_wall_seconds"]
            if current != limit:
                raise WallBudgetError("Operational wall-budget limit differs from its explicit amendments")
            interval = s["interval"]
            if interval is not None:
                if (not isinstance(interval, dict) or interval["identity_sha256"] != s["identity_sha256"] or
                        type(interval["sequence"]) is not int or not 0 < interval["sequence"] < s["next_interval_sequence"] or
                        _number(interval["base_wall_seconds"]) > durable):
                    raise WallBudgetError("Operational wall-budget open interval is invalid")
                _number(interval["start_monotonic_seconds"])
                # Absolute UTC may be negative for a valid injected epoch.
                if (isinstance(interval["start_utc_seconds"], bool) or
                        not isinstance(interval["start_utc_seconds"], numbers.Real) or
                        not math.isfinite(float(interval["start_utc_seconds"]))):
                    raise WallBudgetError("Operational wall-budget interval clock is invalid")
                for name, start in (("last_observed_utc_seconds", interval["start_utc_seconds"]),
                                    ("last_observed_monotonic_seconds", interval["start_monotonic_seconds"])):
                    if name in interval and (isinstance(interval[name], bool) or
                            not isinstance(interval[name], numbers.Real) or not math.isfinite(float(interval[name])) or
                            interval[name] < start):
                        raise WallBudgetError("Operational wall-budget observation high-water clock is invalid")
        except (KeyError, TypeError, OverflowError):
            raise WallBudgetError("Operational wall-budget journal is incomplete; it was not reset") from None

    def _ensure_open(self):
        if self._closed:
            raise WallBudgetError("Operational wall-budget handle is closed")

    def _save(self):
        self._ensure_open()
        _atomic_json(self.path, self._state)

    def _clock(self):
        mono = _number(self._monotonic())
        utc = self._utc_now()
        if isinstance(utc, datetime):
            if utc.tzinfo is None or utc.utcoffset() is None:
                raise WallBudgetError("Operational UTC clock must have an explicit timezone")
            utc = utc.timestamp()
        if isinstance(utc, bool) or not isinstance(utc, numbers.Real) or not math.isfinite(float(utc)):
            raise WallBudgetError("Operational UTC clock must be finite")
        boot = self._boot_id() if callable(self._boot_id) else self._boot_id
        return {"utc_seconds": float(utc), "monotonic_seconds": mono,
                "boot_id": boot if isinstance(boot, str) and boot.strip() else None}

    def _raise_floor(self, floor):
        self._state["checkpoint_wall_high_water_seconds"] = max(self._state["checkpoint_wall_high_water_seconds"], floor)
        self._state["durable_wall_seconds"] = max(self._state["durable_wall_seconds"], floor)

    def _observe_interval(self):
        interval = self._state["interval"]
        try:
            clock = self._clock()
        except Exception:
            self._state["clock_block"] = {"reasons": ["clock_observation_unavailable"], "interval": interval}
            self._save()
            return None
        utc_delta = clock["utc_seconds"] - interval["start_utc_seconds"]
        mono_delta = clock["monotonic_seconds"] - interval["start_monotonic_seconds"]
        same_boot = bool(clock["boot_id"] and interval.get("boot_id") and clock["boot_id"] == interval["boot_id"])
        reasons = []
        if not same_boot: reasons.append("boot_changed_or_unknown")
        last_utc = interval.get("last_observed_utc_seconds", interval["start_utc_seconds"])
        last_mono = interval.get("last_observed_monotonic_seconds", interval["start_monotonic_seconds"])
        if utc_delta < 0 or clock["utc_seconds"] < last_utc: reasons.append("utc_clock_rolled_back")
        if same_boot and (mono_delta < 0 or clock["monotonic_seconds"] < last_mono):
            reasons.append("monotonic_clock_rolled_back")
        if same_boot and abs(utc_delta - mono_delta) > CLOCK_TOLERANCE_SECONDS:
            reasons.append("utc_monotonic_elapsed_disagreement_exceeds_5s")
        known_elapsed = max(0.0, utc_delta, mono_delta if same_boot else 0.0)
        charged = max(self._state["durable_wall_seconds"], interval["base_wall_seconds"] + known_elapsed)
        observation = {"clock": clock, "utc_elapsed_seconds": utc_delta, "monotonic_elapsed_seconds": mono_delta,
                       "charged_wall_seconds": charged, "reasons": reasons}
        if reasons:
            self._state["durable_wall_seconds"] = charged
            self._state["clock_block"] = {**observation, "interval": interval,
                "resolution": "blocked_no_automatic_reset_or_clock_override"}
            self._save()
        else:
            # Publishing a wall observation also publishes its high-water mark.
            # Later clock rollback cannot return a smaller previously seen wall
            # merely because both clocks remain above interval start values.
            self._state["durable_wall_seconds"] = charged
            interval["last_observed_utc_seconds"] = clock["utc_seconds"]
            interval["last_observed_monotonic_seconds"] = clock["monotonic_seconds"]
            self._save()
        return observation

    def _new_intent(self, clock):
        if clock["boot_id"] is None:
            self._state["clock_block"] = {"reasons": ["boot_identity_unavailable_before_work"]}
            self._save()
            raise WallBudgetBlocked("Operational wall budget has no boot identity; work is blocked")
        sequence = self._state["next_interval_sequence"]
        self._state["next_interval_sequence"] += 1
        self._state["interval"] = {"sequence": sequence, "identity_sha256": self.identity_sha256,
            "base_wall_seconds": self._state["durable_wall_seconds"], "start_utc_seconds": clock["utc_seconds"],
            "start_monotonic_seconds": clock["monotonic_seconds"], "boot_id": clock["boot_id"],
            "intent": "persisted_before_work"}

    def _settle_interval(self, *, recovery=False, keep_open=False):
        before = self._state["durable_wall_seconds"]
        observation = self._observe_interval()
        if self.blocked:
            return
        self._state["durable_wall_seconds"] = observation["charged_wall_seconds"]
        if recovery:
            self._state["recovery_events"].append({**observation,
                "charged_increment_seconds": self._state["durable_wall_seconds"] - before,
                "policy": "max_same_boot_UTC_monotonic_elapsed_including_downtime"})
        self._state["interval"] = None
        self._state["closed_intervals"] += 1
        if keep_open:
            self._new_intent(observation["clock"])
        self._save()

    @property
    def blocked(self):
        return self._state["clock_block"] is not None

    @property
    def max_wall_seconds(self):
        return self._state["max_wall_seconds"]

    @property
    def minimum_charged_seconds(self):
        return self._state["durable_wall_seconds"]

    @property
    def wall_seconds(self):
        self._ensure_open()
        if self.blocked:
            raise WallBudgetBlocked("Operational clock history is unknown; use report/minimum_charged_seconds for diagnostics")
        if self._state["interval"] is None:
            return self.minimum_charged_seconds
        observation = self._observe_interval()
        if self.blocked:
            raise WallBudgetBlocked("Operational clock history became unknown; work is blocked")
        return observation["charged_wall_seconds"]

    @property
    def remaining_seconds(self):
        if self.blocked:
            return 0.0
        try:
            return max(0.0, self.max_wall_seconds - self.wall_seconds)
        except WallBudgetBlocked:
            return 0.0

    @property
    def exhausted(self):
        return self.wall_seconds >= self.max_wall_seconds

    def start_interval(self):
        self._ensure_open()
        if self.blocked:
            raise WallBudgetBlocked("Operational clock history is unknown; the durable budget remains blocked")
        if self._state["interval"] is not None:
            raise WallBudgetError("An operational work interval is already open")
        if self.exhausted:
            raise WallBudgetExhausted("Cumulative operational wall budget is exhausted; an explicit amendment is required")
        self._new_intent(self._clock())
        self._save()
        report = self.report()
        if self.blocked:
            raise WallBudgetBlocked("Operational clock history became unknown before work")
        return report

    def checkpoint(self, checkpoint_wall_seconds=0):
        self._ensure_open()
        floor = _number(checkpoint_wall_seconds)
        # Raise the caller's cumulative high-water mark before any durable
        # interval transition; an interruption between writes cannot lose it.
        self._raise_floor(floor)
        if self.blocked:
            self._save()
            raise WallBudgetBlocked("Operational clock history is unknown; checkpoint cannot resume work")
        if self._state["interval"] is not None:
            self._settle_interval(keep_open=True)
        if self._state["interval"] is not None and not self.blocked:
            self._state["interval"]["base_wall_seconds"] = self.minimum_charged_seconds
        self._save()
        if self.blocked:
            raise WallBudgetBlocked("Operational clock history became unknown; work is blocked")
        report = self.report()
        if self.blocked:
            raise WallBudgetBlocked("Operational clock history became unknown; work is blocked")
        return report

    def graceful_close(self, checkpoint_wall_seconds=0):
        self._ensure_open()
        floor = _number(checkpoint_wall_seconds)
        self._raise_floor(floor)
        if self._state["interval"] is not None and not self.blocked:
            self._settle_interval()
        self._save()
        return self.report()

    def amend_limit(self, new_max_wall_seconds, note):
        self._ensure_open()
        maximum = _number(new_max_wall_seconds, positive=True)
        if maximum <= self.max_wall_seconds or not isinstance(note, str) or not note.strip():
            raise WallBudgetError("A larger operational-only wall limit requires an explicit amendment note")
        self._state["amendments"].append({"previous_max_wall_seconds": self.max_wall_seconds,
            "new_max_wall_seconds": maximum, "note": note.strip(), "scientific_identity_changed": False})
        self._state["max_wall_seconds"] = maximum
        self._save()
        return self.report()

    def report(self):
        self._ensure_open()
        try:
            wall = self.wall_seconds
        except WallBudgetBlocked:
            wall = self.minimum_charged_seconds
        return {"schema_version": SCHEMA_VERSION, "identity_sha256": self.identity_sha256,
            "scope": self._state["scope"], "wall_seconds": wall,
            "minimum_charged_seconds": self.minimum_charged_seconds,
            "checkpoint_wall_high_water_seconds": self._state["checkpoint_wall_high_water_seconds"],
            "max_wall_seconds": self.max_wall_seconds,
            "remaining_seconds": 0.0 if self.blocked else max(0.0, self.max_wall_seconds - wall),
            "exhausted": wall >= self.max_wall_seconds, "blocked": self.blocked,
            "clock_block": self._state["clock_block"], "interval_open": self._state["interval"] is not None,
            "amendment_count": len(self._state["amendments"]), "recovered_interval_count": len(self._state["recovery_events"])}

    def close(self, checkpoint_wall_seconds=0):
        if not self._closed:
            try:
                self.graceful_close(checkpoint_wall_seconds)
            finally:
                self._lock.close(); self._closed = True

    def __enter__(self):
        self.start_interval()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        # Releasing a discarded handle is not evidence of a graceful stop.
        # Retain any open intent so a later owner must recover its wall charge.
        if hasattr(self, "_lock") and not self._lock.closed:
            self._lock.close()
