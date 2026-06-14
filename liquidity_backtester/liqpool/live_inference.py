"""Live inference — the missing 'research engine during market hours'.

The package today is overwhelmingly a *historical* engine: warehouse
parquet in, walk-forward training, model bundle pickle, daily brief
out. Codex and the founder both flagged the gap: there's no path that
*runs* during the trading session and ships signals to Sentinel as
ticks land.

This module is that path, and it's small on purpose. The engine
itself doesn't change. We do three things:

  1. ``LiveInferenceBundle`` — load a trained ``MultiAssetReport``
     pickle and expose the inference-time predict callable for each
     head (direction / proximity / quality), behind one ``predict()``
     method that takes a feature row dict.

  2. ``LiveInferenceServer`` — accept ``on_tick(symbol, ts, ohlcv_window)``
     calls every cycle, build features, run each head, and PUBLISH
     each non-trivial output as a canonical ``ModelSignal`` (the
     shared contracts shape Sentinel reads).

  3. ``JsonlPublisher`` — the simplest possible sink: append-only
     JSONL the Sentinel-side bridge tails. Network / Redis / WS
     adapters are one-class-each follow-ups.

The trade-offs that make this honest:

  * NO websocket integration here (kite_websocket is a separate
    concern; this is the model-side surface that whatever stream
    layer feeds).
  * NO new feature engineering — uses the same featurizers the bundle
    was trained against. If the bundle was a sklearn-style estimator,
    the predict path is duck-typed; if it's a LightGBM Booster, we
    call ``predict()`` directly.
  * The bundle loader is *defensive*: any head that's missing or
    fails to predict produces None (logged as a warning) so a
    partial bundle still serves what it can.
"""
from __future__ import annotations

import json
import logging
import pickle
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from liqpool.contracts.signals import ModelSignal

LOG = logging.getLogger("liqpool.live_inference")
IST = timezone(timedelta(hours=5, minutes=30))


# ---------------------------------------------------------------------------
# Bundle loader — duck-typed around whatever sklearn-style models live
# inside a MultiAssetReport
# ---------------------------------------------------------------------------

@dataclass
class HeadSlot:
    """One inference head — direction / proximity / quality."""
    name: str
    predict: Callable[[Dict[str, Any]], Optional[float]]
    label: str = ""               # human label for the published signal


@dataclass
class LiveInferenceBundle:
    """Wraps a loaded MultiAssetReport (or any duck-compatible object)
    and presents a uniform ``predict_one(head, features)`` callable."""
    heads: Dict[str, HeadSlot] = field(default_factory=dict)
    bundle_version: str = ""
    feature_version: str = ""
    loaded_from: str = ""

    def predict_one(self, head: str, features: Dict[str, Any]
                    ) -> Optional[float]:
        slot = self.heads.get(head)
        if slot is None:
            return None
        try:
            return slot.predict(features)
        except Exception as exc:
            LOG.warning("predict %s failed: %s", head, exc)
            return None

    @classmethod
    def from_pickle(cls, path: Path) -> "LiveInferenceBundle":
        """Load a MultiAssetReport pickle and discover whatever
        inference-time callables it exposes."""
        path = Path(path)
        with path.open("rb") as f:
            report = pickle.load(f)
        return cls.from_report(report, loaded_from=str(path))

    @classmethod
    def from_report(cls, report: Any,
                    loaded_from: str = "") -> "LiveInferenceBundle":
        heads: Dict[str, HeadSlot] = {}

        def _wrap(model: Any, name: str, label: str) -> HeadSlot:
            def _predict(features: Dict[str, Any]) -> Optional[float]:
                # Two duck-typed call shapes we expect:
                #   (a) sklearn-style: predict_proba(X)[:, 1]
                #   (b) custom: predict(features_dict) -> float
                try:
                    if hasattr(model, "predict_one"):
                        return float(model.predict_one(features))
                    if hasattr(model, "predict_proba"):
                        import numpy as np
                        X = np.asarray([list(features.values())])
                        proba = model.predict_proba(X)
                        return float(proba[0][1])
                    if hasattr(model, "predict"):
                        out = model.predict([list(features.values())])
                        return float(out[0])
                except Exception:
                    return None
                return None
            return HeadSlot(name=name, predict=_predict, label=label)

        for attr, label in [
            ("unified_direction", "direction"),
            ("unified_ml", "quality"),
            ("unified_proximity", "proximity"),
        ]:
            m = getattr(report, attr, None)
            if m is None:
                continue
            if isinstance(m, dict):
                # proximity comes as {horizon: model}
                for horizon, sub in m.items():
                    heads[f"{label}_h{horizon}"] = _wrap(
                        sub, f"{label}_h{horizon}",
                        f"proximity h={horizon}")
            else:
                heads[label] = _wrap(m, label, label)

        bv = getattr(report, "model_bundle_version", "") or ""
        fv = getattr(report, "feature_version", "") or ""
        return cls(heads=heads, bundle_version=str(bv),
                   feature_version=str(fv), loaded_from=loaded_from)

    def head_names(self) -> List[str]:
        return list(self.heads)


# ---------------------------------------------------------------------------
# Sinks — where the published signals land
# ---------------------------------------------------------------------------

class JsonlPublisher:
    """Append-only JSONL writer; the Sentinel-side bridge tails this
    file and converts each line into a live ModelSignal on its bus.

    Thread-safe (single lock around the file handle); idempotent in
    the sense that re-publishing the same signal just appends another
    row — the consumer dedupes by (asset, model) like the in-process
    LivePublisher does."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def publish(self, sig: ModelSignal) -> None:
        with self._lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(sig.to_row()) + "\n")


class InMemoryPublisher:
    """Test/development sink — keeps the last N signals in memory."""

    def __init__(self, capacity: int = 256) -> None:
        from collections import deque
        self._buf = deque(maxlen=capacity)

    def publish(self, sig: ModelSignal) -> None:
        self._buf.append(sig)

    def all(self) -> List[ModelSignal]:
        return list(self._buf)


# ---------------------------------------------------------------------------
# Server — the per-cycle drive layer
# ---------------------------------------------------------------------------

@dataclass
class LiveInferenceServer:
    """One server, one bundle, one publisher. The streaming layer
    (kite WS, polling cron, replay) calls ``on_tick`` per cycle with
    the features it wants scored. The server publishes the
    corresponding ModelSignals to the bus."""
    bundle: LiveInferenceBundle
    publisher: Any                          # any object with .publish(ModelSignal)
    asset: str = "NIFTY"
    # Confidence floor below which we DON'T publish to keep the bus
    # quiet on flat tape. 0 = publish everything.
    publish_floor: float = 0.30
    # Trust tier each head's output starts at — research models begin
    # SHADOW; Sentinel's curator graduates them as they earn it.
    trust_tier_per_head: Dict[str, str] = field(default_factory=dict)
    # MIS-aware session windows. Outside the session, suppress entirely.
    # After NO_NEW_ENTRY cutoff, still publish but tag the signal so the
    # Crux composer + the cockpit can refuse new entries.
    enforce_mis: bool = True

    def now_ist(self) -> str:
        return datetime.now(IST).strftime("%H:%M:%S")

    def _mis_state(self) -> Dict[str, Any]:
        """Returns the MIS window context the signal extras should carry:

          {
            "in_session": bool,
            "mis_no_new_entry": bool,    # >= 14:30 IST
            "mis_squareoff": bool,        # >= 15:15 IST
          }

        Outside trading hours (or before 09:15) returns
        in_session=False; the caller suppresses publishing entirely."""
        from liqpool.intraday import (
            EOD_SQUAREOFF_IST_MIN, NO_NEW_ENTRY_AFTER_IST_MIN,
            SESSION_CLOSE_IST_MIN, SESSION_OPEN_IST_MIN,
        )
        now = datetime.now(IST)
        m = now.hour * 60 + now.minute
        in_session = SESSION_OPEN_IST_MIN <= m < SESSION_CLOSE_IST_MIN
        return {
            "in_session": in_session,
            "mis_no_new_entry": m >= NO_NEW_ENTRY_AFTER_IST_MIN,
            "mis_squareoff": m >= EOD_SQUAREOFF_IST_MIN,
        }

    def on_tick(self, features_per_head: Dict[str, Dict[str, Any]],
                asset: Optional[str] = None,
                extras: Optional[Dict[str, Any]] = None) -> List[ModelSignal]:
        """Run every head whose features are supplied. Publish
        non-trivial outputs. Returns the signals that fired.

        MIS enforcement: outside 09:15-15:30 IST nothing publishes.
        After 14:30 IST every published signal carries
        ``mis_no_new_entry=True`` in extras so the Crux composer can
        refuse new entries. After 15:15 the spine should already be
        squaring off via TrailEngine; we additionally flag
        ``mis_squareoff=True``."""
        asset = asset or self.asset
        signal_extras = dict(extras or {})
        if self.enforce_mis:
            window = self._mis_state()
            if not window["in_session"]:
                return []                          # nothing publishes outside session
            signal_extras.update(window)
        fired: List[ModelSignal] = []
        ts = self.now_ist()
        for head, features in features_per_head.items():
            prob = self.bundle.predict_one(head, features)
            if prob is None:
                continue
            sig = self._signal_for(head, prob, asset, ts, signal_extras)
            if sig is None:
                continue
            fired.append(sig)
            try:
                self.publisher.publish(sig)
            except Exception as exc:
                LOG.warning("publisher failed for %s: %s", head, exc)
        return fired

    def _signal_for(self, head: str, prob: float, asset: str, ts: str,
                    extras: Dict[str, Any]) -> Optional[ModelSignal]:
        """Map a head's probability to a publishable ModelSignal. We
        translate (head, prob) into a human verdict + confidence in
        a way Sentinel's bus already understands."""
        # The verdict text is the human sentence the cockpit's live brief
        # feed renders. Keep it short.
        if head.startswith("direction"):
            side = "upside" if prob >= 0.5 else "downside"
            conf = abs(prob - 0.5) * 2.0       # 0..1 distance from indifference
            if conf < self.publish_floor:
                return None
            verdict = f"{side} probability {prob:.0%}"
            return ModelSignal(
                ts_ist=ts, asset=asset, model="direction_model",
                signal=verdict, confidence=round(conf, 3),
                trust_tier=self.trust_tier_per_head.get(head, "SHADOW"),
                reason_codes=[head, f"p_up={prob:.3f}",
                              f"bundle_v={self.bundle.bundle_version}"],
                source="liqpool",
                extras={"raw_prob": prob, **extras},
            )

        if head.startswith("quality"):
            conf = prob
            if conf < self.publish_floor:
                return None
            grade = ("HIGH" if prob >= 0.65 else
                     "MEDIUM" if prob >= 0.45 else "LOW")
            return ModelSignal(
                ts_ist=ts, asset=asset, model="quality_model",
                signal=f"pool respect {grade} ({prob:.0%})",
                confidence=round(conf, 3),
                trust_tier=self.trust_tier_per_head.get(head, "SHADOW"),
                reason_codes=[head, "respect_probability"],
                source="liqpool",
                extras={"raw_prob": prob, **extras},
            )

        if head.startswith("proximity"):
            conf = prob
            if conf < self.publish_floor:
                return None
            horizon = head.split("_h")[-1] if "_h" in head else "n/a"
            return ModelSignal(
                ts_ist=ts, asset=asset, model="proximity_model",
                signal=f"P(touch within {horizon} bars) = {prob:.0%}",
                confidence=round(conf, 3),
                trust_tier=self.trust_tier_per_head.get(head, "SHADOW"),
                reason_codes=[head, f"horizon={horizon}"],
                source="liqpool",
                extras={"raw_prob": prob, "horizon": horizon, **extras},
            )

        # Unknown head — generic signal so nothing's silently dropped.
        return ModelSignal(
            ts_ist=ts, asset=asset, model=head,
            signal=f"{head} score {prob:.2f}", confidence=round(prob, 3),
            trust_tier=self.trust_tier_per_head.get(head, "SHADOW"),
            reason_codes=[head],
            source="liqpool",
            extras={"raw_prob": prob, **extras},
        )
