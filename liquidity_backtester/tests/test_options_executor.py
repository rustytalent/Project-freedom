"""Tests for the options executor — CCV adjustments, LCS scalar,
pre-trade and in-trade decision trees.

Each test pins one named contract from
``docs/options_executor_layered_conviction.md``. The doc and these
tests must move together; a doc change without a corresponding test
change is a contract drift bug.
"""
from __future__ import annotations

import math
import unittest

from liqpool.options.ccv import (
    MACRO_GATE_THRESHOLD,
    apply_causal_adjustments,
    compute_lcs_scalar,
    ccv_from_row,
)
from liqpool.options.executor import (
    BROKER_OUTAGE_KILL_SECONDS,
    FORCE_ENTRY_THRESHOLD,
    INVALIDATION_AGREEING_LAYERS_MAX,
    INVALIDATION_LCS_DROP,
    KILL_UNDERLYING_ATR,
    TRAIL_LCS_MIN,
    TRAIL_REALIZED_R_MIN,
    VIX_KILL_SPIKE,
    WAIT_THRESHOLD,
    InTradeState,
    KillConditions,
    agreeing_layer_count,
    causal_warnings,
    in_trade_decision,
    pre_trade_decision,
)


def _ccv(macro=0.0, regime=0.0, pool=0.0, options=0.0, micro=0.0,
         manipulation=0.0):
    return apply_causal_adjustments({
        "macro": macro, "regime": regime, "pool": pool,
        "options": options, "micro": micro,
        "manipulation": manipulation,
    })


# ---------------------------------------------------------------------------
# CCV — local-collapse adjustment matrix
# ---------------------------------------------------------------------------

class LocalCollapseTests(unittest.TestCase):

    def test_micro_forced_when_manip_aligned_at_threshold(self):
        ccv = _ccv(macro=0.5, micro=0.4, manipulation=0.5)
        self.assertTrue(ccv.micro_forced_flag)
        # Damping factor: 1 - 0.6 * min(1.0, 0.5) = 0.7
        self.assertAlmostEqual(ccv.micro_organic_score, 0.4 * 0.7,
                                places=6)

    def test_micro_not_forced_when_manip_opposes(self):
        ccv = _ccv(macro=0.5, micro=0.4, manipulation=-0.5)
        self.assertFalse(ccv.micro_forced_flag)
        self.assertEqual(ccv.micro_organic_score, 0.4)

    def test_micro_not_forced_when_manip_below_threshold(self):
        ccv = _ccv(macro=0.5, micro=0.4, manipulation=0.35)
        self.assertFalse(ccv.micro_forced_flag)
        self.assertEqual(ccv.micro_organic_score, 0.4)

    def test_pool_distorted_when_manip_opposes_at_threshold(self):
        ccv = _ccv(macro=0.5, pool=0.4, manipulation=-0.6)
        self.assertTrue(ccv.pool_distortion_flag)
        self.assertAlmostEqual(ccv.pool_holding_strength, 0.4 * 0.6,
                                places=6)

    def test_pool_not_distorted_when_manip_aligned(self):
        ccv = _ccv(macro=0.5, pool=0.4, manipulation=0.6)
        self.assertFalse(ccv.pool_distortion_flag)
        self.assertEqual(ccv.pool_holding_strength, 0.4)

    def test_macro_and_regime_never_adjusted_by_manipulation(self):
        # Strong manipulation in the opposite direction must not move
        # the upstream layers.
        ccv = _ccv(macro=0.7, regime=0.5, manipulation=-1.0)
        self.assertEqual(ccv.macro_score, 0.7)
        self.assertEqual(ccv.regime_score, 0.5)


class AlignmentFeatureTests(unittest.TestCase):

    def test_alignment_is_signed_min_magnitude(self):
        ccv = _ccv(macro=0.5, manipulation=0.3, micro=0.6)
        # sign(+) * sign(+) * min(0.3, 0.6) = +0.3
        self.assertAlmostEqual(ccv.manip_micro_alignment, +0.3, places=6)

    def test_opposite_signs_give_negative_alignment(self):
        ccv = _ccv(macro=0.5, manipulation=-0.3, pool=0.6)
        # sign(-) * sign(+) * min(0.3, 0.6) = -0.3
        self.assertAlmostEqual(ccv.manip_pool_alignment, -0.3, places=6)

    def test_silent_layer_alignment_is_zero(self):
        ccv = _ccv(macro=0.5, regime=0.0, pool=0.6)
        self.assertEqual(ccv.regime_pool_alignment, 0.0)


# ---------------------------------------------------------------------------
# LCS scalar — properties from §3.4
# ---------------------------------------------------------------------------

class LcsScalarTests(unittest.TestCase):

    def test_macro_gate_closed_returns_zero(self):
        # |macro| < 0.10 → 0 regardless of other layers.
        ccv = _ccv(macro=0.05, regime=0.9, pool=0.9, options=0.9,
                   micro=0.9, manipulation=0.9)
        self.assertEqual(ccv.lcs, 0.0)
        ccv = _ccv(macro=-0.05, regime=-0.9)
        self.assertEqual(ccv.lcs, 0.0)

    def test_sign_follows_macro(self):
        ccv = _ccv(macro=0.5, regime=-0.9, pool=-0.9, options=-0.9,
                   micro=-0.9, manipulation=-0.9)
        self.assertGreater(ccv.lcs, 0.0)
        ccv = _ccv(macro=-0.5, regime=+0.9, pool=+0.9, options=+0.9,
                   micro=+0.9, manipulation=+0.9)
        self.assertLess(ccv.lcs, 0.0)

    def test_full_alignment_saturates_at_one(self):
        ccv = _ccv(macro=0.9, regime=0.9, pool=0.9, options=0.9,
                   micro=0.9, manipulation=0.9)
        self.assertAlmostEqual(ccv.lcs, 1.0, places=2,
            msg=f"expected near-saturation; got {ccv.lcs}")

    def test_geometric_mean_amplifies_more_with_alignment(self):
        aligned = _ccv(macro=0.5, regime=0.5, pool=0.5, options=0.5,
                       micro=0.5, manipulation=0.5).lcs
        mixed = _ccv(macro=0.5, regime=0.5, pool=-0.5, options=0.5,
                     micro=0.5, manipulation=0.5).lcs
        self.assertGreater(aligned, mixed,
            f"aligned LCS ({aligned}) must exceed mixed LCS ({mixed})")

    def test_compute_lcs_scalar_direct_call_matches_path_through_ccv(self):
        # Use manipulation = 0 so no causal adjustment fires and the
        # direct call (raw scores) equals the CCV's adjusted result.
        raw_macro = 0.55
        direct = compute_lcs_scalar(raw_macro, 0.4, 0.4, 0.4, 0.4, 0.0)
        through = _ccv(macro=raw_macro, regime=0.4, pool=0.4,
                       options=0.4, micro=0.4, manipulation=0.0).lcs
        self.assertAlmostEqual(direct, through, places=10)


class CcvFromRowTests(unittest.TestCase):

    def test_default_aliases(self):
        row = {"macro_score": 0.5, "regime_score": 0.3, "pool_score": 0.0,
               "options_score": 0.2, "micro_score": 0.4,
               "manipulation_score": 0.0}
        ccv = ccv_from_row(row)
        self.assertEqual(ccv.macro_score, 0.5)
        self.assertEqual(ccv.regime_score, 0.3)
        self.assertEqual(ccv.options_score, 0.2)
        self.assertEqual(ccv.micro_score, 0.4)
        self.assertGreater(ccv.lcs, 0.0)


# ---------------------------------------------------------------------------
# Pre-trade decision
# ---------------------------------------------------------------------------

class PreTradeTests(unittest.TestCase):

    def test_skip_when_macro_gate_closed(self):
        ccv = _ccv(macro=0.05, regime=0.9)
        d = pre_trade_decision(ccv)
        self.assertEqual(d.action, "SKIP")
        self.assertIn("macro_gate_closed", d.reason)
        self.assertIsNone(d.side_thesis)

    def test_enter_at_or_above_force_entry_threshold(self):
        ccv = _ccv(macro=0.6, regime=0.5, pool=0.5, options=0.5,
                   micro=0.5, manipulation=0.0)
        d = pre_trade_decision(ccv)
        self.assertEqual(d.action, "ENTER")
        self.assertEqual(d.side_thesis, "buy")
        self.assertGreaterEqual(abs(d.lcs), FORCE_ENTRY_THRESHOLD)

    def test_sell_side_when_lcs_negative(self):
        ccv = _ccv(macro=-0.6, regime=-0.5, pool=-0.5, options=-0.5,
                   micro=-0.5, manipulation=0.0)
        d = pre_trade_decision(ccv)
        self.assertEqual(d.action, "ENTER")
        self.assertEqual(d.side_thesis, "sell")

    def test_wait_in_intermediate_band(self):
        # macro=0.25 with all silent layers → factors all = 1.0 →
        # |LCS| = 0.25, which sits strictly between WAIT and
        # FORCE_ENTRY thresholds.
        ccv = _ccv(macro=0.25, regime=0.0, pool=0.0, options=0.0,
                   micro=0.0, manipulation=0.0)
        d = pre_trade_decision(ccv)
        self.assertEqual(d.action, "WAIT")
        self.assertGreater(abs(d.lcs), WAIT_THRESHOLD)
        self.assertLess(abs(d.lcs), FORCE_ENTRY_THRESHOLD)

    def test_skip_below_signal_floor(self):
        # Macro just above gate AND many disagreeing layers — geom-
        # mean damping pulls |LCS| below WAIT.
        ccv = _ccv(macro=0.11, regime=-0.9, pool=-0.9, options=-0.9,
                   micro=-0.9, manipulation=0.0)
        d = pre_trade_decision(ccv)
        self.assertEqual(d.action, "SKIP",
            f"expected SKIP for LCS={d.lcs:.3f}; got {d.action}")
        self.assertIn("lcs_below_signal_floor", d.reason)


# ---------------------------------------------------------------------------
# Kill conditions
# ---------------------------------------------------------------------------

class KillConditionTests(unittest.TestCase):

    def test_calm_bar_triggers_nothing(self):
        self.assertIsNone(KillConditions().triggered())

    def test_exchange_halt_fires_first(self):
        kill = KillConditions(
            exchange_halt=True,
            broker_outage_seconds=120,
            vix_intrabar_spike=10,
            underlying_intrabar_move_atr=10,
        )
        self.assertEqual(kill.triggered(), "exchange_halt")

    def test_broker_outage_threshold(self):
        kill_below = KillConditions(
            broker_outage_seconds=BROKER_OUTAGE_KILL_SECONDS - 0.1)
        self.assertIsNone(kill_below.triggered())
        kill_above = KillConditions(
            broker_outage_seconds=BROKER_OUTAGE_KILL_SECONDS)
        self.assertEqual(kill_above.triggered(), "broker_outage_over_60s")

    def test_vix_spike_either_direction(self):
        kill = KillConditions(vix_intrabar_spike=VIX_KILL_SPIKE + 0.5)
        self.assertIn("vix", kill.triggered())
        kill = KillConditions(vix_intrabar_spike=-(VIX_KILL_SPIKE + 0.5))
        self.assertIn("vix", kill.triggered())

    def test_underlying_4_atr_move(self):
        kill = KillConditions(
            underlying_intrabar_move_atr=KILL_UNDERLYING_ATR + 0.1)
        self.assertIn("underlying_intrabar_move", kill.triggered())


# ---------------------------------------------------------------------------
# In-trade decision tree
# ---------------------------------------------------------------------------

class AgreeingLayerCountTests(unittest.TestCase):

    def test_full_agreement_counts_five(self):
        ccv = _ccv(macro=0.5, regime=0.5, pool=0.5, options=0.5,
                   micro=0.5, manipulation=0.5)
        self.assertEqual(agreeing_layer_count(ccv), 5)

    def test_full_disagreement_counts_zero(self):
        ccv = _ccv(macro=0.5, regime=-0.5, pool=-0.5, options=-0.5,
                   micro=-0.5, manipulation=-0.5)
        self.assertEqual(agreeing_layer_count(ccv), 0)

    def test_macro_gate_closed_gives_zero(self):
        ccv = _ccv(macro=0.05, regime=0.5, pool=0.5)
        self.assertEqual(agreeing_layer_count(ccv), 0)


class InTradeTests(unittest.TestCase):

    def _state(self, **overrides):
        # Default: high-conviction long, profitable, no kill.
        entry = _ccv(macro=0.8, regime=0.8, pool=0.8, options=0.8,
                     micro=0.8, manipulation=0.0)
        defaults = dict(
            ccv_at_entry=entry,
            ccv_now=entry,
            realized_r_atr=0.5,
            predicted_target_r_atr=1.5,
            kill=KillConditions(),
        )
        defaults.update(overrides)
        return InTradeState(**defaults)

    def test_default_returns_hold(self):
        d = in_trade_decision(self._state())
        self.assertEqual(d.action, "HOLD")

    def test_kill_takes_precedence(self):
        state = self._state(
            kill=KillConditions(exchange_halt=True),
            realized_r_atr=10.0,    # would otherwise hit target
        )
        d = in_trade_decision(state)
        self.assertEqual(d.action, "EXIT_KILL")
        self.assertEqual(d.reason, "exchange_halt")

    def test_invalidation_when_lcs_collapses_and_layers_disagree(self):
        entry = _ccv(macro=0.8, regime=0.8, pool=0.8, options=0.8,
                     micro=0.8, manipulation=0.0)
        # Now: layers flip almost entirely against macro AND lcs drops.
        now = _ccv(macro=0.5, regime=-0.5, pool=-0.5, options=-0.5,
                   micro=-0.5, manipulation=0.0)
        state = self._state(ccv_at_entry=entry, ccv_now=now,
                             realized_r_atr=-0.5,
                             predicted_target_r_atr=1.5)
        # LCS drop sanity.
        self.assertGreaterEqual(entry.lcs - now.lcs, INVALIDATION_LCS_DROP)
        self.assertLessEqual(agreeing_layer_count(now),
                              INVALIDATION_AGREEING_LAYERS_MAX)
        d = in_trade_decision(state)
        self.assertEqual(d.action, "EXIT_INVALIDATION")

    def test_drawdown_alone_does_not_exit_when_lcs_holds(self):
        # CCV unchanged from entry; only the realized R is deep red.
        state = self._state(realized_r_atr=-3.0,
                             predicted_target_r_atr=1.5)
        d = in_trade_decision(state)
        self.assertEqual(d.action, "HOLD",
            f"layer-intact drawdown must HOLD; got {d.action} "
            f"with reason {d.reason}")

    def test_exit_target_when_realized_at_or_above_target(self):
        state = self._state(realized_r_atr=1.6,
                             predicted_target_r_atr=1.5)
        d = in_trade_decision(state)
        self.assertEqual(d.action, "EXIT_TARGET")

    def test_trail_stop_when_lcs_high_and_profitable(self):
        ccv_high = _ccv(macro=0.95, regime=0.95, pool=0.95, options=0.95,
                         micro=0.95, manipulation=0.0)
        state = self._state(ccv_now=ccv_high,
                             realized_r_atr=TRAIL_REALIZED_R_MIN + 0.1,
                             predicted_target_r_atr=5.0)
        self.assertGreaterEqual(abs(ccv_high.lcs), TRAIL_LCS_MIN)
        d = in_trade_decision(state)
        self.assertEqual(d.action, "TRAIL_STOP")


# ---------------------------------------------------------------------------
# Causal warnings
# ---------------------------------------------------------------------------

class CausalWarningsTests(unittest.TestCase):

    def test_no_warnings_when_no_flags(self):
        ccv = _ccv(macro=0.5, regime=0.5, pool=0.5)
        self.assertEqual(causal_warnings(ccv), [])

    def test_micro_forced_warning_text(self):
        ccv = _ccv(macro=0.5, micro=0.5, manipulation=0.5)
        warns = causal_warnings(ccv)
        self.assertEqual(len(warns), 1)
        self.assertIn("MICRO FORCED", warns[0])

    def test_pool_distortion_warning_text(self):
        ccv = _ccv(macro=0.5, pool=0.5, manipulation=-0.6)
        warns = causal_warnings(ccv)
        self.assertEqual(len(warns), 1)
        self.assertIn("POOL DISTORTED", warns[0])

    def test_both_flags_emit_both_warnings(self):
        ccv = _ccv(macro=0.5, micro=0.5, pool=0.5, manipulation=0.6)
        # Manip aligned with micro (forcing) and aligned with pool too,
        # so pool distortion does NOT fire (opposite-sign rule).
        warns = causal_warnings(ccv)
        self.assertEqual(len(warns), 1)
        self.assertIn("MICRO FORCED", warns[0])


if __name__ == "__main__":
    unittest.main()
