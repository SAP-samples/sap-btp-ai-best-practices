"""A later eligible week may use repaid capacity without changing legacy calendar behavior."""
from datetime import date
import unittest
import pandas as pd
from app.optimizer.model.limits import ResolvedLimits
from app.optimizer.opt.optimizer_multi_week import optimize_multi_week,MultiWeekOptimizerSettings


class WorkspaceCalendarTests(unittest.TestCase):
    """Use hand-calculated capacity to verify solver calendar integration."""

    def test_release_allows_valid_week_two_only(self):
        """A 50-unit invoice cannot fit 80/100 usage until a 40-unit release."""
        weeks=[pd.Timestamp('2026-02-02'),pd.Timestamp('2026-02-09')]
        frame=pd.DataFrame([{'Company Code':'F1','Customer':'C1','Purchase Price':50,
                             'Offer File Date (UTC)':'2026-02-02','Due Date':'2026-02-10','expected_lifetime_weeks':4}])
        limits=ResolvedLimits({'F1':10000},{'C1':10000},{},{})
        base={weeks[0]:{'facility':{'F1':80},'customer':{'C1':80}},weeks[1]:{'facility':{'F1':40},'customer':{'C1':40}}}
        settings=MultiWeekOptimizerSettings(horizon_weeks=2,attempt_cap=2,calendar_version='monday-v1')
        result=optimize_multi_week(frame,limits,weeks,base,settings)
        self.assertEqual(result.weekly_plan_df['planned_week_start_iso'].tolist(),['2026-02-09'])
        frame['Due Date']='2026-02-08'
        self.assertEqual(len(optimize_multi_week(frame,limits,weeks,base,settings).selected_df),0)
