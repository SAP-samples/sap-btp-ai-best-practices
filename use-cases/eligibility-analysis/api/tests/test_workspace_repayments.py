"""Verify optional repayments, integer-cent balances and overlapping capacity projections."""
from datetime import date
import unittest
from app.optimizer.model.repayments import build_opening_schedule
from app.optimizer.model.planning_calendar import planning_weeks, effective_release_week


class RepaymentTests(unittest.TestCase):
    """Catch incorrect release timing, double subtraction and over-release."""

    def setUp(self):
        """Use a 40,000 opening balance and a three-Monday horizon."""
        self.base={'facility':{'F1':4000000},'customer':{'C1':4000000},'group':{'G1':4000000}}
        self.weeks=planning_weeks(date(2026,2,2),3)

    def test_missing_schedule_keeps_opening_constant(self):
        """Absent optional repayments preserve existing exposure behavior."""
        result=build_opening_schedule(self.base,[],self.weeks,{'C1':'G1'})
        self.assertEqual([result[week]['customer']['C1'] for week in self.weeks],[4000000]*3)

    def test_partial_release_retains_residual_at_all_levels(self):
        """A Tuesday release is available the following Monday and reduces each constraint once."""
        rows=[dict(customer_id='C1',facility_id='F1',release_date=date(2026,2,10),amount_cents=1000000)]
        result=build_opening_schedule(self.base,rows,self.weeks,{'C1':'G1'})
        self.assertEqual(result[self.weeks[1]]['customer']['C1'],4000000)
        self.assertEqual(result[self.weeks[2]],{'facility':{'F1':3000000},'customer':{'C1':3000000},'group':{'G1':3000000}})
        self.assertEqual(self.base['customer']['C1'],4000000)

    def test_invalid_repayments_are_rejected(self):
        """Unknown, duplicated, negative and over-large releases cannot create capacity."""
        valid=dict(customer_id='C1',facility_id='F1',release_date=date(2026,2,9),amount_cents=1000000)
        for rows in ([{**valid,'amount_cents':5000000}],[{**valid,'customer_id':'unknown'}],
                     [{**valid,'amount_cents':-1}],[valid,valid],[{**valid,'release_date':date(2026,1,1)}]):
            with self.subTest(rows=rows),self.assertRaises(ValueError):
                build_opening_schedule(self.base,rows,self.weeks,{'C1':'G1'})

    def test_monday_calendar_is_explicit(self):
        """Week starts must not be silently normalized to the legacy Tuesday period."""
        self.assertEqual(effective_release_week(date(2026,2,9)),date(2026,2,9))
        self.assertEqual(effective_release_week(date(2026,2,10)),date(2026,2,16))
        with self.assertRaises(ValueError): planning_weeks(date(2026,2,3),3)


class RepaymentCurrencyDateTests(unittest.TestCase):
    """Repayment-only currencies follow the same as-of policy as invoice conversions."""

    def test_future_fx_on_repayment_is_not_ready(self):
        """An EUR-only candidate cannot release exposure using a future GBP rate."""
        from app.services.workspace.settings import validate_settings
        settings={'planning_start':'2026-02-02','horizon_weeks':2,'seller_to_facility':{'S':'F'},
            'customer_to_facility':{'C':'F'},'customer_limits':{'C':100},'facility_limits_by_company_code':{'F':100},
            'base_exposure':{'facility':{'F':80},'customer':{'C':80}},
            'currency_rates':{'GBP':{'eur_per_unit':'2','as_of':'2099-01-01'}},
            'expected_repayments':[{'customer_id':'C','facility_id':'F','amount':'40','currency':'GBP','release_date':'2026-02-09'}]}
        rows=[{'row_id':'a','invoice':{'debtor_id':'C','seller_id':'S','original_currency':'EUR'}}]
        result=validate_settings(settings,rows)
        self.assertTrue(any(issue['path']=='currency_rates.GBP' for issue in result['readiness_issues']))
        self.assertEqual(result['weekly_opening_preview'],[])
