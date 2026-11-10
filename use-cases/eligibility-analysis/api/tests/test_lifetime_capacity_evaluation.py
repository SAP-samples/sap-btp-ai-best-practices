"""A forecast-feasible plan is not evidence of observed capacity compliance."""
import unittest
from datetime import date
from app.services.lifecycle.capacity_evaluation import capacity_breaches,replay_capacity


class CapacityTests(unittest.TestCase):
    """Hold funding decisions fixed while changing only lifecycle release timing."""

    def test_later_observed_release_causes_breach(self):
        """Longer realized lifetime overlaps an already committed second funding decision."""
        weeks=[date(2026,2,2),date(2026,2,9)]
        schedule=[dict(row_id='a',funding_week=weeks[0],amount_cents=80,customer_id='C',facility_id='F'),
                  dict(row_id='b',funding_week=weeks[1],amount_cents=60,customer_id='C',facility_id='F')]
        limits={'facility':{'F':100},'customer':{'C':100},'group':{'G':100}}
        forecast=replay_capacity(schedule,{'a':7,'b':7},weeks,limits,{}, {'C':'G'})
        observed=replay_capacity(schedule,{'a':14,'b':7},weeks,limits,{}, {'C':'G'})
        self.assertEqual(forecast['levels']['facility']['F']['breach_weeks'],0)
        self.assertEqual(observed['levels']['facility']['F']['max_excess_cents'],40)
        self.assertEqual(observed['levels']['group']['G']['breach_weeks'],1)

    def test_missing_outcomes_and_opening_remain_explicit(self):
        """Incomplete truth cannot become a zero-risk claim or erase opening exposure."""
        week=date(2026,2,2);schedule=[dict(row_id='missing',funding_week=week,amount_cents=50,customer_id='C',facility_id='F')]
        result=replay_capacity(schedule,{},[week],{'facility':{'F':100}},{week:{'facility':{'F':30}}},{})
        self.assertEqual(result['missing_outcome_row_ids'],['missing'])
        self.assertEqual(result['levels']['facility']['F']['used_cents'],[30])
        with self.assertRaises(ValueError):capacity_breaches([1],[])
