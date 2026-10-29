"""Manual settings validation must precede saving or solver execution."""
import unittest
from app.services.workspace.settings import validate_settings


class WorkspaceSettingsTests(unittest.TestCase):
    """Verify missing capacity/FX and optional opening schedules using saved source rows."""

    def setUp(self):
        """Use a single EUR candidate and explicit manual capacity."""
        self.rows=[{'row_id':'one','invoice':{'seller_id':'S1','debtor_id':'C1','original_currency':'EUR','amount_original':'50'}}]
        self.settings={'planning_start':'2026-02-02','horizon_weeks':3,
            'seller_to_facility':{'S1':'F1'},'customer_to_facility':{'C1':'F1'},
            'facility_limits_by_company_code':{'F1':'100'},'customer_limits':{'C1':'100'},
            'base_exposure':{'facility':{'F1':'80'},'customer':{'C1':'80'},'group':{}}}

    def test_missing_repayments_preserve_constant_opening(self):
        """A valid opening input needs no repayment schedule or automatic assumption."""
        result=validate_settings(self.settings,self.rows)
        self.assertEqual(result['readiness_issues'],[])
        self.assertEqual([row['opening']['customer']['C1'] for row in result['weekly_opening_preview']],['80.00']*3)

    def test_missing_opening_requires_explicit_zero(self):
        """Omitted exposure is not silently interpreted as unused credit."""
        settings={key:value for key,value in self.settings.items() if key!='base_exposure'}
        self.assertTrue(validate_settings(settings,self.rows)['readiness_issues'])
        settings['opening_confirmed_zero']=True
        self.assertEqual(validate_settings(settings,self.rows)['readiness_issues'],[])

    def test_unsupported_currency_and_invalid_mapping_are_not_ready(self):
        """No hidden GBP conversion or customer association can admit a run."""
        self.rows[0]['invoice']['original_currency']='GBP'
        self.assertTrue(validate_settings(self.settings,self.rows)['readiness_issues'])

    def test_yaml_import_retains_optional_schedule_and_date(self):
        """YAML-native dates are accepted and an omitted schedule remains optional."""
        from app.services.workspace.settings_import import import_settings
        draft=import_settings(b'planning_start: 2026-02-02\nexpected_repayments: []\n','settings.yaml',self.settings)
        result=validate_settings(draft,self.rows)
        self.assertEqual(result['readiness_issues'],[])
        self.assertEqual(result['normalized_settings']['planning_start'],'2026-02-02')
        self.settings['currency_rates']={'GBP':{'eur_per_unit':'1.1','as_of':'2026-02-02'}}
        self.settings['expected_repayments']=[{'customer_id':'C1','facility_id':'OTHER',
            'release_date':'2026-02-09','amount':'40','currency':'EUR'}]
        self.assertTrue(validate_settings(self.settings,self.rows)['readiness_issues'])
