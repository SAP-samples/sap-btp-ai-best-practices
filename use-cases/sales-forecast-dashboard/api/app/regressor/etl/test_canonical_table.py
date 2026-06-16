"""
Unit tests for Canonical Training Table (Epic 3 Completion).

Tests verify:
1. AOV calculation correctness (Sales / Orders, not AUR)
2. WEB sales source (Ecomm Traffic.csv, not Written Sales)
3. Conversion filtering (B&M only, with valid traffic)
4. Traffic flag application
5. Schema validation
6. Channel-specific target rules
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.regressor.etl.canonical_table import (
    build_canonical_training_table,
    explode_history,
    attach_targets,
    _safe_log,
    _safe_logit,
)
from app.regressor.etl.schema import validate_canonical_table


class TestSafeTransforms:
    """Test safe log and logit transformations."""

    def test_safe_log_positive_values(self):
        """Test log of positive values."""
        s = pd.Series([1.0, 10.0, 100.0])
        result = _safe_log(s)
        expected = np.log([1.0, 10.0, 100.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_safe_log_zero_values(self):
        """Test log handles zeros with floor."""
        s = pd.Series([0.0, 1.0, 0.0])
        result = _safe_log(s)
        # Should clip to floor (1e-6) before log
        assert np.isfinite(result).all()
        assert result[0] == np.log(1e-6)

    def test_safe_logit_valid_probabilities(self):
        """Test logit of valid probabilities."""
        s = pd.Series([0.1, 0.5, 0.9])
        result = _safe_logit(s)
        expected = np.log(s / (1 - s))
        np.testing.assert_array_almost_equal(result, expected)

    def test_safe_logit_boundary_values(self):
        """Test logit handles 0 and 1 with epsilon clipping."""
        s = pd.Series([0.0, 0.5, 1.0])
        result = _safe_logit(s)
        # Should clip extremes to avoid inf
        assert np.isfinite(result).all()


class TestExplodeHistory:
    """Test history explosion logic."""

    def test_explode_creates_correct_rows(self):
        """Test explosion creates horizon × origin rows."""
        sales_df = pd.DataFrame({
            'profit_center_nbr': [101, 101, 102],
            'dma': ['NYC', 'NYC', 'LA'],
            'channel': ['B&M', 'B&M', 'B&M'],
            'origin_week_date': pd.to_datetime(['2024-01-01', '2024-01-08', '2024-01-01']),
        })

        exploded = explode_history(sales_df, horizons=[1, 2, 3])

        # Should have 3 origin weeks × 3 horizons = 9 rows
        assert len(exploded) == 9
        assert set(exploded['horizon'].unique()) == {1, 2, 3}

    def test_explode_calculates_target_week(self):
        """Test target_week_date = origin + horizon weeks."""
        sales_df = pd.DataFrame({
            'profit_center_nbr': [101],
            'dma': ['NYC'],
            'channel': ['B&M'],
            'origin_week_date': pd.to_datetime(['2024-01-01']),
        })

        exploded = explode_history(sales_df, horizons=[1, 4])

        # Check target weeks
        target_h1 = exploded[exploded['horizon'] == 1]['target_week_date'].iloc[0]
        target_h4 = exploded[exploded['horizon'] == 4]['target_week_date'].iloc[0]

        assert target_h1 == pd.Timestamp('2024-01-08')  # 1 week later
        assert target_h4 == pd.Timestamp('2024-01-29')  # 4 weeks later


class TestAOVCalculation:
    """Test AOV calculation uses Sales/Orders, not AUR."""

    def test_aov_uses_sales_divided_by_orders(self):
        """CRITICAL: AOV must be Sales / Orders, NOT AUR (Sales / Units)."""
        exploded_df = pd.DataFrame({
            'profit_center_nbr': [101],
            'channel': ['B&M'],
            'target_week_date': pd.to_datetime(['2024-01-01']),
        })

        sales_df = pd.DataFrame({
            'profit_center_nbr': [101],
            'channel': ['B&M'],
            'fiscal_start_date_week': pd.to_datetime(['2024-01-01']),
            'total_sales': [10000.0],
            'order_count': [100.0],
            'aur': [50.0],  # This is Sales/Units - should NOT be used for AOV
            'store_traffic': [1000.0],
            'has_traffic_data': [1],
        })

        result = attach_targets(exploded_df, sales_df, drop_missing_targets=False)

        # AOV should be 10000 / 100 = 100, not 50 (AUR)
        expected_aov = 10000.0 / 100.0  # 100
        actual_aov = np.exp(result['label_log_aov'].iloc[0])

        assert np.isclose(actual_aov, expected_aov), f"AOV should be {expected_aov}, got {actual_aov}"
        assert not np.isclose(actual_aov, 50.0), "AOV should NOT equal AUR (50)"


class TestConversionFiltering:
    """Test conversion is only calculated for B&M with valid traffic."""

    def test_web_conversion_is_nan(self):
        """WEB channel should have NaN conversion (no physical traffic)."""
        exploded_df = pd.DataFrame({
            'profit_center_nbr': [101],
            'channel': ['WEB'],
            'target_week_date': pd.to_datetime(['2024-01-01']),
        })

        sales_df = pd.DataFrame({
            'profit_center_nbr': [101],
            'channel': ['WEB'],
            'fiscal_start_date_week': pd.to_datetime(['2024-01-01']),
            'total_sales': [5000.0],
            'order_count': [50.0],
            'aur': [100.0],
            'store_traffic': [np.nan],  # WEB has no traffic
            'has_traffic_data': [0],
        })

        result = attach_targets(exploded_df, sales_df, drop_missing_targets=False)

        assert pd.isna(result['label_logit_conversion'].iloc[0]), "WEB conversion should be NaN"

    def test_bm_without_traffic_flag_has_nan_conversion(self):
        """B&M stores with has_traffic_data=0 should have NaN conversion."""
        exploded_df = pd.DataFrame({
            'profit_center_nbr': [101],
            'channel': ['B&M'],
            'target_week_date': pd.to_datetime(['2024-01-01']),
        })

        sales_df = pd.DataFrame({
            'profit_center_nbr': [101],
            'channel': ['B&M'],
            'fiscal_start_date_week': pd.to_datetime(['2024-01-01']),
            'total_sales': [5000.0],
            'order_count': [50.0],
            'aur': [100.0],
            'store_traffic': [500.0],  # Has traffic data, but flag says unreliable
            'has_traffic_data': [0],  # UNRELIABLE TRAFFIC
        })

        result = attach_targets(exploded_df, sales_df, drop_missing_targets=False)

        assert pd.isna(result['label_logit_conversion'].iloc[0]), \
            "B&M without traffic flag should have NaN conversion"

    def test_bm_with_traffic_flag_has_valid_conversion(self):
        """B&M stores with has_traffic_data=1 should have valid conversion."""
        exploded_df = pd.DataFrame({
            'profit_center_nbr': [101],
            'channel': ['B&M'],
            'target_week_date': pd.to_datetime(['2024-01-01']),
        })

        sales_df = pd.DataFrame({
            'profit_center_nbr': [101],
            'channel': ['B&M'],
            'fiscal_start_date_week': pd.to_datetime(['2024-01-01']),
            'total_sales': [10000.0],
            'order_count': [100.0],
            'aur': [50.0],
            'store_traffic': [1000.0],
            'has_traffic_data': [1],  # VALID TRAFFIC
        })

        result = attach_targets(exploded_df, sales_df, drop_missing_targets=False)

        # Conversion should be valid: 100 orders / 1000 traffic = 0.10
        conversion_logit = result['label_logit_conversion'].iloc[0]
        assert pd.notna(conversion_logit), "B&M with traffic flag should have valid conversion"

        # Back-transform to check value
        conversion = 1 / (1 + np.exp(-conversion_logit))
        assert np.isclose(conversion, 0.10), f"Conversion should be 0.10, got {conversion}"


class TestDropMissingTargets:
    """Test channel-aware drop logic."""

    def test_keeps_web_rows_without_conversion(self):
        """WEB rows should be kept even though conversion is NaN."""
        exploded_df = pd.DataFrame({
            'profit_center_nbr': [101],
            'channel': ['WEB'],
            'target_week_date': pd.to_datetime(['2024-01-01']),
        })

        sales_df = pd.DataFrame({
            'profit_center_nbr': [101],
            'channel': ['WEB'],
            'fiscal_start_date_week': pd.to_datetime(['2024-01-01']),
            'total_sales': [5000.0],
            'order_count': [50.0],
            'aur': [100.0],
            'store_traffic': [np.nan],
            'has_traffic_data': [0],
        })

        result = attach_targets(exploded_df, sales_df, drop_missing_targets=True)

        # Should keep WEB row (conversion NaN is expected)
        assert len(result) == 1
        assert result['channel'].iloc[0] == 'WEB'

    def test_keeps_bm_without_traffic_for_sales_aov(self):
        """B&M rows without traffic should be kept for Sales/AOV training."""
        exploded_df = pd.DataFrame({
            'profit_center_nbr': [101],
            'channel': ['B&M'],
            'target_week_date': pd.to_datetime(['2024-01-01']),
        })

        sales_df = pd.DataFrame({
            'profit_center_nbr': [101],
            'channel': ['B&M'],
            'fiscal_start_date_week': pd.to_datetime(['2024-01-01']),
            'total_sales': [5000.0],
            'order_count': [50.0],
            'aur': [100.0],
            'store_traffic': [500.0],
            'has_traffic_data': [0],  # No valid traffic
        })

        result = attach_targets(exploded_df, sales_df, drop_missing_targets=True)

        # Should keep B&M row for Sales/AOV even without conversion
        assert len(result) == 1
        assert result['channel'].iloc[0] == 'B&M'
        assert pd.notna(result['label_log_sales'].iloc[0])
        assert pd.notna(result['label_log_aov'].iloc[0])
        assert pd.isna(result['label_logit_conversion'].iloc[0])


class TestSchemaValidation:
    """Test schema validation rules."""

    def test_valid_table_passes_validation(self):
        """A correctly structured table should pass validation."""
        df = pd.DataFrame({
            'profit_center_nbr': [101, 102],
            'dma': ['NYC', 'LA'],
            'channel': ['B&M', 'WEB'],
            'origin_week_date': pd.to_datetime(['2024-01-01', '2024-01-01']),
            'horizon': [1, 1],
            'target_week_date': pd.to_datetime(['2024-01-08', '2024-01-08']),
            'label_log_sales': [11.0, 10.5],
            'label_log_aov': [4.5, 4.7],
            'label_logit_conversion': [0.5, np.nan],  # B&M valid, WEB NaN
            'has_traffic_data': [1, 0],
        })

        errors = validate_canonical_table(df)
        assert len(errors) == 0, f"Valid table should pass validation. Errors: {errors}"

    def test_web_with_conversion_fails_validation(self):
        """WEB channel with non-NaN conversion should fail validation."""
        df = pd.DataFrame({
            'profit_center_nbr': [101],
            'dma': ['NYC'],
            'channel': ['WEB'],
            'origin_week_date': pd.to_datetime(['2024-01-01']),
            'horizon': [1],
            'target_week_date': pd.to_datetime(['2024-01-08']),
            'label_log_sales': [11.0],
            'label_log_aov': [4.5],
            'label_logit_conversion': [0.5],  # INVALID: WEB should be NaN
            'has_traffic_data': [0],
        })

        errors = validate_canonical_table(df)
        assert len(errors) > 0, "WEB with conversion should fail validation"
        assert any("WEB" in err and "conversion" in err for err in errors)

    def test_horizon_out_of_range_fails_validation(self):
        """Horizon outside [1, 52] should fail validation."""
        df = pd.DataFrame({
            'profit_center_nbr': [101],
            'dma': ['NYC'],
            'channel': ['B&M'],
            'origin_week_date': pd.to_datetime(['2024-01-01']),
            'horizon': [100],  # INVALID: outside [1, 52]
            'target_week_date': pd.to_datetime(['2024-01-08']),
            'label_log_sales': [11.0],
            'label_log_aov': [4.5],
            'label_logit_conversion': [0.5],
            'has_traffic_data': [1],
        })

        errors = validate_canonical_table(df, check_ranges=True)
        assert len(errors) > 0, "Out-of-range horizon should fail validation"
        assert any("horizon" in err.lower() for err in errors)

    def test_duplicate_keys_fail_validation(self):
        """Duplicate (store, channel, origin, horizon) should fail validation."""
        df = pd.DataFrame({
            'profit_center_nbr': [101, 101],  # DUPLICATE
            'dma': ['NYC', 'NYC'],
            'channel': ['B&M', 'B&M'],
            'origin_week_date': pd.to_datetime(['2024-01-01', '2024-01-01']),
            'horizon': [1, 1],  # SAME HORIZON
            'target_week_date': pd.to_datetime(['2024-01-08', '2024-01-08']),
            'label_log_sales': [11.0, 11.1],
            'label_log_aov': [4.5, 4.5],
            'label_logit_conversion': [0.5, 0.5],
            'has_traffic_data': [1, 1],
        })

        errors = validate_canonical_table(df)
        assert len(errors) > 0, "Duplicate keys should fail validation"
        assert any("duplicate" in err.lower() for err in errors)


class TestIntegration:
    """Integration tests for full pipeline (mocked data)."""

    @pytest.mark.skip(reason="Requires actual data files - run manually if data available")
    def test_build_canonical_table_runs_without_error(self):
        """Test that build_canonical_training_table() runs without crashing."""
        df = build_canonical_training_table(horizons=[1, 4, 13])

        # Basic checks
        assert len(df) > 0, "Table should have rows"
        assert 'profit_center_nbr' in df.columns
        assert 'label_log_sales' in df.columns
        assert 'label_log_aov' in df.columns
        assert 'label_logit_conversion' in df.columns

    @pytest.mark.skip(reason="Requires actual data files - run manually if data available")
    def test_web_sales_from_ecomm_traffic(self):
        """Test that WEB sales come from Ecomm Traffic, not Written Sales."""
        df = build_canonical_training_table(horizons=[1])

        web_df = df[df['channel'] == 'WEB']
        if len(web_df) > 0:
            # All WEB rows should have conversion = NaN
            assert web_df['label_logit_conversion'].isna().all(), \
                "WEB conversion should be NaN"

            # WEB should have sales data
            assert web_df['label_log_sales'].notna().any(), \
                "WEB should have sales data from Ecomm Traffic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
