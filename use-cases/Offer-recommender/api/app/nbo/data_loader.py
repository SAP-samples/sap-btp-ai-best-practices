"""DataStore -- load the HANA-backed runtime tables and expose canonical lookups.

Lookups implemented: L1-L10 and named accessors for the new profile/history
tables. L12 is handled by pdf_extractor, L14 by persona.py.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from app.nbo import hana as hana_runtime
from app.nbo.config import (
    COL_BILLING_ACCOUNT,
    COL_DER_BATTERY_OWNERSHIP,
    COL_DER_BATTERY_PARTNER_BRAND,
    COL_DER_COMPATIBLE_BATTERY,
    COL_DER_DEMAND_MGMT_INCLUDED,
    COL_DER_ELIGIBLE_CONNECTED_DEVICE,
    COL_DER_ELIGIBLE_HOME_EV_CHARGER,
    COL_DER_EV_CHARGER_BRAND,
    COL_DER_NEW_ROOFTOP_SOLAR,
    COL_DER_PREFERRED_SOLAR_INSTALLER,
    COL_DER_PRIOR_REC_ASSIGNMENT,
    COL_DER_QUALIFYING_FACILITY_KW_AC,
    COL_DER_REC_RIGHTS_OWNED,
    COL_DER_SMART_THERMOSTAT_PURCHASE,
    COL_DER_SOLAR_OWNERSHIP,
    COL_DER_STORAGE_ONLY,
    COL_DER_THERMOSTAT_BRAND,
    COL_DER_THERMOSTAT_PROVIDER_ACCOUNT,
    COL_DER_THERMOSTAT_WIFI,
    COL_STATUS,
    COL_RATE_PLAN,
    COL_READ_DATE,
    COL_BILL_TOTAL,
    COL_BILL_PAID,
    COL_ON_PEAK,
    COL_OFF_PEAK,
    COL_METER_USAGE,
    COL_OTHER_PROGRAMS,
    COL_CUSTOMER_TYPE,
    COL_BIZ_OFFERING_NAME,
    COL_SEGMENT_NAME,
    COL_COMM_BILLING_ACCOUNT,
    COL_COMM_SEGMENT_NAME,
    COL_EVENT_DATE,
    COL_PC_PROGRAM,
    COL_PC_SERVICE_OPTION,
    COL_PROFILE_ACCOUNT_GOOD_STANDING,
    COL_PROFILE_ACCOUNT_NAME_TYPE,
    COL_PROFILE_CENTRAL_AC,
    COL_PROFILE_COOLING_SYSTEM_TYPE,
    COL_PROFILE_CONDITIONED_SQFT,
    COL_PROFILE_CONNECTED_UNIT_COUNT,
    COL_PROFILE_CURTAILMENT_CAPABILITY,
    COL_PROFILE_DWELLING_TYPE,
    COL_PROFILE_ELIGIBLE_CONTRACTOR,
    COL_PROFILE_ENERGY_STAR_MFNC,
    COL_PROFILE_LARGE_ENERGY_CONSUMER,
    COL_PROFILE_NEW_CONSTRUCTION,
    COL_PROFILE_OCCUPANCY_STATUS,
    COL_PROFILE_OWNERSHIP_STATUS,
    COL_PROFILE_PROJECT_EXCEEDS_BASELINE_10,
    COL_PROFILE_PROJECT_STAGE,
    COL_PROFILE_QUALIFIED_PRICE_PLAN,
    COL_PROFILE_SAME_ACCOUNT_HOLDER_12M,
    COL_PROFILE_SERVICE_ENTRANCE_AMPS,
    COL_PROFILE_SERVICE_START_DATE,
    USAGE_AVERAGE_PERIODS,
)


class DataStore:
    """Loads and normalises the HANA-backed runtime datasets."""

    def __init__(self, datasets: dict[str, pd.DataFrame] | None = None) -> None:
        loaded = datasets or hana_runtime.load_runtime_datasets()
        missing = set(hana_runtime.DATASET_TABLES) - set(loaded)
        if missing:
            joined = ", ".join(sorted(missing))
            raise ValueError(f"Missing runtime datasets: {joined}")

        self.residential = loaded["residential"].copy(deep=True)
        self.res_segment = loaded["res_segment"].copy(deep=True)
        self.commercial = loaded["commercial"].copy(deep=True)
        self.comm_segment = loaded["comm_segment"].copy(deep=True)
        self.active_offering = loaded["active_offering"].copy(deep=True)
        self.program_contract = loaded["program_contract"].copy(deep=True)
        self.program_samples = loaded["program_samples"].copy(deep=True)
        self.account_profile = loaded["account_profile"].copy(deep=True)
        self.der_profile = loaded["der_profile"].copy(deep=True)
        self.program_event_history = loaded["program_event_history"].copy(deep=True)

        # ── Normalise at load time ────────────────────────────────────────
        self._normalise()

        # ── Pre-compute account sets for fast L1 lookup ───────────────────
        self._res_accounts: set[str] = set(
            self.residential[COL_BILLING_ACCOUNT].astype(str).unique()
        )
        self._comm_accounts: set[str] = set(
            self.commercial[COL_BILLING_ACCOUNT].astype(str).unique()
        )

        # ── Pre-compute the Prepay rate plan set (L5) ────────────────────
        self.mpower_rate_plans: set[str] = self._compute_mpower_set()

        # ── Pre-compute per-account 75th-percentile meter usage ───────────
        all_usage = pd.concat(
            [self.residential[COL_METER_USAGE], self.commercial[COL_METER_USAGE]]
        ).dropna()
        self.usage_p75: float = float(all_usage.quantile(0.75))

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _normalise(self) -> None:
        """Apply one-time data cleaning on every loaded DataFrame."""
        for df in (
            self.residential,
            self.commercial,
            self.active_offering,
            self.res_segment,
            self.comm_segment,
            self.program_contract,
            self.program_samples,
            self.account_profile,
            self.der_profile,
            self.program_event_history,
        ):
            # Strip whitespace from all string columns
            for col in df.select_dtypes(include=["object", "string"]).columns:
                df[col] = df[col].map(
                    lambda v: v.strip() if isinstance(v, str) else v
                )

        # Fix the "COMMERICIAL" typo in Active business offering
        mask = self.active_offering[COL_CUSTOMER_TYPE].str.upper() == "COMMERICIAL"
        self.active_offering.loc[mask, COL_CUSTOMER_TYPE] = "COMMERCIAL"
        # Uppercase customer type for consistent matching
        self.active_offering[COL_CUSTOMER_TYPE] = (
            self.active_offering[COL_CUSTOMER_TYPE].str.upper()
        )

        # Parse READ DATE to datetime
        for df in (self.residential, self.commercial):
            df[COL_READ_DATE] = pd.to_datetime(df[COL_READ_DATE], errors="coerce")
        if COL_PROFILE_SERVICE_START_DATE in self.account_profile.columns:
            self.account_profile[COL_PROFILE_SERVICE_START_DATE] = pd.to_datetime(
                self.account_profile[COL_PROFILE_SERVICE_START_DATE],
                errors="coerce",
            )
        if COL_EVENT_DATE in self.program_event_history.columns:
            self.program_event_history[COL_EVENT_DATE] = pd.to_datetime(
                self.program_event_history[COL_EVENT_DATE],
                errors="coerce",
            )

        # Coerce numeric columns
        numeric_cols = [COL_BILL_TOTAL, COL_BILL_PAID, COL_ON_PEAK, COL_OFF_PEAK, COL_METER_USAGE]
        for df in (self.residential, self.commercial):
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        # Ensure billing account is string everywhere
        self.residential[COL_BILLING_ACCOUNT] = (
            self.residential[COL_BILLING_ACCOUNT].astype(str).str.strip()
        )
        self.commercial[COL_BILLING_ACCOUNT] = (
            self.commercial[COL_BILLING_ACCOUNT].astype(str).str.strip()
        )
        self.res_segment[COL_BILLING_ACCOUNT] = (
            self.res_segment[COL_BILLING_ACCOUNT].astype(str).str.strip()
        )
        self.comm_segment[COL_COMM_BILLING_ACCOUNT] = (
            self.comm_segment[COL_COMM_BILLING_ACCOUNT].astype(str).str.strip()
        )
        if COL_BILLING_ACCOUNT in self.account_profile.columns:
            self.account_profile[COL_BILLING_ACCOUNT] = (
                self.account_profile[COL_BILLING_ACCOUNT].astype(str).str.strip()
            )
        if COL_BILLING_ACCOUNT in self.der_profile.columns:
            self.der_profile[COL_BILLING_ACCOUNT] = (
                self.der_profile[COL_BILLING_ACCOUNT].astype(str).str.strip()
            )
        if COL_BILLING_ACCOUNT in self.program_event_history.columns:
            self.program_event_history[COL_BILLING_ACCOUNT] = (
                self.program_event_history[COL_BILLING_ACCOUNT].astype(str).str.strip()
            )

        for df, numeric_cols in (
            (
                self.account_profile,
                [
                    COL_PROFILE_SERVICE_ENTRANCE_AMPS,
                    COL_PROFILE_CONNECTED_UNIT_COUNT,
                    COL_PROFILE_CONDITIONED_SQFT,
                ],
            ),
            (
                self.der_profile,
                [COL_DER_QUALIFYING_FACILITY_KW_AC],
            ),
        ):
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

    def _compute_mpower_set(self) -> set[str]:
        """Return residential prepay rate plans from the active-offering table."""
        ao = self.active_offering
        mask = (
            (ao[COL_CUSTOMER_TYPE] == "RESIDENTIAL")
            & ao[COL_BIZ_OFFERING_NAME].str.contains("M-POWER|PREPAY", case=False, na=False)
        )
        return set(ao.loc[mask, COL_RATE_PLAN].str.strip().unique())

    @staticmethod
    def _clean_business_offering_name(
        rate_plan: str,
        offering_name: object,
    ) -> str | None:
        """Return a display label from a raw business-offering value.

        Inputs:
            rate_plan: The canonical rate-plan code used by decision logic.
            offering_name: The workbook/HANA value, commonly formatted as
                ``{RATE_ID}-{NAME}``.

        Output:
            The user-facing business offering name without the leading rate-plan
            code, or ``None`` when no usable name is available.
        """
        if pd.isna(offering_name):
            return None
        raw_name = str(offering_name).strip()
        if not raw_name:
            return None

        normalized_rate = str(rate_plan).strip()
        if raw_name.upper().startswith(normalized_rate.upper()):
            suffix = raw_name[len(normalized_rate):].strip()
            if suffix.startswith("-"):
                suffix = suffix[1:].strip()
            return suffix or raw_name
        return raw_name

    def rate_plan_display_name(self, rate_plan: str | None) -> str | None:
        """Return the user-facing business offering name for a rate-plan code.

        Inputs:
            rate_plan: A rate-plan identifier such as ``E21``.

        Output:
            The cleaned ``BUSINESS OFFERING NAME`` from the HANA-backed active
            offering table. Unknown plans fall back to the original identifier so
            the explanation remains complete instead of showing a blank value.
        """
        if rate_plan is None:
            return None
        normalized_rate = str(rate_plan).strip()
        if not normalized_rate:
            return None

        rate_series = self.active_offering[COL_RATE_PLAN].astype(str).str.strip()
        matches = self.active_offering[rate_series == normalized_rate]
        if matches.empty:
            return normalized_rate

        for value in matches[COL_BIZ_OFFERING_NAME]:
            display_name = self._clean_business_offering_name(normalized_rate, value)
            if display_name:
                return display_name
        return normalized_rate

    # ──────────────────────────────────────────────────────────────────────
    # Canonical lookups
    # ──────────────────────────────────────────────────────────────────────

    def l1_account_type(self, account: str) -> str | None:
        """Return 'RESIDENTIAL', 'COMMERCIAL', or None."""
        account = str(account).strip()
        if account in self._res_accounts:
            return "RESIDENTIAL"
        if account in self._comm_accounts:
            return "COMMERCIAL"
        return None

    def _base_df(self, account: str) -> pd.DataFrame:
        """Return the base sheet (Residential or Commercial) for this account."""
        ctype = self.l1_account_type(account)
        if ctype == "RESIDENTIAL":
            return self.residential
        if ctype == "COMMERCIAL":
            return self.commercial
        return pd.DataFrame()

    def l2_current_snapshot(self, account: str) -> dict | None:
        """Latest row by READ DATE for this account.  Returns dict or None."""
        account = str(account).strip()
        df = self._base_df(account)
        if df.empty:
            return None
        rows = df[df[COL_BILLING_ACCOUNT] == account].dropna(subset=[COL_READ_DATE])
        if rows.empty:
            return None
        latest = rows.sort_values(COL_READ_DATE, ascending=False).iloc[0]
        return latest.to_dict()

    def l3_account_history(self, account: str) -> pd.DataFrame:
        """All rows for the account, sorted ascending by READ DATE."""
        account = str(account).strip()
        df = self._base_df(account)
        if df.empty:
            return pd.DataFrame()
        rows = df[df[COL_BILLING_ACCOUNT] == account].dropna(subset=[COL_READ_DATE])
        return rows.sort_values(COL_READ_DATE, ascending=True).reset_index(drop=True)

    def l4_segment(self, account: str) -> str | None:
        """Segment name from persona lookup.  Returns name or None."""
        account = str(account).strip()
        ctype = self.l1_account_type(account)
        if ctype == "RESIDENTIAL":
            match = self.res_segment[
                self.res_segment[COL_BILLING_ACCOUNT] == account
            ]
            if not match.empty:
                val = match.iloc[0][COL_SEGMENT_NAME]
                if pd.notna(val) and str(val).strip().upper() != "NOT FOUND":
                    return str(val).strip()
        elif ctype == "COMMERCIAL":
            match = self.comm_segment[
                self.comm_segment[COL_COMM_BILLING_ACCOUNT] == account
            ]
            if not match.empty:
                val = match.iloc[0][COL_COMM_SEGMENT_NAME]
                if pd.notna(val) and str(val).strip().upper() != "NOT FOUND":
                    return str(val).strip()
        return None

    def l5_mpower_rate_plans(self) -> set[str]:
        """L5 -- residential Prepay rate plan set (pre-computed)."""
        return self.mpower_rate_plans

    def l6_residential_rate_catalog(self) -> pd.DataFrame:
        """All residential rate plans from Active business offering."""
        ao = self.active_offering
        return ao[ao[COL_CUSTOMER_TYPE] == "RESIDENTIAL"].copy()

    def l7_commercial_rate_catalog(self) -> pd.DataFrame:
        """All commercial rate plans from Active business offering."""
        ao = self.active_offering
        return ao[ao[COL_CUSTOMER_TYPE] == "COMMERCIAL"].copy()

    def l8_current_program_codes(self, account: str) -> set[str]:
        """Split OTHER PROGRAMS on ';' from the current snapshot."""
        snap = self.l2_current_snapshot(account)
        if snap is None:
            return set()
        raw = snap.get(COL_OTHER_PROGRAMS)
        if pd.isna(raw) or not isinstance(raw, str):
            return set()
        return {tok.strip() for tok in raw.split(";") if tok.strip()}

    def l9_usage_averages(self, account: str, n: int = USAGE_AVERAGE_PERIODS) -> dict:
        """Average ON-PEAK, OFF-PEAK, METER READ USAGE over last *n* bills.

        On-peak and off-peak averages are computed from the last *n* rows that
        have BOTH on-peak AND off-peak values non-null, ensuring the averages
        are consistent with each other and with the manual calculation method.
        Meter usage average is computed independently from the last *n* rows
        that have non-null meter usage.
        """
        history = self.l3_account_history(account)
        if history.empty:
            return {"on_peak": None, "off_peak": None, "meter_usage": None}

        # Meter usage: last n rows with non-null meter usage
        recent_meter = history.dropna(subset=[COL_METER_USAGE]).tail(n)
        meter_usage = recent_meter[COL_METER_USAGE].mean() if not recent_meter.empty else None

        # On-peak / off-peak: last n rows that have BOTH values non-null
        # This ensures the averages are computed over the same set of rows
        on_peak = None
        off_peak = None
        if COL_ON_PEAK in history.columns and COL_OFF_PEAK in history.columns:
            recent_tou = history.dropna(subset=[COL_ON_PEAK, COL_OFF_PEAK]).tail(n)
            if not recent_tou.empty:
                on_peak = recent_tou[COL_ON_PEAK].mean()
                off_peak = recent_tou[COL_OFF_PEAK].mean()

        return {
            "on_peak": on_peak,
            "off_peak": off_peak,
            "meter_usage": meter_usage,
        }

    def l9_summer_usage_averages(self, account: str) -> dict:
        """Average ON-PEAK, OFF-PEAK, METER READ USAGE over summer months (May-Oct).

        Summer months are when TOU rate differences are most significant,
        so using summer averages provides a fairer comparison between rate plans.

        Summer months: May (5), June (6), July (7), August (8), September (9), October (10)
        """
        history = self.l3_account_history(account)
        if history.empty:
            return {"on_peak": None, "off_peak": None, "meter_usage": None}

        # Filter for summer months (May through October)
        summer_months = {5, 6, 7, 8, 9, 10}
        summer_data = history[history[COL_READ_DATE].dt.month.isin(summer_months)]

        if summer_data.empty:
            return {"on_peak": None, "off_peak": None, "meter_usage": None}

        # Drop rows with null meter usage
        summer_data = summer_data.dropna(subset=[COL_METER_USAGE])

        if summer_data.empty:
            return {"on_peak": None, "off_peak": None, "meter_usage": None}

        on_peak = None
        off_peak = None
        meter_usage = None

        if COL_ON_PEAK in summer_data.columns and not summer_data[COL_ON_PEAK].isna().all():
            on_peak = summer_data[COL_ON_PEAK].mean()

        if COL_OFF_PEAK in summer_data.columns and not summer_data[COL_OFF_PEAK].isna().all():
            off_peak = summer_data[COL_OFF_PEAK].mean()

        if not summer_data.empty:
            meter_usage = summer_data[COL_METER_USAGE].mean()

        return {
            "on_peak": on_peak,
            "off_peak": off_peak,
            "meter_usage": meter_usage,
        }

    def l10_bill_trend(self, account: str) -> dict:
        """Aggregate bill totals and paid amounts over full history."""
        history = self.l3_account_history(account)
        if history.empty:
            return {
                "total_sum": None,
                "paid_sum": None,
                "total_mean": None,
                "paid_mean": None,
                "row_count": 0,
            }
        return {
            "total_sum": history[COL_BILL_TOTAL].sum(),
            "paid_sum": history[COL_BILL_PAID].sum(),
            "total_mean": history[COL_BILL_TOTAL].mean(),
            "paid_mean": history[COL_BILL_PAID].mean(),
            "row_count": len(history),
        }

    def l13_program_catalog(self) -> pd.DataFrame:
        """Return the cleaned Program Contract DataFrame."""
        return self.program_contract.copy()

    def account_profile_row(self, account: str) -> dict | None:
        account = str(account).strip()
        if self.account_profile.empty:
            return None
        match = self.account_profile[self.account_profile[COL_BILLING_ACCOUNT] == account]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    def der_profile_row(self, account: str) -> dict | None:
        account = str(account).strip()
        if self.der_profile.empty:
            return None
        match = self.der_profile[self.der_profile[COL_BILLING_ACCOUNT] == account]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    def program_event_history_rows(self, account: str) -> pd.DataFrame:
        account = str(account).strip()
        if self.program_event_history.empty:
            return pd.DataFrame(columns=self.program_event_history.columns)
        rows = self.program_event_history[
            self.program_event_history[COL_BILLING_ACCOUNT] == account
        ]
        if COL_EVENT_DATE in rows.columns:
            rows = rows.sort_values(COL_EVENT_DATE, ascending=False)
        return rows.reset_index(drop=True)

    def program_code_reference(self) -> pd.DataFrame:
        """Return cleaned program-code workbook rows without repeated header markers."""
        df = self.program_contract.copy()
        if df.empty:
            return df
        if COL_PC_PROGRAM in df.columns:
            df = df[df[COL_PC_PROGRAM].notna()]
            df = df[
                ~df[COL_PC_PROGRAM].astype(str).str.strip().isin(
                    {
                        "",
                        "RESIDENTIAL AND COMMERCIAL",
                        "COMMERCIAL ONLY",
                        "Program/Contract",
                    }
                )
            ]
        return df.reset_index(drop=True)

    def resolve_program_code_aliases(self, match_terms: list[str]) -> list[str]:
        """Resolve workbook program codes by matching program names against terms."""
        if not match_terms:
            return []
        normalized_terms = [term.strip().casefold() for term in match_terms if term.strip()]
        if not normalized_terms:
            return []

        df = self.program_code_reference()
        if df.empty:
            return []

        codes: set[str] = set()
        for _, row in df.iterrows():
            raw_program = str(row.get(COL_PC_PROGRAM, "") or "").strip()
            raw_code = str(row.get(COL_PC_SERVICE_OPTION, "") or "").strip()
            if not raw_program or not raw_code:
                continue
            haystack = raw_program.casefold()
            if any(term in haystack for term in normalized_terms):
                for token in raw_code.replace("\r", "\n").split("\n"):
                    token = token.strip()
                    if token:
                        codes.add(token)
        return sorted(codes)

    # ──────────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────────

    def all_accounts(self) -> list[str]:
        """Return all unique billing accounts across both sheets."""
        return sorted(self._res_accounts | self._comm_accounts)

    def residential_accounts(self) -> list[str]:
        return sorted(self._res_accounts)

    def commercial_accounts(self) -> list[str]:
        return sorted(self._comm_accounts)
