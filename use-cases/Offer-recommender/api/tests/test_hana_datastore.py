from __future__ import annotations

from app.nbo.config import COL_BILLING_ACCOUNT, COL_OTHER_PROGRAMS, COL_RATE_PLAN
from app.nbo.data_loader import DataStore
from synthetic_runtime import synthetic_runtime_datasets


def test_hana_backed_datastore_preserves_normalized_account_sets() -> None:
    datasets = synthetic_runtime_datasets()
    store = DataStore(datasets)

    expected_residential_accounts = sorted(
        datasets["residential"][COL_BILLING_ACCOUNT].astype(str).str.strip().unique()
    )
    expected_commercial_accounts = sorted(
        datasets["commercial"][COL_BILLING_ACCOUNT].astype(str).str.strip().unique()
    )

    assert store.residential_accounts() == expected_residential_accounts
    assert store.commercial_accounts() == expected_commercial_accounts
    assert store.l1_account_type("104") == "RESIDENTIAL"
    assert store.l1_account_type("1004") == "COMMERCIAL"


def test_hana_backed_datastore_preserves_current_snapshot_and_program_codes() -> None:
    datasets = synthetic_runtime_datasets()
    store = DataStore(datasets)

    snapshot = store.l2_current_snapshot("104")

    assert snapshot is not None
    assert snapshot[COL_BILLING_ACCOUNT] == "104"
    assert snapshot[COL_RATE_PLAN]

    raw_programs = snapshot.get(COL_OTHER_PROGRAMS)
    if isinstance(raw_programs, str) and raw_programs.strip():
        expected_codes = {token.strip() for token in raw_programs.split(";") if token.strip()}
    else:
        expected_codes = set()

    assert store.l8_current_program_codes("104") == expected_codes


def test_hana_backed_datastore_resolves_rate_plan_business_offering_names() -> None:
    datasets = synthetic_runtime_datasets()
    store = DataStore(datasets)

    assert (
        store.rate_plan_display_name("E21")
        == "Residential Time of Use 3-6"
    )
    assert (
        store.rate_plan_display_name("E00")
        == "Inactive account"
    )
    assert store.rate_plan_display_name("UNKNOWN") == "UNKNOWN"
