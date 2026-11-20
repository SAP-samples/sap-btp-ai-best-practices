# MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_START: curated overlay contract tests.
# SnapshotService no longer invokes the temporary overlay during validation.
# """Curated demonstration overlay contract tests."""
#
# from app.ecn.mock_overlay import merge_mock_data
#
#
# def test_purchase_prices_and_phase_schedule_keep_their_demo_basis_and_dates():
#     snapshot = merge_mock_data({"ecn_number": "ECN346863", "questions": []})
#
#     price = next(item for item in snapshot["purchase_prices"] if (
#         item["material_number"] == "QKX7702260" and item["plant_code"] == "PL10"
#     ))
#     assert price["valid_from"] == "2023-01-01"
#     assert price["valid_to"] == "2026-12-31"
#     assert price["source"] == "CURATED_DEMONSTRATION"
#
#     schedule = next(item for item in snapshot["phase_schedule"] if item["material_number"] == "QKX5949130")
#     assert schedule["effective_date"] == "2026-05-19"
#     assert schedule["source"] == "CURATED_DEMONSTRATION"
#
#     recommendation = next(item for item in snapshot["make_buy_recommendations"] if (
#         item["material_number"] == "QKX0187890" and item["plant_code"] == "PK01"
#     ))
#     assert recommendation["recommendation"] == "MAKE"
#     assert recommendation["make_cost"] == "0.42"
#     assert recommendation["comparable_history"]
#     assert recommendation["tooling"]
#     assert recommendation["capacity"]
#     assert recommendation["raw_materials"]
#     assert recommendation["source"] == "CURATED_DEMONSTRATION"
# MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_END: curated overlay contract tests.
