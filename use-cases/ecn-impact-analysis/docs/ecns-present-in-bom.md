# ECN numbers present in the BOM workbook

## Purpose

Tells the operator which ECNs from the ECN extract are worth importing. An ECN is
only useful for impact analysis if the BOM workbook carries lines created or changed
by it; an ECN with no BOM footprint has no structure data to reconcile against.

## Inputs and outputs

- ECN source: `data/new_16-09/ECN_full.json` (557 ECNs, one per `Number`).
- BOM source: `data/EBOM MBOM_Q900V_04.09.2026.xlsx`, sheets `EBOM Q900V`,
  `EBOM 130% KM_Q900VPXA`, `MBOM 130% KM_Q900VPXA_PXA`.
- Output: this list. A BOM row belongs to an ECN when its change-number column
  matches the ECN by the same rule the application uses
  (`change_code_matches` in `api/app/ecn/evidence/bom_changes.py`):
  - EBOM sheets: the cell equals the ECN number exactly.
  - MBOM sheet: the code embeds the ECN digits, e.g. `MPXA3452740A` belongs to
    `ECN345274` (pattern `^[A-Z]{4}(\d{6})[0-9A-Z]{2}$`), checked in both the
    "from" and "to" columns.
  - Other change numbers in the workbook (`ECNPXA01`, `EMOD_*`, `M*`, `VM*`, `A*`,
    `F*`, `T*`) never match an ECN from the JSON and are ignored.

| Sheet | Column(s) | Rows | Distinct JSON ECNs found | Used by the application for BOM changes |
|---|---|---:|---:|---|
| `EBOM Q900V` | `N° of modification` | 21,072 | 224 | yes (enriched EBOM) |
| `MBOM 130% KM_Q900VPXA_PXA` | `N° of modification`, `at the number of modif.` | 12,906 | 96 | yes |
| `EBOM 130% KM_Q900VPXA` | `Change No.` | 36,820 | 557 | only part rows the 150% EBOM lacks (see below) |

## Result

- All 557 ECNs have at least one row in `EBOM 130%`. The change builder reads
  enriched EBOM (`EBOM Q900V`) and MBOM rows, and an `EBOM 130%` row only when it is a
  part row with no `EBOM Q900V` row at the same position (see
  [130% EBOM rows in BOM changes](ecn-130-ebom-changes.md)). Document rows never
  produce BOM changes. Example: ECN142964 sits only on two `EBOM 130%` rows of
  the drawing-like component `9981771T1 ZVD 000 05`, not on the material `9981771T1`.
- **228 ECNs** appear in `EBOM Q900V` and/or the MBOM (92 in both, 132 only in
  `EBOM Q900V`, 4 only in the MBOM). **4 more ECNs** (ECN193101, ECN301489, ECN304579,
  ECN307099) appear only as part rows of `EBOM 130%`; since the 130% EBOM rows feature
  (see [130% EBOM rows in BOM changes](ecn-130-ebom-changes.md)) they also produce BOM
  changes. In total **232 ECNs** yield BOM changes.
- **325 ECNs** appear only on document rows of `EBOM 130%` (321 on confirmed documents,
  4 on document-reference candidates: ECN165060, ECN191720, ECN225507, ECN352932). Importing them works, but
  the BOM changes panel states that no part row carries the change number, because
  document rows are not BOM changes. The affected parts can still show plants, through
  part-number where-used.
- State of the 228: 225 RESOLVED,
  3 RELEASED.

Recommendation: import the 228 ECNs of sections 1 to 3 plus the four part-row ECNs named in the Result section (232 in total) to get BOM data.

## 1. In both `EBOM Q900V` and the MBOM (92)

| ECN | State | Resolution date | EBOM Q900V rows | MBOM rows | EBOM 130% rows | Name |
|---|---|---|---:|---:|---:|---|
| ECN268604 | RESOLVED | 2019-09-19 | 7 | 1 | 7 | PXB_Q932V_P1A_Release_S0190 |
| ECN269160 | RESOLVED | 2019-09-16 | 6 | 1 | 1 | PXB_Q932V_P1A_Release_S4010 |
| ECN277892 | RESOLVED | 2020-05-13 | 1 | 1 | 1 | PXB_Q932V_PS Release S4010 content |
| ECN279096 | RESOLVED | 2021-03-12 | 2 | 10 | 2 | Q9-23 S2526 P1A Release new additional light |
| ECN284860 | RESOLVED | 2020-11-09 | 1 | 2 | 4 | PXB_Q900V_P0_S0170_BRAKE_RELEASE |
| ECN284868 | RESOLVED | 2020-10-28 | 4 | 2 | 4 | PXB_Q900V_P0_S0140_DIFFERENTIAL_RELEASE |
| ECN293486 | RESOLVED | 2021-08-26 | 1 | 3 | 7 | Q9-23 S2526+S1002 Release VC with hardware |
| ECN294468 | RESOLVED | 2021-05-27 | 2 | 3 | 3 | Q923V-Release feature unlock |
| ECN296075 | RESOLVED | 2021-11-29 | 3 | 10 | 3 | Q923V S2526 Release road light trafic LHS |
| ECN296328 | RESOLVED | 2021-09-20 | 1 | 4 | 1 | Q9-23 P1A Release 2X on VC group |
| ECN302837 | RESOLVED | 2022-03-01 | 9 | 1 | 11 | PXB_APAJA_R10_B-sample release |
| ECN302838 | RESOLVED | 2022-03-01 | 54 | 5 | 55 | PXB_APAJA_R20_B-sample release |
| ECN304923 | RESOLVED | 2022-07-07 | 2 | 1 | 3 | PXB_Q900V_P1A_Release_of_S0101 |
| ECN304927 | RESOLVED | 2022-06-23 | 3 | 4 | 3 | PXB_Q900V_P1A_Release_of_S0105 |
| ECN304932 | RESOLVED | 2022-06-21 | 9 | 9 | 10 | PXB_Q900V_P1A_Release_of_S0130 |
| ECN304934 | RESOLVED | 2022-04-29 | 25 | 1 | 26 | PXB_Q900V_P1A_Release_of_S0135 |
| ECN304935 | RESOLVED | 2022-09-22 | 5 | 2 | 3 | PXB_Q900V_P1A_Release_of_S0145 |
| ECN304937 | RESOLVED | 2022-04-29 | 1 | 11 | 2 | PXB_Q900V_P1A_Release_of_S0150 |
| ECN304940 | RESOLVED | 2022-08-15 | 9 | 17 | 10 | PXB_Q900V_P1A_Release_of_S0170 |
| ECN304944 | RESOLVED | 2022-08-15 | 21 | 13 | 19 | PXB_Q900V_P1A_Release_of_S0190 |
| ECN305533 | RESOLVED | 2022-12-02 | 6 | 4 | 4 | PXB_Q900V_P1A_Release_of_Q900VS0080 |
| ECN305593 | RESOLVED | 2022-09-30 | 8 | 8 | 8 | PXB_Q900V_P1A_Release_of_S0105 |
| ECN305930 | RESOLVED | 2022-06-28 | 5 | 9 | 5 | Q923 PS Update bi Led road light |
| ECN306072 | RESOLVED | 2022-11-28 | 2 | 12 | 4 | PXB_Q900V_P1A_Release_of_S1513 |
| ECN306073 | RESOLVED | 2022-11-28 | 10 | 38 | 10 | PXB_Q900V_P1A_Release_of_S1540_HVAC_SYS_CHS |
| ECN306091 | RESOLVED | 2022-10-26 | 5 | 17 | 5 | PXB_Q900V_P1A_Release_of_S1525_AIR_INTAKE_SYS |
| ECN306103 | RESOLVED | 2022-10-13 | 36 | 46 | 33 | PXB_Q900V_P1A_Release_of_S1501_ENGINE |
| ECN306105 | RESOLVED | 2022-11-17 | 14 | 65 | 15 | PXB_Q900V_P1A_Release_of_S1530_EXHAUST_SYS |
| ECN306660 | RESOLVED | 2022-06-30 | 15 | 5 | 17 | PXB_Q900V_P1A_PXC_Release_of_S0120_#1 |
| ECN307100 | RESOLVED | 2022-07-04 | 30 | 1 | 30 | PXB_Q900V_P1A_PXC_Release_of_S0140_#1 |
| ECN309109 | RESOLVED | 2023-04-25 | 36 | 39 | 24 | Q900V-First release for 2x feature |
| ECN309206 | RESOLVED | 2022-10-24 | 4 | 1 | 4 | PXB_Q900V_P1A_PXC_Release_of_S0120_#3 |
| ECN309738 | RESOLVED | 2022-12-07 | 5 | 28 | 8 | PXB_Q900V_P1A_Release_of_S0170 |
| ECN310346 | RESOLVED | 2022-11-08 | 1 | 2 | 1 | PXB_Q900V_P1A_Release_of_S0190 |
| ECN312170 | RESOLVED | 2023-01-12 | 6 | 13 | 9 | Q900V - grease sticker releasing for P1A |
| ECN312802 | RESOLVED | 2023-01-27 | 82 | 49 | 46 | PXB_NPI_Q900V_SW: Adding known 2Xs to EBOM |
| ECN314958 | RESOLVED | 2023-10-30 | 1 | 4 | 1 | PXB_Q957V_P1A_release_of_Q957VS0190 |
| ECN315420 | RESOLVED | 2023-10-23 | 55 | 55 | 55 | PXB_Q957V_P1A_release_of_Q957VS0170_VC_groups |
| ECN315601 | RESOLVED | 2023-06-26 | 21 | 6 | 21 | PXB_Q933V_P1A_VS0170_VC release |
| ECN316983 | RESOLVED | 2023-06-13 | 3 | 1 | 3 | PXB_Q900V_P1A_VS0110_corrective release |
| ECN321178 | RESOLVED | 2024-07-05 | 23 | 25 | 32 | PXB_Q900V_P1B_Release_of_S1525_AIR_INTAKE_SYS_FEI |
| ECN321235 | RESOLVED | 2024-06-12 | 41 | 33 | 25 | PXB_Q900V_P1B_Release_of_S0170_FA_BRAKE_LINE_FEI |
| ECN321556 | RESOLVED | 2023-10-23 | 1 | 2 | 1 | PXB_Q957V_P1A_release_of_Q957VS0105_VC_groups |
| ECN322297 | RESOLVED | 2023-12-07 | 8 | 16 | 58 | Q923 SOP S2526 update position of road light on handrail |
| ECN323141 | RESOLVED | 2024-10-18 | 20 | 72 | 24 | PXB_Q900V_P1B_Release_of_S1511_COOLING_COMMON CONTENT |
| ECN323149 | RESOLVED | 2024-08-26 | 30 | 47 | 46 | PXB_Q900V_P1B_Release_of_S1540_HVAC_SYS_CHS_COMMON CONTENT |
| ECN325394 | RESOLVED | 2024-07-04 | 44 | 110 | 46 | PXB_Q900V_VS0170_P1B_release |
| ECN325454 | RESOLVED | 2024-06-26 | 23 | 22 | 27 | PXB_Q900V_S0195_P1B_release |
| ECN325719 | RESOLVED | 2024-06-04 | 16 | 24 | 12 | PXB_Q900V_P1B__VS0101_VCG release |
| ECN325724 | RESOLVED | 2024-05-17 | 3 | 22 | 3 | PXB_Q900V_P1B_VS0135_VCG release |
| ECN325728 | RESOLVED | 2024-05-30 | 6 | 2 | 6 | PXB_Q900V_P1B_VS0145_VCG release |
| ECN325729 | RESOLVED | 2024-06-06 | 27 | 12 | 15 | PXB_Q900V_P1B_VS0150_VCG release |
| ECN326691 | RESOLVED | 2025-01-23 | 1 | 3 | 1 | PXB_Q900V_P1B_VS0105_LLT release |
| ECN326702 | RESOLVED | 2024-06-06 | 3 | 6 | 5 | PXB_Q900V_P1B_VS0190_release |
| ECN327642 | RESOLVED | 2024-08-28 | 2 | 4 | 2 | PXB_Q900V_S0043_P1B_release |
| ECN327644 | RESOLVED | 2024-07-03 | 5 | 3 | 3 | PXB_Q900V_P1B_Release_of_S0080 |
| ECN329509 | RESOLVED | 2024-06-26 | 6 | 21 | 9 | PXB_Q900V_P1B_VS0521_LLT |
| ECN330534 | RESOLVED | 2024-07-04 | 2 | 2 | 1 | PXB_Q900V_P1B_VS0150_LLT release |
| ECN331062 | RESOLVED | 2024-12-19 | 12 | 2 | 13 | PXC_Q900V_P1B_release_S0145_#3 |
| ECN334885 | RESOLVED | 2024-12-05 | 8 | 3 | 6 | PXB_Q900V_PS_service_brake_accumulator_change |
| ECN335126 | RESOLVED | 2024-12-12 | 4 | 8 | 7 | PXB_CPM XT60 Transmission oil replacement |
| ECN338303 | RESOLVED | 2025-03-20 | 2 | 2 | 2 | PXA_Q900V_P1B_Release_of_VS0080_WAX |
| ECN338405 | RESOLVED | 2025-10-15 | 6 | 4 | 7 | PXB_Q957V_P1B_release_of_Q957VS0080_VC_groups |
| ECN338408 | RESOLVED | 2025-09-29 | 50 | 4 | 64 | PXB_Q957V_P1B_release_of_Q957VS0101_VC_groups |
| ECN338458 | RESOLVED | 2025-07-07 | 47 | 1 | 56 | PXB_Q957V_P1B_release_of_Q957VS0150_VC_groups |
| ECN338459 | RESOLVED | 2025-10-01 | 28 | 21 | 43 | PXB_Q957V_P1B_release_of_Q957VS0170_VC_groups |
| ECN338551 | RESOLVED | 2025-10-16 | 33 | 34 | 78 | PXB_Q957V_P1B_release_of_Q957VS3025_VC_groups |
| ECN339372 | RESOLVED | 2025-04-23 | 1 | 2 | 2 | PXB_Q900V Bosch RC18-12 ECU not updated to VC-group |
| ECN339649 | RESOLVED | 2025-06-05 | 1 | 1 | 1 | Q9-10 - SW PARAMETERS UPDATES - 2X PTO_STATYMOD_OFF_W_SEAT_OCCPT |
| ECN339901 | RESOLVED | 2025-05-20 | 1 | 2 | 1 | PXB_CPM: Transmission oil volume incorrectly in the structure |
| ECN340300 | RESOLVED | 2025-09-09 | 7 | 1 | 18 | PXB_Q957V_P1B_release_of_Q957VS0145_VC_groups |
| ECN340311 | RESOLVED | 2025-07-01 | 10 | 12 | 20 | PXB_Q900V_Pre-Series_Release_of_S0521_FRONT_AXLE |
| ECN341059 | RESOLVED | 2025-09-02 | 1 | 2 | 1 | PXB_Q957V_P1B_release_of_Q957VS0130_VC_groups |
| ECN343274 | RESOLVED | 2025-10-14 | 19 | 40 | 38 | PXB_Q957V_P1B_release_of_Q957VS0170_VC_VD_groups |
| ECN344070 | RESOLVED | 2025-12-01 | 2 | 4 | 3 | PXB_Q900V_Pre-Series_Release_of_S0105_INPUT_SHAFT |
| ECN344075 | RESOLVED | 2025-11-25 | 4 | 1 | 3 | PXB_Q900V_Pre-Series_Release_of_S0145_FINAL_DRIVE |
| ECN344281 | RESOLVED | 2025-11-27 | 1 | 3 | 10 | PXB_Q900V_Pre-Series_Release_of_S0170_BRAKE_SYS_CHS |
| ECN344641 | RESOLVED | 2025-11-24 | 1 | 1 | 1 | PXB_Q900V_S0521_PRE_SERIES_UPDATE |
| ECN345256 | RESOLVED | 2025-11-18 | 12 | 2 | 24 | PXB_Q900V_S0145_PRE_SERIES_LOCK_RING_UPDATE |
| ECN345262 | RESOLVED | 2025-11-25 | 2 | 2 | 4 | PXB_Q900V_SENSOR_CABLE_ATTACHMENT |
| ECN345274 | RESOLVED | 2026-01-16 | 4 | 8 | 34 | CPM PCO: Cover change to injection plastic |
| ECN345348 | RESOLVED | 2025-11-28 | 9 | 2 | 20 | PXB_Q900V_S0170_LOP_CORRECTIONS |
| ECN345653 | RESOLVED | 2025-11-24 | 2 | 1 | 1 | PXB_Q900V_Pre-Series_Release_of_S0080 _Content |
| ECN346156 | RESOLVED | 2025-12-18 | 9 | 18 | 16 | PXB_Q900V_S0170_LOP_CORRECTIONS |
| ECN346338 | RESOLVED | 2026-01-22 | 16 | 29 | 25 | PXB_Q900V_V4001001_V0170001_V3010001 |
| ECN346863 | RESOLVED | 2026-03-13 | 12 | 13 | 10 | PXB_Q900V_S0080_CONTENT_CHANGE |
| ECN347498 | RESOLVED | 2026-02-04 | 23 | 27 | 19 | PXB_NPI_Q900V_SW: Q900V SW 2X fixes |
| ECN349781 | RESOLVED | 2026-03-30 | 29 | 5 | 5 | PXB_NPI_Q900V_SW: 2X fixes |
| ECN350019 | RESOLVED | 2026-04-10 | 20 | 10 | 9 | PXB_NPI_Q900V_SW: More 2X fixes |
| ECN350349 | RESOLVED | 2026-05-11 | 1 | 1 | 1 | PXB_Q900V_PS Q900VV0145150_CORRECTION |
| ECN350793 | RESOLVED | 2026-08-28 | 40 | 90 | 49 | EFG_JB_MHP - Q900V RP11 |
| ECN351385 | RESOLVED | 2026-04-30 | 1 | 2 | 2 | PXB_Q900V_PRE_SERIES_S0170_CVT_PARKING_BRAKE_CYLINDER_IMPROVEMENT |

## 2. Only in `EBOM Q900V` (132)

| ECN | State | Resolution date | EBOM Q900V rows | MBOM rows | EBOM 130% rows | Name |
|---|---|---|---:|---:|---:|---|
| ECN126772 | RELEASED |  | 1 | 0 | 1 | Release G3BC3 Project01 Project |
| ECN145929 | RELEASED |  | 32 | 0 | 20 | Release Plant B parts (and units) for G8 Project |
| ECN157037 | RELEASED |  | 6 | 0 | 14 | Release New GB and RA sets for SA |
| ECN207222 | RESOLVED | 2018-01-10 | 2 | 0 | 2 | Cross_Serial_EG_Bearring |
| ECN210095 | RESOLVED | 2017-07-31 | 2 | 0 | 9 | PXB_OPUS Transmission corrections |
| ECN258758 | RESOLVED | 2019-01-25 | 1 | 0 | 4 | PXB_Q947V_P1B_Release_of_Q947VS3525 |
| ECN268603 | RESOLVED | 2019-09-13 | 11 | 0 | 11 | PXB_Q932V_P1A_Release_S0150 |
| ECN269212 | RESOLVED | 2019-09-16 | 1 | 0 | 1 | PXB_Q932V_P1A_Release_S0195 |
| ECN273510 | RESOLVED | 2019-12-02 | 2 | 0 | 2 | PXB_Q933V_S0110_TRANSMISSION |
| ECN276217 | RESOLVED | 2020-11-09 | 71 | 0 | 70 | PXB_Q933V_S0110_TRANSMISSION |
| ECN277835 | RESOLVED | 2020-04-22 | 2 | 0 | 2 | PXB_Q932V_PS Release S0190 content |
| ECN282353 | RESOLVED | 2020-08-13 | 1 | 0 | 3 | PXB_ENG_SYSTEMS_HARDWARE_REPLACEMENT_6 |
| ECN284870 | RESOLVED | 2020-11-04 | 7 | 0 | 7 | PXB_Q900V_P0_S0195_TA_HYD_RELEASE |
| ECN284871 | RESOLVED | 2020-10-28 | 3 | 0 | 3 | PXB_Q900V_P0_S0130_FWD_RELEASE |
| ECN289030 | RESOLVED | 2021-01-21 | 1 | 0 | 1 | Q940P_P1B_EG_Bauteile_S0140_S0145 |
| ECN290546 | RESOLVED | 2021-02-19 | 5 | 0 | 6 | Q9-39 CVT - Serial release for P1A - PXC - S0120 |
| ECN290623 | RESOLVED | 2021-03-19 | 1 | 0 | 1 | T701_Serial_EG_Standardization_brake_actuator |
| ECN292364 | RESOLVED | 2021-06-09 | 1 | 0 | 3 | Q9-39 CVT - Serial release for P1A - PXC - S0120_#2 |
| ECN293799 | RESOLVED | 2021-10-28 | 3 | 0 | 4 | Cross-Serial_EG_axle bevel gear G737150010010 |
| ECN298203 | RESOLVED | 2021-10-01 | 1 | 0 | 2 | PXB_CPM_transm e-eq: QKV8561080 Speed Sensor Replacement |
| ECN299269 | RESOLVED | 2021-12-21 | 4 | 0 | 8 | Cross-Serial_EG_MIP1438_new cold forgged raw part |
| ECN302460 | RESOLVED | 2022-02-22 | 4 | 0 | 4 | PXB_CPM MAKE VD GROUPS FOR APAJA PROJECT |
| ECN302842 | RESOLVED | 2022-04-14 | 3 | 0 | 3 | PXB_APAJA_CP_B-sample release |
| ECN304929 | RESOLVED | 2022-05-02 | 160 | 0 | 161 | PXB_Q900V_P1A_Release_of_S0110 |
| ECN304930 | RESOLVED | 2022-09-28 | 4 | 0 | 4 | PXB_Q900V_P1A_Release_of_S0110 |
| ECN304931 | RESOLVED | 2022-04-29 | 24 | 0 | 24 | PXB_Q900V_P1A_Release_of_S0130 |
| ECN304936 | RESOLVED | 2022-04-29 | 4 | 0 | 4 | PXB_Q900V_P1A_Release_of_S0145 |
| ECN304938 | RESOLVED | 2022-08-15 | 15 | 0 | 17 | PXB_Q900V_P1A_Release_of_S0150 |
| ECN304946 | RESOLVED | 2022-05-25 | 1 | 0 | 1 | PXB_Q900V_P1A_Release_of_S0195 |
| ECN304947 | RESOLVED | 2022-06-23 | 26 | 0 | 29 | PXB_Q900V_P1A_Release_of_S0195 |
| ECN305025 | RESOLVED | 2022-04-06 | 3 | 0 | 5 | PXB_CPM SA-PTO Big Clutch Disc serial release |
| ECN305229 | RESOLVED | 2022-06-06 | 9 | 0 | 10 | PXB_CPM_transm pto: SA-PTO: New Item release |
| ECN305305 | RESOLVED | 2022-06-23 | 8 | 0 | 8 | PXB_CPM:T174_5PS_1B8 _Project 10 PCO workshop |
| ECN305527 | RESOLVED | 2023-01-20 | 8 | 0 | 4 | PXB_Q900V_P1A_Release_of_Q900VS0041 |
| ECN305528 | RESOLVED | 2022-10-13 | 2 | 0 | 1 | PXB_Q900V_P1A_Release_of_Q900VS0042 |
| ECN305877 | RESOLVED | 2022-06-23 | 29 | 0 | 29 | PXB_Q900V_P1A_Release_of_S0110 |
| ECN306661 | RESOLVED | 2022-07-29 | 3 | 0 | 5 | PXB_Q900V_P1A_PXC_Release_of_S0120_#2 |
| ECN306991 | RESOLVED | 2022-08-23 | 5 | 0 | 9 | PXB_CPM_transm pto: Q932VV0150051 SA-PTO proto based problems correction |
| ECN307095 | RESOLVED | 2022-07-04 | 32 | 0 | 32 | PXB_Q900V_P1A_PXC_Release_of_S0130_#1 |
| ECN307096 | RESOLVED | 2022-07-26 | 3 | 0 | 4 | PXB_Q900V_P1A_PXC_Release_of_S0130_#2 |
| ECN307097 | RESOLVED | 2022-06-30 | 16 | 0 | 16 | PXB_Q900V_P1A_PXC_Release_of_S0135_#1 |
| ECN307098 | RESOLVED | 2022-07-08 | 7 | 0 | 9 | PXB_Q900V_P1A_PXC_Release_of_S0135_#2 |
| ECN307102 | RESOLVED | 2022-07-04 | 8 | 0 | 8 | PXB_Q900V_P1A_PXC_Release_of_S0145_#1 |
| ECN307105 | RESOLVED | 2022-07-07 | 12 | 0 | 24 | PXB_Q900V_P1A_PXC_Release_of_S0145_#4 |
| ECN307106 | RESOLVED | 2022-07-05 | 41 | 0 | 42 | PXB_Q900V_P1A_PXC_Release_of_S0170_#1 |
| ECN307107 | RESOLVED | 2022-07-04 | 5 | 0 | 6 | PXB_Q900V_P1A_PXC_Release_of_S0170_#2 |
| ECN307118 | RESOLVED | 2022-10-25 | 3 | 0 | 5 | PXB_Q900V_P1A_PXC_Release_of_S0170_#3 |
| ECN309207 | RESOLVED | 2022-09-21 | 9 | 0 | 9 | PXB_Q900V_P1A_PXC_Release_of_S0130_#3 |
| ECN309208 | RESOLVED | 2022-10-25 | 4 | 0 | 6 | PXB_Q900V_P1A_PXC_Release_of_S0145_#5 |
| ECN311181 | RESOLVED | 2022-12-09 | 3 | 0 | 4 | PXB_CPM_transm pto: QKX3488290 SA-PTO Clutch Housing pulse per cycle update. |
| ECN312299 | RESOLVED | 2023-01-09 | 4 | 0 | 5 | Load_EG_Kolben_kpl |
| ECN313574 | RESOLVED | 2023-03-15 | 4 | 0 | 9 | Cross-Serial_EFS_new ASG hall sensor 40mm to BOM |
| ECN313968 | RESOLVED | 2023-03-03 | 2 | 0 | 2 | Q940P_Serial_EG_S0140_#4 |
| ECN314955 | RESOLVED | 2023-09-21 | 2 | 0 | 2 | PXB_Q957V_P1A_release_of_Q957VS0145 |
| ECN315603 | RESOLVED | 2023-05-15 | 3 | 0 | 3 | PXB_Q933V_P1A_VS0130_VC release |
| ECN315604 | RESOLVED | 2023-05-22 | 4 | 0 | 3 | PXB_Q933V_P1A_VS0110_VC release |
| ECN315808 | RESOLVED | 2023-04-25 | 2 | 0 | 6 | PXB_CPM Documentation updated and part numbers changed during resourcing |
| ECN317061 | RESOLVED | 2023-06-21 | 1 | 0 | 1 | PXB_CPM SA-PTO needs changes in current structures |
| ECN317079 | RESOLVED | 2023-09-12 | 7 | 0 | 9 | PXB_Q957V_P1A_release_of_Q957VS0110 |
| ECN318871 | RESOLVED | 2023-10-05 | 7 | 0 | 7 | PXC_Q957V_P1A_RELEASE_EG_S0140_#1 |
| ECN318876 | RESOLVED | 2023-09-29 | 1 | 0 | 1 | PXC_Q957V_P1A_RELEASE_EG_S0170_#1 |
| ECN319312 | RESOLVED | 2023-08-21 | 1 | 0 | 1 | Load_EG_Q957V_P1A_parts_for_ordering_process |
| ECN321856 | RESOLVED | 2024-01-23 | 4 | 0 | 8 | PXB_CPM: Tolerance change |
| ECN323974 | RESOLVED | 2024-01-11 | 1 | 0 | 1 | PXB_Q933V_P1A_Correction release |
| ECN325363 | RESOLVED | 2024-04-11 | 3 | 0 | 15 | PXB_Q900V_P1B_VS0110_LLT release |
| ECN325365 | RESOLVED | 2024-05-02 | 2 | 0 | 4 | PXB_Q900V_P1B_VS0140_LLT release |
| ECN325366 | RESOLVED | 2024-04-05 | 2 | 0 | 6 | PXB_Q900V_P1B_VS0145_LLT release |
| ECN325528 | RESOLVED | 2024-04-11 | 8 | 0 | 25 | PXB_Q900V_P1B_VS0110_LLT release |
| ECN325721 | RESOLVED | 2024-06-12 | 27 | 0 | 32 | PXB_Q900V_P1B_VS0110_VCG release |
| ECN327638 | RESOLVED | 2024-08-06 | 1 | 0 | 1 | PXB_Q900V_S0042_P1B_release |
| ECN328703 | RESOLVED | 2024-05-14 | 3 | 0 | 2 | PXB_CPM QKV838879B casting missing |
| ECN331061 | RESOLVED | 2024-09-25 | 20 | 0 | 20 | PXC_Q900V_P1B_release_S0145_#2 |
| ECN332077 | RESOLVED | 2024-09-12 | 2 | 0 | 7 | Load_EG_Bremsenteile |
| ECN333413 | RESOLVED | 2024-11-19 | 1 | 0 | 1 | PXB_Q900V_P1B__VS0101_LLT release |
| ECN333415 | RESOLVED | 2024-11-22 | 5 | 0 | 5 | PXB_Q900V_P1B_VS0145_LLT release |
| ECN333540 | RESOLVED | 2024-11-07 | 3 | 0 | 3 | PXB_Q900V_P1B_VS0145_LLT release |
| ECN333702 | RESOLVED | 2024-12-17 | 2 | 0 | 3 | Cross-Serial_EG_groove ball bearing 6211NR update |
| ECN334489 | RESOLVED | 2025-04-10 | 1 | 0 | 2 | Cross-Serial_EG_MIP1452 oil baffle drw update |
| ECN334909 | RESOLVED | 2024-12-03 | 1 | 0 | 1 | PXB_Q900V_P1B_VS0110_LLT release |
| ECN335119 | RESOLVED | 2024-12-18 | 3 | 0 | 2 | PXB_Q900V_P1B_VS0145_LLT release |
| ECN335120 | RESOLVED | 2024-12-03 | 3 | 0 | 6 | PXB_Q900V_P1B_VS0145_LLT release |
| ECN336891 | RESOLVED | 2025-02-13 | 1 | 0 | 1 | PXB_CPM Q932VV0190502  structure not correct from ECN333563 |
| ECN338008 | RESOLVED | 2025-06-27 | 3 | 0 | 16 | PXB_Q900V_Industrialization_changes_PS based parts release |
| ECN338450 | RESOLVED | 2025-07-01 | 5 | 0 | 24 | PXB_Q957V_P1B_release_of_Q957VS0110_VC_groups |
| ECN338453 | RESOLVED | 2025-07-01 | 6 | 0 | 9 | PXB_Q957V_P1B_release_of_Q957VS0135_VC_groups |
| ECN338454 | RESOLVED | 2025-08-11 | 25 | 0 | 31 | PXB_Q957V_P1B_release_of_Q957VS0140_VC_groups |
| ECN338462 | RESOLVED | 2025-07-09 | 4 | 0 | 4 | PXB_Q957V_P1B_release_of_Q957VS0190_VC_groups |
| ECN338463 | RESOLVED | 2025-07-02 | 17 | 0 | 70 | PXB_Q957V_P1B_release_of_Q957VS0195_VC_groups |
| ECN339284 | RESOLVED | 2025-07-01 | 9 | 0 | 15 | PXB_Q957V_P1B_release_of_Q957VS0145_VC-groups |
| ECN340298 | RESOLVED | 2025-10-03 | 13 | 0 | 14 | PXB_Q957V_P1B_release_of_Q957VS0110_VC_groups |
| ECN340301 | RESOLVED | 2025-08-26 | 1 | 0 | 1 | PXB_Q957V_P1B_release_of_Q957VS0150_VC_groups |
| ECN340575 | RESOLVED | 2025-06-27 | 2 | 0 | 25 | PXB_Q957V_P1B_release_of_Chassis_Component |
| ECN340677 | RESOLVED | 2025-09-18 | 3 | 0 | 7 | PXB_CPM Shim rings do not fit to bended one |
| ECN341675 | RESOLVED | 2025-12-04 | 3 | 0 | 6 | Q900V_Pre-Series_PXC_D_S0130_#1 |
| ECN341732 | RESOLVED | 2025-11-27 | 3 | 0 | 8 | Q900V_Pre-Series_PXC_VC_S0135_#1 |
| ECN341734 | RESOLVED | 2025-12-11 | 2 | 0 | 9 | Q900V_Pre-Series_PXC_VC_S0145_#1 |
| ECN341749 | RESOLVED | 2025-12-09 | 2 | 0 | 11 | Q900V_Pre-Series_PXC_D_S0170_#1 |
| ECN343327 | RESOLVED | 2025-10-31 | 1 | 0 | 3 | PXB_CPM: QKX2850390 material and quantity wrong |
| ECN343594 | RESOLVED | 2025-10-08 | 13 | 0 | 13 | PXC_Q957V_P1B_RELEASE_EG_VC_S0170_#1 |
| ECN344073 | RESOLVED | 2025-11-17 | 1 | 0 | 1 | PXB_Q900V_Pre-Series_Release_of_S0130_FRONT_WHEEL_DRIVE |
| ECN344465 | RESOLVED | 2025-10-30 | 1 | 0 | 8 | Q900V_Pre-Series_PXC_S0120_#1 |
| ECN344555 | RESOLVED | 2025-11-24 | 1 | 0 | 1 | PXB_Q900V_DIFFERENTIAL_LOCKING_PLATE_CHANGE |
| ECN344625 | RESOLVED | 2025-11-24 | 2 | 0 | 4 | PXB_Q900V_S0135_PUMP_SUPPORT_MACHINING |
| ECN344626 | RESOLVED | 2025-11-25 | 47 | 0 | 54 | PXB_Q900V_S0150_PTO BEARING_COVER_DRAWING_UPDATE |
| ECN345075 | RESOLVED | 2025-12-01 | 5 | 0 | 11 | PXB_Q900V_S0101_PRE_SERIES_PS_COMPONENTS_LOP_CORRECTIONS |
| ECN345076 | RESOLVED | 2025-11-24 | 5 | 0 | 7 | PXB_Q900V_S0110_PRE_SERIES_RELEASE_CORRECTIONS |
| ECN345559 | RESOLVED | 2026-02-04 | 4 | 0 | 4 | PXB_Q900V_S0040_PreSeries Release |
| ECN346279 | RESOLVED | 2026-02-02 | 117 | 0 | 117 | PXB_Q933V_Pre-Series_Release_of_Q933VV0195001 |
| ECN346576 | RESOLVED | 2025-12-05 | 2 | 0 | 4 | PXB_Q900V_S0190_Q900VV0190010 |
| ECN347729 | RESOLVED | 2026-01-19 | 7 | 0 | 7 | Q933V - New Series 5PS SA |
| ECN348151 | RESOLVED | 2026-01-26 | 2 | 0 | 2 | Q933V - New Series 5PS SA |
| ECN348371 | RESOLVED | 2026-04-20 | 1 | 0 | 3 | PXB_CPM QKX0694700 Planetary carrier balancing update |
| ECN348495 | RESOLVED | 2026-02-23 | 3 | 0 | 1 | PXB_Q900V_PS_CORRECTION_RELEASE_FOR_S0145_FINAL_DRIVE |
| ECN348574 | RESOLVED | 2026-05-27 | 4 | 0 | 5 | PXB_Q900V_PS_CORRECTION_RELEASE_FOR_S0101_TA_HOUSING |
| ECN348575 | RESOLVED | 2026-03-17 | 2 | 0 | 10 | PXB_Q900V_PS_CORRECTION_RELEASE_FOR_S0150_PTO_REAR |
| ECN348576 | RESOLVED | 2026-03-17 | 2 | 0 | 3 | PXB_Q900V_PS_CORRECTION_RELEASE_FOR_S0195_TA_HYD |
| ECN348975 | RESOLVED | 2026-02-25 | 1 | 0 | 1 | PXB_Q900V_S0190_THERMAL_SENSOR_REPLACEMENT |
| ECN349020 | RESOLVED | 2026-05-27 | 37 | 0 | 25 | PXB_Q900V_Q957V_FROM_CATIA_TO_CREO_CONVERSION_RELEASE |
| ECN349473 | RESOLVED | 2026-04-15 | 2 | 0 | 6 | PXB_Q900V_S0101_PS_WET_SPACER_CONTROL_MACHINING |
| ECN349776 | RESOLVED | 2026-05-06 | 16 | 0 | 35 | PXB_Q900V_PS_PTO_SHAFT_AND_ENGAGING_GEARS_UPDATES |
| ECN349819 | RESOLVED | 2026-06-08 | 7 | 0 | 142 | PXB_Q900V_Q957V_FROM_CATIA_TO_CREO_CONVERSION_RELEASE |
| ECN350093 | RESOLVED | 2026-05-27 | 1 | 0 | 23 | PXB_Q900V_PRE_SERIES_TR_CHANGES_TO_DRAWINGS |
| ECN350348 | RESOLVED | 2026-04-16 | 1 | 0 | 6 | PXB_Q900V_corrections_for_S0195_TA_HYD |
| ECN350404 | RESOLVED | 2026-06-22 | 4 | 0 | 5 | PXB_Q900V_PS_GEARBOX_HOUSING_MACHINING_CHANGES |
| ECN350565 | RESOLVED | 2026-05-27 | 10 | 0 | 13 | PXB_Q900V_INTERMEDIATE_TUBE_MACHINING_AND_SEALING_CORRECTION |
| ECN351079 | RESOLVED | 2026-06-02 | 4 | 0 | 14 | PXB_Q900V_PRE_SERIES_S0195_CORRECTIONS |
| ECN352167 | RESOLVED | 2026-06-09 | 1 | 0 | 14 | PXB_CPM: Q932V/Q900V GSPTO sleeve needs to be moved to another vcgroup |
| ECN352311 | RESOLVED | 2026-06-29 | 3 | 0 | 6 | PXB_Q900V_SERIES_RELEASE_S0110_LOP_CORRECTIONS |
| ECN352313 | RESOLVED | 2026-06-18 | 3 | 0 | 5 | PXB_Q900V_SERIES_RELEASE_S0195_LOP_CORRECTIONS_MODIFY_SUCTION_PIPE |
| ECN352868 | RESOLVED | 2026-06-23 | 4 | 0 | 7 | PXB_Q900V_PRE_SERIES_RELEASE_S0101_PS_CORRECTIONS |
| ECN353084 | RESOLVED | 2026-08-10 | 1 | 0 | 2 | QKX4731390 Drawing changes |
| ECN355062 | RESOLVED | 2026-09-04 | 12 | 0 | 33 | PXB_Q900V_SOP_S0521_FRONT_AXLE_UPPDATE |

## 3. Only in the MBOM (4)

| ECN | State | Resolution date | EBOM Q900V rows | MBOM rows | EBOM 130% rows | Name |
|---|---|---|---:|---:|---:|---|
| ECN262354 | RESOLVED | 2019-03-15 | 0 | 10 | 27 | Q938V PS S2538 cab release New seal for rear window and gas spring for door |
| ECN299341 | RESOLVED | 2022-02-28 | 0 | 4 | 2 | Q923 PS S2526 release the  sub for PS |
| ECN302839 | RESOLVED | 2022-03-01 | 0 | 1 | 1 | PXB_APAJA_R30_B-sample release |
| ECN304746 | RESOLVED | 2022-04-26 | 0 | 3 | 1 | PXB_Q900V_P1A_Release_of_S0521 |

## 4. Only in EBOM 130% (329 ECNs: 325 on document rows only, 4 with part rows)

| ECN | State | Resolution date | EBOM Q900V rows | MBOM rows | EBOM 130% rows | Name |
|---|---|---|---:|---:|---:|---|
| ECN107240 | RESOLVED | 2014-06-02 | 0 | 0 | 2 | Décalcomanie de sécurité Z16 |
| ECN117440 | RESOLVED | 2016-08-03 | 0 | 0 | 26 | PXE ISSUE: No space to paste the decal on the trimming G3 & miss decal for passanger seat |
| ECN122869 | RESOLVED | 2016-01-22 | 0 | 0 | 3 | Create and release group unit(s) for the group 334 for plant_b G8 project |
| ECN130705 | RESOLVED | 2015-11-04 | 0 | 0 | 6 | Revision Synchronization - PXA Legacy |
| ECN132162 | RESOLVED | 2015-11-09 | 0 | 0 | 10 | Revision Synchronization - PXA Legacy |
| ECN132186 | RELEASED |  | 0 | 0 | 1 | Release G3BC4 Project01 Project in Mogi |
| ECN134107 | RESOLVED | 2016-01-04 | 0 | 0 | 3 | G7_[05700] Modification to be noise compliant (75Db) |
| ECN134229 | RESOLVED | 2016-01-13 | 0 | 0 | 1 | GIMA_RELEASE_E00597_Delete alternative bearing |
| ECN136809 | RESOLVED | 2016-01-05 | 0 | 0 | 4 | Revision Synchronization - PXA Legacy |
| ECN142964 | RESOLVED | 2016-02-09 | 0 | 0 | 2 | Revision Synchronization - PXA Legacy |
| ECN145266 | RESOLVED | 2016-11-03 | 0 | 0 | 6 | [05820] G7C3/C4 pneumatic trailer brake release for MR |
| ECN145418 | RESOLVED | 2016-03-10 | 0 | 0 | 2 | GIMA_RELEASE_E00834_Add_marking_aera_for_new_ressourcing_to_TAFE |
| ECN145419 | RESOLVED | 2016-04-04 | 0 | 0 | 1 | GIMA_RELEASE_E00778_Remove plug in units gr160 following OEM request |
| ECN145547 | RELEASED |  | 0 | 0 | 2 | ECN for Ariane ESI history |
| ECN147077 | RELEASED |  | 0 | 0 | 10 | Release Plant B parts (and units) for G8 Project |
| ECN151318 | RESOLVED | 2016-07-13 | 0 | 0 | 10 | MR 700 Dyna - need simplification for 700 w/o hyd trailer brake |
| ECN151597 | RESOLVED | 2016-08-03 | 0 | 0 | 1 | Q938V - Release subgroup S0130 - front wheel drive |
| ECN153703 | RELEASED |  | 0 | 0 | 9 | Remove 05149211 from TEA1413 / 1415 & add it on TEA 1412 /1414 |
| ECN154157 | RELEASED |  | 0 | 0 | 7 | Load ECN PXB Project01 for SAP Cornerstone |
| ECN157358 | RESOLVED | 2016-10-26 | 0 | 0 | 25 | JB part/components to be manufactured in Hohenmoelsen |
| ECN159098 | RESOLVED | 2016-09-20 | 0 | 0 | 4 | Standardization copper seal by BS washer |
| ECN159118 | RESOLVED | 2016-09-05 | 0 | 0 | 3 | Q938V - Release JB front lift |
| ECN159797 | RESOLVED | 2017-01-25 | 0 | 0 | 10 | Q9-19 - Grease & opening instruction stickers to modify |
| ECN160378 | RESOLVED | 2016-07-19 | 0 | 0 | 8 | CornerStone ENG/EAT/COOLING |
| ECN163077 | RESOLVED | 2017-03-21 | 0 | 0 | 2 | MR - New sticker to release for MR application |
| ECN165060 | RESOLVED | 2016-08-05 | 0 | 0 | 6 | Correction to Washer Size Information |
| ECN175438 | RESOLVED | 2017-01-16 | 0 | 0 | 6 | Q9-19 Clamp QKW933642A and update qkw366356A create the unit |
| ECN179400 | RESOLVED | 2016-12-01 | 0 | 0 | 1 | Q9-19 LUP323 LUP324, LUP 342, BUG 677 |
| ECN179762 | RESOLVED | 2017-01-25 | 0 | 0 | 10 | Services - Release MT E01103 - Hydraulic distributor |
| ECN187582 | RESOLVED | 2017-03-28 | 0 | 0 | 2 | Cross-Serial_EFS_deliver drawings altern. material |
| ECN190070 | RESOLVED | 2017-02-14 | 0 | 0 | 1 | Q967T_P1A_EG_S0110 |
| ECN191720 | RESOLVED | 2017-03-06 | 0 | 0 | 19 | allg_Prototyp_Materialstammanlage+K-Liste Umbau |
| ECN193101 | RESOLVED | 2017-03-08 | 0 | 0 | 6 | LOAD_EFH_PROTO_Status 20_Q921P |
| ECN204000 | RESOLVED | 2017-11-24 | 0 | 0 | 2 | LRT - Review the dimension on LRT spacer |
| ECN204260 | RESOLVED | 2017-07-07 | 0 | 0 | 37 | T763_P1A_ES release Q938PS2564 |
| ECN212682 | RESOLVED | 2017-10-23 | 0 | 0 | 1 | Cross-Serial_EG deliver drawings |
| ECN216962 | RESOLVED | 2017-07-13 | 0 | 0 | 52 | Q921P_P1B_EFK_K-Liste_Haube_BHM |
| ECN217054 | RESOLVED | 2017-09-04 | 0 | 0 | 13 | Q921P_P1B_EFH delivery_1520/30/40 |
| ECN217525 | RESOLVED | 2017-07-27 | 0 | 0 | 6 | GIMA_RELEASE_E01171_E01505_First Release Gr 169 for Q9-06 |
| ECN220002 | RESOLVED | 2017-07-27 | 0 | 0 | 7 | Release of 24 Degree Cone fittings for Plant C. |
| ECN220926 | RESOLVED | 2017-11-13 | 0 | 0 | 25 | Q938V - Serial release of cardan shaft S0550 |
| ECN225426 | RESOLVED | 2017-09-26 | 0 | 0 | 5 | Update 24 Degree Swivel Fitting CAD Assembly Port Position |
| ECN225507 | RESOLVED | 2017-09-26 | 0 | 0 | 11 | Correct o-ring dimension on some fittings |
| ECN226122 | RESOLVED | 2017-09-07 | 0 | 0 | 11 | Release of Hyd Fitting QKW0800110 |
| ECN227842 | RESOLVED | 2017-09-08 | 0 | 0 | 2 | allg_Prototyp_Materalstammanlage_EG |
| ECN228844 | RESOLVED | 2017-10-03 | 0 | 0 | 1 | PXB_CPM_transm:QKV638576B headless screws for PTO and Final Drive 650 fastening |
| ECN229602 | RESOLVED | 2017-09-15 | 0 | 0 | 1 | Release of Screw for Plant C Designer |
| ECN230942 | RESOLVED | 2017-10-03 | 0 | 0 | 2 | PXB_CPM_hyd transm:31508108 plug of solenoid valve |
| ECN232182 | RESOLVED | 2017-09-26 | 0 | 0 | 1 | Cross-Prototyp_EFK_KW39_26_09_17 |
| ECN234062 | RESOLVED | 2017-11-24 | 0 | 0 | 9 | PXB_Q947V_P1A_Release_of_Q947VS3025 |
| ECN234065 | RESOLVED | 2017-12-13 | 0 | 0 | 6 | PXB_Q947V_P1A_Release_of_Q947VS3501 |
| ECN235802 | RESOLVED | 2017-10-13 | 0 | 0 | 13 | Release of straight stud fitting for tractor projects |
| ECN236605 | RESOLVED | 2018-04-20 | 0 | 0 | 1 | M2AA0_pre-series_EG_release |
| ECN238702 | RESOLVED | 2017-10-24 | 0 | 0 | 7 | Release of 24 degree cone hydraulic fittings for tractor designers |
| ECN239382 | RESOLVED | 2017-10-25 | 0 | 0 | 11 | Release of hydraulic fittings for tractor projects |
| ECN239662 | RESOLVED | 2017-12-04 | 0 | 0 | 7 | PXB_Q947V_P1A_Release_of_Q947VS2001 |
| ECN240922 | RESOLVED | 2018-01-02 | 0 | 0 | 2 | MR: QKW7575980 Emergency Exit decal + hood lock reinforcement + sitcker hood opening direction |
| ECN241520 | RESOLVED | 2017-11-01 | 0 | 0 | 7 | Plant B designer needs fittings for Q947V project. |
| ECN241590 | RESOLVED | 2017-11-02 | 0 | 0 | 12 | Plant B designer needs fittings for Q947V project. |
| ECN241602 | RESOLVED | 2018-03-07 | 0 | 0 | 1 | M2AA0_pre-series_EG_release |
| ECN241661 | RESOLVED | 2017-11-06 | 0 | 0 | 2 | Release of fitting for Plant C designer |
| ECN241668 | RESOLVED | 2017-11-07 | 0 | 0 | 19 | Add reference vendor information to parts |
| ECN241690 | RESOLVED | 2018-05-22 | 0 | 0 | 2 | M2AA0_pre-series_EG_release |
| ECN241695 | RESOLVED | 2017-12-19 | 0 | 0 | 2 | MR: TD modification for Z19B external mirror + decalco freinage remorque |
| ECN241715 | RESOLVED | 2018-04-16 | 0 | 0 | 5 | T954_Serial_EFS_MIP1297 Suction Filter |
| ECN241905 | RESOLVED | 2017-11-10 | 0 | 0 | 1 | Release of 24 degree cone tractor hydraulic fitting for Plant B |
| ECN242191 | RESOLVED | 2018-05-15 | 0 | 0 | 3 | Release of hydraulic fitting for PXB designer |
| ECN243253 | RESOLVED | 2017-12-13 | 0 | 0 | 1 | Release of hardware items for Plant A, Q9-38 |
| ECN247704 | RESOLVED | 2018-03-14 | 0 | 0 | 5 | Q938V-P1B-S0170-Package of LUP resolution for Gima |
| ECN248065 | RESOLVED | 2018-02-13 | 0 | 0 | 52 | Release screw for Plant B designer |
| ECN253382 | RESOLVED | 2018-04-25 | 0 | 0 | 1 | Creation and Release of New Global Hardware Items for Plant C |
| ECN253476 | RESOLVED | 2018-05-29 | 0 | 0 | 9 | PLANT_B GR 820 C2C-20VT P1A RELEASE |
| ECN253647 | RESOLVED | 2018-04-25 | 0 | 0 | 2 | Release of new fittings for tractor designer. |
| ECN254245 | RESOLVED | 2018-07-20 | 0 | 0 | 44 | Q921P_Pre-Series_EC_S1010 |
| ECN254685 | RESOLVED | 2018-06-21 | 0 | 0 | 1 | PXB_CPM_hyd transm: 37187300 tube |
| ECN254847 | RESOLVED | 2018-05-22 | 0 | 0 | 1 | Release of screw for Plant B |
| ECN255066 | RESOLVED | 2018-06-18 | 0 | 0 | 1 | PXB_CPM_transm additional:QKW0952340 TRANSMISSION 3PS Brazil |
| ECN255687 | RESOLVED | 2018-06-15 | 0 | 0 | 1 | Creation and Release of New Global Hardware Items for Plant C |
| ECN255998 | RESOLVED | 2018-06-25 | 0 | 0 | 6 | Creation and Release of New Global Hardware Items for Plant C |
| ECN256139 | RESOLVED | 2018-07-11 | 0 | 0 | 21 | Q9-38V_P1B_S2506_HYD.11 - Release prototype parts to allow P1B build to connect brake pedals and booster |
| ECN256196 | RESOLVED | 2018-07-02 | 0 | 0 | 3 | T077_P1A_EZV_Teamschlepper |
| ECN256291 | RESOLVED | 2018-07-04 | 0 | 0 | 2 | Release of hardware items for Plant A, Q9-38 |
| ECN256445 | RESOLVED | 2018-07-11 | 0 | 0 | 40 | Update Vendor Information for Hydraulic Fitting |
| ECN258453 | RESOLVED | 2018-10-02 | 0 | 0 | 28 | PPM 2697 Mid mounted Several major issues on the midmounted block - Hydraulic std |
| ECN258715 | RESOLVED | 2019-01-07 | 0 | 0 | 5 | PXB_Q947V_P1B_Release_of_Q947VS3005 |
| ECN259024 | RESOLVED | 2019-03-04 | 0 | 0 | 4 | Q938V - Release MT E02197 / E02081 / E02217 / E02230 / E02198 / E02214 |
| ECN259056 | RESOLVED | 2018-10-11 | 0 | 0 | 1 | PXB_CPM_transm gearbox: QKW6898810 Angle sensor shaft |
| ECN259720 | RESOLVED | 2018-11-14 | 0 | 0 | 33 | Move existing projection weld nuts to OEM_Global_Standard_Components library |
| ECN259783 | RESOLVED | 2018-11-07 | 0 | 0 | 1 | PXB_CPM_transm gearbox: QKV3577460 synchro 90mm |
| ECN260094 | RESOLVED | 2018-11-05 | 0 | 0 | 52 | Release of New OEM_Global_Standard_Components for Plant A designer |
| ECN261004 | RESOLVED | 2018-11-28 | 0 | 0 | 11 | Release of New OEM_Global_Standard_Components for Plant B designers |
| ECN261265 | RESOLVED | 2018-12-05 | 0 | 0 | 2 | Release of New OEM_Global_Standard_Components for Plant B designer |
| ECN261305 | RESOLVED | 2018-12-05 | 0 | 0 | 1 | Release of New OEM_Global_Standard_Components for Plant B designer |
| ECN261403 | RESOLVED | 2018-12-07 | 0 | 0 | 1 | Release of New OEM_Global_Standard_Components for Plant B designers |
| ECN261676 | RESOLVED | 2019-03-20 | 0 | 0 | 2 | PXB_CPM_transm gearbox: QKV3111040, QKW9082130 Creeper does not engage |
| ECN262158 | RESOLVED | 2019-02-08 | 0 | 0 | 1 | PXB_CPM_transm gearbox:QKV005830A QKV772098A spacer bush nitrating |
| ECN263691 | RESOLVED | 2019-02-08 | 0 | 0 | 1 | Release of New OEM_Global_Standard_Components for Plant A designer |
| ECN263759 | RESOLVED | 2019-02-20 | 0 | 0 | 4 | Move existing Plant C gasket X540008278000 to the global library |
| ECN264144 | RESOLVED | 2019-03-01 | 0 | 0 | 3 | Transfer of existing Plant C O-rings to the OEM Global Standard Components library |
| ECN264249 | RESOLVED | 2019-03-11 | 0 | 0 | 2 | Standard component request from PXC to support GIMA sourcing |
| ECN268059 | RESOLVED | 2020-01-07 | 0 | 0 | 14 | PXB_Q909V_Pre-Series_Release_of_Q909VS0170 |
| ECN269112 | RESOLVED | 2019-09-12 | 0 | 0 | 1 | T743_Serial_Rekl. 868_EG brake noise_outer clutch disc |
| ECN269599 | RESOLVED | 2019-09-17 | 0 | 0 | 4 | Cross-Serial_EFS_deliver drawings |
| ECN270047 | RESOLVED | 2019-09-18 | 0 | 0 | 44 | CPM-CAB: OV_CHZ_971_Welded nuts on fenders for OIB |
| ECN270739 | RESOLVED | 2019-09-11 | 0 | 0 | 1 | T743_Serial_Rekl. 868_EG actuator brake noise |
| ECN271122 | RESOLVED | 2019-09-25 | 0 | 0 | 1 | PXB_CPM_transm additional: 39254600, 38185014 Cover tolerance |
| ECN271172 | RESOLVED | 2019-11-18 | 0 | 0 | 2 | PXB_Q947V_PS_Release_of_Q947VS0101 |
| ECN271656 | RESOLVED | 2019-11-18 | 0 | 0 | 3 | PXB_Q947V_PS_Release_of_Q947VS3005 |
| ECN272316 | RESOLVED | 2019-10-11 | 0 | 0 | 11 | Release of New OEM_Global_Standard_Component for Plant B designer |
| ECN272713 | RESOLVED | 2019-11-27 | 0 | 0 | 6 | PXB_Q947V_PS_Release_of_Q947VS0105 |
| ECN273070 | RESOLVED | 2019-11-14 | 0 | 0 | 1 | PXB_CPM_transm gearbox: QKV0010600 Pressure limiting valve |
| ECN273807 | RESOLVED | 2019-11-21 | 0 | 0 | 16 | Move existing Plant B retaining rings to the OEM_Global_Standard_Components library |
| ECN273895 | RESOLVED | 2019-12-16 | 0 | 0 | 12 | Move existing Plant C retaining rings to the OEM_Global_Standard_Components library |
| ECN274005 | RESOLVED | 2019-11-25 | 0 | 0 | 1 | Release of New OEM_Global_Standard_Components for Plant A and Plant B designers |
| ECN275210 | RESOLVED | 2020-05-12 | 0 | 0 | 8 | Q9-38V SOP Cost reduction S0170 |
| ECN276194 | RESOLVED | 2020-01-27 | 0 | 0 | 3 | Release of New OEM_Global_Standard_Components for Plant B designer |
| ECN277834 | RESOLVED | 2020-04-24 | 0 | 0 | 1 | PXB_Q932V_PS Release S0150 content |
| ECN278613 | RESOLVED | 2020-07-08 | 0 | 0 | 1 | Cross-Serial_EG_deliver drawings |
| ECN279738 | RESOLVED | 2020-04-07 | 0 | 0 | 6 | Release of New OEM_Global_Standard_Component for Plant B designer |
| ECN280113 | RESOLVED | 2020-04-20 | 0 | 0 | 25 | Release of New OEM_Global_Standard_Component for Plant A designers |
| ECN280423 | RESOLVED | 2020-04-27 | 0 | 0 | 62 | Release of New OEM_Global_Standard_Components for M2 and S2 |
| ECN281931 | RESOLVED | 2020-07-08 | 0 | 0 | 17 | Move 24 Degree Cone Fitting Nuts and Cutting Rings to the Global Standard Components Library |
| ECN281960 | RESOLVED | 2020-06-11 | 0 | 0 | 1 | Release of New OEM_Global_Standard_Components for Plant B designers |
| ECN282065 | RESOLVED | 2020-06-16 | 0 | 0 | 20 | Release of New OEM_Global_Standard_Component for Plant A and Plant B designers |
| ECN282769 | RESOLVED | 2020-07-09 | 0 | 0 | 38 | Release of New OEM_Global_Standard_Components for PXA & PXB |
| ECN283693 | RESOLVED | 2020-09-11 | 0 | 0 | 1 | PXB_Q932V_PS_V0150_ |
| ECN283998 | RESOLVED | 2020-08-25 | 0 | 0 | 2 | Release of New OEM_Global_Standard_Component for Plant B designer |
| ECN284863 | RESOLVED | 2020-10-28 | 0 | 0 | 1 | PXB_Q900V_P0_S0145_FINAL_DRIVE_RELEASE |
| ECN285600 | RESOLVED | 2020-10-06 | 0 | 0 | 1 | Add supplier drawing to QKW6749320 |
| ECN285781 | RESOLVED | 2020-11-20 | 0 | 0 | 1 | PXB_Q909V_SOP_Corrections in S3525 |
| ECN286667 | RESOLVED | 2020-11-10 | 0 | 0 | 3 | Q939V - Serial release of 4WD |
| ECN286743 | RESOLVED | 2020-10-20 | 0 | 0 | 1 | Release of New OEM_Global_Standard_Component for Plant B designers |
| ECN286863 | RESOLVED | 2020-10-22 | 0 | 0 | 1 | Release of New OEM_Global_Standard_Components for Plant B designers |
| ECN286921 | RESOLVED | 2020-11-09 | 0 | 0 | 1 | PXB_Q932V_PS_Coupling_sleeve_replacement |
| ECN287115 | RESOLVED | 2020-12-18 | 0 | 0 | 4 | Q939V - Proto release of 4WD - S0130 |
| ECN287511 | RESOLVED | 2021-02-18 | 0 | 0 | 2 | T842_Serial_EFS_change to global hardware |
| ECN288087 | RESOLVED | 2020-12-10 | 0 | 0 | 1 | PXB_CPM M1 Planeetankannattimien 3D-malleissa ylimääräiset lukkorenkaat |
| ECN288253 | RESOLVED | 2021-01-20 | 0 | 0 | 4 | Cross-Serial_EFS_outer protection add data sheet |
| ECN289315 | RESOLVED | 2020-12-14 | 0 | 0 | 1 | Move existing Plant C clevis pins to the OEM Global Standard Components library |
| ECN289886 | RESOLVED | 2021-01-21 | 0 | 0 | 52 | PXB_Q938V EAS UPDATE |
| ECN290084 | RESOLVED | 2021-01-18 | 0 | 0 | 5 | PXB_Q932V_PS_EXTRA_NUT_IN_4001001 |
| ECN290773 | RESOLVED | 2021-01-21 | 0 | 0 | 3 | Release of New OEM_Global_Standard_Component for Plant B designer |
| ECN290886 | RESOLVED | 2021-02-01 | 0 | 0 | 8 | Move Plant C pin to the OEM Global Standard Component library |
| ECN291153 | RESOLVED | 2021-02-05 | 0 | 0 | 1 | Q9-09 : Update Cornaglia AIO to be aligned with S2 |
| ECN291211 | RESOLVED | 2021-02-01 | 0 | 0 | 1 | Release of Hydraulic Fitting for Plant A designer |
| ECN291495 | RESOLVED | 2021-04-16 | 0 | 0 | 4 | PXB_CPM_Retaining_ring_groove_update |
| ECN293014 | RESOLVED | 2021-04-15 | 0 | 0 | 2 | PXB_CPM_Retaining_ring_groove_update |
| ECN293556 | RESOLVED | 2021-04-14 | 0 | 0 | 1 | PXB_CPM_trasm pto: 37126900 |
| ECN293755 | RESOLVED | 2022-05-30 | 0 | 0 | 1 | Q938P_Serial_EG_axle_bevel_gear_Q938PB0140041 |
| ECN294332 | RESOLVED | 2021-04-22 | 0 | 0 | 1 | Release of New OEM_Global_Standard_Component for Plant B designer |
| ECN294388 | RESOLVED | 2021-04-23 | 0 | 0 | 1 | Release of New OEM_Global_Standard_Component forPlant B Designer |
| ECN294505 | RESOLVED | 2021-06-23 | 0 | 0 | 1 | PXB_Q900V_P0_S0195_Filtration block update |
| ECN295746 | RESOLVED | 2021-06-23 | 0 | 0 | 20 | Cross-Serial_EFS_deliver drawings of supplier |
| ECN297423 | RESOLVED | 2021-09-06 | 0 | 0 | 2 | Q939V & Q938V - Modification of S0150 & S1090 |
| ECN297428 | RESOLVED | 2021-09-13 | 0 | 0 | 1 | Q9-39 CVT - Serial release for P1A - PXC - S0130_#2 |
| ECN297757 | RESOLVED | 2021-08-18 | 0 | 0 | 2 | PXB_CPM:Shaft machining problem and missing data from drawing |
| ECN298055 | RESOLVED | 2021-10-27 | 0 | 0 | 1 | Q9-32, Q9-54, H2-19, Z11 Forbidden Hardware replacement |
| ECN299235 | RESOLVED | 2021-12-02 | 0 | 0 | 2 | Q9-39V_serial_EG_S0120_#3 |
| ECN299884 | RESOLVED | 2021-11-08 | 0 | 0 | 1 | Q9-39V - Serial release of 4WD shaft for JD (with JB impact on VC) |
| ECN299970 | RESOLVED | 2022-06-10 | 0 | 0 | 2 | Cross-Serial_EG_deliver drawings |
| ECN300011 | RESOLVED | 2021-11-11 | 0 | 0 | 22 | Remove O-Ring from WTPart Structure |
| ECN300047 | RESOLVED | 2022-01-21 | 0 | 0 | 1 | Q940P_Pre-Series_EG_S0170_#2 |
| ECN300877 | RESOLVED | 2022-02-10 | 0 | 0 | 4 | Q940P_Pre-Series_EG_S0145_#7 |
| ECN301026 | RESOLVED | 2021-12-16 | 0 | 0 | 2 | Remove O-Ring from WTPart Structure |
| ECN301289 | RESOLVED | 2021-12-13 | 0 | 0 | 1 | Move existing cylindrical pin JE6191 to the OEM Global Standard Component library |
| ECN301489 | RESOLVED | 2022-05-30 | 0 | 0 | 3 | Cross-Serial_EFS_shim set correction |
| ECN301769 | RESOLVED | 2021-12-14 | 0 | 0 | 2 | Release of New OEM_Global_Standard_Components for Plant C designers |
| ECN303301 | RESOLVED | 2022-04-08 | 0 | 0 | 1 | PXB_CPM_Retaining_ring_groove_update |
| ECN304579 | RESOLVED | 2022-04-13 | 0 | 0 | 2 | Q956P_P1B_EG_S0145_#1 |
| ECN305182 | RESOLVED | 2022-04-08 | 0 | 0 | 2 | Release of New OEM_Global_Standard_Components for Plant B designer |
| ECN305569 | RESOLVED | 2022-04-29 | 0 | 0 | 9 | Move existing Plant B hardware components to the OEM Global Standard Components library |
| ECN306291 | RESOLVED | 2022-06-03 | 0 | 0 | 1 | Q940P_Pre-Series_EG_S0145_#11 |
| ECN306528 | RESOLVED | 2022-07-04 | 0 | 0 | 1 | PXB_CPM: Wearing plate tolerancing incorrect |
| ECN306574 | RESOLVED | 2022-09-14 | 0 | 0 | 3 | PXB_CPM: 3mm drilling tolerance synchronization |
| ECN306894 | RESOLVED | 2022-06-22 | 0 | 0 | 7 | GR0263 Drawing fix |
| ECN307099 | RESOLVED | 2022-10-26 | 0 | 0 | 2 | PXB_Q900V_P1A_PXC_Release_of_S0135_#3 |
| ECN307123 | RESOLVED | 2022-07-01 | 0 | 0 | 1 | Move existing Plant B setscrew to the OEM Global Standard Components library |
| ECN307278 | RESOLVED | 2022-06-29 | 0 | 0 | 1 | Release of Hydraulic Fitting for Plant B designer |
| ECN307441 | RESOLVED | 2022-07-26 | 0 | 0 | 7 | Cross-Serial_EFS_deliver drawings |
| ECN307685 | RESOLVED | 2022-07-26 | 0 | 0 | 7 | Q9-39 - supplier change request QKX9397390 |
| ECN307718 | RESOLVED | 2022-08-30 | 0 | 0 | 1 | Q956P_P1B_EFH_S1047 |
| ECN308920 | RESOLVED | 2022-11-07 | 0 | 0 | 2 | Cross-Serial_EG_PTO shaft exchange_M2AA0_Q938V_Q923V |
| ECN308957 | RESOLVED | 2022-10-14 | 0 | 0 | 2 | Q940P_SOP_EG_S0110_#2 |
| ECN308999 | RESOLVED | 2022-11-07 | 0 | 0 | 3 | Current product - QKW8803390: delete Koyo alternative |
| ECN309105 | RESOLVED | 2022-09-07 | 0 | 0 | 1 | Release of New OEM_Global_Standard_Component for Plant C designer |
| ECN309433 | RESOLVED | 2023-08-28 | 0 | 0 | 2 | Cross-Serial_EG_drawing correction marking |
| ECN311779 | RESOLVED | 2023-03-08 | 0 | 0 | 2 | Cross-Serial_EFS_deliver drawings and drw cor. |
| ECN312416 | RESOLVED | 2023-01-09 | 0 | 0 | 1 | PXB_CPM Missing data from QKX4209690 drawing |
| ECN312793 | RESOLVED | 2023-04-18 | 0 | 0 | 5 | Cross-Serial_EFS_WVA deliver drawing cor. |
| ECN314008 | RESOLVED | 2023-05-22 | 0 | 0 | 1 | PXB_CPM Drawings missing on items |
| ECN314607 | RESOLVED | 2023-03-28 | 0 | 0 | 1 | HYD-NPA-PROTO-F4 Rear Suspension |
| ECN314889 | RESOLVED | 2023-06-07 | 0 | 0 | 2 | Q956P_Pre-Series_EG_S0140_#4 |
| ECN315343 | RESOLVED | 2023-08-02 | 0 | 0 | 4 | Q956P_Pre-Series_EG_S0150_#3 |
| ECN315715 | RESOLVED | 2023-05-16 | 0 | 0 | 4 | Q956P_Pre-Series_EG_S0145_#8 |
| ECN315759 | RESOLVED | 2023-07-12 | 0 | 0 | 1 | Cross-Serial_EG_drawing corrections |
| ECN316008 | RESOLVED | 2023-05-09 | 0 | 0 | 2 | PXB_CPM SAPTO Shaft and bellhousing tolerances overlap |
| ECN316539 | RESOLVED | 2024-02-05 | 0 | 0 | 4 | Cross-Serial_EG_manufacturing order spur gears PTO front 1000 |
| ECN318435 | RESOLVED | 2023-12-18 | 0 | 0 | 3 | PXB_CPM: No drawings on old items |
| ECN318718 | RESOLVED | 2024-03-13 | 0 | 0 | 3 | Cross-Serial_EG_bearing deliver updated supplier drawing |
| ECN319844 | RESOLVED | 2023-09-08 | 0 | 0 | 1 | PXB_CPM: oil drain hole tolerance fix for machining |
| ECN319845 | RESOLVED | 2024-03-19 | 0 | 0 | 14 | PXB_CPM: No drawings on old items |
| ECN319846 | RESOLVED | 2024-01-11 | 0 | 0 | 17 | PXB_CPM: No drawings on old items |
| ECN319847 | RESOLVED | 2023-12-14 | 0 | 0 | 23 | PXB_CPM: No drawings on old items |
| ECN319848 | RESOLVED | 2023-11-23 | 0 | 0 | 10 | PXB_CPM: No drawings on old items |
| ECN319849 | RESOLVED | 2023-10-27 | 0 | 0 | 10 | PXB_CPM: No drawings on old items |
| ECN319850 | RESOLVED | 2024-02-13 | 0 | 0 | 12 | PXB_CPM: No drawings on old items |
| ECN319853 | RESOLVED | 2023-11-30 | 0 | 0 | 44 | PXB_CPM: No drawings on old items |
| ECN319856 | RESOLVED | 2023-11-20 | 0 | 0 | 8 | PXB_CPM: No drawings on old items |
| ECN320150 | RESOLVED | 2024-07-11 | 0 | 0 | 6 | Cross-Serial_EFS_drw corrections material gaskets |
| ECN320484 | RESOLVED | 2023-11-13 | 0 | 0 | 4 | PXB_CPM: No drawings on old items |
| ECN320485 | RESOLVED | 2023-11-13 | 0 | 0 | 19 | PXB_CPM: No drawings on old items |
| ECN320533 | RESOLVED | 2024-01-09 | 0 | 0 | 2 | PXB_CPM Stickers, alternative material and supplier |
| ECN321054 | RESOLVED | 2023-11-20 | 0 | 0 | 7 | PXB_CPM: Replace custom set screw with standard part |
| ECN321266 | RESOLVED | 2024-01-09 | 0 | 0 | 1 | PXB_CPM: Graphics update for warning stickers |
| ECN321810 | RESOLVED | 2024-05-30 | 0 | 0 | 11 | PXB_CPM: No drawings on old items |
| ECN321814 | RESOLVED | 2024-02-13 | 0 | 0 | 20 | PXB_CPM: No drawings on old items |
| ECN321815 | RESOLVED | 2024-08-14 | 0 | 0 | 10 | PXB_CPM: No drawings on old items |
| ECN321816 | RESOLVED | 2024-04-10 | 0 | 0 | 16 | PXB_CPM: No drawings on old items |
| ECN321817 | RESOLVED | 2024-02-13 | 0 | 0 | 18 | PXB_CPM: No drawings on old items |
| ECN321818 | RESOLVED | 2024-04-08 | 0 | 0 | 24 | PXB_CPM: No drawings on old items |
| ECN321821 | RESOLVED | 2024-02-13 | 0 | 0 | 18 | PXB_CPM: No drawings on old items |
| ECN321822 | RESOLVED | 2024-02-13 | 0 | 0 | 6 | PXB_CPM: No drawings on old items |
| ECN321825 | RESOLVED | 2024-04-08 | 0 | 0 | 12 | PXB_CPM: No drawings on old items |
| ECN321859 | RESOLVED | 2023-11-13 | 0 | 0 | 1 | PXB_CPM Bushing not tempered |
| ECN322091 | RESOLVED | 2024-04-11 | 0 | 0 | 1 | Q956P_Pre-Series_EG_S0140_#8 |
| ECN322413 | RESOLVED | 2024-02-12 | 0 | 0 | 13 | PXB_CPM Valve coils 3D is not correct |
| ECN322706 | RESOLVED | 2023-12-20 | 0 | 0 | 2 | Cross-Serial_EG_flanged pin geometric drw correction |
| ECN324019 | RESOLVED | 2024-01-10 | 0 | 0 | 2 | Q956P_Pre-Series_EG_S0150_#15 |
| ECN324133 | RESOLVED | 2024-10-23 | 0 | 0 | 1 | PXB CPM: Oil groove standardize |
| ECN324134 | RESOLVED | 2024-06-12 | 0 | 0 | 3 | PXB_CPM: No drawings on old items |
| ECN324482 | RESOLVED | 2024-01-23 | 0 | 0 | 1 | Add Shafer drawing as reference document. |
| ECN325059 | RESOLVED | 2024-06-05 | 0 | 0 | 4 | Cross-Serial_EG_bearing deliver updated supplier drawing |
| ECN325156 | RESOLVED | 2024-06-11 | 0 | 0 | 18 | PXB_CPM: No drawings on old items |
| ECN325331 | RESOLVED | 2024-02-07 | 0 | 0 | 2 | Release of New OEM_Global_Standard_Components for Hesston combine designer |
| ECN325362 | RESOLVED | 2024-05-16 | 0 | 0 | 3 | PXB_Q900V_P1B__VS0101_LLT release |
| ECN325364 | RESOLVED | 2024-04-11 | 0 | 0 | 3 | PXB_Q900V_P1B_VS0135_LLT release |
| ECN325367 | RESOLVED | 2024-04-11 | 0 | 0 | 3 | PXB_Q900V_P1B_VS0150_LLT release |
| ECN325727 | RESOLVED | 2024-05-29 | 0 | 0 | 2 | PXB_Q900V_P1B_VS0140_VCG release |
| ECN326787 | RESOLVED | 2024-04-11 | 0 | 0 | 3 | PXB_Q900V_P1B_VS0110_LLT release |
| ECN327248 | RESOLVED | 2024-04-03 | 0 | 0 | 4 | Release of New OEM_Global_Standard_Component for Plant B designer |
| ECN327659 | RESOLVED | 2024-05-07 | 0 | 0 | 2 | Q910V_PXB_NPI_	Extend Application: Material Option for Brazil - PN QKX3065000 |
| ECN327803 | RESOLVED | 2024-07-09 | 0 | 0 | 12 | Cross-Serial_EFS_translation drawings F4-26L |
| ECN328221 | RESOLVED | 2024-06-19 | 0 | 0 | 3 | PE LIN CORE CPM CH - CORE50 KCL updates |
| ECN328792 | RESOLVED | 2024-06-19 | 0 | 0 | 3 | PXB_CPM KCL’s missing on old parts |
| ECN328794 | RESOLVED | 2024-06-19 | 0 | 0 | 1 | PXB_CPM: LB5013 3D and drawings not correct |
| ECN329029 | RESOLVED | 2024-06-27 | 0 | 0 | 12 | PXB_CPM: No drawings on old items |
| ECN329137 | RESOLVED | 2024-05-21 | 0 | 0 | 2 | Release of Hydraulic Fitting for Plant B designer |
| ECN329354 | RESOLVED | 2024-09-11 | 0 | 0 | 2 | Cross-Serial_EG_S4150 PTO FRONT outer clutch disc 718150220080 |
| ECN329525 | RESOLVED | 2024-06-04 | 0 | 0 | 1 | Release of Hydraulic Fittings |
| ECN329602 | RESOLVED | 2024-09-04 | 0 | 0 | 1 | PXB_CPM QKV147913A incorrect data correction |
| ECN329680 | RESOLVED | 2024-06-20 | 0 | 0 | 2 | Q9-41SA-S4001 - 3_POINTS_SYS_REAR_WAX |
| ECN329959 | RESOLVED | 2024-06-26 | 0 | 0 | 1 | Q956P_Serial_EG_S0140_#1 |
| ECN330468 | RESOLVED | 2025-09-26 | 0 | 0 | 1 | Cross-Serial_Feld-Spurt_P22756_EG_drawing corrections profile bushes |
| ECN330996 | RESOLVED | 2025-09-08 | 0 | 0 | 2 | Cross-Serial_EG_MIP1528_internal clutch disc Q938PB0140020 |
| ECN331687 | RESOLVED | 2024-09-03 | 0 | 0 | 1 | PXB_CPM: warning sticker |
| ECN331688 | RESOLVED | 2024-09-06 | 0 | 0 | 1 | PXB_CPM JE9426 has incorrect data and drawings |
| ECN331774 | RESOLVED | 2025-04-04 | 0 | 0 | 3 | PXB_CPM Revert to old designs on clutch housings |
| ECN332318 | RESOLVED | 2024-09-25 | 0 | 0 | 6 | BOSCH RC40 Bootloader change |
| ECN332554 | RESOLVED | 2025-03-27 | 0 | 0 | 1 | Q950V_Release of Serial CAB Content of Plant B VC-Groups of S0080 |
| ECN332576 | RESOLVED | 2025-02-12 | 0 | 0 | 2 | Cross-Serial_EG_wheel hub torque drw cor |
| ECN333385 | RESOLVED | 2024-10-30 | 0 | 0 | 3 | PXB_CPM: QKV4655180 - Drawing correction |
| ECN333397 | RESOLVED | 2024-10-31 | 0 | 0 | 2 | PXB_CPM: Drawings needs thread pitch to both ends |
| ECN333414 | RESOLVED | 2024-11-05 | 0 | 0 | 4 | PXB_Q900V_P1B_VS0140_LLT release |
| ECN333701 | RESOLVED | 2024-11-25 | 0 | 0 | 1 | Cross-Serial_EG_groove ball bearing 6211NR update |
| ECN334491 | RESOLVED | 2025-07-07 | 0 | 0 | 1 | Cross-Serial_EG_MIP1452 oil baffle drw update |
| ECN334798 | RESOLVED | 2025-01-23 | 0 | 0 | 3 | Cross-Serial_EG_updated supplier drawings |
| ECN334820 | RESOLVED | 2025-07-02 | 0 | 0 | 2 | PXB_Q957V_P1B_release_of_Q957VS0080_VC_groups |
| ECN335323 | RESOLVED | 2025-01-13 | 0 | 0 | 1 | PXB_CPM Std parts missing drawings |
| ECN335751 | RESOLVED | 2025-01-23 | 0 | 0 | 2 | Cross-Serial_EG_axle bevel gear drawing cor |
| ECN336402 | RESOLVED | 2025-03-05 | 0 | 0 | 2 | PXB_CPM handbrake cylinder resource |
| ECN336576 | RESOLVED | 2025-02-12 | 0 | 0 | 2 | PXB_Q900V_charging_valve_serial_release |
| ECN336992 | RESOLVED | 2025-12-11 | 0 | 0 | 1 | Cross-Serial_EG_outer clutch disc drawing update |
| ECN337140 | RESOLVED | 2025-03-03 | 0 | 0 | 3 | Cross-Serial_EG_MIP1564 threaded pin wheel nut |
| ECN337182 | RESOLVED | 2025-03-19 | 0 | 0 | 1 | PXB_CPM_QKX3390220 item number location 2D |
| ECN337190 | RESOLVED | 2025-02-25 | 0 | 0 | 4 | PXB_Q900V_Industrialization_changes_PS_transmission_housing |
| ECN337228 | RESOLVED | 2025-03-05 | 0 | 0 | 1 | PXB_CPM - Pinion shaft lubrication pipeline bushing problem. |
| ECN337694 | RESOLVED | 2025-02-24 | 0 | 0 | 1 | Load_EG_Q900V_release_content_brake-test-bench |
| ECN338916 | RESOLVED | 2025-05-19 | 0 | 0 | 1 | Cross-Serial_EFS_drw correction_ring gears |
| ECN339044 | RESOLVED | 2025-04-07 | 0 | 0 | 1 | Load_EG_Q900V_release_content_brake-test-bench |
| ECN339841 | RESOLVED | 2025-05-26 | 0 | 0 | 4 | CPM : Need homologation stickers for climatisation |
| ECN339869 | RESOLVED | 2025-07-01 | 0 | 0 | 1 | PXB_CPM 32462587 has wrong tooth surface requirement |
| ECN339971 | RESOLVED | 2025-06-11 | 0 | 0 | 4 | Cross-Serial_EG_drawing_correction_bearings_#2 |
| ECN340341 | RESOLVED | 2025-07-01 | 0 | 0 | 1 | PXB_CPM 35420765 CATIA to Creo |
| ECN340655 | RESOLVED | 2025-06-16 | 0 | 0 | 1 | PXB_CPM Replace QKX3165540 KM7 limited version to SKF only |
| ECN341082 | RESOLVED | 2025-09-15 | 0 | 0 | 2 | Cross-Serial_EG_drawing and geometric update |
| ECN341507 | RESOLVED | 2025-07-15 | 0 | 0 | 1 | Load_EG_Q924P_S0150_#1 |
| ECN341537 | RESOLVED | 2025-07-09 | 0 | 0 | 2 | Global Hardware Release |
| ECN341616 | RESOLVED | 2025-08-26 | 0 | 0 | 2 | Cross-Serial_EG_drawing release del hard turned |
| ECN341735 | RESOLVED | 2025-12-11 | 0 | 0 | 2 | Q900V_Pre-Series_PXC_VC_S0145_#2 |
| ECN341740 | RESOLVED | 2025-12-11 | 0 | 0 | 1 | Q900V_Pre-Series_PXC_VC_S0140_#1 |
| ECN341747 | RESOLVED | 2025-12-04 | 0 | 0 | 2 | Q900V_Pre-Series_PXC_S0105_#1 |
| ECN343521 | RESOLVED | 2025-10-02 | 0 | 0 | 5 | Cross-Serial_EG_drawing_correction_bearings_#7 |
| ECN343828 | RESOLVED | 2025-10-07 | 0 | 0 | 1 | Q900V_Pre-Series_PXC_S0170_#2 |
| ECN343870 | RESOLVED | 2025-09-18 | 0 | 0 | 1 | Global hardware release |
| ECN343994 | RESOLVED | 2025-10-01 | 0 | 0 | 1 | Cross-Serial_EG_drawing_correction_bearings_#8 |
| ECN344313 | RESOLVED | 2025-12-17 | 0 | 0 | 2 | PXB_Q900V_Pre-Series_Release_of_S0042_SCHEMATICS_PNE |
| ECN344314 | RESOLVED | 2025-11-24 | 0 | 0 | 2 | PXB_Q900V_Pre-Series_Release_of_S0043_SCHEMATICS_DRIVELINE |
| ECN344315 | RESOLVED | 2025-11-25 | 0 | 0 | 3 | PXB_Q900V_Pre-Series_Release_of_S0044_SCHEMATICS_ENG_INST |
| ECN344442 | RESOLVED | 2025-11-19 | 0 | 0 | 5 | Q900V_Pre-Series_PXC_S0170_#3 |
| ECN344549 | RESOLVED | 2025-11-25 | 0 | 0 | 6 | PXB_Q900V_PS_FILTRATION_PLATE_DOCUMENTATION_UPDATE |
| ECN345315 | RESOLVED | 2025-11-06 | 0 | 0 | 4 | Q940P_EFS_drw correction headlights |
| ECN345905 | RESOLVED | 2025-12-08 | 0 | 0 | 2 | Cross-Serial_EG_drawing_correction_bearings_#9 |
| ECN346091 | RESOLVED | 2025-11-24 | 0 | 0 | 6 | PXB_CPM: Sealing rings to include chamfers |
| ECN347082 | RESOLVED | 2026-02-09 | 0 | 0 | 1 | PXB_CPM QKV8388140 support plate mounting hole tolerance correction |
| ECN347083 | RESOLVED | 2026-02-16 | 0 | 0 | 2 | PXB_CPM KH4171 Description wrong |
| ECN347232 | RESOLVED | 2026-01-16 | 0 | 0 | 1 | PXB_CPM:Bearing dimension correction to drawing |
| ECN347886 | RESOLVED | 2026-01-23 | 0 | 0 | 1 | PXB_Q900V_PRE-SERIES_ROPS |
| ECN347929 | RESOLVED | 2026-02-09 | 0 | 0 | 4 | PXB_Q900V_S0170_ACCUMULATOR_DOCUMENTATION_UPDATE |
| ECN348004 | RESOLVED | 2026-02-05 | 0 | 0 | 1 | PXB_Q957V_P1B_TR_PS_BUSHING_AND_TRANSFERER_SHAFT_REVISION |
| ECN348467 | RESOLVED | 2026-04-14 | 0 | 0 | 18 | PXB_Q900V_HOSE_DRAWING_CORRECTION |
| ECN348536 | RESOLVED | 2026-04-13 | 0 | 0 | 24 | CPM: DM - Functional drawing for 5224388T1 |
| ECN348792 | RESOLVED | 2026-05-27 | 0 | 0 | 1 | PXB_CPM QKV191330B machining values wrong |
| ECN348793 | RESOLVED | 2026-04-30 | 0 | 0 | 2 | PXB_CPM LA9214 drawing not correct |
| ECN349071 | RESOLVED | 2026-04-30 | 0 | 0 | 1 | PXB_APAJA_D-sample_release_corrections for PTO and differential |
| ECN349820 | RESOLVED | 2026-06-05 | 0 | 0 | 37 | PXB_Q900V_Q957V_FROM_CATIA_TO_CREO_CONVERSION_RELEASE |
| ECN350027 | RESOLVED | 2026-04-16 | 0 | 0 | 1 | PXB_CPM JB0937 has wrong hardness class |
| ECN350189 | RESOLVED | 2026-06-16 | 0 | 0 | 1 | PXB_CPM 36993350 machined length, Q933V 5PS |
| ECN350291 | RESOLVED | 2026-08-06 | 0 | 0 | 2 | Cross-Serial_EG_drawing correction_#1 |
| ECN350505 | RESOLVED | 2026-05-06 | 0 | 0 | 1 | Q955V - Series releases |
| ECN350616 | RESOLVED | 2026-06-05 | 0 | 0 | 22 | PXB_Q900V_PRE_SERIES_TR_CHANGES_TO_M01_DRAWINGS |
| ECN350805 | RESOLVED | 2026-07-17 | 0 | 0 | 6 | Cross-Serial_EG_update datasheets_spacer/spacer set |
| ECN350806 | RESOLVED | 2026-06-08 | 0 | 0 | 1 | Cross-Serial_EG_drawing correction_wheel bolts |
| ECN350816 | RESOLVED | 2026-05-27 | 0 | 0 | 1 | Q9-32/Q9-54_PXA_ASY_Fitting vs cap compatibility 3274694W1 |
| ECN351283 | RESOLVED | 2026-06-02 | 0 | 0 | 23 | PXB_Q900V_PRE_SERIES_TR_CHANGES_TO_M05_DRAWINGS |
| ECN351679 | RESOLVED | 2026-05-18 | 0 | 0 | 4 | PXB_Q900V_PRE_SERIES_HYD SCHEMA UPDATE |
| ECN351748 | RESOLVED | 2026-08-27 | 0 | 0 | 47 | PXB_Q900V_PRE_SERIES_TR_CHANGES_ALTERNATIVE MATERIALS |
| ECN352932 | RESOLVED | 2026-07-17 | 0 | 0 | 1 | Q956P_Serial_BAW_EG_use up axle bevel gear |
| ECN353052 | RESOLVED | 2026-06-16 | 0 | 0 | 4 | Cross-Prototyp_EFK_KW25_16_06_26 |
| ECN353499 | RESOLVED | 2026-07-17 | 0 | 0 | 1 | Cross-Serial_EG_BOM correction |

## How to reproduce

```python
import json, re, openpyxl
ecns = {x["Number"] for x in json.load(open("data/new_16-09/ECN_full.json"))}
wb = openpyxl.load_workbook("data/EBOM MBOM_Q900V_04.09.2026.xlsx", read_only=True)

# EBOM Q900V: exact match on "N° of modification".
rows = wb["EBOM Q900V"].iter_rows(values_only=True)
col = [str(h).strip() if h else h for h in next(rows)].index("N° of modification")
in_ebom = {str(r[col]).strip() for r in rows} & ecns

# MBOM: the code embeds the ECN digits (MPXA3452740A -> ECN345274).
code = re.compile(r"^[A-Z]{4}(?P<digits>\d{6})[0-9A-Z]{2}$")
rows = wb["MBOM 130% KM_Q900VPXA_PXA"].iter_rows(values_only=True)
header = [str(h).strip() if h else h for h in next(rows)]
cols = [header.index("N° of modification"), header.index("at the number of modif.")]
digits = {m["digits"] for r in rows for c in cols if (m := code.match(str(r[c] or "").strip()))}
in_mbom = {e for e in ecns if e[3:] in digits}
```

## Related files

- `data/new_16-09/ECN_full.json`
- `data/EBOM MBOM_Q900V_04.09.2026.xlsx`
- `api/app/ecn/evidence/bom_changes.py`
- `api/app/ecn/evidence/raw_ebom.py`
- [130% EBOM rows in BOM changes](ecn-130-ebom-changes.md)
- [ECN BOM changes](ecn-bom-changes.md)
- [Reference datasets](reference-datasets.md)
