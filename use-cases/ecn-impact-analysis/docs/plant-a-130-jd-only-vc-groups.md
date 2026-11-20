# Plant A 130% EBOM: VC groups the conditional rules assign only to Plant B

Prepared for the customer touchpoint after 28-09-2026.

## Why this matters

The application now derives plants from the conditional rules: each level-3 VC
group belongs to the plants of the brands its rules require. With the current
mapping:

| Brand suffix | Plant |
|---|---|
| `JB` | PL10 Plant A |
| `IS` | PL10 Plant A |
| `JD` | PK01 Plant B |

The sheet `EBOM 130% KM_Q900VPXA` is the configured Plant A 130% EBOM, so we
expected every VC group in it to map to Plant A. The 30 VC groups listed below
are exceptions: the rules select them **only** for brand `JD`. The application
therefore assigns them to Plant B (PK01) and not to Plant A, even though the
Plant A 130% EBOM contains them.

**Question for the customer:** are these VC groups built in Plant A? If yes,
the plant logic is more than brand-based for these groups, for example
Plant A building JD-branded tractors, or a country or option condition. We
need the rule to apply.

## Sources and method

- EBOM: `EBOM MBOM_Q900V_04.09.2026.xlsx`, sheet `EBOM 130% KM_Q900VPXA`,
  level `.1` rows (the VC groups). There are 1,295 distinct VC groups.
- Rules: `Règlesconditionnelleslocales~…~04092026_0747GMT.xlsx`, sheet
  `Default`, 4,575 rules.
- A rule selects a VC group through its Include action, for example
  `… --> Inclure (Q900VS3010 = Q900VV3010455)`. The rule's brands are the
  `GEN_E_BRAND = E_BRAND_<X>` terms in its condition.
- The rules are read with the application's own parser
  (`api/app/ecn/rules/plant_scope.py`), with the application default: rules in
  state "To be Obsolete" are ignored.

### How the 1,295 Plant A 130% VC groups map

| Brands required by the rules | VC groups | Plant(s) assigned |
|---|---:|---|
| IS, JB | 658 | Plant A |
| IS, JB, JD | 499 | Plant A + Plant B |
| JB | 72 | Plant A |
| JB, JD | 8 | Plant A + Plant B |
| IS | 7 | Plant A |
| **JD only** | **30** | **Plant B only (listed below)** |
| No active rule | 21 | Not determined (appendix) |

## A. 29 VC groups selected only by JD rules

All of their rules are in state "App For Series". The result does not depend
on the lifecycle filter.

| # | VC group | Description (130% EBOM) | Rule number(s) |
|---:|---|---|---|
| 1 | `Q932VV0150051` | PTO_REAR SA-PTO | RL0000041506 |
| 2 | `Q932VV0150302` | PTO_REAR GSPTO_50KMH | RL0000079151 |
| 3 | `Q900VV0041100` | SCHEMATICS_HYD TRANSMISSION_5PS | RL0000047777 |
| 4 | `Q900VV0130103` | FRONT_WHEEL_DRIVE W/O_CREEPER_50KMH_PS | RL0000078966, RL0000009248 |
| 5 | `Q900VV0130104` | FRONT_WHEEL_DRIVE W/_CREEPER_50KMH_PS | RL0000078967, RL0000009249 |
| 6 | `Q900VV1005261` | ELC_CHS REAR_HYD_HARNESS_3_VALVES | RL0000038442 |
| 7 | `Q900VV1005264` | ELC_CHS REAR_HYD_HARNESS_6_VALVES | RL0000038445 |
| 8 | `Q900VV2501501` | STEERING_SYS_OE ELC_UNIT_SW | RL0000062637 |
| 9 | `Q900VV2564801` | ELN_CTRL_DEVICES_OE TC1_PS_SW | RL0000046736 |
| 10 | `Q900VV2564811` | ELN_CTRL_DEVICES_OE IC2_SW | RL0000046738 |
| 11 | `Q900VV2564831` | ELN_CTRL_DEVICES_OE DGW_SW | RL0000046740 |
| 12 | `Q900VV2564871` | ELN_CTRL_DEVICES_OE C-PILAR_DISPLAY_SW | RL0000046742 |
| 13 | `Q900VV2578501` | PRECISION_FARMING ACM_SW | RL0000043733 |
| 14 | `Q900VV2578511` | PRECISION_FARMING VD03_SW | RL0000043736 |
| 15 | `Q900VV2578521` | PRECISION_FARMING SC11786_SW | RL0000043737 |
| 16 | `Q900VV3010451` | AUX_HYD_VALV_SYS_CHS EHS4_RE_1_SW | RL0000037833 |
| 17 | `Q900VV3010452` | AUX_HYD_VALV_SYS_CHS EHS4_RE_2_SW | RL0000037834 |
| 18 | `Q900VV3010453` | AUX_HYD_VALV_SYS_CHS EHS4_RE_3_SW | RL0000037835 |
| 19 | `Q900VV3010454` | AUX_HYD_VALV_SYS_CHS EHS4_RE_4_SW | RL0000037836 |
| 20 | `Q900VV3010455` | AUX_HYD_VALV_SYS_CHS EHS4_RE_5_SW | RL0000037837 |
| 21 | `Q900VV3010456` | AUX_HYD_VALV_SYS_CHS EHS4_RE_6_SW | RL0000045150 |
| 22 | `Q900VV3010731` | AUX_HYD_VALV_SYS_CHS EHS4_FR_1_SW | 17 rules: RL0000080371–RL0000080385, RL0000080436, RL0000037838 |
| 23 | `Q900VV3010732` | AUX_HYD_VALV_SYS_CHS EHS4_FR_2_SW | 14 rules: RL0000080386–RL0000080397, RL0000080437, RL0000037839 |
| 24 | `Q900VV3010733` | AUX_HYD_VALV_SYS_CHS EHS4_FR_3_SW | 7 rules: RL0000080398–RL0000080403, RL0000037840 |
| 25 | `Q900VV3010734` | AUX_HYD_VALV_SYS_CHS EHS4_FR_4_SW | 13 rules: RL0000080405–RL0000080415, RL0000080438, RL0000037841 |
| 26 | `Q900VV3010735` | AUX_HYD_VALV_SYS_CHS EHS4_FR_5_SW | 7 rules: RL0000080418–RL0000080422, RL0000080439, RL0000043355 |
| 27 | `Q900VV3010736` | AUX_HYD_VALV_SYS_CHS EHS4_FR_6_SW | RL0000043356 |
| 28 | `Q900VV4101606` | 3POINT_SYS_FRONT R_CTRL_LINE5_VALVES_PS | RL0000042072 |
| 29 | `Q938PV0150803` | PTO_REAR CVT_PTO_SHAFT_1_3/4"_6-SPLINE | RL0000036660 |

The groups fall into a few families:

- electronic software units (`…_SW`): steering, control devices, precision
  farming, auxiliary hydraulic valves EHS4 rear and front;
- rear PTO variants;
- front-wheel drive at 50 km/h;
- rear hydraulic harnesses;
- one front three-point linkage control line.

### Example rule texts

```text
RL0000046738  GEN_E_BRAND = E_BRAND_JD --> Inclure (Q900VS2564 = Q900VV2564811)
RL0000047777  GEN_E_BRAND = E_BRAND_JD + GEN_E_GEARBOX = E_GEARBOX_5PS --> Inclure (Q900VS0041 = Q900VV0041100)
RL0000037837  GEN_E_BRAND = E_BRAND_JD + (GEN_E_ELECTRICAL_BASELINE = E_ELEC_BLINE_005 / GEN_E_ELECTRICAL_BASELINE = E_ELEC_BLINE_006)
              --> Inclure (Q900VS3010 = Q900VV3010455)
RL0000036660  GEN_E_BRAND = E_BRAND_JD + GEN_E_GEARBOX = E_GEARBOX_CVT + GEN_E_REAR_PTO_ADD_6_45 = E_R_PTO_ADD_6_45_W
              --> Inclure (Q900VS0150 = Q938PV0150803)
```

## B. One VC group that is JD-only because its JB/IS rules are being retired

| VC group | Description | Active rules (App For Series) | Rules "To be Obsolete" |
|---|---|---|---|
| `Q900VV3005750` | HYD_SUPPLY W/O_F_FREE_RETURN_JD | RL0000046334, RL0000046335 (JD) | RL0000046331, RL0000046332, RL0000046333, RL0000036089 (IS/JB) |

With the default filter this group is Plant B only. If "To be Obsolete" rules
are included in Settings, it becomes Plant A + Plant B. Its description already
ends in `_JD`.

**Question for the customer:** should "To be Obsolete" rules still count for
plant assignment?

## Appendix: Plant A 130% VC groups without an active rule

The application shows these as "Plant not determined".

| VC group | Description | Why |
|---|---|---|
| `Q900VV1545208` | HOOD DECAL_7S.155 | Only "To be Obsolete" rules |
| `Q900VV2564300` | ELN_CTRL_DEVICES_OE JB_SRC_5PS_SFT | Only "To be Obsolete" rules |
| `Q938VV2598154` | ACCESSORIES_OE MONITOR_BRACKET | Only "To be Obsolete" rules |
| `Q900VVMPXA004` | Trans CVT Q9-00 Flange PTO 4 speed | No rule |
| `Q900VVMPXA005` | Trans CVT Q9-00 Short bar PTO 4 speed | No rule |
| `Q900VVMPXA006` | Trans CVT Q9-00 Long bar PTO 4 speed | No rule |
| `Q900VVMPXA101` | VC GROUP UNIT TEMPLATE CREATION | No rule |
| `Q900VVMPXA111` | Q900VV3005152 spé CVT | No rule |
| `Q900VVMPXA121` | Q900VV4010115 spé CVT | No rule |
| `Q900VVMPXA131` | Q900VV0523110 Spé CVT | No rule |
| `Q900VVMPXA201` | VC GROUP UNIT TEMPLATE CREATION | No rule |
| `Q900VVMPXA301` | VC GROUP UNIT TEMPLATE CREATION | No rule |
| `Q900VVMPS401` | VC GROUP UNIT TEMPLATE CREATION | No rule |
| `Q900VVMPXA501` | VC GROUP UNIT TEMPLATE CREATION | No rule |
| `Q900VVMPXA502` | VC GROUP UNIT TEMPLATE CREATION | No rule |
| `Q900VVMPXA601` | VC GROUP UNIT TEMPLATE CREATION | No rule |
| `Q900VVMPXA701` | VC GROUP UNIT TEMPLATE CREATION | No rule |
| `Q900VVMPXA801` | Q900V - cab W/O DEICING + LESS KEY | No rule |
| `Q900VVMPXA802` | Q900V - cab W/ DEICING + LESS KEY | No rule |
| `Q900VVMPXA803` | Q900V - cab W/O DEICING + START BUTTON | No rule |
| `Q900VVMPXA804` | Q900V - cab W/ DEICING + START BUTTON | No rule |

The `Q900VVMPXA…` groups look like Plant A-local VC groups created on the SAP
side ("MPXA", "spé CVT", unit templates). They do not exist in the Windchill
rules.

**Question for the customer:** should Plant A-local `…MPXA…` groups always
belong to Plant A?

## Questions to raise

1. Are the 29 JD-only VC groups in section A built in Plant A? If so, which
   condition, other than brand, places them there?
2. Should "To be Obsolete" rules count for plant assignment (section B and the
   first three rows of the appendix)?
3. Should the Plant A-local `Q900VVMPXA…` VC groups, which have no Windchill
   rule, always be assigned to Plant A?
4. Is there a complete brand-to-plant key beyond JB, IS → Plant A and
   JD → Plant B?
