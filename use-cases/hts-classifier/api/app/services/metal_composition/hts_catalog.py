"""Structured HTS catalog compilation, HANA refresh, and runtime resolution."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from app.utils.hana import HANAConnection

from .config import MetalCompositionSettings, get_settings


logger = logging.getLogger(__name__)

_CURRENT_CODE_PATTERNS = (
    re.compile(r"\b\d{4}\.\d{2}\.\d{2}\.\d{2}\b"),
    re.compile(r"\b\d{4}\.\d{2}\.\d{4}\b"),
    re.compile(r"\b\d{4}\.\d{2}\.\d{2}\b"),
    re.compile(r"\b\d{4}\.\d{2}\b"),
    re.compile(r"\b\d{4}\b"),
    re.compile(r"\b\d{10}\b"),
    re.compile(r"\b\d{8}\b"),
    re.compile(r"\b\d{6}\b"),
    re.compile(r"\b\d{4}\b"),
)
_CHAPTER_99_RE = re.compile(r"\b99\d{2}(?:\.\d{2}){1,2}\b")
_SEARCH_TOKEN_RE = re.compile(r"[A-Za-z0-9#./-]+")
_SEARCH_TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "chapter",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "with",
}

CHAPTER_TITLES: Dict[int, str] = {
    1: "Live animals",
    2: "Meat and edible meat offal",
    3: "Fish and crustaceans, molluscs and other aquatic invertebrates",
    4: "Dairy produce; birds eggs; natural honey; edible products of animal origin, not elsewhere specified or included",
    5: "Products of animal origin, not elsewhere specified or included",
    6: "Live trees and other plants; bulbs, roots and the like; cut flowers and ornamental foliage",
    7: "Edible vegetables and certain roots and tubers",
    8: "Edible fruit and nuts; peel of citrus fruit or melons",
    9: "Coffee, tea, mat\u00e9 and spices",
    10: "Cereals",
    11: "Products of the milling industry; malt; starches; inulin; wheat gluten",
    12: "Oil seeds and oleaginous fruits; miscellaneous grains, seeds and fruits; industrial or medicinal plants; straw and fodder",
    13: "Lac; gums, resins and other vegetable saps and extracts",
    14: "Vegetable plaiting materials; vegetable products not elsewhere specified or included",
    15: "Animal or vegetable fats and oils and their cleavage products; prepared edible fats; animal or vegetable waxes",
    16: "Preparations of meat, of fish or of crustaceans, molluscs or other aquatic invertebrates",
    17: "Sugars and sugar confectionery",
    18: "Cocoa and cocoa preparations",
    19: "Preparations of cereals, flour, starch or milk; pastrycooks' products",
    20: "Preparations of vegetables, fruit, nuts or other parts of plants",
    21: "Miscellaneous edible preparations",
    22: "Beverages, spirits and vinegar",
    23: "Residues and waste from the food industries; prepared animal fodder",
    24: "Tobacco and manufactured tobacco substitutes",
    25: "Salt; sulfur; earths and stone; plastering materials, lime and cement",
    26: "Ores, slag and ash",
    27: "Mineral fuels, mineral oils and products of their distillation; bituminous substances; mineral waxes",
    28: "Inorganic chemicals; organic or inorganic compounds of precious metals, of rare-earth metals, of radioactive elements or of isotopes",
    29: "Organic chemicals",
    30: "Pharmaceutical products",
    31: "Fertilizers",
    32: "Tanning or dyeing extracts; tannins and their derivatives; dyes, pigments and other coloring matter; paints and varnishes; putty and other mastics; inks",
    33: "Essential oils and resinoids; perfumery, cosmetic or toilet preparations",
    34: "Soap, organic surface-active agents, washing preparations, lubricating preparations, artificial waxes, prepared waxes, polishing or scouring preparations, candles and similar articles, modeling pastes, dental waxes and dental preparations with a basis of plaster",
    35: "Albuminoidal substances; modified starches; glues; enzymes",
    36: "Explosives; pyrotechnic products; matches; pyrophoric alloys; certain combustible preparations",
    37: "Photographic or cinematographic goods",
    38: "Miscellaneous chemical products",
    39: "Plastics and articles thereof",
    40: "Rubber and articles thereof",
    41: "Raw hides and skins (other than furskins) and leather",
    42: "Articles of leather; saddlery and harness; travel goods, handbags and similar containers; articles of animal gut (other than silkworm gut)",
    43: "Furskins and artificial fur; manufactures thereof",
    44: "Wood and articles of wood; wood charcoal",
    45: "Cork and articles of cork",
    46: "Manufactures of straw, of esparto or of other plaiting materials; basketware and wickerwork",
    47: "Pulp of wood or of other fibrous cellulosic material; recovered (waste and scrap) paper or paperboard",
    48: "Paper and paperboard; articles of paper pulp, of paper or of paperboard",
    49: "Printed books, newspapers, pictures and other products of the printing industry; manuscripts, typescripts and plans",
    50: "Silk",
    51: "Wool, fine or coarse animal hair; horsehair yarn and woven fabric",
    52: "Cotton",
    53: "Other vegetable textile fibers; paper yarn and woven fabrics of paper yarn",
    54: "Man-made filaments; strip and the like of man-made textile materials",
    55: "Man-made staple fibers",
    56: "Wadding, felt and nonwovens; special yarns; twine, cordage, ropes and cables and articles thereof",
    57: "Carpets and other textile floor coverings",
    58: "Special woven fabrics; tufted textile fabrics; lace; tapestries; trimmings; embroidery",
    59: "Impregnated, coated, covered or laminated textile fabrics; textile articles of a kind suitable for industrial use",
    60: "Knitted or crocheted fabrics",
    61: "Articles of apparel and clothing accessories, knitted or crocheted",
    62: "Articles of apparel and clothing accessories, not knitted or crocheted",
    63: "Other made-up textile articles; sets; worn clothing and worn textile articles; rags",
    64: "Footwear, gaiters and the like; parts of such articles",
    65: "Headgear and parts thereof",
    66: "Umbrellas, sun umbrellas, walking sticks, seatsticks, whips, riding-crops and parts thereof",
    67: "Prepared feathers and down and articles made of feathers or of down; artificial flowers; articles of human hair",
    68: "Articles of stone, plaster, cement, asbestos, mica or similar materials",
    69: "Ceramic products",
    70: "Glass and glassware",
    71: "Natural or cultured pearls, precious or semiprecious stones, precious metals, metals clad with precious metal, and articles thereof; imitation jewelry; coin",
    72: "Iron and steel",
    73: "Articles of iron or steel",
    74: "Copper and articles thereof",
    75: "Nickel and articles thereof",
    76: "Aluminum and articles thereof",
    78: "Lead and articles thereof",
    79: "Zinc and articles thereof",
    80: "Tin and articles thereof",
    81: "Other base metals; cermets; articles thereof",
    82: "Tools, implements, cutlery, spoons and forks, of base metal; parts thereof of base metal",
    83: "Miscellaneous articles of base metal",
    84: "Nuclear reactors, boilers, machinery and mechanical appliances; parts thereof",
    85: "Electrical machinery and equipment and parts thereof; sound recorders and reproducers, television image and sound recorders and reproducers, and parts and accessories of such articles",
    86: "Railway or tramway locomotives, rolling stock and parts thereof; railway or tramway track fixtures and fittings and parts thereof; mechanical (including electro-mechanical) traffic signaling equipment of all kinds",
    87: "Vehicles other than railway or tramway rolling stock, and parts and accessories thereof",
    88: "Aircraft, spacecraft, and parts thereof",
    89: "Ships, boats and floating structures",
    90: "Optical, photographic, cinematographic, measuring, checking, precision, medical or surgical instruments and apparatus; parts and accessories thereof",
    91: "Clocks and watches and parts thereof",
    92: "Musical instruments; parts and accessories of such articles",
    93: "Arms and ammunition; parts and accessories thereof",
    94: "Furniture; bedding, mattresses, mattress supports, cushions and similar stuffed furnishings; lamps and lighting fittings, not elsewhere specified or included; illuminated signs, illuminated nameplates and the like; prefabricated buildings",
    95: "Toys, games and sports requisites; parts and accessories thereof",
    96: "Miscellaneous manufactured articles",
    97: "Works of art, collectors' pieces and antiques",
    98: "Special classification provisions",
    99: "Temporary legislation; temporary modifications proclaimed pursuant to trade agreements legislation; additional import restrictions proclaimed pursuant to section 22 of the agricultural adjustment act, as amended",
}


CATALOG_COLUMNS = [
    "code",
    "raw_code",
    "digits",
    "chapter_number",
    "heading_code",
    "family_6_code",
    "family_8_code",
    "indent",
    "parent_code",
    "description",
    "path_description",
    "unit_of_quantity",
    "general_rate_of_duty",
    "special_rate_of_duty",
    "column_2_rate_of_duty",
    "quota_quantity",
    "additional_duties",
    "searchable_text",
    "sort_order",
]
CODE_MAP_COLUMNS = [
    "source_code",
    "target_code",
    "mapping_type",
    "source_basis",
    "effective_note",
]


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).split()).strip()


def _lower_text(value: Any) -> str:
    return _normalize_text(value).lower()


def _tokenize(value: Any) -> List[str]:
    tokens: List[str] = []
    seen = set()
    for match in _SEARCH_TOKEN_RE.findall(_normalize_text(value).lower()):
        token = match.strip(" .,:;()[]{}")
        if len(token) < 2 or token in _SEARCH_TOKEN_STOPWORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _truncate_text(value: Any, *, max_chars: int = 220) -> str:
    text = _normalize_text(value)
    if len(text) <= max_chars:
        return text
    trimmed = text[: max_chars - 3].rstrip()
    return f"{trimmed}..."


def _unique_context_strings(values: Iterable[Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        normalized = _normalize_text(value)
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def _digit_count(code: str) -> int:
    return sum(1 for char in str(code) if char.isdigit())


def canonicalize_hts_code(value: Any) -> str:
    """Normalize HTS codes to 4, 6, 8, or 10 digits using a 4.2.4 leaf form."""

    text = _normalize_text(value)
    if not text:
        return ""

    candidate = ""
    for pattern in _CURRENT_CODE_PATTERNS:
        match = pattern.search(text)
        if match:
            candidate = match.group(0)
            break
    if not candidate:
        return ""

    if re.fullmatch(r"\d{10}", candidate):
        return f"{candidate[:4]}.{candidate[4:6]}.{candidate[6:10]}"
    if re.fullmatch(r"\d{8}", candidate):
        return f"{candidate[:4]}.{candidate[4:6]}.{candidate[6:8]}"
    if re.fullmatch(r"\d{6}", candidate):
        return f"{candidate[:4]}.{candidate[4:6]}"
    if re.fullmatch(r"\d{4}\.\d{2}\.\d{2}\.\d{2}", candidate):
        parts = candidate.split(".")
        return f"{parts[0]}.{parts[1]}.{parts[2]}{parts[3]}"
    return candidate


def _parent_code_for(code: str) -> str:
    digits = _digit_count(code)
    if digits >= 10:
        parts = code.split(".")
        if len(parts) >= 3 and len(parts[2]) == 4:
            return f"{parts[0]}.{parts[1]}.{parts[2][:2]}"
    if digits == 8:
        parts = code.split(".")
        if len(parts) >= 2:
            return f"{parts[0]}.{parts[1]}"
    if digits == 6:
        return code.split(".")[0]
    return ""


def _family_code(code: str, digits: int) -> str:
    normalized = canonicalize_hts_code(code)
    if not normalized:
        return ""
    if digits == 4:
        return normalized.split(".")[0]
    if digits == 6:
        parts = normalized.split(".")
        return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else ""
    if digits == 8:
        parts = normalized.split(".")
        if len(parts) >= 3:
            third = parts[2]
            return f"{parts[0]}.{parts[1]}.{third[:2]}"
    return ""


def _family_candidates(code: str) -> List[str]:
    normalized = canonicalize_hts_code(code)
    if not normalized:
        return []
    digits = _digit_count(normalized)
    candidates: List[str] = []
    if digits >= 10:
        family_8 = _family_code(normalized, 8)
        if family_8:
            candidates.append(family_8)
    if digits >= 8:
        family_6 = _family_code(normalized, 6)
        if family_6:
            candidates.append(family_6)
    return candidates


def _chapter_from_filename(path: Path) -> int:
    match = re.search(r"chapter(\d{2})\.csv$", path.name)
    if not match:
        raise ValueError(f"Unexpected HTS chapter filename: {path.name}")
    return int(match.group(1))


def _string_or_empty(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return _normalize_text(value)


def _int_or_zero(value: Any) -> int:
    if value is None or pd.isna(value):
        return 0
    return int(value)


def compile_hts_catalog_frame(*, csv_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    csv_paths = sorted(csv_dir.glob("chapter*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No HTS chapter CSV files found in {csv_dir}")

    sort_order = 0
    for csv_path in csv_paths:
        chapter_number = _chapter_from_filename(csv_path)
        frame = pd.read_csv(csv_path)
        outline_stack: List[str] = []
        for raw_row in frame.to_dict(orient="records"):
            raw_code = raw_row.get("HTS Number")
            code = canonicalize_hts_code(raw_code)
            description = _string_or_empty(raw_row.get("Description"))
            if not description:
                continue

            indent = _int_or_zero(raw_row.get("Indent"))
            outline_stack = outline_stack[:indent]
            outline_path_bits = [*outline_stack, description]
            outline_stack = outline_path_bits

            if not code:
                continue

            digits = _digit_count(code)
            heading_code = code.split(".")[0]
            rows.append(
                {
                    "code": code,
                    "raw_code": _string_or_empty(raw_code),
                    "digits": digits,
                    "chapter_number": chapter_number,
                    "heading_code": heading_code,
                    "family_6_code": _family_code(code, 6),
                    "family_8_code": _family_code(code, 8),
                    "indent": indent,
                    "parent_code": "",
                    "description": description,
                    "path_description": "",
                    "unit_of_quantity": _string_or_empty(raw_row.get("Unit of Quantity")),
                    "general_rate_of_duty": _string_or_empty(raw_row.get("General Rate of Duty")),
                    "special_rate_of_duty": _string_or_empty(raw_row.get("Special Rate of Duty")),
                    "column_2_rate_of_duty": _string_or_empty(raw_row.get("Column 2 Rate of Duty")),
                    "quota_quantity": _string_or_empty(raw_row.get("Quota Quantity")),
                    "additional_duties": _string_or_empty(raw_row.get("Additional Duties")),
                    "searchable_text": "",
                    "sort_order": sort_order,
                    "outline_path_description": " > ".join(outline_path_bits),
                }
            )
            sort_order += 1

    if not rows:
        raise ValueError(f"No coded HTS rows were compiled from {csv_dir}")

    outline_columns = [*CATALOG_COLUMNS, "outline_path_description"]
    catalog = pd.DataFrame(rows, columns=outline_columns)
    catalog = catalog.drop_duplicates(subset=["code"], keep="first").sort_values("sort_order").reset_index(drop=True)

    existing_codes = set(catalog["code"].values)
    synthetic_rows: List[Dict[str, Any]] = []
    seen_synthetic: set = set()
    for row in catalog.itertuples(index=False):
        if int(row.digits) < 8:
            continue
        family_6 = _family_code(str(row.code), 6)
        if not family_6 or family_6 in existing_codes or family_6 in seen_synthetic:
            continue
        seen_synthetic.add(family_6)
        synthetic_rows.append(
            {
                "code": family_6,
                "raw_code": family_6,
                "digits": 6,
                "chapter_number": int(row.chapter_number),
                "heading_code": str(row.heading_code),
                "family_6_code": family_6,
                "family_8_code": "",
                "indent": max(int(row.indent) - 1, 1),
                "parent_code": "",
                "description": str(row.description),
                "path_description": "",
                "unit_of_quantity": str(row.unit_of_quantity),
                "general_rate_of_duty": str(row.general_rate_of_duty),
                "special_rate_of_duty": str(row.special_rate_of_duty),
                "column_2_rate_of_duty": str(row.column_2_rate_of_duty),
                "quota_quantity": str(row.quota_quantity),
                "additional_duties": str(row.additional_duties),
                "searchable_text": "",
                "sort_order": int(row.sort_order),
                "outline_path_description": "",
            }
        )
    if synthetic_rows:
        synthetic_df = pd.DataFrame(synthetic_rows, columns=outline_columns)
        catalog = pd.concat([catalog, synthetic_df], ignore_index=True)
        catalog = catalog.sort_values("sort_order").reset_index(drop=True)

    descriptions = {str(row.code): str(row.description) for row in catalog.itertuples(index=False)}
    code_set = set(descriptions)

    parent_codes: List[str] = []
    path_descriptions: List[str] = []
    searchable_texts: List[str] = []
    for row in catalog.itertuples(index=False):
        parent_code = _parent_code_for(str(row.code))
        while parent_code and parent_code not in code_set:
            parent_code = _parent_code_for(parent_code)
        parent_codes.append(parent_code)

        outline_path_description = _normalize_text(getattr(row, "outline_path_description", ""))
        if outline_path_description:
            path_description = outline_path_description
        else:
            path_bits: List[str] = [str(row.description)]
            cursor = parent_code
            while cursor:
                parent_description = descriptions.get(cursor)
                if parent_description:
                    path_bits.append(parent_description)
                cursor = _parent_code_for(cursor)
            path_description = " > ".join(reversed(path_bits))
        path_descriptions.append(path_description)

        searchable_texts.append(
            _normalize_text(
                " ".join(
                    [
                        str(row.code),
                        str(row.description),
                        path_description,
                        f"chapter {int(row.chapter_number)}",
                        str(row.unit_of_quantity),
                        str(row.general_rate_of_duty),
                        str(row.special_rate_of_duty),
                        str(row.column_2_rate_of_duty),
                        str(row.additional_duties),
                    ]
                )
            )
        )

    catalog["parent_code"] = parent_codes
    catalog["path_description"] = path_descriptions
    catalog["searchable_text"] = searchable_texts
    return catalog[CATALOG_COLUMNS].copy()


def compile_hts_code_map_frame(*, code_map_path: Path) -> pd.DataFrame:
    if not code_map_path.exists():
        return pd.DataFrame(columns=CODE_MAP_COLUMNS)

    frame = pd.read_csv(code_map_path)
    rows: List[Dict[str, Any]] = []
    for raw_row in frame.to_dict(orient="records"):
        source_code = canonicalize_hts_code(raw_row.get("source_code"))
        target_code = canonicalize_hts_code(raw_row.get("target_code"))
        if not source_code or not target_code:
            continue
        rows.append(
            {
                "source_code": source_code,
                "target_code": target_code,
                "mapping_type": _string_or_empty(raw_row.get("mapping_type")) or "explicit_official_map",
                "source_basis": _string_or_empty(raw_row.get("source_basis")),
                "effective_note": _string_or_empty(raw_row.get("effective_note")),
            }
        )

    return pd.DataFrame(rows, columns=CODE_MAP_COLUMNS).drop_duplicates(subset=["source_code"], keep="first")


def refresh_hts_catalog_tables(
    *,
    settings: Optional[MetalCompositionSettings] = None,
    csv_dir: Optional[Path] = None,
    code_map_path: Optional[Path] = None,
    hana_schema: Optional[str] = None,
    catalog_table: Optional[str] = None,
    code_map_table: Optional[str] = None,
) -> Dict[str, Any]:
    config = settings or get_settings()
    catalog_frame = compile_hts_catalog_frame(csv_dir=(csv_dir or config.hts_catalog_dir).resolve())
    code_map_frame = compile_hts_code_map_frame(code_map_path=(code_map_path or config.hts_code_map_path).resolve())

    schema = (hana_schema if hana_schema is not None else config.hts_hana_schema).strip() or None
    catalog_table_name = (catalog_table or config.hts_catalog_hana_table).strip()
    code_map_table_name = (code_map_table or config.hts_code_map_hana_table).strip()

    connection = HANAConnection()
    catalog_result = connection.refresh_serving_table(
        frame=catalog_frame,
        table=catalog_table_name,
        schema=schema,
        primary_key="code",
        index_columns=("chapter_number", "heading_code", "family_6_code", "family_8_code"),
    )

    code_map_result: Dict[str, Any]
    if code_map_frame.empty:
        code_map_result = {"table": code_map_table_name, "schema": schema, "row_count": 0}
    else:
        code_map_result = connection.refresh_serving_table(
            frame=code_map_frame,
            table=code_map_table_name,
            schema=schema,
            primary_key="source_code",
            index_columns=("target_code",),
        )

    return {
        "status": "completed",
        "catalog": catalog_result,
        "code_map": code_map_result,
        "catalog_row_count": int(len(catalog_frame)),
        "code_map_row_count": int(len(code_map_frame)),
        "csv_dir": str((csv_dir or config.hts_catalog_dir).resolve()),
        "code_map_path": str((code_map_path or config.hts_code_map_path).resolve()),
    }


@dataclass(frozen=True)
class HTSCodeResolution:
    requested_code: str
    resolved_code: str
    validation_status: str
    normalized_from: Optional[str]
    resolution_basis: str
    catalog_row: Dict[str, Any]
    mapping_row: Optional[Dict[str, Any]]


class HanaHTSCatalogResolver:
    """Resolve and validate HTS candidates against the HANA-backed catalog."""

    def __init__(
        self,
        *,
        settings: Optional[MetalCompositionSettings] = None,
        catalog_frame: Optional[pd.DataFrame] = None,
        code_map_frame: Optional[pd.DataFrame] = None,
    ) -> None:
        self.settings = settings or get_settings()
        schema = self.settings.hts_hana_schema or None
        connection = HANAConnection()
        self.catalog_frame = (
            catalog_frame.copy()
            if catalog_frame is not None
            else connection.fetch_dataframe(self.settings.hts_catalog_hana_table, schema=schema)
        )
        self.code_map_frame = (
            code_map_frame.copy()
            if code_map_frame is not None
            else (
                connection.fetch_dataframe(self.settings.hts_code_map_hana_table, schema=schema)
                if connection.table_exists(self.settings.hts_code_map_hana_table, schema=schema)
                else pd.DataFrame(columns=CODE_MAP_COLUMNS)
            )
        )

        if self.catalog_frame.empty:
            raise ValueError("HTS catalog table is empty or unavailable in HANA.")

        self.catalog_frame["code"] = self.catalog_frame["code"].map(canonicalize_hts_code)
        self.catalog_frame = self.catalog_frame[self.catalog_frame["code"] != ""].copy()
        self.catalog_frame["description_norm"] = self.catalog_frame["description"].map(_lower_text)
        self.catalog_frame["path_description_norm"] = self.catalog_frame["path_description"].map(_lower_text)
        self.catalog_frame["searchable_text_norm"] = self.catalog_frame["searchable_text"].map(_lower_text)
        self.catalog_by_code = {
            str(row["code"]): dict(row)
            for row in self.catalog_frame.to_dict(orient="records")
        }
        self.code_map_by_source = {
            str(row["source_code"]): dict(row)
            for row in self.code_map_frame.to_dict(orient="records")
            if canonicalize_hts_code(row.get("source_code"))
        }
        self.heading_rows = self.catalog_frame[self.catalog_frame["digits"] == 4].copy()
        self.family_rows = self.catalog_frame[self.catalog_frame["digits"] == 6].copy()

    def _build_search_context(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        raw = dict(context or {})
        phrases = _unique_context_strings(
            [
                raw.get("article_summary"),
                raw.get("function_summary"),
                raw.get("part_description"),
                raw.get("new_part_description"),
                *(raw.get("material_clues", []) or []),
                *(raw.get("standard_cues", []) or []),
                *(raw.get("discriminator_notes", []) or []),
                *(raw.get("phrases", []) or []),
            ]
        )
        tokens = _unique_context_strings(
            [
                *[token for value in raw.get("tokens", []) or [] for token in _tokenize(value)],
                *[token for phrase in phrases for token in _tokenize(phrase)],
            ]
        )
        heading_hypotheses = [
            str(item).strip()
            for item in raw.get("heading_hypotheses", []) or []
            if re.fullmatch(r"\d{4}", str(item).strip())
        ]

        likely_chapters = {
            int(heading[:2])
            for heading in heading_hypotheses
            if len(heading) >= 2 and heading[:2].isdigit()
        }
        token_set = {token.lower() for token in tokens}
        material_text = " ".join(_lower_text(value) for value in raw.get("material_clues", []) or [])
        function_text = " ".join(_lower_text(value) for value in phrases)

        if "steel" in token_set or "iron" in token_set or "steel" in material_text:
            likely_chapters.update({72, 73})
        if "aluminum" in token_set or "aluminium" in token_set:
            likely_chapters.add(76)
        if token_set.intersection({"valve", "pump", "filter", "filtering", "purifying", "machinery", "appliance"}):
            likely_chapters.add(84)
        if token_set.intersection({"electrical", "electric", "connector", "switch", "circuit", "relay", "diffuser"}):
            # "diffuser" is ambiguous; the router still sees the real options later.
            likely_chapters.add(85)
        if "section 232" in function_text or "additional duties" in function_text:
            likely_chapters.update({98, 99})

        return {
            "phrases": phrases[:10],
            "tokens": tokens[:80],
            "heading_hypotheses": heading_hypotheses[:8],
            "likely_chapters": sorted(likely_chapters),
        }

    def _chapter_bias_score(self, chapter_number: int, search_context: Dict[str, Any]) -> float:
        score = 0.0
        likely_chapters = set(search_context.get("likely_chapters", []) or [])
        if chapter_number in likely_chapters:
            score += 3.0
        headings = search_context.get("heading_hypotheses", []) or []
        if any(heading.startswith(f"{chapter_number:02d}") for heading in headings):
            score += 4.0
        tokens = set(search_context.get("tokens", []) or [])
        if chapter_number in {72, 73} and tokens.intersection({"steel", "iron", "rolled", "plate", "coil", "disc"}):
            score += 1.5
        if chapter_number == 84 and tokens.intersection({"valve", "pump", "filter", "filtering", "purifying"}):
            score += 1.5
        if chapter_number == 85 and tokens.intersection({"electrical", "electric", "connector", "switch", "circuit"}):
            score += 1.5
        return score

    def _row_prefilter_details(self, row: Dict[str, Any], search_context: Dict[str, Any]) -> Dict[str, Any]:
        description = _lower_text(row.get("description"))
        path_description = _lower_text(row.get("path_description"))
        searchable = _lower_text(row.get("searchable_text"))
        matched_terms = [
            token
            for token in search_context.get("tokens", []) or []
            if token and (token in description or token in path_description or token in searchable)
        ][:10]
        matched_phrases = [
            phrase
            for phrase in search_context.get("phrases", []) or []
            if phrase and phrase.lower() in searchable
        ][:6]

        score = float(len(matched_phrases) * 3.0 + len(matched_terms) * 0.45)
        if str(row.get("heading_code") or "") in set(search_context.get("heading_hypotheses", []) or []):
            score += 6.0
        score += self._chapter_bias_score(int(row.get("chapter_number") or 0), search_context)

        if score <= 0.0:
            # Keep broad recall available even when the text does not overlap cleanly.
            score = self._chapter_bias_score(int(row.get("chapter_number") or 0), search_context)

        return {
            "prefilter_score": round(score, 4),
            "matched_terms": matched_terms,
            "matched_phrases": matched_phrases,
        }

    def list_chapter_options(self, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        search_context = self._build_search_context(context)
        chapters = sorted(int(value) for value in self.catalog_frame["chapter_number"].dropna().unique())
        options: List[Dict[str, Any]] = []
        for chapter_number in chapters:
            heading_rows = self.heading_rows[self.heading_rows["chapter_number"] == chapter_number]
            sample_headings = [
                _truncate_text(value, max_chars=80)
                for value in heading_rows["description"].head(2).tolist()
                if _normalize_text(value)
            ]
            title = CHAPTER_TITLES.get(chapter_number, f"Chapter {chapter_number}")
            options.append(
                {
                    "chapter_number": chapter_number,
                    "title": title,
                    "summary": _truncate_text(
                        f"{title}. {len(heading_rows)} heading rows available."
                        + (f" Examples: {'; '.join(sample_headings)}." if sample_headings else ""),
                        max_chars=240,
                    ),
                    "prefilter_score": round(self._chapter_bias_score(chapter_number, search_context), 4),
                }
            )
        return sorted(
            options,
            key=lambda item: (float(item["prefilter_score"]), -int(item["chapter_number"])),
            reverse=True,
        )

    def list_heading_options(
        self,
        chapters: Sequence[int | str],
        context: Optional[Dict[str, Any]] = None,
        *,
        per_chapter: int = 25,
    ) -> List[Dict[str, Any]]:
        search_context = self._build_search_context(context)
        chapter_numbers = {
            int(str(chapter))
            for chapter in chapters
            if str(chapter).strip().isdigit()
        }
        rows = self.heading_rows[self.heading_rows["chapter_number"].isin(chapter_numbers)]
        options: List[Dict[str, Any]] = []
        for row in rows.to_dict(orient="records"):
            details = self._row_prefilter_details(row, search_context)
            options.append(
                {
                    "chapter_number": int(row["chapter_number"]),
                    "heading_code": str(row["code"]),
                    "description": _normalize_text(row["description"]),
                    "path_description": _normalize_text(row["path_description"]),
                    "path_summary": _truncate_text(row["path_description"], max_chars=240),
                    **details,
                }
            )

        selected: List[Dict[str, Any]] = []
        for chapter_number in sorted(chapter_numbers):
            chapter_options = [item for item in options if item["chapter_number"] == chapter_number]
            chapter_options.sort(
                key=lambda item: (
                    float(item["prefilter_score"]),
                    len(item["matched_phrases"]),
                    len(item["matched_terms"]),
                    item["heading_code"],
                ),
                reverse=True,
            )
            selected.extend(chapter_options[:per_chapter])
        return selected

    def list_family_options(
        self,
        headings: Sequence[str],
        context: Optional[Dict[str, Any]] = None,
        *,
        per_heading: int = 5,
    ) -> List[Dict[str, Any]]:
        search_context = self._build_search_context(context)
        heading_codes = {
            str(code).strip()
            for code in headings
            if re.fullmatch(r"\d{4}", str(code).strip())
        }
        rows = self.family_rows[self.family_rows["heading_code"].isin(heading_codes)]
        options: List[Dict[str, Any]] = []
        for row in rows.to_dict(orient="records"):
            details = self._row_prefilter_details(row, search_context)
            options.append(
                {
                    "code": str(row["code"]),
                    "digits": int(row["digits"]),
                    "chapter_number": int(row["chapter_number"]),
                    "heading_code": str(row["heading_code"]),
                    "description": _normalize_text(row["description"]),
                    "path_description": _normalize_text(row["path_description"]),
                    **details,
                }
            )

        selected: List[Dict[str, Any]] = []
        for heading_code in sorted(heading_codes):
            heading_options = [item for item in options if item["heading_code"] == heading_code]
            heading_options.sort(
                key=lambda item: (
                    float(item["prefilter_score"]),
                    len(item["matched_phrases"]),
                    len(item["matched_terms"]),
                    item["code"],
                ),
                reverse=True,
            )
            selected.extend(heading_options[:per_heading])
        return selected

    def expand_children(
        self,
        families: Sequence[str],
        context: Optional[Dict[str, Any]] = None,
        *,
        per_family: int = 3,
    ) -> Dict[str, List[Dict[str, Any]]]:
        search_context = self._build_search_context(context)
        expanded: Dict[str, List[Dict[str, Any]]] = {}
        family_codes: List[str] = []
        seen_family_codes = set()
        for code in families:
            family_code = canonicalize_hts_code(code)
            if not family_code or family_code in seen_family_codes:
                continue
            seen_family_codes.add(family_code)
            family_codes.append(family_code)
        for family_code in family_codes:
            family_row = self.catalog_by_code.get(family_code, {})
            rows = self.catalog_frame[
                (self.catalog_frame["family_6_code"] == family_code)
                & (self.catalog_frame["digits"].isin([8, 10]))
                & (self.catalog_frame["code"] != family_code)
            ]
            options: List[Dict[str, Any]] = []
            for row in rows.to_dict(orient="records"):
                if _lower_text(row.get("path_description")) == _lower_text(family_row.get("path_description")):
                    continue
                details = self._row_prefilter_details(row, search_context)
                options.append(
                    {
                        "code": str(row["code"]),
                        "digits": int(row["digits"]),
                        "family_code": family_code,
                        "chapter_number": int(row["chapter_number"]),
                        "heading_code": str(row["heading_code"]),
                        "description": _normalize_text(row["description"]),
                        "path_description": _normalize_text(row["path_description"]),
                        **details,
                    }
                )
            options.sort(
                key=lambda item: (
                    float(item["prefilter_score"]),
                    len(item["matched_phrases"]),
                    len(item["matched_terms"]),
                    item["digits"],
                    item["code"],
                ),
                reverse=True,
            )
            expanded[family_code] = options[:per_family]
        return expanded

    def resolve_code(self, candidate_code: Any) -> HTSCodeResolution:
        requested_code = canonicalize_hts_code(candidate_code)
        if not requested_code:
            return HTSCodeResolution(
                requested_code="",
                resolved_code="",
                validation_status="invalid",
                normalized_from=None,
                resolution_basis="No parseable HTS code was available to validate.",
                catalog_row={},
                mapping_row=None,
            )

        current_row = self.catalog_by_code.get(requested_code)
        if current_row is not None:
            return HTSCodeResolution(
                requested_code=requested_code,
                resolved_code=requested_code,
                validation_status="current_exact",
                normalized_from=None,
                resolution_basis=f"Current HTS catalog contains {requested_code}.",
                catalog_row=current_row,
                mapping_row=None,
            )

        mapping_row = self.code_map_by_source.get(requested_code)
        if mapping_row is not None:
            mapped_code = canonicalize_hts_code(mapping_row.get("target_code"))
            mapped_row = self.catalog_by_code.get(mapped_code)
            if mapped_row is not None:
                basis = _normalize_text(mapping_row.get("source_basis")) or "explicit official HTS mapping"
                return HTSCodeResolution(
                    requested_code=requested_code,
                    resolved_code=mapped_code,
                    validation_status="legacy_mapped_exact",
                    normalized_from=requested_code,
                    resolution_basis=f"Mapped {requested_code} to current code {mapped_code} using {basis}.",
                    catalog_row=mapped_row,
                    mapping_row=mapping_row,
                )
            requested_code = mapped_code or requested_code
        elif _digit_count(requested_code) == 8:
            padded_10 = requested_code + "00"
            padded_row = self.catalog_by_code.get(padded_10)
            if padded_row is not None:
                return HTSCodeResolution(
                    requested_code=canonicalize_hts_code(candidate_code),
                    resolved_code=padded_10,
                    validation_status="current_exact",
                    normalized_from=requested_code,
                    resolution_basis=f"Current HTS catalog contains {padded_10}.",
                    catalog_row=padded_row,
                    mapping_row=None,
                )
        elif _digit_count(requested_code) >= 10:
            return HTSCodeResolution(
                requested_code=canonicalize_hts_code(candidate_code),
                resolved_code="",
                validation_status="invalid",
                normalized_from=canonicalize_hts_code(candidate_code),
                resolution_basis=f"{canonicalize_hts_code(candidate_code)} is not present in the current HTS catalog.",
                catalog_row={},
                mapping_row=None,
            )

        for family_code in _family_candidates(requested_code):
            family_row = self.catalog_by_code.get(family_code)
            if family_row is None:
                continue
            family_digits = _digit_count(family_code)
            return HTSCodeResolution(
                requested_code=canonicalize_hts_code(candidate_code),
                resolved_code=family_code,
                validation_status=f"downgraded_to_{family_digits}",
                normalized_from=canonicalize_hts_code(candidate_code),
                resolution_basis=f"Deepest current HTS family found for {canonicalize_hts_code(candidate_code)} is {family_code}.",
                catalog_row=family_row,
                mapping_row=mapping_row,
            )

        return HTSCodeResolution(
            requested_code=canonicalize_hts_code(candidate_code),
            resolved_code="",
            validation_status="invalid",
            normalized_from=canonicalize_hts_code(candidate_code),
            resolution_basis=f"{canonicalize_hts_code(candidate_code)} is not present in the current HTS catalog or official code map.",
            catalog_row={},
            mapping_row=mapping_row,
        )

    def chapter_99_candidates_for(self, code: str) -> List[str]:
        current = canonicalize_hts_code(code)
        seen = set()
        values: List[str] = []
        while current:
            row = self.catalog_by_code.get(current)
            if row is not None:
                values.extend(
                    match
                    for match in _CHAPTER_99_RE.findall(_normalize_text(row.get("additional_duties")))
                    if match not in seen
                )
            current = _parent_code_for(current)

        candidates: List[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            candidates.append(value)
        return candidates

    def row_for_code(self, code: str) -> Dict[str, Any]:
        return dict(self.catalog_by_code.get(canonicalize_hts_code(code), {}))
