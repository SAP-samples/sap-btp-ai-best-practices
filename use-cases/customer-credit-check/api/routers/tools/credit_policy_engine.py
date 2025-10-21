
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Literal, Any
import json

# -------------------------------
# Constants derived from the policy
# -------------------------------

# Entity whitelist for auto-classification in classify_group.
GROUP_A_ENTITIES = {
    "Sample Foods S.A. de C.V.",
    "Sample Foods LLC",
    "Sample Ingredients S.A. de C.V.",
    "Sample Gluten Free Ingredients S.A. de C.V.",
    "Sample Gluten Free Jalisco S. de R.L. de C.V.",
}

# Entity whitelist for auto-classification in classify_group.
GROUP_B_ENTITIES = {
    "Sample Pastas de Occidente S.A. de C.V.",
}

# Days-late scoring matrix consumed by cp_score / compute_payment_scores.
CP_THRESHOLDS = {
    "A": [
        ("Excellent",  -10**9, -1, 10),
        ("Excellent",  0, 0, 10),
        ("Good",       1, 5, 8),
        ("Regular",    6, 10, 6),
        ("Poor",       11, 15, 4),
        ("Critical",   16, 10**9, 0),
    ],
    "B": [
        ("Excellent",  -10**9, -1, 10),
        ("Excellent",  0, 0, 10),
        ("Good",       1, 3, 8),
        ("Regular",    4, 6, 6),
        ("Poor",       7, 9, 4),
        ("Critical",   10, 10**9, 0),
    ],
}

# CH percentage to CAL label mapping used in compute_payment_scores.
CAL_THRESHOLDS = [
    ("Excellent", 95, 100.0001),
   ("Good", 80, 95),
    ("Regular", 65, 80),
    ("Poor", 50, 65),
    ("Critical", -1, 50),
]

# Role/persona caps for new credit approvals in within_role_max_new.
ROLE_MAX_NEW = {
    "analyst": {"PF": {"MXN": 620_000, "USD": 31_000, "EUR": 34_000},
                "PM": {"MXN": 1_050_000, "USD": 52_000, "EUR": 58_000}},
    "coordinator": {"PF": {"MXN": 1_250_000, "USD": 62_000, "EUR": 69_000},
                    "PM": {"MXN": 2_100_000, "USD": 105_000, "EUR": 138_000}},
}

# Role/CAL caps for update approvals in within_role_max_update.
ROLE_MAX_UPDATE = {
    "analyst": {
        "Excellent": {"MXN": 1_250_000, "USD": 62_000, "EUR": 69_000},
        "Good":      {"MXN": 1_250_000, "USD": 62_000, "EUR": 69_000},
        "Regular":   {"MXN": 620_000,  "USD": 31_000, "EUR": 34_000},
        "Poor":      {"MXN": 0, "USD": 0, "EUR": 0},
        "Critical":  {"MXN": 0, "USD": 0, "EUR": 0},
    },
    "coordinator": {
        "Excellent": {"MXN": 2_600_000, "USD": 130_000, "EUR": 144_000},
        "Good":      {"MXN": 2_600_000, "USD": 130_000, "EUR": 144_000},
        "Regular":   {"MXN": 1_550_000, "USD": 78_000, "EUR": 85_000},
        "Poor":      {"MXN": 0, "USD": 0, "EUR": 0},
        "Critical":  {"MXN": 0, "USD": 0, "EUR": 0},
    },
}

# Role/CAL caps for exception approvals in check_exception_caps.
ROLE_MAX_EXCEPTION = {
    "analyst": {
        "Excellent": {"MXN": 620_000,  "USD": 31_000, "EUR": 34_000},
        "Good":      {"MXN": 620_000,  "USD": 31_000, "EUR": 34_000},
        "Regular":   {"MXN": 310_000,  "USD": 16_000, "EUR": 17_000},
        "Poor":      {"MXN": 0,        "USD": 0,      "EUR": 0},
        "Critical":  {"MXN": 0,        "USD": 0,      "EUR": 0},
    },
    "coordinator": {
        "Excellent": {"MXN": 1_250_000, "USD": 62_000, "EUR": 69_000},
        "Good":      {"MXN": 1_250_000, "USD": 62_000, "EUR": 69_000},
        "Regular":   {"MXN": 620_000,   "USD": 31_000, "EUR": 34_000},
        "Poor":      {"MXN": 310_000,   "USD": 16_000, "EUR": 17_000},
        "Critical":  {"MXN": 0,         "USD": 0,      "EUR": 0},
    },
}

# Trigger value for coordinator LA logic within check_la_caps.
VL = {"MXN": 320_000, "USD": 16_000, "EUR": 17_000}
# LA% tuples (analyst, coordinator<=VL, coordinator>VL) read by check_la_caps.
LA_CAPS = {
    "Excellent": (0.20, 1.00, 0.50),
    "Good":      (0.20, 1.00, 0.50),
    "Regular":   (0.15, 1.00, 0.30),
    "Poor":      (0.00, 0.00, 0.00),
    "Critical":  (0.00, 0.00, 0.00),
}

# Absolute exception caps by group/currency used in check_exception_caps.
EXC_ABS_CAPS = {
    "A": {"MXN": 1_250_000, "USD": 62_000, "EUR": 69_000},
    "B": {"MXN": 720_000, "USD": None, "EUR": None},
}

# Maximum term authority per role enforced in check_terms_authority.
TERMS_MAX = {"analyst": 32, "coordinator": 47}

# Used by months_between to translate day deltas into month fractions.
APPROX_DAYS_IN_MONTH = 30.0
# C3M window in compute_payment_scores when evaluating recent invoices.
C3M_LOOKBACK_DAYS = 92  # ≈3 months
# CP score to percentage conversion in compute_payment_scores.
CP_TO_PERCENT_MULTIPLIER = 10
# Weight sequence for CH calculation in compute_payment_scores.
CH_WEIGHT_SEQUENCE = [3, 6, 8, 10]
# Table D check for minimum upfront purchases or existing credit.
MIN_ADVANCE_PURCHASES_OR_ACTIVE = 3
# MX-specific pagaré / guarantor logic in check_table_d.
COUNTRY_MX = "MX"
# Persona PF pagaré validation thresholds in check_table_d.
PF_MIN_GUARANTORS_ALLOWED = 0
PF_GUARANTOR_THRESHOLD_MEDIUM = 620_000
PF_GUARANTOR_THRESHOLD_HIGH = 1_250_000
PF_MIN_GUARANTORS_MEDIUM = 1
PF_MIN_GUARANTORS_HIGH = 2
PF_GUARANTORS_INVALID_REASON = "Dato inválido de avales"
PF_GUARANTORS_MEDIUM_REASON = "Persona física: requiere 1 aval para 620k–<1.25M"
PF_GUARANTORS_HIGH_REASON = "Persona física: requiere 2 avales para ≥1.25M"
# Document recency limits enforced in check_docs_recency.
KYC_MAX_AGE_DAYS = 730  # 2 years
DOC_PROOF_MAX_AGE_DAYS = 92
TAX_CERT_MAX_AGE_DAYS = 92
# Minimum acceptable C3M percent in update eligibility.
C3M_REGULAR_MIN_PCT = 68
# Minimum days between update requests in update eligibility.
MIN_DAYS_BETWEEN_UPDATES = 90
# Exception request limiters in check_exception_caps.
EXCEPTION_MAX_MULTIPLIER = 2.0
MAX_EXCEPTIONS_PER_SEMESTER = 3

# Appendix C banding applied in derive_late_payment_reinstatement.
LATE_PAYMENT_BAND_RULES = [
    {
        "name": "15-<30",
        "lower_bound": 15,
        "upper_bound": 30,
        "lower_inclusive": False,
        "upper_inclusive": False,
        "requirements": [
            "Presentar nueva solicitud de crédito (sección 6.1.1).",
            "Actualizar investigación de crédito.",
            "Visita a las instalaciones del cliente por parte de Crédito y Cobranza (opcional).",
            "Si el CGV no contempla interés moratorio, actualizar los términos para incluirlo y obtener la firma del cliente.",
        ],
        "waiting_period_months": 0,
        "site_visit_required": False,
    },
    {
        "name": "30-<60",
        "lower_bound": 30,
        "upper_bound": 60,
        "lower_inclusive": True,
        "upper_inclusive": False,
        "requirements": [
            "Esperar 6 meses desde la liquidación de la última factura, el cliente debe permanecer activo y pagando por adelantado.",
            "Presentar nueva solicitud de crédito (sección 6.1.1).",
            "Compartir estados financieros actuales.",
            "Visita a las instalaciones del cliente por parte de Crédito y Cobranza (opcional).",
            "Si el CGV no contempla interés moratorio, actualizar los términos para incluirlo y obtener la firma del cliente.",
        ],
        "waiting_period_months": 6,
        "site_visit_required": False,
    },
    {
        "name": "60-<90",
        "lower_bound": 60,
        "upper_bound": 90,
        "lower_inclusive": True,
        "upper_inclusive": False,
        "requirements": [
            "Esperar 12 meses desde la liquidación de la última factura, el cliente debe permanecer activo y pagando por adelantado.",
            "Presentar nueva solicitud de crédito (sección 6.1.1).",
            "Compartir estados financieros actuales.",
            "Visita a las instalaciones del cliente por parte de Crédito y Cobranza.",
            "Si el CGV no contempla interés moratorio, actualizar los términos para incluirlo y obtener la firma del cliente.",
        ],
        "waiting_period_months": 12,
        "site_visit_required": True,
    },
]

# -------------------------------
# Data models
# -------------------------------

@dataclass
class Invoice:
    invoice_id: str
    issue_date: datetime
    due_date: datetime
    paid_date: Optional[datetime]
    amount: float
    currency: str = "MXN"

@dataclass
class CustomerInput:
    customer_id: str
    legal_name: str
    persona: Literal["PF","PM"]
    country: str
    entity_name: Optional[str] = None
    customer_group: Optional[Literal["A","B"]] = None
    cgv_signed_date: Optional[datetime] = None
    pagare_signed: bool = False
    guarantors: int = 0
    insurance_full_credit: bool = False

@dataclass
class DocsInput:
    kyc_date: Optional[datetime] = None
    seller_comments_present: bool = True
    address_proof_date: Optional[datetime] = None
    tax_cert_date: Optional[datetime] = None

@dataclass
class CreditRequest:
    use_case: Literal["new","update","exception"]
    requested_amount: float
    requested_currency: str = "MXN"
    requested_terms_days: int = 30
    last_update_date: Optional[datetime] = None
    current_credit_line: float = 0.0
    current_credit_currency: str = "MXN"

@dataclass
class Investigation:
    mmr_amount: Optional[float] = None
    mmr_currency: str = "MXN"
    legal_risk: Optional[Literal["low","medium","high"]] = None
    external_investigation_date: Optional[datetime] = None
    onsite_visit_done: bool = False

@dataclass
class BehaviorInput:
    invoices: List[Invoice]
    has_overdue_invoices: bool = False
    advance_purchases_count: int = 0
    has_active_credit: bool = False
    exceptions_in_semester: int = 0

@dataclass
class RoleContext:
    role: Literal["analyst","coordinator"] = "analyst"


# -------------------------------
# Helpers
# -------------------------------

def months_between(a: datetime, b: datetime) -> float:
    return abs((a.year - b.year) * 12 + (a.month - b.month) + (a.day - b.day)/APPROX_DAYS_IN_MONTH)

def classify_group(entity_name: Optional[str], explicit_group: Optional[str]) -> str:
    if explicit_group in ("A","B"):
        return explicit_group
    if entity_name in GROUP_A_ENTITIES:
        return "A"
    if entity_name in GROUP_B_ENTITIES:
        return "B"
    return "A"

def cp_score(days_late: int, group: str) -> Dict[str, Any]:
    for label, lo, hi, score in CP_THRESHOLDS[group]:
        if lo <= days_late <= hi:
            return {"label": label, "score": score}
    return {"label": "Critical", "score": 0}

def compute_payment_scores(invoices: List[Invoice], group: str, as_of: Optional[datetime]=None) -> Dict[str, Any]:
    """
    Compute payment behavior scores based on invoice payment history.
    
    This function calculates various payment metrics including:
    - CP (Credit Payment) scores per invoice
    - CA (Credit Average) by year
    - C3M (Credit 3 Months) recent performance
    - CH (Credit Historical) weighted average
    - CAL (Credit Assessment Level) classification
    """
    # Set default reference date to current time if not provided
    if as_of is None:
        as_of = datetime.now()
    
    # Calculate CP (Credit Payment) scores for each invoice
    cp_entries = []
    for inv in invoices:
        # Calculate days late: positive if paid after due date, negative if paid early
        if inv.paid_date is not None:
            days_late = (inv.paid_date.date() - inv.due_date.date()).days
        else:
            # For unpaid invoices, calculate days late from reference date
            days_late = (as_of.date() - inv.due_date.date()).days
        
        # Get CP score based on days late and entity group (A or B)
        s = cp_score(days_late, group)
        cp_entries.append({"invoice_id": inv.invoice_id, "days_late": days_late, **s})
    
    # Group CP scores by year for CA (Credit Average) calculation
    by_year: Dict[int, list] = {}
    for e in cp_entries:
        # Find corresponding invoice to get due date year
        inv = next(i for i in invoices if i.invoice_id == e["invoice_id"])
        y = inv.due_date.year
        by_year.setdefault(y, []).append(e["score"])
    
    # Helper function to convert score to percentage (multiply by 10)
    def to_pct(x): return x * CP_TO_PERCENT_MULTIPLIER
    
    # Calculate C3M (Credit 3 Months) - average score for invoices due within last 92 days
    C3M_list = []
    for e in cp_entries:
        inv = next(i for i in invoices if i.invoice_id == e["invoice_id"])
        # Check if invoice was due within last 92 days (approximately 3 months)
        if (as_of.date() - inv.due_date.date()).days <= C3M_LOOKBACK_DAYS:
            C3M_list.append(e["score"])
    
    # Calculate C3M percentage: average of recent scores converted to percentage
    C3M = to_pct(sum(C3M_list)/len(C3M_list)) if C3M_list else None
    
    # Calculate CA (Credit Average) by year: average score per year converted to percentage
    CA = {y: to_pct(sum(v)/len(v)) for y,v in by_year.items()} if by_year else {}
    
    # Calculate CH (Credit Historical) - weighted average of up to 4 most recent years
    # Weights increase for more recent years: [3, 6, 8, 10]
    weights = CH_WEIGHT_SEQUENCE
    sorted_years = sorted(CA.keys())[-4:]  # Get up to 4 most recent years
    
    if not sorted_years:
        CH = None
    else:
        # Select years and their CA values
        selected = [(y, CA[y]) for y in sorted_years]
        # Use weights for the selected years (most recent get higher weights)
        w = weights[-len(selected):]
        # Calculate weighted average
        num = sum(val * w[i] for i, (_, val) in enumerate(selected))
        den = sum(w)
        CH = num/den if den else None
    
    # Helper function to classify CAL (Credit Assessment Level) based on CH percentage
    def classify_cal(value: Optional[float]) -> Optional[str]:
        if value is None:
            return None
        # Find the appropriate classification based on CAL_THRESHOLDS
        for label, lo, hi in CAL_THRESHOLDS:
            if lo <= value < hi:
                return label
        return "Critical"  # Default to Critical if no threshold matches
    
    # Return comprehensive payment behavior analysis
    return {
        "cp_by_invoice": cp_entries,      # Individual invoice CP scores and days late
        "CA_by_year_pct": CA,             # Average payment score by year (as percentage)
        "C3M_pct": C3M,                   # Recent 3-month payment performance (as percentage)
        "CH_pct": CH,                     # Historical weighted average (as percentage)
        "CAL": classify_cal(CH),          # Credit Assessment Level classification
    }

def check_table_d(ci: CustomerInput, docs: DocsInput, cr: CreditRequest, inv: Investigation, beh: BehaviorInput, group: str, now: Optional[datetime]=None) -> Dict[str, Any]:
    now = now or datetime.now()
    findings = {}
    if inv.mmr_amount is None:
        findings["commercial_investigation"] = {"ok": False, "reason": "Sin MMR externo"}
    else:
        ok = inv.mmr_amount >= cr.requested_amount
        findings["commercial_investigation"] = {"ok": ok, "reason": None if ok else f"MMR {inv.mmr_amount} < crédito solicitado {cr.requested_amount}"}
    ok_ap = beh.advance_purchases_count >= MIN_ADVANCE_PURCHASES_OR_ACTIVE or beh.has_active_credit
    findings["advance_purchases_or_active"] = {"ok": ok_ap, "reason": None if ok_ap else f"Se requieren ≥{MIN_ADVANCE_PURCHASES_OR_ACTIVE} compras de contado o crédito activo"}
    if inv.legal_risk is None:
        findings["legal_investigation"] = {"ok": False, "reason": "Sin resultado de investigación legal"}
    else:
        ok_legal = inv.legal_risk in ("low","medium")
        findings["legal_investigation"] = {"ok": ok_legal, "reason": None if ok_legal else f"Riesgo legal {inv.legal_risk} no permitido"}
    pagare_ok = True
    reason_pag = None
    if ci.country.upper() == COUNTRY_MX:
        if not ci.insurance_full_credit:
            if not ci.pagare_signed:
                pagare_ok = False
                reason_pag = "Pagaré requerido salvo seguro 100%"
            else:
                if ci.persona == "PF":
                    if ci.guarantors < PF_MIN_GUARANTORS_ALLOWED:
                        pagare_ok = False
                        reason_pag = PF_GUARANTORS_INVALID_REASON
                    elif cr.requested_amount >= PF_GUARANTOR_THRESHOLD_HIGH and ci.guarantors < PF_MIN_GUARANTORS_HIGH:
                        pagare_ok = False
                        reason_pag = PF_GUARANTORS_HIGH_REASON
                    elif cr.requested_amount >= PF_GUARANTOR_THRESHOLD_MEDIUM and ci.guarantors < PF_MIN_GUARANTORS_MEDIUM:
                        pagare_ok = False
                        reason_pag = PF_GUARANTORS_MEDIUM_REASON
        else:
            pagare_ok = True
    findings["pagare"] = {"ok": pagare_ok, "reason": reason_pag}
    ok_cgv = ci.cgv_signed_date is not None
    findings["cgv_signed"] = {"ok": ok_cgv, "reason": None if ok_cgv else "Falta firma de CGV"}
    return findings

def check_docs_recency(docs: DocsInput, now: Optional[datetime]=None) -> Dict[str, Any]:
    now = now or datetime.now()
    res = {}
    if docs.kyc_date is None:
        res["kyc"] = {"ok": False, "reason": "Sin KYC (FO-FIN-006)"}
    else:
        ok = abs((now - docs.kyc_date).days) <= KYC_MAX_AGE_DAYS
        res["kyc"] = {"ok": ok, "reason": None if ok else "KYC con antigüedad > 2 años"}
    if docs.address_proof_date is None:
        res["address_proof"] = {"ok": False, "reason": "Sin comprobante de domicilio"}
    else:
        ok = abs((now - docs.address_proof_date).days) <= DOC_PROOF_MAX_AGE_DAYS
        res["address_proof"] = {"ok": ok, "reason": None if ok else "Comprobante de domicilio > 3 meses"}
    if docs.tax_cert_date is None:
        res["tax_cert"] = {"ok": False, "reason": "Sin constancia de situación fiscal"}
    else:
        ok = abs((now - docs.tax_cert_date).days) <= TAX_CERT_MAX_AGE_DAYS
        res["tax_cert"] = {"ok": ok, "reason": None if ok else "Constancia de situación fiscal > 3 meses"}
    res["seller_comments"] = {"ok": bool(docs.seller_comments_present), "reason": None if docs.seller_comments_present else "Falta FO-FIN-007"}
    return res

def derive_late_payment_reinstatement(invoices: List[Invoice], scores: Dict[str, Any], as_of: datetime) -> Dict[str, Any]:
    """Build Appendix C late-payment reinstatement guidance based on delinquency bands.

    Bands considered (days late based on current `as_of` reference):
    - >15 and <30
    - >=30 and <60
    - >=60 and <90

    The function returns a structured object with:
    - band: textual band identifier or "none" when not applicable
    - max_days_late: the maximum observed days late across invoices (>=0)
    - months_since_last_settlement: months between `as_of` and the most recent paid_date
    - requirements: human-readable list of required steps per Appendix C
    - checks: automatic validations when possible (e.g., waiting period satisfied)
    """
    cp_list = scores.get("cp_by_invoice") or []
    max_days_late = 0
    for e in cp_list:
        try:
            d = int(e.get("days_late", 0))
        except Exception:
            d = 0
        if d > max_days_late:
            max_days_late = d

    # Determine delinquency band
    band = "none"
    band_rule = None
    for rule in LATE_PAYMENT_BAND_RULES:
        lower_ok = max_days_late >= rule["lower_bound"] if rule.get("lower_inclusive") else max_days_late > rule["lower_bound"]
        upper_bound = rule.get("upper_bound")
        if upper_bound is None:
            upper_ok = True
        else:
            upper_ok = max_days_late <= upper_bound if rule.get("upper_inclusive") else max_days_late < upper_bound
        if lower_ok and upper_ok:
            band = rule["name"]
            band_rule = rule
            break

    # Determine months since last settlement (most recent paid_date)
    last_paid: Optional[datetime] = None
    for inv in invoices:
        if inv.paid_date is not None:
            if (last_paid is None) or (inv.paid_date > last_paid):
                last_paid = inv.paid_date
    months_since_last_settlement: Optional[float] = None
    if last_paid is not None:
        months_since_last_settlement = months_between(as_of, last_paid)

    requirements: List[str] = []
    checks: Dict[str, Any] = {}

    if band_rule:
        requirements = band_rule["requirements"]
        required_wait = band_rule["waiting_period_months"]
        checks["waiting_period_required_months"] = required_wait
        if required_wait == 0:
            checks["waiting_period_ok"] = True
        else:
            checks["waiting_period_ok"] = (months_since_last_settlement is None) or (months_since_last_settlement >= required_wait)
        checks["site_visit_required"] = band_rule["site_visit_required"]
    else:
        requirements = []
        checks["waiting_period_required_months"] = 0
        checks["waiting_period_ok"] = True
        checks["site_visit_required"] = False

    return {
        "band": band,
        "max_days_late": max_days_late,
        "months_since_last_settlement": months_since_last_settlement,
        "requirements": requirements,
        "checks": checks,
    }

def within_role_max_new(role: str, persona: str, currency: str, amount: float) -> Dict[str, Any]:
    cap = ROLE_MAX_NEW.get(role, {}).get(persona, {}).get(currency, 0)
    ok = amount <= cap
    return {"ok": ok, "cap": cap, "reason": None if ok else f"Excede tope {role}/{persona} ({cap} {currency})"}

def within_role_max_update(role: str, cal: str, currency: str, amount: float) -> Dict[str, Any]:
    cap = ROLE_MAX_UPDATE.get(role, {}).get(cal or "Critical", {}).get(currency, 0)
    ok = amount <= cap
    return {"ok": ok, "cap": cap, "reason": None if ok else f"Excede tope {role}/{cal} ({cap} {currency})"}

def check_la_caps(role: str, cal: str, currency: str, current_line: float, requested_line: float) -> Dict[str, Any]:
    # LA_CAPS order per CLA: (analyst_any, coordinator_if_current_le_VL, coordinator_if_current_gt_VL)
    # - Analysts: fixed % by CLA; VL does not change their cap.
    # - Coordinators: 100% if current_line <= VL; else 50% (Exc/Good) or 30% (Regular); 0% for Poor/Critical.
    caps = LA_CAPS.get(cal or "Critical", (0.0, 0.0, 0.0))
    if role == "analyst":
        pct_cap = caps[0]
    else:
        pct_cap = caps[1] if current_line <= VL[currency] else caps[2]

    max_increase = current_line * pct_cap
    max_allowed = current_line + max_increase
    ok = requested_line <= max_allowed
    return {
        "ok": ok,
        "pct_cap": pct_cap,
        "max_allowed": max_allowed,
        "reason": None if ok else f"Incremento solicitado excede LA% ({pct_cap*100:.0f}%)",
    }

def check_terms_authority(role: str, requested_terms_days: int) -> Dict[str, Any]:
    cap = TERMS_MAX.get(role, 0)
    ok = requested_terms_days <= cap
    return {"ok": ok, "cap_days": cap, "reason": None if ok else f"Plazo solicitado {requested_terms_days} > {cap} días (requiere Director)"}

def check_exception_caps(group: str, cal: str, role: str, currency: str, current_line: float, requested_amount: float, exceptions_in_semester: int, has_overdue: bool) -> Dict[str, Any]:
    res = {}
    res["no_overdue"] = {"ok": not has_overdue, "reason": None if not has_overdue else "Cliente con vencidos"}
    ok_overage = requested_amount <= current_line * EXCEPTION_MAX_MULTIPLIER
    overage_limit_pct = (EXCEPTION_MAX_MULTIPLIER - 1) * 100
    res["overage_le_100pct"] = {
        "ok": ok_overage,
        "reason": None if ok_overage else f"Excede +{overage_limit_pct:.0f}% sobre la línea actual",
    }
    abs_cap = EXC_ABS_CAPS[group][currency]
    ok_abs = True if abs_cap is None else (requested_amount <= abs_cap)
    res["absolute_cap"] = {"ok": ok_abs, "cap": abs_cap, "reason": None if ok_abs else f"Excede tope absoluto grupo {group}: {abs_cap} {currency}"}
    ok_sem = exceptions_in_semester < MAX_EXCEPTIONS_PER_SEMESTER
    res["max_3_per_semester"] = {
        "ok": ok_sem,
        "reason": None if ok_sem else f"Límite de {MAX_EXCEPTIONS_PER_SEMESTER} excepciones/semestre superado",
    }
    role_cap = ROLE_MAX_EXCEPTION.get(role, {}).get(cal or "Critical", {}).get(currency, 0)
    ok_role = requested_amount <= role_cap
    res["role_cap"] = {"ok": ok_role, "cap": role_cap, "reason": None if ok_role else f"Excede tope por rol/CAL ({role}/{cal})"}
    return res

def evaluate_credit(customer: CustomerInput, docs: DocsInput, request: CreditRequest, investigation: Investigation, behavior: BehaviorInput, role: RoleContext, as_of: Optional[datetime]=None) -> Dict[str, Any]:
    as_of = as_of or datetime.now()
    group = classify_group(customer.entity_name, customer.customer_group)
    scores = compute_payment_scores(behavior.invoices, group, as_of=as_of)
    table_d = check_table_d(customer, docs, request, investigation, behavior, group, now=as_of)
    docs_check = check_docs_recency(docs, now=as_of)
    results = {"use_case": request.use_case, "group": group, "as_of": as_of.isoformat()}
    results["scores"] = scores
    results["checks"] = {"table_d": table_d, "docs": docs_check}
    cal = scores["CAL"]
    c3m = scores["C3M_pct"]
    has_overdue = behavior.has_overdue_invoices
    # Always attach late-payment reinstatement guidance per Appendix C
    results["late_payment_reinstatement"] = derive_late_payment_reinstatement(
        invoices=behavior.invoices, scores=scores, as_of=as_of
    )

    if request.use_case == "new":
        role_ok = within_role_max_new(role.role, customer.persona, request.requested_currency, request.requested_amount)
        terms_ok = check_terms_authority(role.role, request.requested_terms_days)
        results["checks"]["new_credit"] = {
            "within_role_max": role_ok,
            "terms_authority": terms_ok,
        }
        results["decision_hint"] = {
            "needs_director": (not role_ok["ok"]) or (not terms_ok["ok"]) or (not table_d["commercial_investigation"]["ok"]),
            "notes": "Sigue aprobaciones según matriz; Director si excede topes/tenor o MMR<CS."
        }
    elif request.use_case == "update":
        elig = {}
        elig["cal_regular_or_better"] = {"ok": cal in ("Excellent","Good","Regular"), "reason": None if cal in ("Excellent","Good","Regular") else f"CAL={cal}"}
        if c3m is not None:
            ok_c3m = c3m >= C3M_REGULAR_MIN_PCT
            elig["c3m_regular_or_better"] = {"ok": ok_c3m, "reason": None if ok_c3m else f"C3M={c3m:.1f}% < {C3M_REGULAR_MIN_PCT}%"}
        else:
            elig["c3m_regular_or_better"] = {"ok": True, "reason": "Sin ventas últimos 3 meses"}
        elig["no_overdue"] = {"ok": not has_overdue, "reason": None if not has_overdue else "Con vencidos"}
        if request.last_update_date:
            ok_cad = abs((as_of - request.last_update_date).days) >= MIN_DAYS_BETWEEN_UPDATES
            elig["last_update_ge_3m"] = {"ok": ok_cad, "reason": None if ok_cad else f"Última actualización < {MIN_DAYS_BETWEEN_UPDATES//30} meses"}
        else:
            elig["last_update_ge_3m"] = {"ok": True, "reason": "Sin registro de actualización previa"}
        la = check_la_caps(role.role, cal or "Critical", request.requested_currency, request.current_credit_line, request.requested_amount)
        role_ok = within_role_max_update(role.role, cal or "Critical", request.requested_currency, request.requested_amount)
        terms_ok = check_terms_authority(role.role, request.requested_terms_days)
        results["checks"]["update_terms"] = {
            "eligibility": elig,
            "la_caps": la,
            "within_role_max": role_ok,
            "terms_authority": terms_ok,
        }
        results["decision_hint"] = {
            "needs_director": (not la["ok"]) or (not role_ok["ok"]) or (not terms_ok["ok"]),
            "notes": "Aplicar LA% y topes por rol/CAL; Director si excede."
        }
    elif request.use_case == "exception":
        elig = {}
        elig["cal_regular_or_better"] = {"ok": cal in ("Excellent","Good","Regular"), "reason": None if cal in ("Excellent","Good","Regular") else f"CAL={cal}"}
        elig["no_overdue"] = {"ok": not has_overdue, "reason": None if not has_overdue else "Con vencidos"}
        exc_caps = check_exception_caps(group, cal or "Critical", role.role, request.requested_currency, request.current_credit_line, request.requested_amount, behavior.exceptions_in_semester, behavior.has_overdue_invoices)
        results["checks"]["credit_exception"] = {
            "eligibility": elig,
            "exception_caps": exc_caps,
        }
        results["decision_hint"] = {
            "needs_director": (not exc_caps["role_cap"]["ok"]) or (not exc_caps["absolute_cap"]["ok"]) or (not exc_caps["overage_le_100pct"]["ok"]),
            "notes": "Excepción sujeta a topes por rol/CAL y límites absolutos; máx. 3/semestre."
        }
    else:
        results["checks"]["error"] = "use_case inválido"
    return results

# -------------------------------
# Demo
# -------------------------------

if __name__ == "__main__":
    now = datetime(2025, 9, 5)
    invoices = []
    for i in range(1, 9):
        issue = now - timedelta(days=30*i + 10)
        due = issue + timedelta(days=30)
        if i % 4 == 0:
            paid = due - timedelta(days=2)
        elif i % 4 == 1:
            paid = due + timedelta(days=2)
        elif i % 4 == 2:
            paid = due + timedelta(days=7)
        else:
            paid = due + timedelta(days=16)
        invoices.append(Invoice(invoice_id=f"INV{i:03d}", issue_date=issue, due_date=due, paid_date=paid, amount=50_000))
    from dataclasses import asdict
    customer = CustomerInput(
        customer_id="CUST001",
        legal_name="Cliente Demo SA de CV",
        persona="PF",
        country="MX",
        entity_name="Sample Foods S.A. de C.V.",
        cgv_signed_date=now - timedelta(days=30),
        pagare_signed=True,
        guarantors=1,
        insurance_full_credit=False,
    )
    docs = DocsInput(
        kyc_date=now - timedelta(days=300),
        seller_comments_present=True,
        address_proof_date=now - timedelta(days=40),
        tax_cert_date=now - timedelta(days=60),
    )
    request = CreditRequest(
        use_case="new",
        requested_amount=820_000,
        requested_currency="MXN",
        requested_terms_days=32,
    )
    investigation = Investigation(
        mmr_amount=1_200_000,
        mmr_currency="MXN",
        legal_risk="low",
        external_investigation_date=now - timedelta(days=10),
    )
    behavior = BehaviorInput(
        invoices=invoices,
        has_overdue_invoices=False,
        advance_purchases_count=3,
        has_active_credit=False,
        exceptions_in_semester=1,
    )
    role = RoleContext(role="analyst")
    res = evaluate_credit(customer, docs, request, investigation, behavior, role, as_of=now)
    print(json.dumps(res, indent=2, default=str))
