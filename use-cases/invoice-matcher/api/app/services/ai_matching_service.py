import logging
import math
import re
import json
import csv
import os
from typing import Any, Callable

from .csv_parser import parse_csv, parse_amount

logger = logging.getLogger(__name__)


def save_csv(filepath: str, rows: list[dict], fieldnames: list[str] | None = None):
    if not rows:
        with open(filepath, "w") as f:
            f.write("")
        return
    if not fieldnames:
        fieldnames = list(rows[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_json(filepath: str, data: Any):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def save_text(filepath: str, text: str):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

NAME_MATCH_PROMPT = (
    "You are a payment-invoice matching assistant for a Japanese enterprise.\n\n"
    "TASK: Match bank PAYER identities to CUSTOMER identities. These represent the same entities but are recorded differently:\n"
    "- Payer info comes from bank transfers (romanized Japanese, abbreviated, split across fields)\n"
    "- Customer info comes from invoices (formal company names, addresses)\n\n"
    "MATCHING SIGNALS (use ANY combination):\n"
    "- Name similarity: (KA)=株式会社, field-width splits (e.g. 'SHIM OHARA'='SHIMOHARA'), romanization variants (OU=O, SHI=SI, CHI=TI, TSU=TU)\n"
    "- City/location: bank branch location matching customer city\n"
    "- Company suffixes: Corp, Co., Ltd., Inc., Holdings are interchangeable\n"
    "- Partial name overlap: even 3+ character substring matches are meaningful\n"
    "- If both payer and customer names are anonymized/placeholder (e.g. '____', 'COMPANY XXX'), match based on city or bank location\n\n"
    "A payer can match MULTIPLE customers (same company, different billing entities).\n"
    "Match aggressively — prefer LOW confidence over missing a real match.\n\n"
    "Respond ONLY with valid JSON:\n"
    '{"matches": [{"payerIdx": <0-based>, "customerIdx": <0-based>, '
    '"confidence": "HIGH"|"MEDIUM"|"LOW", "reason": "<explanation>"}]}\n'
    'If no matches found: {"matches": []}'
)

DIRECT_MATCH_PROMPT = (
    "You are a payment-invoice matching assistant for a Japanese enterprise.\n\n"
    "TASK: Given a list of bank PAYMENTS and a list of unmatched INVOICES, find which payment "
    "corresponds to which invoice.\n\n"
    "MANDATORY RULES:\n"
    "1. Amount MUST match within the stated tolerance — do NOT match if amounts differ by more than tolerance\n"
    "2. Once amounts match, confirm the pair using ANY of these signals:\n"
    "   - Payer name ↔ Customer name similarity (any substring overlap counts)\n"
    "   - Bank branch city ↔ Customer city match\n"
    "   - Bank name ↔ any customer identifier\n"
    "   - If names are anonymized/masked (e.g. '____', 'COMPANY XXX'), use city/bank location as primary signal\n"
    "3. Multiple invoices may sum to one payment (aggregated) — sum must be within tolerance\n\n"
    "IMPORTANT:\n"
    "- Text in {braces} is the romanized version of preceding katakana\n"
    "- (KA) or KA) suffix means 株式会社 (company)\n"
    "- Names may be split oddly due to field width limits\n"
    "- Romanization differences: OU=O, SHI=SI, CHI=TI, TSU=TU, FU=HU\n"
    "- Match aggressively — it's better to return a LOW confidence match than miss it entirely\n\n"
    "Confidence levels (amount within tolerance is ALWAYS required):\n"
    "- HIGH: amount within tolerance + clear name OR city/bank match\n"
    "- MEDIUM: amount within tolerance + partial name OR location overlap\n"
    "- LOW: amount within tolerance + weak signal (same region, similar structure)\n\n"
    "Do NOT return matches where amount exceeds tolerance, even if names match.\n\n"
    "Respond ONLY with valid JSON:\n"
    '{"matches": [{"paymentIdx": <0-based>, "invoiceIdx": <0-based or array for aggregated>, '
    '"confidence": "HIGH"|"MEDIUM"|"LOW", "reason": "<explanation>"}]}\n'
    'If no matches: {"matches": []}'
)


async def call_llm(system_prompt: str, user_prompt: str) -> dict:
    try:
        from gen_ai_hub.orchestration.models.llm import LLM
        from gen_ai_hub.orchestration.models.message import SystemMessage, UserMessage
        from gen_ai_hub.orchestration.models.template import Template, TemplateValue
        from gen_ai_hub.orchestration.service import OrchestrationService
    except ImportError:
        raise RuntimeError(
            "generative-ai-hub-sdk not available. Install it and configure SAP AI Core credentials."
        )

    llm = LLM(name="gpt-4o", version="latest", parameters={"max_tokens": 16000, "temperature": 0.0})
    template = Template(
        messages=[
            SystemMessage(system_prompt),
            UserMessage("{{?user_message}}"),
        ]
    )

    logger.info(f"[AI Match] LLM call started, prompt length: {len(user_prompt)} chars")

    import time
    t0 = time.time()

    response = OrchestrationService(llm=llm, template=template).run(
        template_values=[TemplateValue(name="user_message", value=user_prompt)]
    )

    content = response.orchestration_result.choices[0].message.content
    logger.info(f"[AI Match] LLM call completed in {(time.time() - t0) * 1000:.0f}ms")
    logger.info(f"[AI Match] Response length: {len(content)} chars")

    json_match = re.search(r"\{[\s\S]*\}", content)
    if not json_match:
        raise ValueError("LLM response did not contain valid JSON")
    return json.loads(json_match.group(0))


def normalize_for_comparison(name: str) -> str:
    if not name:
        return ""
    result = name.upper()
    result = re.sub(r"\s+", "", result)
    result = re.sub(r"[.,\-()（）株式会社有限会社]", "", result)
    result = re.sub(r"\(KA\)|KA\)|KABUSHIKI|KAISHA|CO\.?|LTD\.?|INC\.?|CORP\.?", "", result, flags=re.IGNORECASE)
    result = result.replace("OU", "O")
    result = result.replace("SHI", "SI")
    result = result.replace("CHI", "TI")
    result = result.replace("TSU", "TU")
    result = result.replace("FU", "HU")
    return result.strip()


def fuzzy_name_match(payer_name: str, customer_name: str) -> float:
    p_norm = normalize_for_comparison(payer_name)
    c_norm = normalize_for_comparison(customer_name)
    if not p_norm or not c_norm or len(p_norm) < 3 or len(c_norm) < 3:
        return 0

    if p_norm == c_norm:
        return 1.0

    if p_norm in c_norm or c_norm in p_norm:
        return 0.8

    shorter = p_norm if len(p_norm) <= len(c_norm) else c_norm
    longer = c_norm if len(p_norm) <= len(c_norm) else p_norm
    window_size = max(3, int(len(shorter) * 0.6))
    for i in range(len(shorter) - window_size + 1):
        segment = shorter[i:i + window_size]
        if segment in longer:
            return 0.5

    return 0


def city_match(bank_branch: str, city: str) -> bool:
    if not bank_branch or not city or city in ("__ __", "___") or len(city) < 3:
        return False
    b_up = bank_branch.upper()
    c_up = city.upper().replace(" ", "")
    if len(c_up) < 3:
        return False
    return c_up in b_up


def compute_match_score(payer_name: str, customer_name: str, bank_branch: str, city: str) -> float:
    name_score = fuzzy_name_match(payer_name, customer_name)
    loc_score = 0.4 if city_match(bank_branch, city) else 0
    return max(name_score, loc_score)


def extract_payer_info(p: dict) -> dict:
    full_text = (p.get("BY_ORD_OF_NAME", "") or "") + " " + (p.get("BY_ORD_OF_ADDR", "") or "")
    roman_match = re.search(r"\{([^}]+)\}", full_text)
    payer_name = roman_match.group(1).lstrip("0123456789").replace(",", "").strip() if roman_match else ""
    if not payer_name:
        payer_name = re.sub(r"^\d+", "", full_text)
        payer_name = re.sub(r"[｡-ﾟ]+", " ", payer_name)
        payer_name = re.sub(r"\{[^}]*\}", "", payer_name)
        payer_name = payer_name.replace(",", "").strip()

    bank_field = (p.get("BANK_NAME", "") or "") + " " + (p.get("BANK_ADDR", "") or "")
    bank_match = re.search(r"\{([^}]+)\}", bank_field)
    bank_name = bank_match.group(1).replace(",", "").strip() if bank_match else ""

    remit_name = (p.get("REMIT_NAME", "") or "").strip()

    return {
        "payerName": re.sub(r"\s+", " ", payer_name).strip(),
        "bankBranch": re.sub(r"\s+", " ", bank_name).strip(),
        "remitName": remit_name,
    }


def build_direct_prompt(payments: list, invoices: list, tolerance: float) -> str:
    lines = ["=== PAYMENTS ==="]
    for i, p in enumerate(payments):
        lines.append(
            f"{i}. {p['jpyAmount']:.2f} JPY | Payer: {p['payerName']}"
            f" | Bank: {p['bankBranch']} | Ref: {p['bankRef']}"
        )
    lines.append("")
    lines.append("=== UNMATCHED INVOICES ===")
    for j, inv in enumerate(invoices):
        cust_info = inv["customerName"]
        if inv.get("nameCo") and inv["nameCo"] != inv["customerName"]:
            cust_info += " / " + inv["nameCo"]
        lines.append(
            f"{j}. {inv['amount']:.2f} JPY | {inv['invoiceNumber']}"
            f" | Customer: {cust_info} | City: {inv['city']}"
        )
    lines.append("")
    lines.append(f"TOLERANCE: ±{tolerance:.2f} JPY.")
    lines.append("RULES:")
    lines.append("1. Amount difference MUST be within tolerance — reject if exceeded")
    lines.append("2. Confirm match using ANY signal: payer↔customer name, bank↔city, or bank↔customer location")
    lines.append("3. If multiple invoices have the same amount, use name/city/bank to pick the right one")
    lines.append("4. Match aggressively — LOW confidence is better than missing a match")
    return "\n".join(lines)


async def run_ai_matching(
    invoice_csv: str,
    payment_csv: str,
    unmatched_invoice_numbers: list[str],
    col_config: dict,
    on_progress: Callable[[dict], None],
    output_dir: str | None = None,
) -> list[dict]:
    import time
    import asyncio

    total_start = time.time()
    logger.info("=== Starting AI matching ===")
    logger.info(f"Invoice CSV size: {len(invoice_csv) / 1024:.1f} KB")
    logger.info(f"Payment CSV size: {len(payment_csv) / 1024:.1f} KB")
    logger.info(f"Unmatched invoices to process: {len(unmatched_invoice_numbers)}")

    t0 = time.time()
    invoice_parsed = parse_csv(invoice_csv)
    logger.info(f"Invoice CSV parsed in {(time.time() - t0) * 1000:.0f}ms ({len(invoice_parsed['rows'])} rows)")

    t0 = time.time()
    payment_parsed = parse_csv(payment_csv)
    logger.info(f"Payment CSV parsed in {(time.time() - t0) * 1000:.0f}ms ({len(payment_parsed['rows'])} rows)")

    invoice_rows = invoice_parsed["rows"]
    payment_rows = payment_parsed["rows"]

    on_progress({"phase": "preparing", "message": "Preparing data..."})

    t0 = time.time()
    unmatched_set = set(unmatched_invoice_numbers)

    invoice_number_col = col_config.get("invoiceNumber", "Invoice#")
    invoice_amount_col = col_config.get("invoiceAmount", "Invoice Amt")
    tolerance = col_config.get("tolerance", 0.01)

    unmatched_invoices = []
    for i, row in enumerate(invoice_rows):
        inv_no = (row.get(invoice_number_col) or "").strip()
        if inv_no not in unmatched_set:
            continue
        amt = parse_amount(row.get(invoice_amount_col))
        if math.isnan(amt):
            continue
        unmatched_invoices.append({
            "rowIndex": i,
            "invoiceNumber": inv_no,
            "amount": amt,
            "customerName": (row.get("Mc_name") or row.get("Name1") or "").strip(),
            "name1": (row.get("Name1") or "").strip(),
            "nameCo": (row.get("Name Co") or "").strip(),
            "city": (row.get("City1") or "").strip(),
            "soldTo": (row.get("SoldTo") or "").strip(),
            "street": (row.get("Street") or "").strip(),
            "postCode": (row.get("Post code1") or "").strip(),
        })
    logger.info(f"Unmatched invoices collected: {len(unmatched_invoices)} in {(time.time() - t0) * 1000:.0f}ms")

    if output_dir:
        save_csv(os.path.join(output_dir, "01_unmatched_invoices.csv"), unmatched_invoices)

    # Group by customer name
    t0 = time.time()
    customer_index: dict[str, dict] = {}
    for i, inv in enumerate(unmatched_invoices):
        key = inv["customerName"] or inv["soldTo"] or f"row_{i}"
        if key not in customer_index:
            customer_index[key] = {"name": inv["customerName"], "city": inv["city"], "invoices": []}
        customer_index[key]["invoices"].append(inv)

    # Collect all payments, group by payer
    payer_index: dict[str, dict] = {}
    for i, p in enumerate(payment_rows):
        raw_amt = parse_amount(p.get("TRANSACTION_AMT"))
        if math.isnan(raw_amt):
            continue
        info = extract_payer_info(p)
        payer_key = info["payerName"] or f"payer_{i}"

        if payer_key not in payer_index:
            payer_index[payer_key] = {"payerName": info["payerName"], "bankBranch": info["bankBranch"], "payments": []}
        payer_index[payer_key]["payments"].append({
            "rowIndex": i,
            "rawAmount": raw_amt,
            "jpyAmount": raw_amt / 100,
            "payerName": info["payerName"],
            "bankBranch": info["bankBranch"],
            "bankRef": (p.get("BANK_REF") or "").strip(),
            "remitName": info["remitName"],
        })

    payer_names = list(payer_index.keys())
    customer_names = list(customer_index.keys())
    logger.info(f"Grouped: {len(payer_names)} payers, {len(customer_names)} customers in {(time.time() - t0) * 1000:.0f}ms")

    if output_dir:
        payer_rows = [{"payerName": p_info["payerName"], "bankBranch": p_info["bankBranch"], "paymentCount": len(p_info["payments"])} for p_info in payer_index.values()]
        save_csv(os.path.join(output_dir, "02_payer_groups.csv"), payer_rows)
        customer_rows = [{"customerName": c_info["name"], "city": c_info["city"], "invoiceCount": len(c_info["invoices"])} for c_info in customer_index.values()]
        save_csv(os.path.join(output_dir, "03_customer_groups.csv"), customer_rows)

    on_progress({
        "phase": "filtering",
        "message": f"{len(payer_names)} unique payers, {len(customer_names)} unique customers. Finding amount overlaps...",
    })

    # Build invoice amount index
    t0 = time.time()
    tolerance_cents = round(tolerance * 100)

    inv_amt_to_customers: dict[int, set] = {}
    for c_name, c_info in customer_index.items():
        for inv in c_info["invoices"]:
            key = round(inv["amount"] * 100)
            if key not in inv_amt_to_customers:
                inv_amt_to_customers[key] = set()
            inv_amt_to_customers[key].add(c_name)

    inv_by_amount: dict[int, list[int]] = {}
    for i, inv in enumerate(unmatched_invoices):
        key = round(inv["amount"] * 100)
        if key not in inv_by_amount:
            inv_by_amount[key] = []
        inv_by_amount[key].append(i)

    # For each payer, find customers with overlapping amounts
    payer_candidates: dict[str, list[str]] = {}
    for p_name, p_info in payer_index.items():
        candidates: set = set()
        for payment in p_info["payments"]:
            jpy_key = round(payment["jpyAmount"] * 100)
            for delta in range(-tolerance_cents, tolerance_cents + 1):
                bucket = inv_amt_to_customers.get(jpy_key + delta)
                if bucket:
                    candidates.update(bucket)
        if candidates:
            payer_candidates[p_name] = list(candidates)

    logger.info(f"Amount overlap filtering done in {(time.time() - t0) * 1000:.0f}ms")

    payers_with_candidates = list(payer_candidates.keys())
    logger.info(f"Payers with candidates: {len(payers_with_candidates)}")

    if output_dir:
        overlap_rows = [{"payer": p, "candidates": ", ".join(cs)} for p, cs in payer_candidates.items()]
        save_csv(os.path.join(output_dir, "04_amount_overlap_candidates.csv"), overlap_rows)

    if not payers_with_candidates:
        logger.info("No amount overlap. Aborting.")
        on_progress({"phase": "complete", "message": "No amount overlap found between payers and customers", "matchCount": 0})
        return []

    # Local fuzzy matching
    t0 = time.time()
    all_results: list[dict] = []
    matched_invoice_nums: set = set()
    matched_bank_refs: set = set()

    for p_name, payer in payer_index.items():
        for payment in payer["payments"]:
            if payment["bankRef"] in matched_bank_refs:
                continue
            pmt_amt = payment["jpyAmount"]
            jpy_key = round(pmt_amt * 100)
            found = False
            for delta in range(-tolerance_cents, tolerance_cents + 1):
                if found:
                    break
                bucket = inv_by_amount.get(jpy_key + delta)
                if not bucket:
                    continue
                for bi in bucket:
                    inv = unmatched_invoices[bi]
                    if inv["invoiceNumber"] in matched_invoice_nums:
                        continue
                    diff = abs(pmt_amt - inv["amount"])
                    if diff > tolerance:
                        continue
                    score = compute_match_score(payer["payerName"], inv["customerName"], payer["bankBranch"], inv["city"])
                    if score >= 0.4:
                        matched_bank_refs.add(payment["bankRef"])
                        matched_invoice_nums.add(inv["invoiceNumber"])
                        conf = "HIGH" if score >= 0.8 else ("MEDIUM" if score >= 0.5 else "LOW")
                        reason = f"Name match (score={score:.2f}) + amount" if score >= 0.5 else "City/bank match + amount"
                        all_results.append({
                            "invoiceNumber": inv["invoiceNumber"],
                            "invoiceAmount": inv["amount"],
                            "paymentAmount": pmt_amt,
                            "bankRef": payment["bankRef"],
                            "matchStatus": "ai_matched",
                            "matched": True,
                            "confidence": conf,
                            "matchReason": reason,
                            "branchProximity": city_match(payer["bankBranch"], inv["city"]),
                            "payerName": payer["payerName"],
                            "customerName": inv["customerName"],
                        })
                        found = True
                        break
    logger.info(f"Local fuzzy matching done: {len(all_results)} matches in {(time.time() - t0) * 1000:.0f}ms")

    if output_dir:
        save_csv(os.path.join(output_dir, "05_local_fuzzy_matches.csv"), all_results)

    # Build candidate sets for LLM
    all_candidate_names = list(set(
        c for p in payers_with_candidates for c in payer_candidates[p]
    ))
    logger.info(f"Total unique candidate customers: {len(all_candidate_names)}")

    # Phase 1 prompt
    lines = ["=== PAYERS (from bank payments) ==="]
    payer_list = []
    for i, p_name in enumerate(payers_with_candidates):
        p_info = payer_index[p_name]
        payer_list.append(p_info)
        sample_amts = ", ".join(f"{p['jpyAmount']:.0f}" for p in p_info["payments"][:5])
        lines.append(
            f"{i}. Payer: {p_info['payerName']} | Bank: {p_info['bankBranch']}"
            f" | Payments: {len(p_info['payments'])} | Sample amounts: {sample_amts}"
        )

    lines.append("")
    lines.append("=== CUSTOMERS (from invoices) ===")
    customer_list = []
    for i, c_name in enumerate(all_candidate_names):
        c_info = customer_index[c_name]
        customer_list.append(c_info)
        cust_names = c_info["name"]
        inv0 = c_info["invoices"][0]
        if inv0.get("nameCo") and inv0["nameCo"] != c_info["name"]:
            cust_names += " / " + inv0["nameCo"]
        lines.append(
            f"{i}. Customer: {cust_names} | City: {c_info['city']}"
            f" | Invoices: {len(c_info['invoices'])}"
        )

    lines.append("")
    lines.append("Match each payer to the customer(s) they most likely are. A payer can match multiple customers.")
    lines.append("SIGNALS (use ANY combination — a match on ANY signal is valid):")
    lines.append("1. Name similarity (romanization, abbreviation, company suffix differences are expected)")
    lines.append("2. Bank branch city/region matching customer city")
    lines.append("3. If names are all masked/anonymized, use location as primary signal")
    lines.append("Match aggressively — mark as LOW confidence rather than omitting potential matches.")

    name_prompt = "\n".join(lines)
    logger.info(f"Phase 1 prompt: {len(payer_list)} payers × {len(customer_list)} customers ({len(name_prompt)} chars)")

    if output_dir:
        save_text(os.path.join(output_dir, "06_phase1_prompt.txt"), f"=== SYSTEM PROMPT ===\n{NAME_MATCH_PROMPT}\n\n=== USER PROMPT ===\n{name_prompt}")

    # Phase 2 prompt
    candidate_payments = []
    for p_name in payers_with_candidates:
        for payment in payer_index[p_name]["payments"]:
            if payment["bankRef"] in matched_bank_refs:
                continue
            jpy_key = round(payment["jpyAmount"] * 100)
            found_bucket = False
            for delta in range(-tolerance_cents, tolerance_cents + 1):
                if inv_by_amount.get(jpy_key + delta):
                    found_bucket = True
                    break
            if found_bucket:
                candidate_payments.append(payment)

    p2_inv_idx_set: set = set()
    for cp in candidate_payments:
        jpy_key = round(cp["jpyAmount"] * 100)
        for delta in range(-tolerance_cents, tolerance_cents + 1):
            bucket = inv_by_amount.get(jpy_key + delta)
            if bucket:
                for k in bucket:
                    if unmatched_invoices[k]["invoiceNumber"] not in matched_invoice_nums:
                        p2_inv_idx_set.add(k)

    p2_invoices = [unmatched_invoices[idx] for idx in p2_inv_idx_set]
    p2_payments = candidate_payments[:300]
    p2_invoices = p2_invoices[:300]

    direct_prompt = build_direct_prompt(p2_payments, p2_invoices, tolerance) if (p2_payments and p2_invoices) else None
    logger.info(
        f"Phase 2 prompt: {len(p2_payments)} payments × {len(p2_invoices)} invoices"
        f" ({len(direct_prompt)} chars)" if direct_prompt else f"Phase 2 prompt: skipped"
    )

    if output_dir and direct_prompt:
        save_text(os.path.join(output_dir, "08_phase2_prompt.txt"), f"=== SYSTEM PROMPT ===\n{DIRECT_MATCH_PROMPT}\n\n=== USER PROMPT ===\n{direct_prompt}")

    on_progress({
        "phase": "matching",
        "message": "Sending both LLM calls in parallel...",
        "totalPayers": 2,
        "completedPayers": 0,
    })

    # Fire both LLM calls in parallel
    async def safe_call_llm(system_prompt, user_prompt):
        try:
            return await call_llm(system_prompt, user_prompt)
        except Exception as e:
            logger.warning(f"LLM call failed: {e}. Retrying...")
            try:
                return await call_llm(system_prompt, user_prompt)
            except Exception:
                return {"matches": []}

    name_match_task = asyncio.create_task(safe_call_llm(NAME_MATCH_PROMPT, name_prompt))
    direct_match_task = asyncio.create_task(
        safe_call_llm(DIRECT_MATCH_PROMPT, direct_prompt) if direct_prompt else asyncio.coroutine(lambda: {"matches": []})()
    ) if direct_prompt else None

    name_result = await name_match_task
    direct_result = await direct_match_task if direct_match_task else {"matches": []}

    logger.info(
        f"Both LLM calls done. Phase 1: {len(name_result.get('matches', []))} name matches, "
        f"Phase 2: {len(direct_result.get('matches', []))} direct matches"
    )

    if output_dir:
        save_json(os.path.join(output_dir, "07_phase1_response.json"), name_result)
        save_json(os.path.join(output_dir, "09_phase2_response.json"), direct_result)

    on_progress({
        "phase": "linking",
        "message": "LLM calls complete. Linking invoices to payments...",
        "totalPayers": 2,
        "completedPayers": 2,
    })

    # Process Phase 1
    t0 = time.time()
    if name_result and name_result.get("matches"):
        for m in name_result["matches"]:
            p_idx = m.get("payerIdx", -1)
            c_idx = m.get("customerIdx", -1)

            if p_idx < 0 or p_idx >= len(payer_list):
                continue
            if c_idx < 0 or c_idx >= len(customer_list):
                continue

            payer = payer_list[p_idx]
            customer = customer_list[c_idx]
            confidence = m.get("confidence", "MEDIUM")
            reason = m.get("reason", "AI name match")

            logger.info(f"Name match: {payer['payerName']} → {customer['name']} ({confidence})")

            payer_payments = payer_index[payers_with_candidates[p_idx]]["payments"]
            cust_invoices = customer_index[all_candidate_names[c_idx]]["invoices"]

            used_payments: set = set()
            used_invoices: set = set()
            linked = 0

            for pi, pay in enumerate(payer_payments):
                if pi in used_payments or pay["bankRef"] in matched_bank_refs:
                    continue
                for ii, inv in enumerate(cust_invoices):
                    if ii in used_invoices or inv["invoiceNumber"] in matched_invoice_nums:
                        continue
                    diff = abs(pay["jpyAmount"] - inv["amount"])
                    if diff <= tolerance:
                        used_payments.add(pi)
                        used_invoices.add(ii)
                        matched_bank_refs.add(pay["bankRef"])
                        matched_invoice_nums.add(inv["invoiceNumber"])
                        linked += 1
                        all_results.append({
                            "invoiceNumber": inv["invoiceNumber"],
                            "invoiceAmount": inv["amount"],
                            "paymentAmount": pay["jpyAmount"],
                            "bankRef": pay["bankRef"],
                            "matchStatus": "ai_matched",
                            "matched": True,
                            "confidence": confidence,
                            "matchReason": reason,
                            "branchProximity": bool(
                                payer["bankBranch"] and customer["city"]
                                and customer["city"].upper() in payer["bankBranch"].upper()
                            ),
                            "payerName": payer["payerName"],
                            "customerName": customer["name"],
                        })
                        break
            logger.info(f"  Linked {linked} invoice-payment pairs")

    logger.info(f"Phase 1 linking done: {len(all_results)} matches in {(time.time() - t0) * 1000:.0f}ms")

    if output_dir:
        phase1_matches = [r for r in all_results if r.get("matchReason") != "AI direct match"]
        save_csv(os.path.join(output_dir, "10_phase1_linked_matches.csv"), phase1_matches)

    # Process Phase 2
    if direct_result and direct_result.get("matches"):
        for m in direct_result["matches"]:
            p_idx = m.get("paymentIdx", -1)
            i_idx = m.get("invoiceIdx")

            if isinstance(i_idx, int):
                if p_idx < 0 or p_idx >= len(p2_payments):
                    continue
                if i_idx < 0 or i_idx >= len(p2_invoices):
                    continue
                pay = p2_payments[p_idx]
                inv = p2_invoices[i_idx]
                if pay["bankRef"] in matched_bank_refs or inv["invoiceNumber"] in matched_invoice_nums:
                    continue

                amt_diff = abs(pay["jpyAmount"] - inv["amount"])
                if amt_diff > tolerance:
                    logger.info(f"Phase 2 rejected (diff {amt_diff:.2f} > {tolerance}): {inv['invoiceNumber']}")
                    continue

                matched_bank_refs.add(pay["bankRef"])
                matched_invoice_nums.add(inv["invoiceNumber"])
                all_results.append({
                    "invoiceNumber": inv["invoiceNumber"],
                    "invoiceAmount": inv["amount"],
                    "paymentAmount": pay["jpyAmount"],
                    "bankRef": pay["bankRef"],
                    "matchStatus": "ai_matched",
                    "matched": True,
                    "confidence": m.get("confidence", "MEDIUM"),
                    "matchReason": m.get("reason", "AI direct match"),
                    "branchProximity": bool(
                        pay["bankBranch"] and inv["city"]
                        and inv["city"].upper() in pay["bankBranch"].upper()
                    ),
                    "payerName": pay["payerName"],
                    "customerName": inv["customerName"],
                })

            elif isinstance(i_idx, list):
                if p_idx < 0 or p_idx >= len(p2_payments):
                    continue
                pay = p2_payments[p_idx]
                if pay["bankRef"] in matched_bank_refs:
                    continue

                sum_amt = 0
                all_valid = True
                for k in i_idx:
                    if k < 0 or k >= len(p2_invoices) or p2_invoices[k]["invoiceNumber"] in matched_invoice_nums:
                        all_valid = False
                        break
                    sum_amt += p2_invoices[k]["amount"]
                if not all_valid:
                    continue
                agg_diff = abs(pay["jpyAmount"] - sum_amt)
                if agg_diff > tolerance:
                    continue

                matched_bank_refs.add(pay["bankRef"])
                for k in i_idx:
                    inv = p2_invoices[k]
                    matched_invoice_nums.add(inv["invoiceNumber"])
                    all_results.append({
                        "invoiceNumber": inv["invoiceNumber"],
                        "invoiceAmount": inv["amount"],
                        "paymentAmount": pay["jpyAmount"],
                        "bankRef": pay["bankRef"],
                        "matchStatus": "ai_matched",
                        "matched": True,
                        "confidence": m.get("confidence", "LOW"),
                        "matchReason": (m.get("reason", "Aggregated payment") + " (partial)"),
                        "branchProximity": False,
                        "payerName": pay["payerName"],
                        "customerName": inv["customerName"],
                    })

    total_time = time.time() - total_start
    logger.info(f"=== Complete: {len(all_results)} matches in {total_time * 1000:.0f}ms ===")

    if output_dir:
        phase2_matches = [r for r in all_results if "direct match" in (r.get("matchReason") or "").lower() or "partial" in (r.get("matchReason") or "").lower()]
        save_csv(os.path.join(output_dir, "11_phase2_direct_matches.csv"), phase2_matches)
        save_csv(os.path.join(output_dir, "12_final_results.csv"), all_results)

    on_progress({
        "phase": "complete",
        "message": f"AI matching complete: {len(all_results)} invoices matched ({total_time:.1f}s)",
        "matchCount": len(all_results),
    })

    return all_results
