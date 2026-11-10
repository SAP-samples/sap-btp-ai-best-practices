---
name: deduction-skill-authoring
description: Create a customer-specific HANA deduction playbook from caller-authorized Word and Excel business-rule documents; use for authoring or evaluating stored customer logic, not for interpreting an incoming payment advice.
---

# Deduction Skill Authoring

Create one evidence-backed customer playbook that can be passed to
`save_deduction_rules(client_key, playbook_text, anchors, ...)`. In this workflow,
the generated customer playbook is the HANA **skill**. This file is the reusable
agent-local **agent skill** that explains how to author it.

## Read the evidence

1. Call `list_rule_sources()` once.
2. Read every listed file with `read_rule_source(source_name, cursor=0)`. When
   `has_more` is true, continue with `next_cursor` until the entire source is read.
3. Treat paragraph, table-row, sheet, and cell labels as evidence locators.
4. Give direct conversion rules and special-features documents priority over
   examples or generic mapping templates. Repetition corroborates a rule; it does
   not create a duplicate rule.
5. Keep customer accounts, supplier/vendor identifiers, company codes, branch
   accounts, and market names distinct.

Do not infer a business meaning for a reason code unless a source defines it. Do
not resolve contradictions silently. Record uncertain, missing, conflicting, or
apparently stale material in `open_questions`.

## Build the HANA skill

Return the requested structured object with `playbook_text` and `anchors`.

- `playbook_text`: concise natural-language instructions for the interpretation
  agent, including scope, precedence, defaults, and exceptional handling.
- `source_scope`: documented **incoming payment-advice** formats, sender patterns,
  and filename patterns. Never add the authoring DOCX/XLSX file type or evidence
  file name merely because that file was supplied as a source.
- `default_customer_account`: the documented default or `null`.
- `customer_account_rules`: one ordered rule per documented condition that selects
  among customer accounts. Keep a single repeated fixed account as
  `default_customer_account` and/or a field mapping; do not invent tautological
  account-selection conditions from repeated assignments.
- `default_company_code`: the documented fixed payee company code (e.g. the
  "ILN recipient / payee" constant `CA01`), or `null`.
- `company_code_rules`: one ordered rule per documented condition that selects the
  company code (e.g. a header field or country marker), same shape as account rules.
- `field_mappings`: source-to-target mappings needed to understand the document.
- `deduction_rules`: one ordered rule per condition. Preserve the source's
  `equals`, `starts_with`, `ends_with`, `contains`, sign, and fallback semantics.

Rules are evaluated deterministically by code on the extracted columns, so write
each rule's `field` as the column label printed on the remittance (e.g. `Rg`,
`Your Invoice No`, `Invoice Description`) or as a canonical field name
(`invoice_reference`, `deduction_reason`, `company_code`). Use only the operators
`equals`, `starts_with`, `ends_with`, `contains` or `regex`; express anything else
(compound conditions, amount splits, several residuals per line) in
`playbook_text` for the interpretation agent instead of as a rule.
- `default_rule`: the documented fallback. Use reason code `323` only when the
  source documents it or when explicitly described as the existing escalation
  sentinel; identify the latter as requiring confirmation.
- `processing_notes`: other operational instructions that affect interpretation.
- `evidence`: cite the source file and the most precise available locator for each
  rule group.
- `open_questions`: unresolved facts that must not be presented as confirmed logic.

Preserve rule order. Specific exceptions precede broader prefixes or descriptions,
and the default is last. Never turn an example-only pattern into a confirmed rule.

## Storage boundary

Generation produces a candidate only. Do not claim it is stored in HANA. A caller
may compare and review it, then use the existing staged
`save_deduction_rules(..., user_confirmed=False)` and explicit-confirmation flow to
persist it.
