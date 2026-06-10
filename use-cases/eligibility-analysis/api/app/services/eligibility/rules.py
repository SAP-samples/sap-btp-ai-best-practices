"""
Eligibility Rules

Implementation of all eligibility rules for invoice analysis.
Each rule is a class with a check() method that returns a RejectionReason if failed.
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import List, Optional, Set

from ...config.eligibility_config import EligibilitySettings
from ...models.eligibility import (
    RULE_DESCRIPTIONS,
    OfferInvoice,
    RejectionReason,
    RuleCode,
    RuleDiagnostic,
)


class EligibilityRule(ABC):
    """Abstract base class for eligibility rules."""

    @property
    @abstractmethod
    def rule_code(self) -> RuleCode:
        """Return the rule code for this rule."""
        pass

    @abstractmethod
    def check(
        self,
        invoice: OfferInvoice,
        purchase_date: date,
        settings: EligibilitySettings,
        context: Optional[dict] = None,
    ) -> Optional[RejectionReason]:
        """
        Check if the invoice passes this rule.

        Args:
            invoice: The invoice to check
            purchase_date: The purchase date for calculations
            settings: Configuration settings with thresholds
            context: Optional shared context for stateful rules (e.g., duplicate tracking)

        Returns:
            RejectionReason if the rule fails, None if it passes
        """
        pass


def _format_date(value: date) -> str:
    return value.isoformat()


def _diagnostic(
    rule_code: RuleCode,
    bullets: List[str],
    invoice_values: Optional[dict] = None,
    settings_values: Optional[dict] = None,
    computed_values: Optional[dict] = None,
) -> RuleDiagnostic:
    return RuleDiagnostic(
        rule_code=rule_code,
        description=RULE_DESCRIPTIONS.get(rule_code, "Unknown rule"),
        bullets=bullets,
        invoice_values=invoice_values or {},
        settings_values=settings_values or {},
        computed_values=computed_values or {},
    )


class R1_DueDateRule(EligibilityRule):
    """
    Rule R1: Due Date - Purchase Date >= NDDT

    The due date must be at least NDDT days after the purchase date.
    This ensures there's enough time between purchase and when payment is due.
    """

    @property
    def rule_code(self) -> RuleCode:
        return RuleCode.R1

    def check(
        self,
        invoice: OfferInvoice,
        purchase_date: date,
        settings: EligibilitySettings,
        context: Optional[dict] = None,
    ) -> Optional[RejectionReason]:
        days_to_due = (invoice.due_date - purchase_date).days

        if days_to_due < settings.nddt:
            diagnostic = _diagnostic(
                rule_code=self.rule_code,
                bullets=[
                    f"Purchase date: {_format_date(purchase_date)}",
                    f"Due date: {_format_date(invoice.due_date)}",
                    f"Days to due date: {days_to_due} (minimum NDDT: {settings.nddt})",
                ],
                invoice_values={
                    "purchase_date": _format_date(purchase_date),
                    "due_date": _format_date(invoice.due_date),
                },
                settings_values={"nddt": settings.nddt},
                computed_values={"days_to_due": days_to_due},
            )
            return RejectionReason.from_rule(
                self.rule_code,
                details=f"Days to due date ({days_to_due}) < minimum required ({settings.nddt})",
                diagnostic=diagnostic,
            )
        return None


class R2_DuplicateInvoiceRule(EligibilityRule):
    """
    Rule R2: No duplicate (doc_number + fiscal_year + reference_number)

    Each combination of doc_number, fiscal_year, and reference_number must be unique
    within the batch being processed. This prevents double-funding of the same invoice.
    """

    @property
    def rule_code(self) -> RuleCode:
        return RuleCode.R2

    def check(
        self,
        invoice: OfferInvoice,
        purchase_date: date,
        settings: EligibilitySettings,
        context: Optional[dict] = None,
    ) -> Optional[RejectionReason]:
        if context is None:
            return None

        if not (invoice.doc_number and invoice.fiscal_year and invoice.reference_number):
            return None

        # Get or create the set of seen invoices
        seen_invoices: Set[str] = context.setdefault("seen_invoices", set())

        # Create unique key from doc_number, fiscal_year, and reference_number
        invoice_key = f"{invoice.doc_number}|{invoice.fiscal_year}|{invoice.reference_number}"

        if invoice_key in seen_invoices:
            diagnostic = _diagnostic(
                rule_code=self.rule_code,
                bullets=[
                    "Invoice key: doc number + fiscal year + reference number",
                    f"doc_number={invoice.doc_number}",
                    f"fiscal_year={invoice.fiscal_year}",
                    f"reference_number={invoice.reference_number}",
                    "Duplicate found within the current batch",
                    "Rule requires unique doc number + fiscal year + reference number per batch",
                ],
                invoice_values={
                    "doc_number": invoice.doc_number,
                    "fiscal_year": invoice.fiscal_year,
                    "reference_number": invoice.reference_number,
                },
            )
            return RejectionReason.from_rule(
                self.rule_code,
                details=(
                    "Duplicate invoice: doc number="
                    f"{invoice.doc_number}, fiscal year={invoice.fiscal_year}, "
                    f"reference number={invoice.reference_number}"
                ),
                diagnostic=diagnostic,
            )

        # Add to seen set for future checks
        seen_invoices.add(invoice_key)
        return None


class R11_EligibleCurrencyRule(EligibilityRule):
    """
    Rule R11: Currency in allowed list

    The invoice currency must be in the list of eligible currencies.
    Default eligible currencies are EUR and USD.
    """

    @property
    def rule_code(self) -> RuleCode:
        return RuleCode.R11

    def check(
        self,
        invoice: OfferInvoice,
        purchase_date: date,
        settings: EligibilitySettings,
        context: Optional[dict] = None,
    ) -> Optional[RejectionReason]:
        currency = invoice.original_currency.upper()

        if currency not in settings.eligible_currencies:
            diagnostic = _diagnostic(
                rule_code=self.rule_code,
                bullets=[
                    f"Invoice currency: {currency}",
                    f"Eligible currencies: {', '.join(settings.eligible_currencies)}",
                    "Rule requires the invoice currency to be in the eligible list",
                ],
                invoice_values={"currency": currency},
                settings_values={"eligible_currencies": settings.eligible_currencies},
            )
            return RejectionReason.from_rule(
                self.rule_code,
                details=f"Currency '{currency}' not in eligible list: {settings.eligible_currencies}",
                diagnostic=diagnostic,
            )
        return None


class R13_NotOverdueRule(EligibilityRule):
    """
    Rule R13: Due Date > Purchase Date (not overdue)

    The invoice must not be overdue at the time of purchase.
    The due date must be strictly after the purchase date.
    """

    @property
    def rule_code(self) -> RuleCode:
        return RuleCode.R13

    def check(
        self,
        invoice: OfferInvoice,
        purchase_date: date,
        settings: EligibilitySettings,
        context: Optional[dict] = None,
    ) -> Optional[RejectionReason]:
        if invoice.due_date <= purchase_date:
            diagnostic = _diagnostic(
                rule_code=self.rule_code,
                bullets=[
                    f"Purchase date: {_format_date(purchase_date)}",
                    f"Due date: {_format_date(invoice.due_date)}",
                    "Rule requires due date to be after purchase date",
                ],
                invoice_values={
                    "purchase_date": _format_date(purchase_date),
                    "due_date": _format_date(invoice.due_date),
                },
            )
            return RejectionReason.from_rule(
                self.rule_code,
                details=f"Invoice is overdue: due date={invoice.due_date}, purchase date={purchase_date}",
                diagnostic=diagnostic,
            )
        return None


class R16_TenorCheckRule(EligibilityRule):
    """
    Rule R16: Due Date - Issuance Date < TEIH

    The tenor (days from issuance to due date) must be less than TEIH days.
    This limits the maximum payment term of invoices.
    """

    @property
    def rule_code(self) -> RuleCode:
        return RuleCode.R16

    def check(
        self,
        invoice: OfferInvoice,
        purchase_date: date,
        settings: EligibilitySettings,
        context: Optional[dict] = None,
    ) -> Optional[RejectionReason]:
        tenor = (invoice.due_date - invoice.issuance_date).days

        if tenor >= settings.teih:
            diagnostic = _diagnostic(
                rule_code=self.rule_code,
                bullets=[
                    f"Issuance date: {_format_date(invoice.issuance_date)}",
                    f"Due date: {_format_date(invoice.due_date)}",
                    f"Tenor: {tenor} days (maximum TEIH: {settings.teih})",
                ],
                invoice_values={
                    "issuance_date": _format_date(invoice.issuance_date),
                    "due_date": _format_date(invoice.due_date),
                },
                settings_values={"teih": settings.teih},
                computed_values={"tenor_days": tenor},
            )
            return RejectionReason.from_rule(
                self.rule_code,
                details=f"Tenor ({tenor} days) >= maximum allowed ({settings.teih} days)",
                diagnostic=diagnostic,
            )
        return None


class R17_IssuancePurchaseRule(EligibilityRule):
    """
    Rule R17: Purchase Date - Issuance Date >= ISSPUR

    The invoice must be issued at least ISSPUR days before the purchase date.
    This ensures the invoice has been in existence for a minimum period.
    """

    @property
    def rule_code(self) -> RuleCode:
        return RuleCode.R17

    def check(
        self,
        invoice: OfferInvoice,
        purchase_date: date,
        settings: EligibilitySettings,
        context: Optional[dict] = None,
    ) -> Optional[RejectionReason]:
        days_since_issuance = (purchase_date - invoice.issuance_date).days

        if days_since_issuance < settings.isspur:
            diagnostic = _diagnostic(
                rule_code=self.rule_code,
                bullets=[
                    f"Issuance date: {_format_date(invoice.issuance_date)}",
                    f"Purchase date: {_format_date(purchase_date)}",
                    f"Days since issuance: {days_since_issuance} (minimum ISSPUR: {settings.isspur})",
                ],
                invoice_values={
                    "issuance_date": _format_date(invoice.issuance_date),
                    "purchase_date": _format_date(purchase_date),
                },
                settings_values={"isspur": settings.isspur},
                computed_values={"days_since_issuance": days_since_issuance},
            )
            return RejectionReason.from_rule(
                self.rule_code,
                details=f"Days since issuance ({days_since_issuance}) < minimum required ({settings.isspur})",
                diagnostic=diagnostic,
            )
        return None


# Registry of all rules in the order they should be checked
ALL_RULES: List[EligibilityRule] = [
    R13_NotOverdueRule(),  # Check overdue first (fundamental check)
    R1_DueDateRule(),  # Then days to due date
    R16_TenorCheckRule(),  # Tenor check
    R17_IssuancePurchaseRule(),  # Issuance timing
    R11_EligibleCurrencyRule(),  # Currency check
    R2_DuplicateInvoiceRule(),  # Duplicate check last (needs context)
]
