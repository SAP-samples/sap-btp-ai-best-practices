"""
Eligibility Engine

Orchestrates the eligibility analysis by applying all rules to invoices
and categorizing them as funded or non-funded.
"""

import logging
from datetime import date
from typing import List, Optional, Tuple

from ...config.eligibility_config import EligibilitySettings
from ...models.eligibility import (
    EligibilityResult,
    FundedInvoice,
    NonFundedInvoice,
    OfferInvoice,
    RuleCode,
)
from .rules import ALL_RULES, EligibilityRule

logger = logging.getLogger(__name__)


class EligibilityEngine:
    """
    Engine for processing invoice eligibility.

    This class orchestrates the eligibility analysis by:
    1. Applying all configured rules to each invoice
    2. Tracking shared context across invoices (for duplicate detection)
    3. Categorizing results into funded and non-funded invoices
    """

    def __init__(
        self,
        settings: EligibilitySettings,
        rules: Optional[List[EligibilityRule]] = None,
    ):
        """
        Initialize the eligibility engine.

        Args:
            settings: Configuration settings with rule thresholds
            rules: Optional list of rules to use (defaults to ALL_RULES)
        """
        self.settings = settings
        self.rules = rules if rules is not None else ALL_RULES

    def analyze_invoice(
        self,
        invoice: OfferInvoice,
        purchase_date: date,
        context: Optional[dict] = None,
    ) -> EligibilityResult:
        """
        Analyze a single invoice against all eligibility rules.

        Args:
            invoice: The invoice to analyze
            purchase_date: The purchase date for calculations
            context: Shared context for stateful rules (e.g., duplicate tracking)

        Returns:
            EligibilityResult with pass/fail status and details
        """
        passed_rules: List[RuleCode] = []
        failed_rules = []

        for rule in self.rules:
            rejection = rule.check(
                invoice=invoice,
                purchase_date=purchase_date,
                settings=self.settings,
                context=context,
            )

            if rejection is None:
                passed_rules.append(rule.rule_code)
            else:
                failed_rules.append(rejection)

        is_eligible = len(failed_rules) == 0

        return EligibilityResult(
            invoice_ref=invoice.invoice_ref,
            seller_id=invoice.seller_id,
            is_eligible=is_eligible,
            passed_rules=passed_rules,
            failed_rules=failed_rules,
        )

    def analyze_batch(
        self,
        invoices: List[OfferInvoice],
        purchase_date: date,
    ) -> Tuple[List[EligibilityResult], List[FundedInvoice], List[NonFundedInvoice]]:
        """
        Analyze a batch of invoices.

        This method processes all invoices in the batch, maintaining shared
        context for duplicate detection and other stateful rules.

        Args:
            invoices: List of invoices to analyze
            purchase_date: The purchase date for calculations

        Returns:
            Tuple of:
            - List of all EligibilityResult objects
            - List of FundedInvoice objects (passed all rules)
            - List of NonFundedInvoice objects (failed one or more rules)
        """
        logger.info(f"Analyzing batch of {len(invoices)} invoices")

        # Shared context for stateful rules (e.g., duplicate tracking)
        context: dict = {}

        results: List[EligibilityResult] = []
        funded: List[FundedInvoice] = []
        non_funded: List[NonFundedInvoice] = []

        for invoice in invoices:
            result = self.analyze_invoice(
                invoice=invoice,
                purchase_date=purchase_date,
                context=context,
            )
            results.append(result)

            if result.is_eligible:
                funded_invoice = FundedInvoice.from_invoice(
                    invoice=invoice,
                    purchase_date=purchase_date,
                )
                funded.append(funded_invoice)
            else:
                non_funded_invoice = NonFundedInvoice.from_invoice(
                    invoice=invoice,
                    purchase_date=purchase_date,
                    result=result,
                )
                non_funded.append(non_funded_invoice)

        logger.info(
            f"Batch analysis complete: {len(funded)} funded, {len(non_funded)} non-funded"
        )

        return results, funded, non_funded

    def get_rule_statistics(
        self,
        results: List[EligibilityResult],
    ) -> dict:
        """
        Calculate statistics about rule failures across results.

        Args:
            results: List of eligibility results to analyze

        Returns:
            Dictionary with rule failure counts and percentages
        """
        total = len(results)
        eligible_count = sum(1 for r in results if r.is_eligible)
        rejected_count = total - eligible_count

        # Count failures by rule
        rule_failures: dict = {}
        for result in results:
            for rejection in result.failed_rules:
                rule_code = rejection.rule_code.value
                rule_failures[rule_code] = rule_failures.get(rule_code, 0) + 1

        # Calculate percentages
        rule_stats = {}
        for rule_code, count in rule_failures.items():
            rule_stats[rule_code] = {
                "count": count,
                "percentage_of_rejections": (
                    round(count / rejected_count * 100, 2) if rejected_count > 0 else 0
                ),
                "percentage_of_total": round(count / total * 100, 2) if total > 0 else 0,
            }

        return {
            "total_invoices": total,
            "eligible_count": eligible_count,
            "rejected_count": rejected_count,
            "eligibility_rate": round(eligible_count / total * 100, 2) if total > 0 else 0,
            "rule_failures": rule_stats,
        }
