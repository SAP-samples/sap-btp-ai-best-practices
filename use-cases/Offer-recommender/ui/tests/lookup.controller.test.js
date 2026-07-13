import { beforeEach, describe, expect, it, vi } from "vitest";

import { initLookupPage } from "../src/pages/lookup/lookup.controller.js";

describe("lookup controller", () => {
  beforeEach(() => {
    document.body.innerHTML = `
      <input id="lookup-account-input" />
      <button id="lookup-submit"></button>
      <button id="lookup-recalculate"></button>
      <div id="lookup-summary"></div>
      <div id="lookup-questions"></div>
    `;
  });

  it("evaluates an account and renders the final offer with explanation", async () => {
    const api = {
      evaluateAccount: vi.fn().mockResolvedValue({
        billing_account: "104",
        customer_type: "RESIDENTIAL",
        final_offer: {
          display_name: "Plan Savings Review",
          confidence: "HIGH",
          explanation: {
            summary: "Offer-level reason: Manage Demand 5-10pm and Save lowers the estimated bill.",
            details: ["Current plan EZ-3(3-6) Residential Time of Use is estimated at $117.09."],
            facts_used: [
              "Current rate plan: EZ-3(3-6) Residential Time of Use.",
              "Recommended rate plan: Manage Demand 5-10pm and Save."
            ],
            rules_used: ["lower_estimated_bill"],
            blockers: [],
            source_documents: [],
            polish_status: "fallback"
          }
        },
        eligible_offers: [],
        blocked_offers: [
          {
            program_id: "income_qualified_discount",
            display_name: "Household Assistance Discount",
            confidence: "MEDIUM",
            explanation: {
              summary: "household assistance discount needs income and customer-of-record facts.",
              details: ["Missing facts: household_income_qualified."],
              facts_used: [],
              rules_used: ["missing_documented_eligibility_facts"],
              blockers: ["household_income_qualified"],
              source_documents: [],
              polish_status: "fallback"
            }
          }
        ],
        questions: [],
        facts: {},
        decision_trace: [],
        flags: [],
        explanation: {
          summary: "Plan Savings Review remains the best current fit.",
          facts_used: ["Estimated monthly cost comparison: current plan $146.83 versus recommended plan $130.59."],
          missing_core_facts: [],
          next_step: "No additional questions are required for the current recommendation."
        }
      })
    };

    await initLookupPage(api);
    document.getElementById("lookup-account-input").value = "104";
    document.getElementById("lookup-submit").click();
    await Promise.resolve();

    expect(api.evaluateAccount).toHaveBeenCalledWith("104", {}, []);
    expect(document.getElementById("lookup-summary").textContent).toContain("Plan Savings Review");
    expect(document.getElementById("lookup-summary").textContent).toContain(
      "Plan Savings Review remains the best current fit."
    );
    expect(document.getElementById("lookup-summary").textContent).toContain(
      "Offer-level reason: Manage Demand 5-10pm and Save lowers the estimated bill."
    );
    expect(document.getElementById("lookup-summary").textContent).toContain(
      "Current rate plan: EZ-3(3-6) Residential Time of Use."
    );
    expect(document.getElementById("lookup-summary").textContent).not.toContain("E16");
    expect(document.getElementById("lookup-summary").textContent).not.toContain("E21");
    expect(document.getElementById("lookup-summary").textContent).not.toContain(
      "Fact: Current rate plan"
    );
    expect(document.getElementById("lookup-summary").textContent).toContain(
      "household assistance discount needs income and customer-of-record facts."
    );
    expect(document.getElementById("lookup-summary").textContent).toContain("Facts Used");
  });

  it("loads saved answers before evaluating an account", async () => {
    const api = {
      getSavedAnswers: vi.fn().mockResolvedValue({
        billing_account: "6001",
        answers: {
          customer_wants_followup: true,
          account_name_type: "PERSONAL"
        }
      }),
      evaluateAccount: vi.fn().mockResolvedValue({
        billing_account: "6001",
        customer_type: "RESIDENTIAL",
        final_offer: {
          display_name: "Plan Savings Review",
          confidence: "HIGH"
        },
        eligible_offers: [],
        blocked_offers: [],
        questions: [],
        facts: {},
        decision_trace: [],
        flags: [],
        explanation: {
          summary: "Plan Savings Review remains the best current fit.",
          facts_used: [],
          missing_core_facts: [],
          next_step: null
        }
      })
    };

    await initLookupPage(api);
    document.getElementById("lookup-account-input").value = "6001";
    document.getElementById("lookup-submit").click();
    await Promise.resolve();
    await Promise.resolve();

    expect(api.getSavedAnswers).toHaveBeenCalledWith("6001");
    expect(api.evaluateAccount).toHaveBeenCalledWith(
      "6001",
      {
        customer_wants_followup: true,
        account_name_type: "PERSONAL"
      },
      []
    );
  });

  it("renders answerable questions before system blockers and preserves selected answers across recalculation", async () => {
    const api = {
      evaluateAccount: vi
        .fn()
        .mockResolvedValueOnce({
          billing_account: "103",
          customer_type: "RESIDENTIAL",
          final_offer: null,
          eligible_offers: [],
          blocked_offers: [],
          questions: [
            {
              expected_fact: "has_current_snapshot",
              prompt: "Retrieve a current billing snapshot for this account before making an offer decision.",
              source: "system",
              answer_options: []
            },
            {
              expected_fact: "payments_on_time",
              prompt: "Are this customer's payments currently being made on time?",
              source: "customer",
              explanation: {
                summary: "This answer determines whether payment-sensitive offers can be evaluated.",
                details: ["Candidate program: BYOT."],
                facts_used: [],
                rules_used: ["fact_payments_on_time"],
                blockers: ["payments_on_time"],
                source_documents: [],
                polish_status: "fallback"
              },
              answer_options: [
                { label: "Yes", value: true },
                { label: "No", value: false },
                { label: "Not sure / Prefer not to answer", value: null }
              ]
            }
          ],
          facts: {},
          decision_trace: [],
          flags: [],
          explanation: {
            summary: "No primary recommendation is ready yet because the account lacks a current billing snapshot.",
            facts_used: [],
            missing_core_facts: ["Current billing snapshot: no dated current billing snapshot is available for this account."],
            next_step: "Resolve the missing current billing snapshot before asking downstream program-specific questions."
          }
        })
        .mockResolvedValueOnce({
          billing_account: "103",
          customer_type: "RESIDENTIAL",
          final_offer: null,
          eligible_offers: [],
          blocked_offers: [],
          questions: [
            {
              expected_fact: "has_current_snapshot",
              prompt: "Retrieve a current billing snapshot for this account before making an offer decision.",
              source: "system",
              answer_options: []
            }
          ],
          facts: {},
          decision_trace: [],
          flags: [],
          explanation: {
            summary: "No primary recommendation is ready yet because the account lacks a current billing snapshot.",
            facts_used: [],
            missing_core_facts: ["Current billing snapshot: no dated current billing snapshot is available for this account."],
            next_step: "Resolve the missing current billing snapshot before asking downstream program-specific questions."
          }
        })
    };

    await initLookupPage(api);
    document.getElementById("lookup-account-input").value = "103";
    document.getElementById("lookup-submit").click();
    await Promise.resolve();

    expect(document.getElementById("lookup-questions").textContent).toContain(
      "Are this customer's payments currently being made on time?"
    );
    expect(document.getElementById("lookup-questions").textContent).not.toContain(
      "This answer determines whether payment-sensitive offers can be evaluated."
    );
    expect(document.getElementById("lookup-questions").textContent).not.toContain(
      "Retrieve a current billing snapshot for this account before making an offer decision."
    );

    const select = document.querySelector('[data-fact="payments_on_time"]');
    select.value = "true";
    document.getElementById("lookup-recalculate").click();
    await Promise.resolve();

    expect(api.evaluateAccount).toHaveBeenLastCalledWith(
      "103",
      { payments_on_time: true },
      []
    );
    expect(document.getElementById("lookup-questions").textContent).not.toContain(
      "Are this customer's payments currently being made on time?"
    );
    expect(document.getElementById("lookup-questions").textContent).toContain(
      "A current billing snapshot is needed before this recommendation can continue."
    );
  });

  it("renders questionless system steps and avoids PDF path evidence leakage in the summary", async () => {
    const api = {
      evaluateAccount: vi.fn().mockResolvedValue({
        billing_account: "103",
        customer_type: "RESIDENTIAL",
        final_offer: {
          display_name: "Household Assistance Discount",
          confidence: "HIGH"
        },
        eligible_offers: [],
        blocked_offers: [],
        questions: [
          {
            expected_fact: "has_current_snapshot",
            prompt: "Retrieve a current billing snapshot for this account before making an offer decision.",
            source: "system",
            answer_options: []
          }
        ],
        facts: {},
        decision_trace: [],
        flags: [],
        explanation: {
          summary: "Household Assistance Discount remains the best current fit based on the known account facts.",
          facts_used: [
            "Customer type: RESIDENTIAL.",
            "Decision basis: Deterministic tariff simulation."
          ],
          missing_core_facts: [],
          next_step: "No additional questions are required for the current recommendation."
        }
      })
    };

    await initLookupPage(api);
    document.getElementById("lookup-account-input").value = "103";
    document.getElementById("lookup-submit").click();
    await Promise.resolve();

    expect(document.getElementById("lookup-questions").textContent).toContain(
      "A current billing snapshot is needed before this recommendation can continue."
    );
    expect(document.getElementById("lookup-questions").textContent).not.toContain(
      "Retrieve a current billing snapshot for this account before making an offer decision."
    );
    expect(document.getElementById("lookup-questions").textContent).toContain(
      "This step requires a system or data update before the recommendation can continue."
    );
    expect(document.getElementById("lookup-summary").textContent).not.toContain(".pdf");
    expect(document.getElementById("lookup-summary").textContent).not.toContain("data/" + "Programs/");
  });

  it("clears structured answers when the lookup account changes", async () => {
    const prepayQuestion = {
      expected_fact: "prepay_advance_offers_this_month",
      prompt: "How many Prepay Advance offers has this customer already received this month?",
      source: "customer",
      answer_options: [
        { label: "0 (None)", value: 0 },
        { label: "1", value: 1 },
        { label: "2 or more", value: 2 }
      ]
    };
    const api = {
      evaluateAccount: vi.fn().mockResolvedValue({
        billing_account: "10106",
        customer_type: "RESIDENTIAL",
        final_offer: null,
        eligible_offers: [],
        blocked_offers: [],
        questions: [prepayQuestion],
        facts: {},
        decision_trace: [],
        flags: [],
        explanation: {
          summary: "No primary recommendation is ready yet.",
          facts_used: [],
          missing_core_facts: [],
          next_step: "Ask next."
        }
      })
    };

    await initLookupPage(api);
    document.getElementById("lookup-account-input").value = "10106";
    document.getElementById("lookup-submit").click();
    await Promise.resolve();

    document.querySelector('[data-fact="prepay_advance_offers_this_month"]').value = "1";
    document.getElementById("lookup-recalculate").click();
    await Promise.resolve();

    expect(api.evaluateAccount).toHaveBeenLastCalledWith(
      "10106",
      { prepay_advance_offers_this_month: 1 },
      []
    );

    document.getElementById("lookup-account-input").value = "105";
    document.getElementById("lookup-submit").click();
    await Promise.resolve();

    expect(api.evaluateAccount).toHaveBeenLastCalledWith("105", {}, []);
  });
});
