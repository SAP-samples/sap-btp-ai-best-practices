import { beforeEach, describe, expect, it, vi } from "vitest";

import { initChatPage } from "../src/pages/chat/chat.controller.js";

describe("chat controller", () => {
  let storage;

  beforeEach(() => {
    storage = {
      data: new Map(),
      getItem(key) {
        return this.data.has(key) ? this.data.get(key) : null;
      },
      setItem(key, value) {
        this.data.set(key, value);
      },
      removeItem(key) {
        this.data.delete(key);
      }
    };
    document.body.innerHTML = `
      <div id="chat-status"></div>
      <div id="chat-primary-summary"></div>
      <div id="chat-explanation"></div>
      <div id="chat-guidance"></div>
      <div id="chat-additional-offers"></div>
      <div id="chat-messages"></div>
      <div id="chat-options"></div>
      <div id="chat-conversation-declines"></div>
      <div id="chat-summary-declines"></div>
      <textarea id="chat-input"></textarea>
      <button id="chat-send"></button>
      <button id="chat-new-thread"></button>
    `;
  });

  it("creates a thread and renders the initial assistant message", async () => {
    const api = {
      createChatThread: vi.fn().mockResolvedValue({
        thread_id: "thread-1",
        state: {
          thread_id: "thread-1",
          billing_account: null,
          customer_type: null,
          messages: [
            {
              role: "assistant",
              content: "Welcome to Customer Offer Advisor.",
              timestamp: "2026-04-17T00:00:00Z"
            }
          ],
          pending_questions: [],
          current_question: null,
          decision_result: null,
          declined_programs: [],
          status_phase: "awaiting_account",
          user_answers: {}
        }
      }),
      getChatThread: vi.fn(),
      postChatMessage: vi.fn(),
      declineChatProgram: vi.fn()
    };

    await initChatPage(api, storage);

    expect(api.createChatThread).toHaveBeenCalledTimes(1);
    expect(document.getElementById("chat-messages").textContent).toContain("Welcome to Customer Offer Advisor.");
    expect(document.getElementById("chat-status").textContent).toContain("awaiting_account");
    expect(document.getElementById("chat-additional-offers").textContent).toBe("");
  });

  it("renders primary and additional offers outside the transcript and keeps decline actions human-readable", async () => {
    const api = {
      createChatThread: vi.fn().mockResolvedValue({
        thread_id: "thread-1",
        state: {
          thread_id: "thread-1",
          billing_account: "6001",
          customer_type: "RESIDENTIAL",
          messages: [
            {
              role: "assistant",
              content: "I found a primary recommendation for this account.",
              timestamp: "2026-04-17T00:00:00Z"
            }
          ],
          pending_questions: [],
          current_question: {
            expected_fact: "eligible_connected_devices",
            prompt: "Does the customer have an eligible connected device?",
            explanation: {
              summary: "This answer checks whether connected-device programs can be evaluated.",
              details: ["Candidate program: BYOT."],
              facts_used: [],
              rules_used: ["fact_eligible_connected_devices"],
              blockers: ["eligible_connected_devices"],
              source_documents: [],
              polish_status: "fallback"
            },
            answer_options: [
              { label: "Yes", value: true },
              { label: "No", value: false }
            ]
          },
          decision_result: {
            final_offer: {
              program_id: "rate_plan_optimization",
              display_name: "Plan Savings Review",
              confidence: "HIGH",
              metadata: {
                estimated_monthly_savings: 16.24
              },
              explanation: {
                summary: "Offer-level reason: Manage Demand 5-10pm and Save lowers the estimated bill.",
                details: ["Current plan EZ-3(3-6) Residential Time of Use is estimated at $146.83."],
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
            eligible_offers: [
              {
                program_id: "rate_plan_optimization",
                display_name: "Plan Savings Review",
                confidence: "HIGH",
                metadata: {
                  estimated_monthly_savings: 16.24
                },
                explanation: {
                  summary: "Offer-level reason: Manage Demand 5-10pm and Save lowers the estimated bill.",
                  details: ["Current plan EZ-3(3-6) Residential Time of Use is estimated at $146.83."],
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
              {
                program_id: "income_qualified_discount",
                display_name: "Household Assistance Discount",
                confidence: "HIGH",
                metadata: {},
                explanation: {
                  summary: "household assistance discount is eligible after the supplied qualification facts.",
                  details: ["Documented eligibility logic is satisfied."],
                  facts_used: ["Customer type: RESIDENTIAL."],
                  rules_used: ["documented_eligibility_satisfied"],
                  blockers: [],
                  source_documents: [],
                  polish_status: "fallback"
                }
              }
            ],
            workflow_stage: "primary_offer_with_followup",
            explanation: {
              summary: "Plan Savings Review remains the best current fit.",
              facts_used: [
                "Estimated monthly cost comparison: current plan $146.83 versus recommended plan $130.59.",
                "Latest 3-bill average total usage: 911.67 kWh."
              ],
              missing_core_facts: [
                "Service charge tier: account details do not include the property or service information needed to determine this yet."
              ],
              next_step: "Use the next question to see whether more programs are now eligible."
            }
          },
          declined_programs: [],
          status_phase: "questioning",
          user_answers: {}
        }
      }),
      getChatThread: vi.fn(),
      postChatMessage: vi.fn(),
      declineChatProgram: vi.fn().mockResolvedValue({
        thread_id: "thread-1",
        billing_account: "6001",
        customer_type: "RESIDENTIAL",
        messages: [],
        pending_questions: [],
        current_question: null,
        decision_result: null,
        declined_programs: ["rate_plan_optimization"],
        status_phase: "complete",
        user_answers: {}
      })
    };

    await initChatPage(api, storage);

    expect(document.getElementById("chat-primary-summary").textContent).toContain(
      "Plan Savings Review"
    );
    expect(document.getElementById("chat-primary-summary").textContent).toContain(
      "Plan Savings Review remains the best current fit."
    );
    expect(document.getElementById("chat-primary-summary").textContent).toContain(
      "Offer-level reason: Manage Demand 5-10pm and Save lowers the estimated bill."
    );
    expect(document.getElementById("chat-primary-summary").textContent).toContain(
      "Current rate plan: EZ-3(3-6) Residential Time of Use."
    );
    expect(document.getElementById("chat-primary-summary").textContent).not.toContain("E16");
    expect(document.getElementById("chat-primary-summary").textContent).not.toContain("E21");
    expect(document.getElementById("chat-primary-summary").textContent).not.toContain(
      "Fact: Current rate plan"
    );
    expect(document.getElementById("chat-primary-summary").textContent).not.toContain(
      "lower_estimated_bill"
    );
    expect(document.getElementById("chat-guidance").textContent).toBe("");
    expect(document.getElementById("chat-explanation").textContent).toContain("Facts Used");
    expect(document.getElementById("chat-explanation").textContent).toContain(
      "Estimated monthly cost comparison: current plan $146.83 versus recommended plan $130.59."
    );
    expect(document.getElementById("chat-explanation").textContent).toContain(
      "Missing Core Facts"
    );
    expect(document.getElementById("chat-explanation").textContent).toContain(
      "Service charge tier"
    );
    expect(document.getElementById("chat-explanation").textContent).not.toContain(
      "Question Reason"
    );
    expect(document.getElementById("chat-explanation").textContent).not.toContain(
      "This answer checks whether connected-device programs can be evaluated."
    );
    expect(document.getElementById("chat-additional-offers").textContent).toContain(
      "Additional programs now eligible"
    );
    expect(document.getElementById("chat-additional-offers").textContent).toContain(
      "Household Assistance Discount"
    );
    expect(document.getElementById("chat-additional-offers").textContent).toContain(
      "household assistance discount is eligible after the supplied qualification facts."
    );
    expect(document.getElementById("chat-additional-offers").textContent).not.toContain(
      "Plan Savings Review"
    );
    expect(document.getElementById("chat-messages").textContent).not.toContain(
      "Plan Savings Review"
    );
    expect(document.getElementById("chat-summary-declines").textContent).toContain(
      "Remove an offer from this session:"
    );
    expect(document.getElementById("chat-summary-declines").textContent).toContain(
      "Remove Plan Savings Review for this session"
    );
    expect(document.getElementById("chat-summary-declines").textContent).toContain(
      "Remove Household Assistance Discount for this session"
    );
    expect(document.getElementById("chat-conversation-declines").textContent).toBe("");
  });

  it("does not render the additional-programs section before a recommendation exists", async () => {
    const api = {
      createChatThread: vi.fn().mockResolvedValue({
        thread_id: "thread-2",
        state: {
          thread_id: "thread-2",
          billing_account: null,
          customer_type: null,
          messages: [],
          pending_questions: [],
          current_question: null,
          decision_result: null,
          declined_programs: [],
          status_phase: "awaiting_account",
          user_answers: {}
        }
      }),
      getChatThread: vi.fn(),
      postChatMessage: vi.fn(),
      declineChatProgram: vi.fn()
    };

    await initChatPage(api, storage);

    expect(document.getElementById("chat-additional-offers").textContent).toBe("");
    expect(document.getElementById("chat-additional-offers").textContent).not.toContain(
      "Additional programs now eligible"
    );
    expect(document.getElementById("chat-explanation").textContent).toBe("");
  });

  it("blocks duplicate overlapping answer, send, and new-thread requests", async () => {
    let resolveMessage;
    const messagePromise = new Promise((resolve) => {
      resolveMessage = resolve;
    });

    const api = {
      createChatThread: vi
        .fn()
        .mockResolvedValueOnce({
          thread_id: "thread-3",
          state: {
            thread_id: "thread-3",
            billing_account: "6001",
            customer_type: "RESIDENTIAL",
            messages: [],
            pending_questions: [],
            current_question: {
              expected_fact: "payments_on_time",
              prompt: "Are payments on time?",
              answer_options: [
                { label: "Yes", value: true },
                { label: "No", value: false }
              ]
            },
            decision_result: {
              final_offer: {
                program_id: "rate_plan_optimization",
                display_name: "Plan Savings Review",
                confidence: "HIGH",
                metadata: {}
              },
              eligible_offers: [
                {
                  program_id: "rate_plan_optimization",
                  display_name: "Plan Savings Review",
                  confidence: "HIGH",
                  metadata: {}
                }
              ]
            },
            declined_programs: [],
            status_phase: "questioning",
            user_answers: {}
          }
        }),
      getChatThread: vi.fn(),
      postChatMessage: vi.fn().mockImplementation(() => messagePromise),
      declineChatProgram: vi.fn()
    };

    await initChatPage(api, storage);

    const answerButton = document.querySelector("#chat-options button");
    const sendButton = document.getElementById("chat-send");
    const newThreadButton = document.getElementById("chat-new-thread");
    const input = document.getElementById("chat-input");

    input.value = "my billing account number is 104";
    answerButton.click();
    answerButton.click();
    sendButton.click();
    newThreadButton.click();

    expect(api.postChatMessage).toHaveBeenCalledTimes(1);
    expect(api.createChatThread).toHaveBeenCalledTimes(1);
    expect(answerButton.disabled).toBe(true);
    expect(sendButton.disabled).toBe(true);
    expect(newThreadButton.disabled).toBe(true);

    resolveMessage({
      thread_id: "thread-3",
      billing_account: "6001",
      customer_type: "RESIDENTIAL",
      messages: [],
      pending_questions: [],
      current_question: null,
      decision_result: {
        final_offer: {
          program_id: "rate_plan_optimization",
          display_name: "Plan Savings Review",
          confidence: "HIGH",
          metadata: {}
        },
        eligible_offers: [
          {
            program_id: "rate_plan_optimization",
            display_name: "Plan Savings Review",
            confidence: "HIGH",
            metadata: {}
          }
        ]
      },
      declined_programs: [],
      status_phase: "complete",
      user_answers: {}
    });

    await messagePromise;
    await Promise.resolve();

    expect(sendButton.disabled).toBe(false);
    expect(newThreadButton.disabled).toBe(false);
  });

  it("keeps the stored thread id and surfaces a resume failure for non-404 errors", async () => {
    storage.setItem("customer_offer_advisor_thread_id", "thread-stale");

    const api = {
      createChatThread: vi.fn(),
      getChatThread: vi.fn().mockRejectedValue({ status: 500 }),
      postChatMessage: vi.fn(),
      declineChatProgram: vi.fn()
    };

    await initChatPage(api, storage);

    expect(api.getChatThread).toHaveBeenCalledWith("thread-stale");
    expect(api.createChatThread).not.toHaveBeenCalled();
    expect(storage.getItem("customer_offer_advisor_thread_id")).toBe("thread-stale");
    expect(document.getElementById("chat-status").textContent).toContain("resume_failed");
    expect(document.getElementById("chat-messages").textContent).toContain(
      "We couldn't reconnect to your saved chat thread."
    );
  });

  it("clears a stale stored thread id and recreates the thread on a 404-shaped resume error", async () => {
    storage.setItem("customer_offer_advisor_thread_id", "thread-missing");

    const api = {
      createChatThread: vi.fn().mockResolvedValue({
        thread_id: "thread-404-recovery",
        state: {
          thread_id: "thread-404-recovery",
          billing_account: null,
          customer_type: null,
          messages: [
            {
              role: "assistant",
              content: "Welcome to Customer Offer Advisor.",
              timestamp: "2026-04-17T00:00:00Z"
            }
          ],
          pending_questions: [],
          current_question: null,
          decision_result: null,
          declined_programs: [],
          status_phase: "awaiting_account",
          user_answers: {}
        }
      }),
      getChatThread: vi
        .fn()
        .mockRejectedValue(Object.assign(new Error("HTTP error! status: 404"), { status: 404 })),
      postChatMessage: vi.fn(),
      declineChatProgram: vi.fn()
    };

    await initChatPage(api, storage);

    expect(api.getChatThread).toHaveBeenCalledWith("thread-missing");
    expect(api.createChatThread).toHaveBeenCalledTimes(1);
    expect(storage.getItem("customer_offer_advisor_thread_id")).toBe("thread-404-recovery");
    expect(document.getElementById("chat-status").textContent).toContain("awaiting_account");
    expect(document.getElementById("chat-messages").textContent).toContain("Welcome to Customer Offer Advisor.");
  });
});
