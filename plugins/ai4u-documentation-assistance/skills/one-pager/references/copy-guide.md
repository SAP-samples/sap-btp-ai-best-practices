# One-pager copy guide

How to write each slot so it reads like the existing AI4U one-pagers and fits the template. Word limits come from test renders: at the upper limit the renderer shrinks the font to fit; above it, the renderer stops with an overflow message.

## Slots and limits

| Slot | Content | Target | Hard limit |
|---|---|---|---|
| `title` | Use-case name, outcome-oriented ("AI-Powered Claims Validation") | 3-6 words | one line, ~55 characters |
| `subtitle` | Leave the default "Powered by SAP Business AI Platform (BAIP)" unless the user asks otherwise | - | one line |
| `app_name` | Name of the agent or app shown on the laptop ("Claims Validation Agent") | 2-5 words | one line, ~40 characters |
| `left_persona.name` / `right_persona.name` | First name only, from `pick_personas.py` | 1 word | one line |
| `left_persona.role` | Role of the person who does the work and uses the app | 2-5 words | one line, ~45 characters |
| `right_persona.role` | Role of the stakeholder or leader who receives the outcome | 2-5 words | one line, ~45 characters |
| `left_persona.caption` | What the left persona does today, or the task the app performs for them | 8-16 words | ~22 words |
| `right_persona.caption` | What the right persona gets or is accountable for | 8-16 words | ~22 words |
| `challenge` | First-person problem statement from the left persona | 45-65 words | ~75 words |
| `solution` | What the app does, for whom, with which SAP Business AI capability | 30-50 words | ~60 words |
| `benefits` | Exactly 3 lines, each starting with a verb | 10-15 words each | ~17 words each |

## Voice and rules

- **Challenge** speaks as the left persona: "As an AR claims specialist, I need to ... Today, ... This makes it difficult to ...". Name the concrete sources of pain (documents, systems, manual steps), then the impact.
- **Solution** is third person and concrete: the input the app takes, what it produces, and who decides. Mention SAP Business AI where it fits naturally; don't list model names.
- **Benefits** start with one verb, which the renderer colors magenta. Use the imperative form ("Reduce", "Improve", "Create", "Accelerate", "Increase", "Standardize") or the past tense when stating measured results ("Reduced", "Optimized"). Keep the same form in all three lines.
- **Numbers** (FTE, $, %, hours) only when the user or the repository states them. Never invent or estimate figures.
- **No customer names**, customer codes, plant or system identifiers, or real people's names anywhere on the image. Personas use generic first names from the picker.
- **Left vs right persona**: left = the hands-on user (specialist, clerk, planner, rep). Right = the reviewer, manager or leader who consumes the outcome. The arrows run left to app to right.
- Prefer plain words over jargon; acronyms the business audience knows (BOL, POD, AR, SKU) are fine.

## Worked examples (from the deck)

### AI-Powered Claims Validation

- **App:** Claims Validation Agent
- **Left:** Maya, AR Claims Specialist. Caption: "Determine whether the claim is valid by reconciling payment, claim details, shipment evidence, delivery records, and financial impact."
- **Right:** Jordan, Finance / Operations Reviewer. Caption: "Provides the reviewer with the decisive evidence and a clear recommended next step."
- **Challenge (52 words):** "As an AR claims specialist, I need to turn a customer deduction into a defensible decision. Today, claim details can be buried in remittance advice, payment records, invoices, and shipping documents. Manually reconciling these sources takes time, delays cross-functional review, and can lead to inconsistent outcomes."
- **Solution (32 words):** "The Claims Validation Agent brings claim and supporting-document context into one review workspace. It generates an evidence-based assessment that helps the reviewer approve, reject, or investigate with confidence."
- **Benefits:**
  - Reduce manual reconciliation across claim, invoice reference, BOL, and POD evidence.
  - Improve confidence with explicit validation of quantities, locations, delivery exceptions, and signatures.
  - Create a defensible record with supporting documents, structured analysis results, and reviewer actions.

### AI-Driven Sales Order Automation (dense example, with measured figures supplied by the project team)

- **App:** AI Sales Order Intake Agent
- **Left:** Kimberly, Order Management & Customer Support. Caption: "Receives multi-format orders and manually keys them into SAP."
- **Right:** Jim, Customer Operations Leadership. Caption: "Accountable for order accuracy, processing speed, and team capacity."
- **Challenge (73 words, upper limit):** "As a customer service rep, I process incoming orders every day, but they arrive in every format imaginable: emails, PDFs, Excel files, even images. Every order must be manually read, interpreted, and re-keyed into SAP. Ambiguous SKUs, incomplete addresses, and missing account details require follow-up before entry can even begin. Most of my time goes to data entry instead of actually helping customers, and when volume spikes, the only answer is more people."
- **Solution (56 words):** "Using SAP Business AI, we gave customer service teams an AI agent that ingests orders from any format: email, PDF, Excel, or image, automatically extracts the data needed to create an SAP sales order and applies confidence scoring to route clean orders straight through and flag ambiguous ones for human review. CSRs only touch the exceptions."
- **Benefits:**
  - Reduced manual CSR workload, 4 FTE equivalent recovered from repetitive data entry, worth $260K/year
  - Optimized order quality, 50% fewer clarification loops and 25% fewer PO adjustments, saving $444K/year combined
  - Improved scalability, 12 FTE headcount growth avoided over 3 years as volume grows, $1.56M in cost avoidance

## Regenerating the base plate

`assets/base_plate.png` is the template master slide without any slide-level content. If the background, logos or panels change in the source deck:

1. Copy the deck to an empty temp folder. With python-pptx, delete every slide except one that uses the one-pager layout, then delete all shapes on that slide (the master keeps the artwork and the Challenge/Solution/Benefits headings).
2. `soffice --headless --convert-to pdf plate.pptx`
3. `pdftoppm -png -scale-to-x 3840 -scale-to-y 2160 -singlefile plate.pdf base_plate`
4. Re-measure the laptop screen and circles if they moved, and update `scripts/layout.py`.
