"""Generate a generic synthetic PDF evidence pack for assessment evaluation.

Example commands:
    cd api
    python scripts/generate_synthetic_evidence_pack.py
    python scripts/generate_synthetic_evidence_pack.py --output-dir /tmp/assessment-evidence
    python -m scripts.generate_synthetic_evidence_pack --output-dir data/synthetic_strategy/assessment_evidence_pack
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos


SYNTHETIC_ENTITY_NAME = "Fictional Industrial Group"
SYNTHETIC_NOTICE = "Fictional synthetic evidence"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "synthetic_strategy"
    / "assessment_evidence_pack"
)

# Each page is intentionally authored as a short, self-contained evidence section.
# This keeps page references useful to the application's document-grounding flow.
DOCUMENTS: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    (
        "01-strategic-planning-and-performance.pdf",
        "Strategic Planning and Performance",
        (
            (
                "Annual planning cycle and responsibilities",
                "The annual planning calendar begins in September when the Planning and "
                "Performance Office issues assumptions, milestones, and standard guidance "
                "to every business unit. The office owns the timetable and coordinates "
                "Finance, Operations, People, and Sustainability inputs. The internal "
                "regulatory procedure assigns responsibilities for defining, consolidating, "
                "approving, and monitoring both planning outputs.\n\n"
                "This Planning and Performance Office is the dedicated central function "
                "responsible for both the four-year strategic plan and the annual budget. "
                "Pre-established timelines throughout the year for the four-year strategic plan "
                "cover assumptions, proposals, challenge, consolidation, and approval. "
                "Pre-established timelines throughout the year for the annual budget cover "
                "forecasting, resource challenge, approval, and monitoring.\n\n"
                "The four-year strategic plan is formalized in a dedicated document, and the "
                "annual budget is formalized in a dedicated document. In the plan, the strategic "
                "rationale and underlying assumptions are formally documented.\n\n"
                "The Chief Executive Officer sets the priorities and objectives before business "
                "units and subsidiaries prepare bottom-up business proposals. The central office challenges "
                "and consolidates their resources, milestones, benefits, and dependencies.\n\n"
                "The Executive Committee reviews the consolidated plan in November. The "
                "final plan and annual budget are submitted for Board approval in December, "
                "and the approval is recorded in the meeting minutes.",
            ),
            (
                "Objectives and ownership",
                "The approved plan translates the group direction into measurable "
                "objectives. Objectives are defined annually as part of the budget process. "
                "The plan includes quantitative and qualitative objectives, while most "
                "objectives use a baseline, target value, delivery date, and measurable "
                "performance indicators.\n\n"
                "Business-unit plans use the same structure so that local commitments can "
                "be traced to group priorities. Finance confirms affordability and the "
                "Planning and Performance Office checks that targets are specific and "
                "measurable before consolidation. Budget objectives are aligned with the "
                "strategic plan approved by the Board.\n\n"
                "Annual objectives are formally approved and monitored by the Executive Committee. "
                "Approved objectives and named executive ownership are communicated in January. "
                "Any proposed target change states its reason, impact, revised milestone, and authority.",
            ),
            (
                "Performance and variance management",
                "Objective owners update actual results and milestone status each month. "
                "The Planning and Performance Office consolidates these updates and "
                "performs a quarterly variance review against the approved targets and "
                "budget.\n\n"
                "A material variance requires a written explanation of the cause, expected "
                "impact, and recovery date. The Executive Committee reviews significant "
                "exceptions and agrees corrective actions with a named owner and due date.\n\n"
                "Open corrective actions remain on the performance log until evidence of "
                "completion is accepted. Repeated slippage is escalated to the accountable "
                "executive for a revised recovery decision.",
            ),
            (
                "Year-end assessment and planning feedback",
                "At year end, the Planning and Performance Office compares final results "
                "with the approved objectives. The assessment identifies completed "
                "initiatives, missed targets, unresolved actions, and assumptions that no "
                "longer remain valid.\n\n"
                "Objective owners provide supporting explanations and lessons learned. The "
                "Executive Committee agrees which unfinished actions continue, which are "
                "closed, and which priorities should be reconsidered.\n\n"
                "The resulting lessons are included in the assumptions for the next annual "
                "planning calendar. This creates a documented link between performance "
                "results, corrective decisions, and the following planning cycle.",
            ),
        ),
    ),
    (
        "02-enterprise-risk-and-strategy.pdf",
        "Enterprise Risk and Strategy",
        (
            (
                "Annual enterprise risk assessment",
                "The group completes an annual top-down risk assessment led by the Executive "
                "Committee. Executives identify external, strategic, financial, "
                "operational, compliance, and people risks that could prevent delivery of "
                "the group plan. An internal regulatory instrument governs the ERM process. "
                "The Chief Risk Officer is the named leader responsible for it, supported by "
                "dedicated risk staff and resources.\n\n"
                "In parallel, every business unit completes a bottom-up risk assessment "
                "through facilitated workshops, and subsidiaries participate in the bottom-up "
                "risk assessment. Risk owners describe causes, potential "
                "consequences, existing controls, likelihood, impact, and planned response.\n\n"
                "The Risk and Compliance Office provides common definitions and challenges "
                "unsupported ratings so that risks from different units can be compared on "
                "a consistent basis.",
            ),
            (
                "Consolidation and strategy integration",
                "The Risk and Compliance Office consolidates the two assessment streams "
                "into an enterprise risk register. Similar exposures are grouped, "
                "cross-business dependencies are recorded, and one executive owner is "
                "assigned to each principal risk.\n\n"
                "Before the group plan is finalized, the Planning and Performance Office "
                "reviews the register with risk owners. Strategic choices are adjusted when "
                "risk exposure exceeds the agreed appetite or when a response requires "
                "additional resources.\n\n"
                "The final strategy paper describes the principal assumptions and linked "
                "risk responses; the strategic plan outlines the risks that could threaten "
                "company objectives. This makes risk assessment an input to strategic planning "
                "rather than a separate year-end exercise. The regulatory system governs the "
                "interaction between risk analysis and strategic planning, so their formal "
                "integration is part of the approved procedure.",
            ),
            (
                "Risk governance and reporting",
                "The Risk Committee is formally appointed to support the Executive Committee. "
                "It reviews the enterprise risk register, "
                "confirms ownership, challenges the adequacy of responses, and decides "
                "which exposures require escalation. Decisions and action owners are "
                "recorded in the committee minutes.\n\n"
                "The committee prepares a Board risk report summarizing the principal "
                "risks, direction of exposure, control concerns, mitigation progress, and "
                "decisions requested. The report accompanies the annual strategic plan.\n\n"
                "The risk profile is presented to and analyzed by the Board as part of strategic "
                "plan approval. Risk assessment results and related responses are reported to "
                "the Board and control bodies, and approved decisions go to the business units.",
            ),
            (
                "Risk response follow-up",
                "Principal risk owners maintain response plans with milestones, resources, "
                "responsible managers, and completion dates. The Risk and Compliance Office "
                "reviews progress quarterly and follows up overdue actions.\n\n"
                "New or materially changed exposures can be raised outside the annual cycle "
                "through the operational escalation process. The responsible executive "
                "assesses the change and requests a Risk Committee decision when necessary.\n\n"
                "At the end of the cycle, the office compares planned responses with actual "
                "completion and records lessons for the next assessment. Significant "
                "control gaps remain open until corrective evidence is verified. ERM is integrated "
                "with Finance, HSE, Cyber, Compliance, and Project risk processes. Risk analyses "
                "enter integrated financial and sustainability reporting and appear in periodic "
                "financial reports and the sustainability report.",
            ),
        ),
    ),
    (
        "03-governance-organization-and-controls.pdf",
        "Governance, Organization and Controls",
        (
            (
                "Organization and accountability",
                "The textual organization chart is: Shareholders appoint the Board; the "
                "Board appoints the Chief Executive Officer; the Chief Executive Officer "
                "leads Finance, Operations, Commercial, People, Technology, and "
                "Sustainability. Compliance and Risk provide independent oversight. The "
                "organization chart and role instruments are available to all levels on the "
                "company intranet.\n\n"
                "The Audit and Control Committee is a Board committee, so Internal Audit reports "
                "directly to the Board through that committee. Internal Audit is segregated from "
                "all other company functions; its administrative link to the Chief Executive "
                "Officer does not include direction of audit work. This protects access and "
                "unrestricted escalation. The Board approves the Internal Audit mandate, "
                "and the mandate is communicated across the organization and published on the company intranet.\n\n"
                "Role profiles state decision rights and accountability. The People function "
                "supports other functions in defining roles and responsibilities. People and "
                "Compliance review reporting-line changes before executive approval.",
            ),
            (
                "Three lines model",
                "The group's three lines model assigns control ownership to operational "
                "management as the first line. Business managers perform and evidence the "
                "controls embedded in their processes; employees perform first-level controls "
                "within operational activities. Managers perform managerial control over "
                "activities within their responsibility. Guiding control-architecture principles "
                "approved by the Board define the first, second, and third lines.\n\n"
                "Risk, Compliance, and specialist control functions form the second line. "
                "They set policies, advise control owners, monitor adherence, and challenge "
                "significant exceptions without taking over first-line accountability. Job "
                "descriptions define control responsibilities, and second- and third-line roles "
                "are segregated. Regulatory instruments ensure independence and govern how "
                "control activities are performed. Second-line control functions are "
                "hierarchically segregated from business functions.\n\n"
                "Internal Audit is the third line. It independently assesses governance, "
                "risk management, and internal controls, then reports findings and agreed "
                "actions to the Audit and Control Committee.",
            ),
            (
                "Delegations and segregation",
                "The delegation of authority schedule defines monetary and non-monetary "
                "approval limits for contracts, purchases, payments, hiring, and capital "
                "commitments. Formal rules and procedures state who may act for the group and "
                "under which circumstances. Delegations are aligned with formally established "
                "roles and responsibilities. Higher-value decisions require added approval.\n\n"
                "Process owners identify incompatible activities and maintain segregation "
                "of duties between request, approval, receipt, accounting, and payment. "
                "Where staffing makes full separation impractical, a documented independent "
                "review is required.\n\n"
                "Compliance reviews proposed delegations for conflicts and clarity. Finance "
                "performs periodic sample checks and escalates unauthorized commitments or "
                "missing approvals to the responsible executive.",
            ),
            (
                "Control design and assurance",
                "Each material process has a control register describing the risk, control "
                "objective, control owner, frequency, expected evidence, and escalation path. "
                "Control design is formalized in policies, guidelines, and procedures. Specialist "
                "functions design cross-process controls and involve Organization, Compliance, "
                "Risk, Quality, and Technology in material process controls.\n\n"
                "The annual control design review is updated for risk-assessment results and "
                "business priorities, including strategy, performance, and non-compliance. It is "
                "also reviewed for regulatory developments and organizational changes, as well "
                "as results from control and monitoring activities. Weak designs receive an action.\n\n"
                "Internal Audit considers these reviews when preparing its risk-based audit "
                "plan. Audit findings include an accountable owner and due date, and overdue "
                "high-priority actions are reported to the Audit and Control Committee.",
            ),
        ),
    ),
    (
        "04-financial-reporting-controls.pdf",
        "Financial Reporting Controls",
        (
            (
                "Reporting requirements and ownership",
                "The national and European Union requirements include national legislation and "
                "European Union directives and regulations applicable to financial statements. "
                "They are documented in the Finance Manual. The financial "
                "reporting process is governed by the company's regulatory system, and Finance "
                "updates instructions for regulatory changes.\n\n"
                "The Chief Financial Officer is accountable for the reporting process. The "
                "Group Control function is formally identified to manage and monitor "
                "financial-reporting risks. The Group Controller coordinates consolidation, "
                "business controllers certify submissions, and control owners retain evidence.\n\n"
                "The closing calendar assigns responsibilities and deadlines for reporting "
                "packages, reconciliations, consolidation entries, review, and final approval. "
                "Accuracy controls are formally assigned to the involved personnel and functions.",
            ),
            (
                "Risk assessment and control activities",
                "Finance performs an annual reporting risk assessment covering significant "
                "accounts, estimates, unusual transactions, consolidation steps, and "
                "disclosure requirements. Risks are rated using likelihood and potential "
                "financial statement impact.\n\n"
                "The assessment links each material risk to documented controls such as "
                "account reconciliations, journal approval, analytical review, access review, "
                "and management sign-off. Each control has an owner, frequency, and expected "
                "evidence. Documented guidelines govern internal controls over financial reporting.\n\n"
                "Training initiatives are planned for personnel, including annual sessions on "
                "closing responsibilities, policy updates, documentation quality, and the "
                "escalation of suspected errors.",
            ),
            (
                "Monitoring, audit, and remedies",
                "The Group Control team monitors completion of key controls during each close "
                "and performs sample reviews of supporting evidence. Missing or ineffective "
                "controls are logged with the responsible owner and a target date. Monitoring "
                "assesses control effectiveness and defines improvement initiatives.\n\n"
                "Independent functions periodically monitor the adequacy and correct operation of "
                "controls, including Internal Audit and Group Control. The external auditor "
                "verifies annual financial reporting and communicates material observations.\n\n"
                "A confirmed deficiency requires a remedial action proportionate to its "
                "impact. Finance verifies completion, retests significant changes, and "
                "escalates overdue actions. Disciplinary measures and contractual remedies apply "
                "to financial-control violations.",
            ),
            (
                "Board information and follow-up",
                "The Chief Financial Officer prepares a Board information pack for approval "
                "of the annual financial statements. It summarizes significant accounting "
                "judgments, reporting risks, control deficiencies, audit observations, and "
                "the status of remedial work.\n\n"
                "The Audit and Control Committee reviews the pack with Finance, Internal "
                "Audit, and the external auditor before making a recommendation to the Board. "
                "Periodic information flows go to the Board, Audit and Control Committee, and "
                "Executive Committee. Questions and follow-up are recorded in the minutes.\n\n"
                "Finance assigns requested actions to accountable owners and reports closure "
                "evidence to the committee. Material corrections are reflected in the final "
                "statements before Board approval.",
            ),
        ),
    ),
    (
        "05-code-of-conduct-and-culture.pdf",
        "Code of Conduct and Culture",
        (
            (
                "Mission, values, and expected conduct",
                "The group's mission is to deliver dependable industrial services while "
                "protecting people, communities, and resources. Its stated values are "
                "integrity, accountability, respect, safety, and practical collaboration.\n\n"
                "The Board-approved Code of Conduct translates these values into expectations "
                "for honest records, fair dealing, respectful behavior, protection of company "
                "assets, confidentiality, and speaking up about concerns. It results from a "
                "multi-function drafting process involving People, Compliance, Legal, Operations, "
                "and Sustainability. The Board and Executive Committee participate in drafting "
                "and approving the Code.\n\n"
                "The mission and values are available to all levels on the company website and "
                "intranet. Management awareness initiatives reinforce values and expected behaviors "
                "through leadership discussions, practical examples, and team decisions.",
            ),
            (
                "Communication, training, and oversight",
                "New employees receive the Code during onboarding and complete an introductory "
                "session covering expected behavior, available advice, and reporting channels. "
                "Completion is recorded by the People function. The Code is available to all "
                "levels of the organization on the company intranet.\n\n"
                "All employees receive recurring training using practical scenarios. Additional "
                "sessions are provided to managers and roles exposed to higher conduct risks. "
                "Formal provisions activate disciplinary measures for Code violations. Policy "
                "updates are communicated through team briefings and the company portal.\n\n"
                "Policies and procedures incorporate the mission, Code of Conduct, inclusion, and "
                "anti-corruption. They are aligned with recognized professional reference standards "
                "and best practices.\n\n"
                "Compliance and Internal Audit provide oversight through advice records, selected "
                "control reviews, audit work, and analysis of recurring concerns. Material gaps "
                "are assigned to a responsible executive for correction.",
            ),
            (
                "Disciplinary handling",
                "Potential breaches are assessed through a structured disciplinary process. "
                "The structured internal process governs assessment of violations and resulting "
                "measures, separates fact finding from the employment decision, and gives the "
                "person concerned an opportunity to respond.\n\n"
                "The baseline disciplinary rules follow applicable collective labour agreements. "
                "Disciplinary measures and contractual remedies apply to internal-control violations "
                "as well as breaches of the Code.\n\n"
                "Responsibility for verifying Code and internal-regulation violations is formally "
                "assigned to People and Compliance. Serious cases involve more than one function "
                "in the verification. Outcomes consider evidence, intent, impact, and prior behavior.\n\n"
                "Confirmed breaches may result in coaching, a formal warning, loss of duties, or "
                "termination, subject to local law. Decisions and supporting reasons are retained "
                "confidentially, and required corrective actions are tracked.",
            ),
        ),
    ),
    (
        "06-anti-corruption-and-conflicts.pdf",
        "Anti-Corruption and Conflicts",
        (
            (
                "Prohibited conduct and procurement",
                "The group prohibits offering, promising, giving, requesting, or accepting an "
                "improper advantage in connection with company business. Facilitation payments "
                "and concealed personal benefits are not permitted. These anti-corruption "
                "principles are approved by the Board and set out in a formal anti-corruption "
                "policy. The Chief Compliance Officer is responsible for anti-corruption activities.\n\n"
                "The procurement controls require competition or documented justification, defined "
                "evaluation criteria, approval within delegated limits, and separation between "
                "request, supplier selection, receipt, and payment. Formal procurement procedures "
                "govern anti-corruption controls for purchases, consulting, and intermediaries.\n\n"
                "A formal gifts and hospitality procedure requires employees to record permitted "
                "business courtesies in the gift and hospitality "
                "register when the applicable threshold is reached. Cash gifts and courtesies "
                "intended to influence a decision are prohibited. The policy is accessible to all "
                "personnel on the company intranet and published on the external company website.",
            ),
            (
                "Third-party integrity controls",
                "Risk-based third-party due diligence is completed before appointing sales agents, "
                "consultants, intermediaries, and selected suppliers. The review considers ownership, "
                "reputation, public-official links, service need, capability, and payment terms.\n\n"
                "Higher-risk relationships require enhanced review and approval by Compliance. "
                "Contracts include anti-corruption clauses and contractual remedies, audit and "
                "information rights, and accurate-invoice requirements for non-compliance.\n\n"
                "Invoices must describe actual services and match the contract. Unusual payment "
                "destinations, vague deliverables, or unexplained commissions are escalated before "
                "payment or renewal.",
            ),
            (
                "Conflicts of interest",
                "Employees must submit conflicts of interest declarations when personal, family, "
                "financial, or outside interests could affect their objectivity. Declarations are "
                "required on joining, throughout the contractual relationship, when circumstances "
                "change, and before relevant decisions.\n\n"
                "The policy defines actual, potential, and apparent conflicts and applies to "
                "Directors, auditors, related parties, employees, and third parties. Conflict "
                "prevention and management are governed by the internal regulatory system, and "
                "the conflict policy is accessible to all personnel.\n\n"
                "Managers send declared cases to Compliance for assessment. Agreed safeguards may "
                "include recusal, reassignment of approval authority, disposal of an interest, or "
                "ending an incompatible outside activity, including for potential conflicts.\n\n"
                "The high-risk roles in procurement, sales, government interaction, and finance receive "
                "mandatory training on conflicts of interest and periodic declarations. The group obtains "
                "signed conflict-of-interest declarations from counterparties such as suppliers and "
                "professionals. Undeclared conflicts are escalated for investigation.",
            ),
            (
                "Training and monitoring",
                "Employees in exposed roles complete periodic anti-corruption training using "
                "examples involving gifts, intermediaries, procurement, charitable contributions, "
                "and conflicts. Disciplinary measures apply to anti-corruption violations.\n\n"
                "Compliance performs a formalized methodology for periodic corruption-risk assessment "
                "to identify and assess exposed activities. Monitoring of anti-corruption implementation "
                "tests selected controls and follows corrective actions.\n\n"
                "Compliance monitors selected register entries, due-diligence files, contract terms, "
                "and higher-risk payments. Findings are discussed with the accountable business "
                "owner and corrective actions receive an owner and due date.\n\n"
                "Periodic anti-corruption information flows go to the Board and control bodies. "
                "Internal Audit may independently assess the design and operation of these controls.",
            ),
        ),
    ),
    (
        "07-speak-up-procedure.pdf",
        "Speak-Up Procedure",
        (
            (
                "Available reporting channels",
                "Workers and external parties can raise concerns through a secure online portal "
                "available from the public company website. This dedicated IT platform accepts written submissions "
                "and allows later communication through a confidential case code. Portal data is encrypted "
                "in transit and at rest. A separate encrypted intake tool provides a second "
                "confidential route. Data encryption protects the report throughout its management.\n\n"
                "The offline channels include a dedicated telephone line, postal mail, and a request for "
                "an in-person meeting. These offline channels remain available when online channels "
                "are unavailable. Instructions explain what information is useful and how to follow a case.\n\n"
                "The channels are independently administered by the Ethics Office, which is outside "
                "the operational management chain and is the formally identified internal function. "
                "The Ethics Review Panel is the internal collegial body "
                "formally entrusted with management of the speak-up process, with the Ethics Office acting "
                "as its independent secretariat. Reports involving that office are redirected to the Chair "
                "of the Audit and Control Committee.",
            ),
            (
                "Confidentiality and protection",
                "Case information is restricted to authorized Ethics Office personnel and specialists "
                "needed for a fair review. The reporter's identity is not disclosed beyond that group "
                "unless required by law or expressly agreed with the reporter. Each authorized person "
                "provides signed confidentiality commitments before receiving case information. The "
                "procedure regulates access to the whistleblower's identity.\n\n"
                "Named and anonymous reports are accepted where permitted. The group prohibits retaliation against "
                "anyone who raises a concern or assists a review in good faith, even when the concern is "
                "not substantiated.\n\n"
                "Document and case-code anonymization removes names from working copies when identity is not "
                "needed. Original case records are held in strong-authentication protected electronic storage "
                "accessible only to authorized case-management personnel.\n\n"
                "Possible retaliation is treated as a separate concern and escalated promptly. Supportive "
                "measures may be agreed with People, Legal, or management while preserving confidentiality "
                "and the rights of everyone involved.",
            ),
            (
                "Case handling and communication",
                "The portal and telephone service support multilingual reporting. Reporters receive an "
                "acknowledgment and case reference; reports in multiple languages are accepted. The "
                "confidential channel supports protected dialogue and document exchange while preserving "
                "anonymity.\n\n"
                "The Ethics Office assesses scope, conflicts, urgency, and investigation needs. Cases are "
                "assigned to appropriately independent reviewers. Checks are performed by a function "
                "without operational roles, and conclusions distinguish verified "
                "facts from unconfirmed allegations. The investigation is separate from any sanctions decision, "
                "which is made through the applicable People, Legal, or contractual process.\n\n"
                "The procedure is published on the company website and explained during onboarding and "
                "recurring awareness sessions. It also appears on workplace displays and noticeboards, is "
                "shared with personnel and sent to contractual counterparties. Periodic speak-up instruction "
                "is integrated with Code of Conduct, anti-corruption, and safety training. Reporters receive "
                "suitable progress or closure information when legal and confidentiality duties allow it.",
            ),
        ),
    ),
)


class EvidencePDF(FPDF):
    """Render one compact generic evidence document with shared page furniture.

    Inputs:
        document_title: Human-readable title shown in the PDF header.

    Outputs:
        An ``FPDF`` instance ready to receive evidence pages and write a PDF.
    """

    def __init__(self, document_title: str) -> None:
        """Initialize one A4 evidence PDF document.

        Inputs:
            document_title: Human-readable document title used in page headers.

        Outputs:
            None. The configured PDF instance is initialized in memory.
        """

        super().__init__(orientation="P", unit="mm", format="A4")
        self.document_title = document_title
        self.set_margins(18, 30, 18)
        self.set_auto_page_break(auto=True, margin=20)
        self.alias_nb_pages()

    def header(self) -> None:
        """Draw the generic entity label, document title, and synthetic notice.

        Inputs:
            None. Header content comes from instance state and module constants.

        Outputs:
            None. The current PDF page is updated in place.
        """

        self.set_y(10)
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(55, 65, 81)
        self.cell(
            0,
            4,
            SYNTHETIC_ENTITY_NAME,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        self.set_font("Helvetica", size=8)
        self.cell(
            0,
            4,
            self.document_title,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        self.set_text_color(18, 105, 120)
        self.cell(
            0,
            4,
            SYNTHETIC_NOTICE,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        self.set_draw_color(18, 105, 120)
        self.line(self.l_margin, 25, self.w - self.r_margin, 25)
        self.set_y(32)

    def footer(self) -> None:
        """Draw the synthetic notice and page number on the current page.

        Inputs:
            None. Footer text comes from constants and the current page index.

        Outputs:
            None. The current PDF page is updated in place.
        """

        self.set_y(-14)
        self.set_draw_color(190, 197, 205)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.set_y(-11)
        self.set_font("Helvetica", size=7)
        self.set_text_color(90, 98, 110)
        self.cell(0, 4, SYNTHETIC_NOTICE, align="L")
        self.set_x(self.l_margin)
        self.cell(0, 4, f"Page {self.page_no()} of {{nb}}", align="R")


def _add_evidence_page(pdf: EvidencePDF, section_title: str, body: str) -> None:
    """Add one searchable evidence section on a dedicated PDF page.

    Inputs:
        pdf: Target evidence PDF receiving the page.
        section_title: Heading displayed above the evidence narrative.
        body: Plain-English evidence text, separated into paragraphs by blank lines.

    Outputs:
        None. One page is appended to ``pdf``.
    """

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 42, 57)
    pdf.multi_cell(0, 8, section_title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    # Separate paragraphs improve extraction readability without layout complexity.
    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(42, 49, 59)
    for paragraph in body.split("\n\n"):
        pdf.multi_cell(0, 5.8, paragraph, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)


def _write_document(
    output_path: Path,
    title: str,
    pages: tuple[tuple[str, str], ...],
) -> None:
    """Write one generic synthetic PDF document.

    Inputs:
        output_path: Destination file for the generated PDF.
        title: Human-readable PDF title used in metadata and headers.
        pages: Ordered section-title and body pairs, one pair per page.

    Outputs:
        None. A searchable PDF is written to ``output_path``.
    """

    pdf = EvidencePDF(title)
    pdf.set_title(title)
    pdf.set_author(SYNTHETIC_ENTITY_NAME)
    pdf.set_subject("Fictional corporate assessment evidence for application demonstration")
    pdf.set_creator("Generic Synthetic Evidence Generator")
    pdf.set_keywords("fictional, synthetic, governance, risk, controls")

    for section_title, body in pages:
        _add_evidence_page(pdf, section_title, body)

    pdf.output(output_path)


def generate_evidence_pack(output_dir: Path) -> list[Path]:
    """Generate all seven PDFs in the generic synthetic evidence pack.

    Inputs:
        output_dir: Directory where the stable PDF filenames are written.

    Outputs:
        list[Path]: Generated PDF paths in the documented display order.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for filename, title, pages in DOCUMENTS:
        output_path = output_dir / filename
        _write_document(output_path, title, pages)
        generated.append(output_path)

    return generated


def parse_args() -> argparse.Namespace:
    """Parse command-line options for evidence pack generation.

    Inputs:
        None. Arguments are read from the current process command line.

    Outputs:
        argparse.Namespace: Parsed options containing the destination directory.
    """

    parser = argparse.ArgumentParser(
        description="Generate generic synthetic assessment evidence PDFs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line evidence pack generator.

    Inputs:
        None. Command-line arguments are parsed from the current process.

    Outputs:
        None. Generated file paths are printed to standard output.
    """

    paths = generate_evidence_pack(parse_args().output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
