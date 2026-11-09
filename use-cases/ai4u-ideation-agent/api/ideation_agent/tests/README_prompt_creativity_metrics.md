# Prompt Creativity Metrics

## Purpose

This document explains the prompt creativity evaluation used for post-final refinement questions. The eval checks whether the newer prompt asks better follow-up questions after an initial final report has already been produced.

The goal is not to reward surprising questions for their own sake. The goal is to reward questions that help turn a business idea into a stronger SAP-oriented implementation plan.

## How The A/B Eval Works

The test runs the same business scenario through two variants:

- the old prompt behavior
- the new prompt behavior

After the agent reaches a final report, the eval sends a broad post-final refinement request such as "I'd like to go deeper on this". Each variant then produces refinement questions.

An LLM judge compares the two variants for the same scenario. The judge scores both variants from 1 to 5 on every metric and chooses a winner. The run saves the prompts, scores, winner, reasoning, repair details, per-scenario aggregates, and examples in a JSON result file.

## Scoring

Each metric uses a 1 to 5 scale:

- 1 = poor
- 3 = acceptable
- 5 = excellent

We try to maximize every metric. We also inspect per-scenario warnings and concrete examples, because a high global score can still hide a weak pattern in one scenario.

## Metrics

### novelty

What it means: A question introduces a new useful angle that was not already obvious from the conversation.

Good example: "What exception cases would make the automation unsafe even when the AI confidence score is high?"

Weak example: "Can you describe the process again?"

What the judge rewards: Useful new angles that extend the solution.

What the judge penalizes: Questions that repeat known facts or ask for obvious details.

### usefulness

What it means: A question helps improve the implementation plan, workflow, architecture, business decision, or next action.

Good example: "Should low-confidence matches be routed to a buyer work queue, blocked from posting, or saved as draft suggestions?"

Weak example: "Would AI be helpful here?"

What the judge rewards: Questions that change design decisions, workflow steps, controls, or delivery priorities.

What the judge penalizes: Interesting but unactionable questions.

### different_pov

What it means: A question reframes the problem from another useful perspective, such as risk, operations, data quality, adoption, governance, value, or user trust.

Good example: "How should procurement managers audit AI-assisted assignments when a supplier disputes the selected catalog code?"

Weak example: "What else should the app do?"

What the judge rewards: Practical reframing that reveals a stakeholder, risk, data, or operating concern.

What the judge penalizes: Vague broadening that does not add a useful perspective.

### provocative_grounded

What it means: A question challenges an assumption or exposes a failure mode while staying practical and relevant.

Good example: "If the reference data contains stale or conflicting records, should the AI still suggest a match or force human review?"

Weak example: "Could something go wrong?"

What the judge rewards: Grounded challenge questions about assumptions, failure modes, constraints, approvals, or trade-offs.

What the judge penalizes: Abstract, hostile, random, or alarmist questions.

### comprehensiveness

What it means: The question set broadens the solution by covering edge cases, controls, adoption, data quality, lifecycle, or operational concerns.

Good example: "How should the solution handle unreadable files, missing reference records, user overrides, and later audit reviews?"

Weak example: "What color should the dashboard use?"

What the judge rewards: Coverage of practical areas that make the solution more complete.

What the judge penalizes: Narrow chains that only optimize one feature and ignore operations or controls.

### non_redundancy

What it means: Questions do not repeat earlier questions or ask for information already answered.

Good example: "You already said clerks approve uncertain matches; should their correction be captured as training feedback?"

Weak example: "Should clerks approve uncertain matches?" after the user already answered that.

What the judge rewards: Questions that build on known answers.

What the judge penalizes: Repetition, reworded duplicates, and questions that ignore prior answers.

### context_continuity

What it means: The question chain stays anchored in the same business scenario and prior answers. It may broaden the discussion, but every broadening move must be traceable to the original use case.

Good example: "Since the quotes arrive by email, should the intake monitor a shared mailbox, a supplier portal, or both?"

Weak example: "Should we add a chatbot for customer marketing?"

What the judge rewards: Connected broadening that follows from the scenario, prior answers, or current recommendation.

What the judge penalizes: Topic drift, hallucinated systems, generic AI ideas, or disconnected suggestions.

### progressive_depth

What it means: Across multiple refinement questions, the chain develops. Each question adds a meaningful layer instead of repeating the same shallow routing question.

Good example: "Q1 asks about low-confidence handling, Q2 asks about approval rules, Q3 asks how override reasons should improve the reference data."

Weak example: "Q1 asks which area to focus on, Q2 asks which area to focus on next, Q3 asks what else to discuss."

What the judge rewards: A chain that becomes more specific, connected, and implementation-relevant over time.

What the judge penalizes: Flat chains, repeated routing questions, or disconnected jumps.

## Overall Score

`overall` is the average of all metric scores.

It is useful for comparing prompt iterations, but it is not a perfect absolute truth. We compare iterations carefully, inspect per-scenario aggregates, and read saved examples before deciding what to change next.

## Deep Refinement

Depth `2` checks a short go-deeper interaction. Depth `5` checks a longer chain.

`--refinement-depth 5` means "up to five questions". The agent may produce a final report earlier if it believes it has enough information.

Deep refinement evaluates the whole chain, not just one question. A strong chain is coherent, connected, and progressively deeper. Topic drift is penalized. Useful broadening is rewarded when it remains connected to the business scenario and prior answers.

## Examples

Good: "If the AI cannot confidently match an item to the reference catalog, should it block automation, route to human review, or accept with a low-confidence warning?"

Weak: "Which area should we focus on?"

Good: "When evidence is partial or contradictory, what should count as enough proof for automated verification?"

Weak: "What document types do you use?"

Good: "When users override the AI recommendation, should the system capture a reason code, learn from the correction, or require approval for high-risk changes?"

Weak: "Do users like the dashboard?"

## How To Interpret Results

A pass means the new prompt performed better under the configured global criteria:

- baseline regression passes
- new wins more often than old
- new overall average is greater than or equal to the threshold
- every global new metric is greater than or equal to `min_metric`

A warning means a scenario or metric needs attention. Per-scenario warnings are diagnostic signals, not always hard failures.

A failed run is useful because it shows where the prompt or eval needs more work. LLM evals are nondeterministic, so we look at repeated runs, saved JSON, and examples before drawing conclusions.
