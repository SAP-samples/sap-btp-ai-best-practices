# Migration Analysis: Extra-Tools Features → Ideation Agent

**Date**: 2026-05-26  
**Purpose**: Identify key functionalities from `ideation-agent-extra-tools` to implement in `ideation-agent`

---

## Context

- **ideation-agent**: Preferred by users for its clean, bounded interaction (3 questions with suggested answers → final recommendation card)
- **ideation-agent-extra-tools**: Rich in intelligence/grounding tools but poor interaction UX (open-ended, overwhelming, slow)

**Goal**: Port the *intelligence layers* from extra-tools into ideation-agent without breaking its structured interaction model.

---

## Why Users Prefer Ideation Agent

| Aspect | Ideation Agent | Extra-Tools |
|--------|----------------|-------------|
| Response format | Structured JSON contract (always predictable) | Free-form LLM text (unpredictable) |
| Questions | Max 3, one at a time, with 3 suggested answers | Adaptive, unlimited, open-ended |
| UI feel | Guided game (splash → chips → final card) | Traditional chat (overwhelming) |
| Output | Clean final recommendation card | Raw markdown exports |
| Pacing | Fast, decisive, bounded | Slow, iterative, unbounded |

---

## Features to Port (Prioritized)

### Tier 1 — High Value, Low UX Disruption

These run in the background after company/context is captured. They don't change the 3-question flow — they just make the final answer dramatically better.

#### 1. External Market Research (Perplexity Grounding)

- **Source**: `ideation_deep_agent_extra/tools/perplexity_grounding.py`
- **What**: Dual-model web research (Sonar finds sources, Sonar Pro synthesizes)
- **Value**: Real-time market context, competitor landscape, industry trends
- **Impact**: Recommendations grounded in actual market data rather than generic LLM knowledge
- **Integration point**: After first user message provides company/industry context
- **Output feeds**: `market_context` field in enhanced final card

#### 2. LinkedIn Position Analysis

- **Source**: `ideation_deep_agent_extra/tools/linkedin_open_position.py`
- **What**: Analyzes open job postings for AI maturity signals, technology demand, skill gaps, change readiness
- **Value**: Objective signal about where a company is investing/struggling
- **Impact**: Reveals organizational pain points through hiring patterns
- **Integration point**: Triggered when company name is captured
- **Output feeds**: `company_signals` field in enhanced final card

#### 3. Company Search Engine (CSE)

- **Source**: `ideation_deep_agent_extra/tools/company_search.py`
- **What**: Company identity resolution, EDGAR filings, customer stories, competitive intelligence
- **Value**: Grounds recommendations in the specific company's reality
- **Impact**: Moves from generic advice to company-specific insights
- **Integration point**: Triggered when company name is captured (parallel with LinkedIn)
- **Output feeds**: Company context for LLM, `company_profile` in final card

---

### Tier 2 — Medium Value, Moderate Effort

#### 4. RBA/RSA Architecture Mapping

- **Source**: `ideation_deep_agent_extra/tools/rba_rsa_tool.py`
- **What**: Maps use cases to SAP Reference Business Architecture / Reference Solution Architecture
- **Value**: TCM 2510 AI feature catalog with Discovery Center links, Mermaid process diagrams
- **Impact**: Makes capability selection precise and enterprise-credible
- **Integration point**: After final recommendation is generated (enrichment layer)
- **Output feeds**: `architecture_fit` field, optional diagram artifacts

#### 5. SAP AI Core Document Grounding

- **Source**: `ideation_deep_agent_extra/tools/ai_core_grounding.py`
- **What**: RAG over SAP-internal knowledge bases (market intelligence, technical docs)
- **Value**: Evidence-backed answers with source citations
- **Impact**: Prevents hallucinated recommendations
- **Integration point**: Before final response generation
- **Output feeds**: `sources` and `evidence` fields in final card

#### 6. Structured Export

- **Source**: `ideation_deep_agent_extra/tools/internal_tools.py` → `export_ideation_json()`
- **What**: Full ideation payload export as downloadable JSON/PDF
- **Value**: Actionable artifact for stakeholders
- **Impact**: Ideation Agent's final card is nice but not exportable
- **Integration point**: Post-final action button
- **Output feeds**: `export_url` in final response metadata

---

### Tier 3 — Nice-to-Have

#### 7. Voice Transcription

- **Source**: AI Core audio-to-text deployment
- **What**: Hands-free input via microphone
- **Value**: Accessibility and convenience for workshop scenarios
- **Impact**: Low — most users type; useful for demos

---

## Features to NOT Port

| Feature | Why Skip |
|---------|----------|
| 55-slot adaptive interview | Too complex, ruins "3 questions" structured pacing |
| DeepAgent orchestration | LangGraph is simpler and already works well |
| Multi-role facilitation (GTM, Technical Lead, etc.) | Overcomplicates single-agent UX |
| Unlimited questions / open-ended flow | Exactly what people dislike |
| Command system (`/role`, `/mode`, `/export`) | Power-user complexity breaks ideation |
| MCP server infrastructure | Over-engineered for what's needed |

---

## Implementation Strategy

### Phase 1 — Background Enrichment (High Impact, Low UX Disruption)

Add tools that run silently in the background after user provides company/context:

1. Perplexity research → enriches final recommendation with market data
2. LinkedIn analysis → adds market signals to final card
3. Company search → validates company context, adds profile data

**Architecture**:
```
User Input → [3 clarification questions with chips]
                    ↓ (background, in parallel)
         ┌─────────┼─────────────┐
    Perplexity   LinkedIn    Company Search
         └─────────┼─────────────┘
                    ↓
            Final Recommendation Card
          (enriched with market intelligence)
```

**Key decisions**:
- Tools fire after first message (company context available)
- Results cached with TTL (Perplexity: 600s, LinkedIn: 900s, CSE: 180s)
- If tools haven't completed by final turn, include whatever is ready
- Never block the conversation flow waiting for tools

### Phase 2 — Enhanced Final Output

Extend the `final` response contract to include:

```json
{
  "kind": "final",
  "industry": "...",
  "title": "...",
  "problem": "...",
  "selected_ai_capabilities": [...],
  "solution": "...",
  "market_context": "Summary from Perplexity research (nullable)",
  "company_signals": "Key findings from LinkedIn + CSE (nullable)",
  "architecture_fit": "RBA/RSA mapping (nullable)",
  "sources": ["url1", "url2"],
  "export_url": "/api/ideation-agent/exports/{id}.json"
}
```

**UI changes**:
- Final card gains collapsible sections for market context and company signals
- Sources rendered as clickable links
- Export button triggers download

### Phase 3 — Optional Deep-Dive

After the final card, offer "Want to go deeper?" action that unlocks:
- RBA/RSA process diagram visualization
- Detailed market research report
- Full structured export (55-slot format via additional questions)

---

## Technical Considerations

### Tool Integration Pattern

Tools should be added as LangGraph tool nodes but invoked via a **parallel background runner** (not part of the main conversation loop):

```python
# New state fields
class AgentState(MessagesState):
    ...
    enrichment_results: Optional[Dict[str, Any]]  # Background tool results
    enrichment_status: Optional[str]              # "pending" | "ready" | "partial"
```

### Dependencies to Add

```
# Additional requirements for enrichment tools
httpx>=0.27.0        # Async HTTP for Perplexity, LinkedIn, CSE
beautifulsoup4       # HTML parsing for LinkedIn
aioboto3             # If using AWS-based services
```

### Configuration

New environment variables needed:
```bash
# Perplexity (via GenAI Hub)
PERPLEXITY_ENABLED=true

# LinkedIn
LINKEDIN_ENABLED=true
IDEATION_LINKEDIN_SOURCES=  # Optional CSV paths

# Company Search Engine
CSE_ENABLED=true
CSE_API_BASE_URL=

# AI Core Grounding
GROUNDING_ENABLED=true
AICORE_GROUNDING_COLLECTION_ID_MARKET_EXTERNAL=
AICORE_GROUNDING_COLLECTION_ID_SAP_MARKET_INTERNAL=
```

---

## Files to Study (Reference Implementations)

| Tool | Source File |
|------|------------|
| Perplexity | `api/ideation_deep_agent_extra/tools/perplexity_grounding.py` |
| LinkedIn | `api/ideation_deep_agent_extra/tools/linkedin_open_position.py` |
| Company Search | `api/ideation_deep_agent_extra/tools/company_search.py` |
| RBA/RSA | `api/ideation_deep_agent_extra/tools/rba_rsa_tool.py` |
| AI Core Grounding | `api/ideation_deep_agent_extra/tools/ai_core_grounding.py` |
| Export | `api/ideation_deep_agent_extra/tools/internal_tools.py` |
| Engine orchestration | `api/ideation_deep_agent_extra/ideation/engine.py` |

---

## Success Criteria

1. Users still experience the same clean 3-question flow
2. Final recommendations are noticeably richer (market data, company context)
3. Response time stays under 15s for the final card (background tools don't block)
4. New features are opt-in via env vars (graceful degradation if disabled)
5. No breaking changes to existing A2A protocol or UI contract
