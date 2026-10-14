# Diagrams and screenshot placeholders

All diagrams are PNGs rendered from JSON specs you write, so they are editable and consistent. Keep each spec next to its PNG in `figures/` so the team can regenerate it. Screens are **placeholders** that the author replaces with real screenshots later.

```bash
python $TOOLS/scripts/render_diagram.py figures/arch_current.json figures/arch_current.png
python $TOOLS/scripts/make_placeholder.py figures/scr_01_datasets.png --id SCR-01 --title "Datasets page" --capture "…"
```

Always open the PNG and look at it before using it. Fix overlapping labels, clipped text and edges crossing boxes by moving nodes, or with `from_side`/`to_side`, `via_x`/`via_y` or `label_at`.

### Contents
1. `platform` — SAP Business AI Platform reference-architecture layout (default for architecture chapters)
2. `graph` — generic grid of components
3. `steps` — numbered process flow
4. `sequence` — runtime interaction flow
5. Node, group and edge options (shared by `platform` and `graph`)
6. Screenshot placeholders

---

### 1. `platform` — the default architecture layout

This layout mirrors the SAP Business AI Platform reference architecture:
- users and application clients on the left;
- a light-blue **SAP Business AI Platform** frame containing a white **Subaccount** frame. Inside it sit the application router, XSUAA ("Trust"), the app runtimes, SAP HANA Cloud and a **Generative AI Hub** frame around SAP AI Core, whose nested capability pills include Orchestration (Grounding, Templating, Translation…) and Foundation Model Access;
- a vertical **NETWORK** divider;
- on the right, a dashed box of external systems: SAP on-premise via Cloud Connector, 3rd-party applications, SAP cloud solutions, and partner-hosted foundation models.

Connector labels render as pills (HTTPS, Trust, Destination, Harmonized API).

Use it twice in every handover:
- **Current PoC architecture** (Part II) — only what exists.
- **Target production architecture** (Part VI) — adds approuter, XSUAA/IAS, Destination, Connectivity/Cloud Connector, Credential Store, Audit Log, etc. with `"status": "planned"` so they draw dashed with a *TO BE BUILT* badge.

A worked example is in `$TOOLS/assets/examples/platform_target_architecture.json`.

```json
{
  "type": "platform",
  "title": "Current PoC architecture",
  "subtitle": "What runs today",
  "subaccount": {"title": "Subaccount", "sub": "Cloud Foundry · eu10"},
  "clients": [{"id": "user", "title": "Planners", "sub": "Browser", "icon": "users"}],
  "nodes": [
    {"id": "ui",  "col": 0, "row": 0, "title": "Optimizer UI", "tech": "CF app · Vite", "items": ["Dataset import", "Agent chat"], "color": "blue"},
    {"id": "api", "col": 1, "row": 0, "title": "Optimizer API", "tech": "CF app · FastAPI", "items": ["REST API", "LangGraph agent"], "color": "purple"},
    {"id": "hana","col": 1, "row": 1, "title": "SAP HANA Cloud", "icon": "database", "items": ["Datasets", "Results"], "color": "green"},
    {"id": "aicore","col": 2, "row": 0, "rowspan": 2, "title": "SAP AI Core", "icon": "ai", "color": "orange",
     "items": ["Foundation model access"],
     "sections": [{"title": "Orchestration", "items": ["Templating", "Data masking", "I/O filtering", "Grounding"], "cols": 2}]},
    {"id": "dest", "col": 3, "row": 0, "kind": "service", "icon": "destination", "title": "Destination Service", "status": "planned", "color": "teal"}
  ],
  "groups": [{"id": "aihub", "label": "Generative AI Hub", "nodes": ["aicore"], "color": "red", "pad": 24}],
  "external": [
    {"id": "s4", "title": "SAP S/4HANA", "sub": "on-premise · Cloud Connector", "icon": "building", "row": 0, "status": "planned"},
    {"id": "llm", "title": "Foundation models", "sub": "Partner hosted", "icon": "cloud", "row": 1}
  ],
  "edges": [
    {"from": "user", "to": "ui", "label": "HTTPS"},
    {"from": "ui", "to": "api", "label": "REST"},
    {"from": "api", "to": "hana", "label": "SQL"},
    {"from": "api", "to": "aicore", "label": "Harmonized API", "color": "#C400C4"},
    {"from": "aicore", "to": "llm", "label": "HTTPS"}
  ],
  "note": "Optional footer note",
  "legend": [{"color": "blue", "label": "Existing", "kind": "edge"}, {"color": "grey", "style": "dashed", "label": "To be built"}]
}
```

- `nodes` use columns/rows **inside the subaccount** (0-based); clients and external systems are placed automatically.
- `groups` inside the subaccount default to white frames (use for *Generative AI Hub*, *SAP Build*, *Joule*, *Kyma cluster*…). Nest with `"groups": ["childId"]`.
- Colour conventions matching the reference: Trust edges `green`, harmonized AI API magenta `#C400C4`, everything else `blue`/`grey`; planned edges `"status": "planned"`.
- Layout knobs: `col_weights` (relative widths of inner columns), `col_gap`, `row_gap`, `client_gap`, `network_gap`, `external_label`, `platform_label`, `width` (default 2600 for ≥4 inner columns).
- Keep it readable: ≤ 4 inner columns, ≤ 12 inner nodes. Put detail in pills (`items`) instead of extra boxes.

### 2. `graph`
Same node/edge/group options without the SAP Business AI Platform frame. Use for component deep-dives, data-flow views, or deployments outside SAP Business AI Platform. Keys: `nodes`, `edges`, `groups` (`style`: `boundary` dashed with label chip, or `frame` titled box), `dividers` (`[{"after_col": 2, "label": "NETWORK"}]`), `col_weights`, `col_gaps`, `row_gaps`, `fit_default` (`cell` stretches boxes to row height, `content` sizes to text).

### 3. `steps`
```json
{"type": "steps", "title": "Optimization run lifecycle", "subtitle": "…", "per_row": 4,
 "steps": [{"title": "Upload workbook", "text": "Stored as HANA artifact"}, {"title": "Solve", "text": "MILP per block", "color": "orange"}]}
```
Use for the end-to-end journey (Part I) and pipeline explanations. 4–10 steps.

### 4. `sequence`
```json
{"type": "sequence", "title": "Agent-driven solve",
 "participants": [{"id": "ui", "title": "UI", "sub": "Vite SPA"}, {"id": "api", "title": "FastAPI", "color": "purple"}],
 "messages": [{"from": "ui", "to": "api", "label": "POST /api/chat"}, {"from": "api", "to": "ui", "label": "202 job_id", "kind": "return"},
              {"from": "api", "to": "api", "label": "run tool", "note": "self-call"}],
 "sections": [{"at": 2, "label": "Asynchronous worker"}]}
```
`kind`: `sync` (solid), `return` (grey dashed), `async` (dashed). Messages are auto-numbered (`"autonumber": false` to disable). ≤ 7 participants, ≤ 14 messages.

**Section dividers.** `sections[].at` is the **0-based index of the message the divider is drawn before**. In the example above, `{"at": 2, "label": "Asynchronous worker"}` draws the line between message 2 and message 3 (numbered from 1 on the diagram), i.e. above the self-call `run tool`. To start a section at the diagram's message number *n*, use `"at": n - 1`.

**Close every loop.** Each synchronous request needs a response drawn back to its caller, hop by hop, until the initiator (usually the UI) receives the final result. For example, a chat turn needs *model → agent runtime → chat router → UI*, not just the model's answer arriving at the runtime. The renderer prints `WARNING (sequence)` for every unanswered request; fix them, or mark a truly fire-and-forget call with `"kind": "async"`.

### 5. Node, group and edge options

**Node** (`platform`/`graph`):
| Key | Meaning |
|---|---|
| `id`, `col`, `row`, `colspan`, `rowspan` | grid placement |
| `title`, `tech` (small caps line), `sub` (italic), `lines` (plain text lines) | text |
| `items` + `item_cols` | pill list (sub-capabilities, modules) |
| `sections` | `[{"title", "items", "cols"}]` nested blocks, e.g. AI Core › Orchestration |
| `kind` | `box` (default), `service` (round icon + label, for platform services), `actor` (people/clients) |
| `icon` | user, users, device, browser, router, shield, key, destination, plug, database, ai, agent, cloud, building, gear, code, chat, cpu, optimizer, queue, file, chart, lock, globe, monitor, pipeline, api, mail, search |
| `color` | blue, navy, teal, green, purple, orange, red, grey |
| `status` | `planned` (dashed + TO BE BUILT), `new`, `external` |
| `fit` | `content` or `cell` |

**Edge**: `from`, `to`, `label`, `color` (palette name or `#hex`), `style: dashed`, `status: planned`, `both: true` (two arrowheads), `from_side`/`to_side` (`left|right|top|bottom`), `via_x` / `via_y` (force the corridor through an absolute coordinate), `label_at` (0–1 position along the path), `label_style` (`pill`|`plain`).

Routing tips:
- Edges are orthogonal and don't avoid boxes. The best fix for a crossing is layout: put connected nodes in adjacent cells, and keep services that talk to external systems in the right-most inner column.
- `via_x` / `via_y` are absolute pixel coordinates on the full-resolution canvas (the PNG's real size, e.g. 2600 px wide for `platform`), not the downscaled preview. The generated `.html` next to the PNG shows the layout if you need exact positions.
- Don't route through the top band: keep `via_y` below the frame titles (for `platform`, roughly > 300) and above the last row's bottom.
- Edges leaving a service icon downward start below its label automatically.

`service` and `actor` nodes draw their label below the icon; the frames of their groups (Subaccount, Generative AI Hub…) grow to include it, so a service in the bottom row keeps its label inside the frame.

Icons are generic line icons, not SAP's official icon set. If the user supplies official SVG icons, they can be inlined by extending `ICONS` in `render_diagram.py`.

### 6. Screenshot placeholders

Usage chapters show each important screen as a **WIP placeholder** that the author later replaces with a real screenshot. Don't draw mockups and don't screenshot live systems yourself.

1. Read the real UI code for the screen (templates/components, labels, tabs, buttons, dialogs) so the placeholder names the screen exactly as the app does.
2. Create one placeholder per important screen or state (typically 4–8):
   ```bash
   python $TOOLS/scripts/make_placeholder.py figures/scr_02_workspace.png --id SCR-02 \
     --title "Optimizer workspace — frontier after a completed run" \
     --capture "Run list with one completed run selected" \
     --capture "Frontier chart with the chosen point highlighted" --capture "Assistant panel open"
   ```
3. Use it like any figure, with a caption describing what the user does there: `![Optimizer workspace after a completed run (screenshot to be added)](figures/scr_02_workspace.png "width=0.9")`.
4. List all placeholders (ID, screen, what to capture) in the delivery message, so the author knows which screenshots to take. They are 2000×1250 (16:10), so a real screenshot can replace them one-to-one in Word.
