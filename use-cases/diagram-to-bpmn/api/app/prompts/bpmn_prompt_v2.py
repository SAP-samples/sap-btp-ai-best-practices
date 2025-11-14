"""Focused prompts for a two‑stage image→BPMN pipeline (v2).

Stage A (Vision→Graph): The model receives the diagram image and returns a
strict JSON graph of lanes, nodes and edges. Output must be valid JSON only.

Stage B (Graph→BPMN): The model receives the Stage‑A JSON and returns BPMN 2.0
XML compatible with Signavio. One participant/pool with a single process that
contains a laneSet with lanes. Use sequenceFlow across lanes inside the pool;
use messageFlow only across different participants (not typical for
cross‑functional diagrams).

These prompts are intentionally concise and consistent to reduce variance
across providers.
"""

# =========================
# Stage A: Vision → Graph
# =========================

GRAPH_JSON_SCHEMA = r"""
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["lanes", "nodes", "edges"],
  "properties": {
    "title": {"type": "string"},
    "lanes": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "name"],
        "properties": {
          "id": {"type": "string"},
          "name": {"type": "string"}
        },
        "additionalProperties": false
      }
    },
    "nodes": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "laneId", "type"],
        "properties": {
          "id": {"type": "string"},
          "laneId": {"type": "string"},
          "type": {
            "type": "string",
            "enum": [
              "startEvent", "endEvent", "task", "userTask",
              "serviceTask", "exclusiveGateway", "parallelGateway"
            ]
          },
          "label": {"type": "string"}
        },
        "additionalProperties": false
      }
    },
    "edges": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "kind", "sourceId", "targetId"],
        "properties": {
          "id": {"type": "string"},
          "kind": {"type": "string", "enum": ["sequence", "message"]},
          "sourceId": {"type": "string"},
          "targetId": {"type": "string"},
          "label": {"type": "string"}
        },
        "additionalProperties": false
      }
    }
  },
  "additionalProperties": false
}
"""


VISION_TO_GRAPH_PROMPT = f"""
You convert process diagram IMAGES into a compact JSON graph.

Rules
- Read labels carefully, but prioritize topology correctness.
- Detect lanes (horizontal or vertical bands) as actors/roles.
- Nodes: startEvent, endEvent, task, userTask, serviceTask,
  exclusiveGateway, parallelGateway.
- Edges: kind = "sequence" for solid arrows (even when they cross lanes
  inside the same pool); kind = "message" only when the diagram shows
  distinct pools/participants communicating.
- If a task is clearly manual (human icon or verbs like approve/review), use
  userTask; if it’s obviously system/automation, use serviceTask; otherwise use
  task. Keep node labels short and faithful.

Output
- Return STRICT JSON only. No Markdown, no commentary.
- Must validate against this JSON Schema:
{GRAPH_JSON_SCHEMA}

Examples (tiny)
Image shows two lanes: Customer, Agent. Start → "Request booking" → XOR
gateway "Accepted?"; Yes → "Confirm booking" → End; No → "Notify rejection" → End.

Expected JSON (illustrative):
{{
  "title": "Cab Booking Request",
  "lanes": [
    {{"id": "lane-1", "name": "Customer"}},
    {{"id": "lane-2", "name": "Agent"}}
  ],
  "nodes": [
    {{"id": "n-start", "laneId": "lane-1", "type": "startEvent"}},
    {{"id": "n-req",   "laneId": "lane-1", "type": "userTask", "label": "Request booking"}},
    {{"id": "n-dec",   "laneId": "lane-2", "type": "exclusiveGateway", "label": "Accepted?"}},
    {{"id": "n-conf",  "laneId": "lane-2", "type": "task", "label": "Confirm booking"}},
    {{"id": "n-rej",   "laneId": "lane-2", "type": "task", "label": "Notify rejection"}},
    {{"id": "n-end-y", "laneId": "lane-2", "type": "endEvent"}},
    {{"id": "n-end-n", "laneId": "lane-2", "type": "endEvent"}}
  ],
  "edges": [
    {{"id": "e1", "kind": "sequence", "sourceId": "n-start", "targetId": "n-req"}},
    {{"id": "e2", "kind": "sequence", "sourceId": "n-req",   "targetId": "n-dec"}},
    {{"id": "e3", "kind": "sequence", "sourceId": "n-dec",   "targetId": "n-conf", "label": "Yes"}},
    {{"id": "e4", "kind": "sequence", "sourceId": "n-dec",   "targetId": "n-rej",  "label": "No"}},
    {{"id": "e5", "kind": "sequence", "sourceId": "n-conf",  "targetId": "n-end-y"}},
    {{"id": "e6", "kind": "sequence", "sourceId": "n-rej",   "targetId": "n-end-n"}}
  ]
}}

Return only the final JSON for the provided image.
"""


# =========================
# Stage B: Graph → BPMN XML
# =========================

GRAPH_TO_BPMN_PROMPT = """
You convert a process GRAPH (lanes, nodes, edges) into complete BPMN 2.0 XML
with Diagram Interchange (DI) that imports cleanly in Signavio.

Modeling rules
- Use a single collaboration with a single participant (one pool) unless the
  graph explicitly models multiple pools. Inside that participant, create a
  single process with a laneSet containing all lanes from the graph.
- Use sequenceFlow across lanes within this process. Only use messageFlow when
  there are different participants/pools.
- Map node types directly:
  startEvent → <startEvent/>
  endEvent → <endEvent/>
  task → <task/>
  userTask → <userTask/>
  serviceTask → <serviceTask/>
  exclusiveGateway → <exclusiveGateway/>
  parallelGateway → <parallelGateway/>
- Give every flow node a unique, readable id (e.g., sid-task-1) and include it
  in the lane's <flowNodeRef> list.
- Include labels as @name on tasks/gateways; arrow labels become sequenceFlow
  @name.

BPMN DI requirements (mandatory)
- ALWAYS include a complete <bpmndi:BPMNDiagram> section with:
  - <bpmndi:BPMNPlane> referencing the collaboration or process
  - <bpmndi:BPMNShape> for every flow node and lane with bounds
  - <bpmndi:BPMNEdge> for every sequenceFlow and messageFlow with waypoints
- Use consistent left-to-right layout:
  - Lanes stacked vertically, 200 units apart starting at y=100
  - Elements within lanes spaced 200 units horizontally starting at x=100
  - Standard sizes: startEvent/endEvent 36×36, gateways 50×50, tasks 160×80
  - Keep all elements within their lane's vertical bounds

Output
- Return complete BPMN 2.0 XML with both semantic and DI sections.
- Start with the XML declaration and <definitions>, end with </definitions>.
- No Markdown or commentary.
"""


# =========================
# Stage C: BPMN Validation against Graph JSON
# =========================

BPMN_VALIDATION_PROMPT = """
You validate and correct BPMN 2.0 XML against its source GRAPH JSON structure
to ensure the BPMN accurately represents the original process diagram.

Inputs (provided in user message):
1. GRAPH JSON - the source of truth for process structure (lanes, nodes, edges)
2. BPMN XML - the generated BPMN that may contain errors or inconsistencies

Validation tasks (in order of priority):

1. Structural integrity - verify BPMN matches GRAPH JSON:
   - Every lane in GRAPH JSON has exactly one <lane> element in BPMN laneSet
   - Every node in GRAPH JSON has a corresponding flow node in BPMN process
   - Every edge in GRAPH JSON has a corresponding sequenceFlow or messageFlow
   - Node types match exactly (task→<task>, exclusiveGateway→<exclusiveGateway>, etc.)
   - Lane assignments match (node.laneId in JSON = flowNodeRef in BPMN lane)

2. Reference integrity - ensure all BPMN references are valid:
   - All IDs are unique across the entire document
   - All <flowNodeRef> elements reference existing flow nodes
   - All sourceRef/targetRef in flows point to valid flow node IDs
   - All laneSet contains all lanes from GRAPH JSON
   - All flow nodes are referenced in exactly one lane

3. BPMN DI completeness - ensure visual representation exists:
   - <bpmndi:BPMNDiagram> exists with <bpmndi:BPMNPlane>
   - Every semantic element (lane, flow node, flow) has exactly one DI element
   - All bpmnElement attributes reference valid semantic elements
   - Shapes have proper bounds (x, y, width, height)
   - Edges have at least 2 waypoints (start and end coordinates)
   - Coordinates keep elements within their lane bounds

4. XML well-formedness - ensure valid XML structure:
   - Valid XML syntax with proper escaping
   - Correct BPMN 2.0 namespaces declared
   - Proper element nesting per BPMN specification
   - All elements properly closed

Correction approach:
- If BPMN structure doesn't match GRAPH JSON, FIX BPMN to match GRAPH JSON
  (GRAPH JSON is the authoritative source)
- If elements are missing, add them based on GRAPH JSON
- If elements are extra (not in GRAPH JSON), remove them
- If DI is missing/incomplete, generate complete DI with left-to-right layout:
  - Lanes stacked vertically, 200 units apart starting at y=100
  - Elements within lanes spaced 200 units horizontally starting at x=100
  - Standard sizes: startEvent/endEvent 36×36, gateways 50×50, tasks 160×80
- If IDs conflict, regenerate unique IDs while maintaining all references
- Preserve labels, names, and other semantic attributes when possible

Output format (diff-first):
- Prefer returning a minimal JSON patch that fixes issues instead of rewriting the whole XML.
- JSON object schema:
  {
    "text_edits": [  // optional; apply to raw XML string BEFORE XML parsing
      {"op":"regex_sub","pattern":"...","repl":"...","flags":"ims?"},
      {"op":"insert_before","needle":"<...>","text":"..."},
      {"op":"insert_after","needle":"</...>","text":"..."}
    ],
    "edits": [       // optional; apply to parsed XML tree (id-targeted)
      {"op":"set_attr","id":"...","name":"...","value":"..."},
      {"op":"delete_element","id":"..."},
      {"op":"replace_element","id":"...","new_xml":"<bpmn:.../>"},
      {"op":"insert_child","parent_id":"...","new_xml":"<bpmndi:...>","before_id":"...","after_id":"..."}
    ]
  }
- Use standard prefixes in snippets: bpmn, bpmndi, di, dc.
- If no changes are required, return {"text_edits":[], "edits":[]} (or {"edits":[]}).
- If a patch cannot express all necessary fixes (e.g., extensive restructuring), return the full corrected BPMN 2.0 XML only, with no commentary.
"""


__all__ = [
    "GRAPH_JSON_SCHEMA",
    "VISION_TO_GRAPH_PROMPT",
    "GRAPH_TO_BPMN_PROMPT",
    "BPMN_VALIDATION_PROMPT",
]
