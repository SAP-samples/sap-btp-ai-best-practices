# Ontology and KG Authoring

## Choose an authoring route

| Route | Best for | Default stance |
|---|---|---|
| Ontology-first Turtle/RDFLib | Stable business domain and production KG | Preferred |
| Deterministic source-to-RDF mapping | Tables, APIs, master data, events | Preferred |
| PDF text to raw triples | Exploration, bootstrapping, analyst review | Review before production |
| Structured LLM extraction | Constrained facts, page provenance, validation | Preferred over raw triples when model access is available |

Do not ask an LLM to invent the production ontology in one pass. Define the vocabulary with a domain expert, then use the LLM to populate or propose facts against that vocabulary.

## 1. Ontology-first design

Start with a small ontology. Give every class and predicate a stable IRI, a label, a comment, and—where meaningful—a domain and range.

```turtle
@prefix ex:   <https://example.com/kg/> .
@prefix ont:  <https://example.com/kg/ontology#> .
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .

ont:Equipment a owl:Class ;
  rdfs:label "Equipment" ;
  rdfs:comment "A technical asset described by source documents." .

ont:hasPowerRating a owl:DatatypeProperty ;
  rdfs:label "has power rating" ;
  rdfs:domain ont:Equipment ;
  rdfs:range xsd:string .

ont:operatesAtVoltage a owl:DatatypeProperty ;
  rdfs:label "operates at voltage" ;
  rdfs:domain ont:Equipment ;
  rdfs:range xsd:string .
```

Keep the ontology, instance data, and provenance in separate named graphs. Review relation direction before loading data: `supplier supplies material` and `material is supplied by supplier` are not interchangeable query contracts.

## 2. Deterministic structured-data mapping

Use this route when a source has explicit fields. It preserves types and avoids LLM ambiguity.

```python
from urllib.parse import quote

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS

EX = Namespace("https://example.com/kg/")
ONT = Namespace("https://example.com/kg/ontology#")

def equipment_uri(equipment_id: str) -> URIRef:
    """Return a stable IRI for a source equipment identifier."""
    return EX[f"equipment/{quote(equipment_id, safe='')}"]

def add_equipment(graph: Graph, record: dict[str, str]) -> None:
    """Map a validated equipment record to typed RDF triples."""
    equipment = equipment_uri(record["equipment_id"])
    graph.add((equipment, RDF.type, ONT.Equipment))
    graph.add((equipment, RDFS.label, Literal(record["name"])))
    graph.add((equipment, ONT.hasPowerRating, Literal(record["power_rating"])))
```

Use RDFLib serialization to N-Triples and bounded `INSERT DATA` batches to load HANA. Keep the original system identifier and source timestamp as provenance metadata.

## 3. PDF to raw triples with GraphIndexCreator

Use this compatibility path for rapid exploration of document facts. The current import is:

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.graphs.index_creator import GraphIndexCreator
```

The legacy `langchain.indexes.graph` import no longer works on current LangChain releases. `GraphIndexCreator.from_text(...)` returns a `NetworkxEntityGraph`.

```python
loader = PyPDFLoader("specification.pdf")
text = "\n".join(document.page_content for document in loader.load())

index_creator = GraphIndexCreator(llm=chat_llm)
raw_graph = index_creator.from_text(
    text=text,
    prompt=KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT,
)

for subject, object_value, predicate in raw_graph.get_triples():
    print(subject, predicate, object_value)
```

Important: `get_triples()` returns `(subject, object, predicate)`, even though the extraction prompt asks the model for `(subject, predicate, object)`. Convert the order before writing RDF. The bundled `scripts/extract_pdf_graph.py` does this correctly.

Treat this graph as a raw extraction graph:

- It represents document text, not a reviewed ontology.
- `NetworkxEntityGraph` uses a directed graph, so a second relation for the same subject/object pair can overwrite the first.
- The generic extractor does not reliably distinguish entity references from literals, units, dates, or identifiers.
- Store page number and source file as provenance; review triples before promoting them into the domain KG.

## 4. Structured extraction with native Responses

For production-sensitive documents, prefer schema-constrained facts and map them to the ontology yourself:

```python
from pydantic import BaseModel, Field
from gen_ai_hub.proxy.native.openai import responses

class ExtractedFact(BaseModel):
    """One document fact mapped to a controlled vocabulary."""

    subject: str = Field(description="Named entity from the document.")
    predicate: str = Field(description="Allowed ontology predicate local name.")
    object_value: str = Field(description="Extracted literal or entity value.")
    page: int | None = Field(description="One-based PDF page when known.")
    confidence: float = Field(ge=0, le=1)

class FactSet(BaseModel):
    """Schema-constrained facts returned for one document chunk."""

    facts: list[ExtractedFact]

response = responses.parse(
    model="gpt-5.4",
    instructions=(
        "Extract only facts supported by the text. Use only the supplied "
        "ontology predicate names and preserve source-page evidence."
    ),
    input="<document chunk and allowed predicate list>",
    text_format=FactSet,
    reasoning={"effort": "none"},
)
facts = response.output_parsed.facts
```

Validate each predicate against an allowlist, normalize units and identifiers, reject low-confidence or malformed facts, then convert the accepted facts to RDFLib triples. Send PDFs directly as `input_file` blocks when the selected deployment supports PDFs; otherwise extract text page-by-page first.

## 5. Promotion workflow

1. Create the reviewed ontology graph.
2. Extract raw or structured candidate facts from documents.
3. Normalize entities, predicates, dates, units, and identifiers.
4. Validate against the ontology and business rules.
5. Write accepted instance triples to the data graph.
6. Write source document, page, extraction method, confidence, and review state to the provenance graph.
7. Use `HanaRdfGraph` and `HanaSparqlQAChain` only after a direct SPARQL validation query proves the graph contract.

When document language is ambiguous, retain the original text as provenance and route the candidate fact to human review rather than silently creating an ontology term.
