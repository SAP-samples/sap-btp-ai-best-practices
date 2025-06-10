
from langchain_community.graphs.networkx_graph import KG_TRIPLE_DELIMITER
from langchain_core.prompts.prompt import PromptTemplate

_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE = (  
    f"""# # Knowledge Graph Instructions for triplets extraction from text

## 1. Overview
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
- **Nodes** represent entities and concepts. They're akin to Wikipedia nodes.
- The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.

## 2. Labeling Nodes
- **Consistency**: Ensure you use basic or elementary types for node labels.
- For example, when you identify an entity representing an Equipment, always label it as **"Equipment"**. Avoid using more specific terms like "1500-HP Equipment".
- **Node IDs**: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers found in the text.
- **Allowed Node Labels**: Equipment, Voltage, Power, Frequency, Temperature, Current, Efficiency, Manufacturer, Enclosure, Bearing, Test, Location
- **Allowed Relationship Types**: is a, has, includes, is rated for, operates at, measures, is equipped with, is tested for, provides, is designed for

## 3. Identifying and Processing Tables
- **Table Detection**: Identify tables by the keyword "Table" in the text document.
- **Entity and Relationship Extraction**: From tables, extract entities and their relationships. Consider rows, columns, and headers for contextual understanding.

## 4. Handling Numerical Data and Dates
- Numerical data, like voltage, current, efficiency, etc., should be incorporated as attributes or properties of the respective nodes.
- **No Separate Nodes for Dates/Numbers**: Do not create separate nodes for dates or numerical values. Always attach them as attributes or properties of nodes.
- **Property Format**: Properties must be in a key-value format.
- **Naming Convention**: Use camelCase for property keys, e.g., `ratedVoltage`.

## 5. Extracting Models as Separate Equipment Nodes
- **Model Identification**: All mentions of specific equipment models should be extracted as separate **"Equipment"** nodes. For example, if "Motor Model MAC 1-1" is mentioned in the text, it should be labeled as an **Equipment** node with the model name as the node ID.
- **Consistency**: If a model is referenced multiple times with different names, ensure the most complete identifier is used consistently in the knowledge graph.

## 6. Conference Resolution
- **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency. If an entity, such as "Equipment Model X", is mentioned multiple times in the text but is referred to by different names or descriptions, always use the most complete identifier for that entity throughout the knowledge graph.

## 7. Strict Compliance
Adhere to the rules strictly. Non-compliance will result in termination.
 """

    "EXAMPLE\n"  
    "Equipment Model A, a 1125-HP induction Equipment, is rated for 4160V at 60Hz with a full load speed of 3222 RPM. The Equipment is designed with Class F insulation and a Class B temperature rise. It is equipped with RTDs for temperature detection and vibration sensors. "  
    "Efficiency: 95% at full load. Power factor at full load: 0.90. Noise level: 85 dBA at 1 meter.\n\n"  
    f"Output: (Equipment Model A, is a, Equipment){KG_TRIPLE_DELIMITER}"  
    f"(Equipment Model A, has, 1125-HP power){KG_TRIPLE_DELIMITER}"  
    f"(Equipment Model A, operates at, 4160V){KG_TRIPLE_DELIMITER}"  
    f"(Equipment Model A, operates at, 60Hz){KG_TRIPLE_DELIMITER}"  
    f"(Equipment Model A, has, full load speed of 3222 RPM){KG_TRIPLE_DELIMITER}"  
    f"(Equipment Model A, is equipped with, RTDs for temperature detection){KG_TRIPLE_DELIMITER}"  
    f"(Equipment Model A, is equipped with, vibration sensors){KG_TRIPLE_DELIMITER}"  
    f"(Equipment Model A, has, efficiency of 95% at full load){KG_TRIPLE_DELIMITER}"  
    f"(Equipment Model A, has, power factor 0.90 at full load){KG_TRIPLE_DELIMITER}"  
    f"(Equipment Model A, has, noise level of 85 dBA at 1 meter){KG_TRIPLE_DELIMITER}"  
    f"(Equipment Model A, is designed with, Class F insulation){KG_TRIPLE_DELIMITER}"  
    f"(Equipment Model A, is designed with, Class B temperature rise)\n"  
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "{text}"
    "Output:"
)

KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE,
)




_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE_KG_RAG = (  
    f"""# # Knowledge Graph Instructions for triplets extraction from text (Selective Attributes Mode)

## 1. Overview
You are a top-tier algorithm specialized in extracting specific, predefined attributes from texts into a structured knowledge graph.

- **Nodes** represent entities and concepts, akin to Wikipedia nodes.
- Focus ONLY on extracting information that matches the provided allowed attributes list.

## 2. Labeling Nodes
- Same rules as before:
  - Use general node types.
  - No integers as IDs.
  - Allowed Node Labels: Equipment, Voltage, Power, Frequency, Temperature, Current, Efficiency, Manufacturer, Enclosure, Bearing, Test, Location

## 3. Allowed Relationship Types
- is a, has, includes, is rated for, operates at, measures, is equipped with, is tested for, provides, is designed for

## 4. Attribute Filtering
- **IMPORTANT**: Extract ONLY information that corresponds exactly or closely matches the allowed attributes list below:
  {{allowedAttributes}}
- If a property, standard, or compliance from the text matches any of the allowed attributes (case insensitive, partial matching allowed), extract it.
- Ignore all other attributes or properties.

## 5. Tables and Numerical Data
- Same treatment as before, but filter extracted properties using allowedAttributes.
- No new nodes for numbers or dates.

## 6. Strict Compliance
Strictly adhere to the allowed attributes list.

EXAMPLE
Equipment Model B complies with API 541 standards and has an enclosure type NEMA WP-I.

Output: 
(Equipment Model B, is a, Equipment){KG_TRIPLE_DELIMITER}
(Equipment Model B, is designed for, API 541 standards){KG_TRIPLE_DELIMITER}
(Equipment Model B, has, enclosure type NEMA WP-I)
END OF EXAMPLE

EXAMPLE
{{text}}
Output:
"""
)

KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT_KG_RAG = PromptTemplate(
    input_variables=["text", "allowedAttributes"],
    template=_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE_KG_RAG,
)
