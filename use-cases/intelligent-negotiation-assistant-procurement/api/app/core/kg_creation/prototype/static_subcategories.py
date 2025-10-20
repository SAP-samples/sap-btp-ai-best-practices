"""
Static TQSDC Subcategory Definitions for Metric Classification

This module contains the predefined subcategories for each TQSDC category,
used for consistent classification of metrics across different suppliers.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class Subcategory(BaseModel):
    """Model for a TQSDC subcategory."""
    name: str
    description: str
    examples: List[str]
    category: str  # Parent TQSDC category (T, Q, S, D, C)


class TQSDCSubcategories(BaseModel):
    """Container for all TQSDC subcategories."""
    technology: List[Subcategory] = Field(default_factory=list)
    quality: List[Subcategory] = Field(default_factory=list)
    sustainability: List[Subcategory] = Field(default_factory=list)
    delivery: List[Subcategory] = Field(default_factory=list)
    cost: List[Subcategory] = Field(default_factory=list)
    
    def get_subcategories_by_category(self, category: str) -> List[Subcategory]:
        """Get all subcategories for a given TQSDC category."""
        mapping = {
            'T': self.technology,
            'Q': self.quality,
            'S': self.sustainability,
            'D': self.delivery,
            'C': self.cost
        }
        return mapping.get(category.upper(), [])
    
    def get_all_subcategories(self) -> Dict[str, List[Subcategory]]:
        """Get all subcategories organized by category."""
        return {
            'T': self.technology,
            'Q': self.quality,
            'S': self.sustainability,
            'D': self.delivery,
            'C': self.cost
        }


# Define all subcategories based on clusters.txt
TQSDC_SUBCATEGORIES = TQSDCSubcategories(
    technology=[
        Subcategory(
            name="Product Specifications & Information",
            description="Details about the product's technical features, performance metrics, design characteristics, and general product data.",
            examples=[
                "DW5 clutch system", "Clutch CBE 2027", "Clutch Disc TD285 GEN II.5", 
                "Tilted facing", "Nickel coated hub", "Max inertia", "Wear capacity",
                "Material Data Sheet (MDS)", "Technical Specification PD of the RFQ",
                "Max torque", "stiffness", "burst speed", "dimensions", "material composition"
            ],
            category="T"
        ),
        Subcategory(
            name="Design & Development Capabilities",
            description="The supplier's ability to conduct R&D, simulations, prototyping, and design new components and systems.",
            examples=[
                "R&D Capability", "Development Capability", "CAD System Usage",
                "Creo 2.0", "CATIA", "FTP link", "Simulation Methods",
                "FEM strength simulations", "NVH Simulation", "Patents"
            ],
            category="T"
        ),
        Subcategory(
            name="Prototyping & Testing",
            description="Methods, facilities, and equipment used for creating prototypes and validating product performance, durability, and compliance with technical standards.",
            examples=[
                "Internal Prototyping Facilities", "On-site Test Facility/Prototyping",
                "On-site Laboratory Equipment", "Hardness measurements", "Electronic microscope",
                "3D measurements", "On-site Functional Test Rigs", "Caradyn test",
                "Judder bench test", "On-site Endurance Test Rigs", "Rotec Vibration Analysis Equipment",
                "System Testing", "Test Methodologies", "Klassifikationsbaummethode (KB)",
                "Time Partition Testing"
            ],
            category="T"
        ),
        Subcategory(
            name="Manufacturing Processes & Equipment",
            description="The specific processes, machinery, and tooling used in production, including special treatments and assembly methods.",
            examples=[
                "Electric Arc Furnace (EAF)", "H2 DRI + EAF Process", "Induction Furnace",
                "Machining", "Heat Treatment", "Assembly", "Welding", "Automatic Riveting",
                "Automatic Caulking", "Laser graving", "On-site Tool Making Facility"
            ],
            category="T"
        ),
        Subcategory(
            name="Technical Data Exchange & Documentation",
            description="Management of technical data, drawings, and communication protocols for technical information exchange.",
            examples=[
                "CAD Data Transfer Capability", "Odette-file-transfer-system",
                "Technical Documentation Provision", "Internal wiring diagram"
            ],
            category="T"
        )
    ],
    quality=[
        Subcategory(
            name="Quality Standards & Certifications",
            description="Adherence to international and industry-specific quality standards, management systems, and certifications.",
            examples=[
                "IATF16949", "ISO 9001", "PurchasingOrganization CVS10", "Formel Q Konkret",
                "VDA Standard", "AIAG Standard", "ISO 26262", "SPICE", "CMMI"
            ],
            category="Q"
        ),
        Subcategory(
            name="Quality Management Processes",
            description="Measures and checks implemented throughout the product lifecycle to ensure quality, consistency, and adherence to requirements.",
            examples=[
                "100% functional check", "Prototype Control Plans", "Special Characteristics",
                "Process Capability Requirements", "Cpk >= 1.33", "Continuous assurance of process capability",
                "Test Equipment Calibration Requirement", "Quality performance records",
                "Quality checks", "Rivet type", "Presence of all the rivets",
                "Riveting diameter", "Run Out inspection", "Cushion and thickness under load inspection",
                "Batch traceability and FIFO guarantee"
            ],
            category="Q"
        ),
        Subcategory(
            name="Problem & Complaint Management",
            description="Processes for identifying, managing, and resolving product defects, non-conformities, and customer complaints.",
            examples=[
                "Problem recognition and management", "Complaint Handling Process",
                "8D Problem Solving Method", "eQuality System", "KPM–Halle system",
                "Analysis of defective parts", "Early warning system",
                "Serial complaints and recall actions", "Rework of non-conforming parts"
            ],
            category="Q"
        ),
        Subcategory(
            name="Audits & Assessments",
            description="Evaluation and monitoring of supplier quality performance, capabilities, and system compliance through audits and assessments.",
            examples=[
                "VDA 6.3 Audit", "Supplier assessment", "Quality capability",
                "Quality performance", "C-Rating", "Technical Supplier reviews",
                "Critical Supplier program"
            ],
            category="Q"
        ),
        Subcategory(
            name="Compliance & Safety",
            description="Adherence to legal, regulatory, and safety requirements related to products and processes, ensuring product integrity and user safety.",
            examples=[
                "Product safety/product liability Requirement",
                "Functional safety after SOP", "Hazard analyses and risk assessments"
            ],
            category="Q"
        )
    ],
    sustainability=[
        Subcategory(
            name="Environmental & Social Governance (ESG) Frameworks",
            description="Adherence to environmental management standards, internal environmental policies, and broader ESG initiatives.",
            examples=[
                "ISO 14001:2015", "SupplierA annual report 2020",
                "2030 Supply Chain Decarbonisation Strategy"
            ],
            category="S"
        ),
        Subcategory(
            name="Decarbonisation & Energy Management",
            description="Initiatives and targets related to reducing carbon emissions, especially concerning energy sources and material production processes.",
            examples=[
                "Climate Neutral by 2040", "Fossil-free electricity", "Wind Power",
                "Solar Power", "Hydropower", "Geothermal Power", "Tidal Power",
                "Biofuels", "Nuclear Power", "H2 DRI + EAF Process",
                "Electric Arc Furnace (EAF)", "Blast Furnace (BF)",
                "Waste-based Biofuels", "Crop-based Biofuels", "Fossil-Free Energy"
            ],
            category="S"
        ),
        Subcategory(
            name="Material Circularity & Substance Compliance",
            description="Efforts and capabilities related to using recycled materials, managing waste, and ensuring compliance with material and substance regulations.",
            examples=[
                "GADSL (Global Automotive Declarable Substance List)", "IMDS Reporting",
                "Recycling Management and Waste Law", "Post-consumer scrap aluminium",
                "Secondary Aluminium", "Steel scrap iron", "Pig iron", "Sponge iron"
            ],
            category="S"
        ),
        Subcategory(
            name="Supply Chain Sustainability",
            description="Practices and requirements for ensuring sustainability throughout the supply chain, including sub-suppliers and ethical sourcing.",
            examples=[
                "Supplier database (LDB)", "Sub-supplier policy",
                "Supplier sources flat steel from a steel mill", "Tier 1 suppliers"
            ],
            category="S"
        ),
        Subcategory(
            name="Product Life Cycle Sustainability",
            description="Aspects of the product itself that contribute to sustainability across its lifecycle, from design to end-of-life.",
            examples=[
                "Life cycle assessment (LCA)", "Disposal of defective field parts"
            ],
            category="S"
        )
    ],
    delivery=[
        Subcategory(
            name="Logistics & Transportation",
            description="Methods and capabilities for transporting and delivering parts, including specific delivery concepts and return processes.",
            examples=[
                "Return of defective field parts", "Logistics Concept-STA (Ship-to-stock)",
                "Just-In-Time (JIT) Delivery Capability", "Just-In-Sequence (JIS) Delivery Capability",
                "Delivery in PurchasingOrganization2 load carriers", "Logistics Overseas Worksheet"
            ],
            category="D"
        ),
        Subcategory(
            name="Capacity & Production Planning",
            description="The supplier's ability to plan, monitor, and scale production capacity to meet demand fluctuations and ensure supply.",
            examples=[
                "Capacity Requirements", "Supplier Flexibility",
                "Calculated nominal capacity per year", "Technical capacity per week",
                "Group/Family Capacity (TGR)", "Nominal capacity per week",
                "Leadtime to reach technical capacity", "Quoted estimated max peak annual volume",
                "Operators per shift", "Weeks per year", "Shifts per day",
                "Shifts per week", "Max shifts per week", "Days per week",
                "Days per year", "Net hours per shift", "Systemic capacity monitoring",
                "Peak Demand Management Capability", "Capacity increase for urgent re-order"
            ],
            category="D"
        ),
        Subcategory(
            name="Lead Times & Scheduling",
            description="Timeframes for various stages of the product lifecycle, from tool construction to initial sample and serial part delivery.",
            examples=[
                "PT-Tool construction lead-time", "B-sample supply lead-time",
                "C-sample supply lead-time", "Tool construction lead-time",
                "Tool optimization lead-time", "Initial Sample supply lead-time",
                "Series preparation lead-time", "Serial part planned delivery time",
                "Standard Delivery Lead Time", "Escalation Lead Time"
            ],
            category="D"
        ),
        Subcategory(
            name="Order & Information Management",
            description="Systems and processes for managing orders, tracking inventory, and exchanging delivery-related information efficiently.",
            examples=[
                "EDI Capability (VDA 4984/4987, EDIFACT DELFOR/DESADV, EDIFACT DELJIT)",
                "Real-time warehouse stock tracking in ERP", "End-to-End ERP Process",
                "Supplier ID", "Supplier name", "Part Number", "Part Description",
                "Supplier order"
            ],
            category="D"
        ),
        Subcategory(
            name="Packaging & Handling",
            description="Specific requirements and methods for packaging, labeling, and handling of parts during transit and storage.",
            examples=[],  # No explicit examples in the provided snippets
            category="D"
        ),
        Subcategory(
            name="Supply Chain Resilience",
            description="Measures and plans to ensure continuity of supply and mitigate risks in the supply chain, including contingency and relocation strategies.",
            examples=[
                "Continuity Plan", "Internal substitutes for machines and tools",
                "Production Relocation Right", "Daily batch delivery within 24 hours in case of escalation"
            ],
            category="D"
        )
    ],
    cost=[
        Subcategory(
            name="Pricing & Quotations",
            description="Details related to the pricing of parts, services, and the terms and conditions of quotations.",
            examples=[
                "Part Price Offer", "Cost of Initial Samples", "Selling Price",
                "Price Condition Revision Status", "Price Basis Net Untaxed",
                "Prototype Clutch Cover Price", "Prototype Clutch Disc Price",
                "Prototype Releaser Price"
            ],
            category="C"
        ),
        Subcategory(
            name="Development & Tooling Costs",
            description="Specific costs associated with product development, engineering, and the creation of necessary tooling.",
            examples=[
                "Development Costs", "Application and project costs",
                "Service Design Cost", "Service Process Cost", "Service Quality Cost",
                "Service Purchasing Cost", "Service Project Management Cost",
                "Service Testing Cost", "Tooling Costs Participation"
            ],
            category="C"
        ),
        Subcategory(
            name="Fees & Surcharges",
            description="Additional charges, handling fees, or special release fees that may apply.",
            examples=[
                "Handling Surcharge", "Special Release Handling Fee"
            ],
            category="C"
        ),
        Subcategory(
            name="Warranty & Reimbursement Costs",
            description="Agreements and policies for sharing or reimbursing costs related to product defects, warranty claims, and field issues.",
            examples=[
                "No Trouble Found (NTF) Cost Sharing", "Commercial agreement for costs",
                "Defective Part Analysis Costs", "Cost Reimbursement Agreement for Defective Parts",
                "0km Complaint Reimbursement", "Field Claim Reimbursement",
                "Removal and installation costs", "Identification costs",
                "Processing costs", "Landing costs of the replacement goods",
                "Administration costs", "Logistics and handling costs",
                "Additional Costs for Field Claims"
            ],
            category="C"
        ),
        Subcategory(
            name="Financial Terms & Agreements",
            description="Payment terms, long-term financial agreements, and clauses related to volume, material, energy, and inflation.",
            examples=[
                "Payment Terms", "Long Term Agreement (LTA) linked with volume",
                "Price Negotiation Clause", "Material Cost Clause", "Energy Cost Clause",
                "Inflation Costs Clause", "Supply Chain Finance Interest", "Contract Period"
            ],
            category="C"
        ),
        Subcategory(
            name="Turnover & Business Share",
            description="Financial metrics related to the supplier's business volume and distribution, providing insight into their market position and revenue streams.",
            examples=[
                "Export Business Turnover Share", "Domestic Business Turnover Share",
                "Product Net Value Share (Customers)", "Product Net Value Share (Components)",
                "Product Net Value Share (Others)", "Annual Turnover",
                "Turnover sold to DWC", "Turnover from Truck + Bus segment"
            ],
            category="C"
        )
    ]
)


def get_subcategories_for_category(category: str) -> List[Subcategory]:
    """Get all subcategories for a given TQSDC category."""
    return TQSDC_SUBCATEGORIES.get_subcategories_by_category(category)


def get_subcategory_names_for_category(category: str) -> List[str]:
    """Get just the names of subcategories for a given TQSDC category."""
    subcategories = get_subcategories_for_category(category)
    return [sub.name for sub in subcategories]


def find_subcategory_by_name(category: str, name: str) -> Optional[Subcategory]:
    """Find a specific subcategory by name within a category."""
    subcategories = get_subcategories_for_category(category)
    for sub in subcategories:
        if sub.name.lower() == name.lower():
            return sub
    return None


def create_classification_prompt(
    node_content: str,
    category: str,
    subcategories: List[Subcategory]
) -> str:
    """Create a prompt for LLM classification of a node into subcategories."""
    subcategory_text = ""
    for i, sub in enumerate(subcategories, 1):
        subcategory_text += f"\n{i}. **{sub.name}**\n"
        subcategory_text += f"   - Description: {sub.description}\n"
        subcategory_text += f"   - Examples: {', '.join(sub.examples[:5])}...\n"
    
    prompt = f"""Classify the following node into the most appropriate subcategory for TQSDC category '{category}'.

Node Content:
{node_content}

Available Subcategories:
{subcategory_text}

Instructions:
1. Analyze the node content carefully
2. Compare it against the descriptions and examples of each subcategory
3. Choose the single most appropriate subcategory
4. If multiple subcategories seem applicable, choose the primary/most relevant one

Respond with ONLY the subcategory name, nothing else."""
    
    return prompt