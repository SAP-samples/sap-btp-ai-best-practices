#!/usr/bin/env python3
"""Generate synthetic reference datasets and optionally load them into HANA."""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional helper
    load_dotenv = None


CATALOG_COLUMNS = {
    "Procurement Group": "PROCUREMENT_GROUP",
    "Description - high level": "DESCRIPTION_HIGH_LEVEL",
    "Code": "CODE",
    "Description - low level": "DESCRIPTION_LOW_LEVEL",
    "Detailed description. Keywords": "DETAILED_DESCRIPTION_KEYWORDS",
    "Increased CSR Risk": "INCREASED_CSR_RISK",
    "Environmental Impact": "ENVIRONMENTAL_IMPACT",
    "High Comm Code": "HIGH_LEVEL_REFERENCE",
    "Global/Glocal/Local": "GLOBAL_SCOPE",
    "Global lead": "GLOBAL_LEAD",
    "Spend Type": "SPEND_TYPE",
    "Cluster": "CATEGORY_CLUSTER",
}

UNSPSC_REFERENCE_CODE_COLUMNS = (
    "REFERENCE_CODE",
    "Legacy Reference Code",
)

UNSPSC_REFERENCE_DESCRIPTION_COLUMNS = (
    "REFERENCE_CODE_DESCRIPTION",
    "Legacy Reference Code Description",
)

SYNTHETIC_TAXONOMY = [
    {
        "CODE": "RC1001",
        "PROCUREMENT_GROUP": "DIGITAL WORKPLACE",
        "DESCRIPTION_HIGH_LEVEL": "Workplace Devices and Peripherals",
        "DESCRIPTION_LOW_LEVEL": "Employee laptops, handheld devices, printers and workplace accessories",
        "DETAILED_DESCRIPTION_KEYWORDS": "laptop, printer, handset, monitor, scanner, workplace device, accessory",
        "INCREASED_CSR_RISK": "LOW",
        "ENVIRONMENTAL_IMPACT": "MEDIUM",
        "HIGH_LEVEL_REFERENCE": "RC10",
        "GLOBAL_SCOPE": "GLOBAL",
        "GLOBAL_LEAD": "Digital Operations",
        "SPEND_TYPE": "Hardware",
        "CATEGORY_CLUSTER": "Corporate and Digital",
        "MATCH_TERMS": ["laptop", "workplace", "phone", "printer", "scanner", "desktop", "computer", "hand scanner"],
    },
    {
        "CODE": "RC1002",
        "PROCUREMENT_GROUP": "DIGITAL INFRASTRUCTURE",
        "DESCRIPTION_HIGH_LEVEL": "Server, Storage and Data Center Equipment",
        "DESCRIPTION_LOW_LEVEL": "Core compute, storage and data center infrastructure hardware",
        "DETAILED_DESCRIPTION_KEYWORDS": "server, storage, data center, backup appliance, compute node, rack hardware",
        "INCREASED_CSR_RISK": "LOW",
        "ENVIRONMENTAL_IMPACT": "HIGH",
        "HIGH_LEVEL_REFERENCE": "RC10",
        "GLOBAL_SCOPE": "GLOBAL",
        "GLOBAL_LEAD": "Digital Infrastructure",
        "SPEND_TYPE": "Hardware",
        "CATEGORY_CLUSTER": "Corporate and Digital",
        "MATCH_TERMS": ["server", "storage", "datacenter", "data center", "backup", "rack"],
    },
    {
        "CODE": "RC1003",
        "PROCUREMENT_GROUP": "DIGITAL INFRASTRUCTURE",
        "DESCRIPTION_HIGH_LEVEL": "Network and Communication Infrastructure",
        "DESCRIPTION_LOW_LEVEL": "LAN, WAN and telecom hardware used for secure site connectivity",
        "DETAILED_DESCRIPTION_KEYWORDS": "network, router, switch, telecom, lan, wan, communication hardware",
        "INCREASED_CSR_RISK": "LOW",
        "ENVIRONMENTAL_IMPACT": "MEDIUM",
        "HIGH_LEVEL_REFERENCE": "RC10",
        "GLOBAL_SCOPE": "GLOBAL",
        "GLOBAL_LEAD": "Digital Infrastructure",
        "SPEND_TYPE": "Hardware",
        "CATEGORY_CLUSTER": "Corporate and Digital",
        "MATCH_TERMS": ["lan", "wan", "router", "network", "telecom", "communication", "connectivity"],
    },
    {
        "CODE": "RC1004",
        "PROCUREMENT_GROUP": "DIGITAL WORKPLACE",
        "DESCRIPTION_HIGH_LEVEL": "Audio Visual and Collaboration Equipment",
        "DESCRIPTION_LOW_LEVEL": "Conference room, event and digital signage equipment for shared spaces",
        "DETAILED_DESCRIPTION_KEYWORDS": "audio visual, conference room, digital signage, projector, collaboration room",
        "INCREASED_CSR_RISK": "LOW",
        "ENVIRONMENTAL_IMPACT": "MEDIUM",
        "HIGH_LEVEL_REFERENCE": "RC10",
        "GLOBAL_SCOPE": "REGIONAL",
        "GLOBAL_LEAD": "Workplace Experience",
        "SPEND_TYPE": "Hardware",
        "CATEGORY_CLUSTER": "Corporate and Digital",
        "MATCH_TERMS": ["audio visual", "conference room", "digital signage", "projector", "screen", "video room"],
    },
    {
        "CODE": "RC1101",
        "PROCUREMENT_GROUP": "DIGITAL SERVICES",
        "DESCRIPTION_HIGH_LEVEL": "Cloud Platform and Hosting Services",
        "DESCRIPTION_LOW_LEVEL": "Managed cloud, platform and hosting services for shared digital workloads",
        "DETAILED_DESCRIPTION_KEYWORDS": "cloud platform, hosting, platform service, devops, infrastructure service",
        "INCREASED_CSR_RISK": "LOW",
        "ENVIRONMENTAL_IMPACT": "MEDIUM",
        "HIGH_LEVEL_REFERENCE": "RC11",
        "GLOBAL_SCOPE": "GLOBAL",
        "GLOBAL_LEAD": "Digital Platform Services",
        "SPEND_TYPE": "Service",
        "CATEGORY_CLUSTER": "Corporate and Digital",
        "MATCH_TERMS": ["cloud", "platform", "hosting", "devops", "infrastructure services", "server integration"],
    },
    {
        "CODE": "RC1102",
        "PROCUREMENT_GROUP": "DIGITAL SOFTWARE",
        "DESCRIPTION_HIGH_LEVEL": "Business and Engineering Software Licenses",
        "DESCRIPTION_LOW_LEVEL": "Packaged software, SaaS and engineering application licenses",
        "DETAILED_DESCRIPTION_KEYWORDS": "software license, saas, application subscription, engineering software, standard software",
        "INCREASED_CSR_RISK": "LOW",
        "ENVIRONMENTAL_IMPACT": "LOW",
        "HIGH_LEVEL_REFERENCE": "RC11",
        "GLOBAL_SCOPE": "GLOBAL",
        "GLOBAL_LEAD": "Software Asset Management",
        "SPEND_TYPE": "Software",
        "CATEGORY_CLUSTER": "Corporate and Digital",
        "MATCH_TERMS": ["software", "licence", "license", "saas", "subscription", "application"],
    },
    {
        "CODE": "RC1103",
        "PROCUREMENT_GROUP": "DIGITAL SERVICES",
        "DESCRIPTION_HIGH_LEVEL": "Cybersecurity and Digital Operations",
        "DESCRIPTION_LOW_LEVEL": "Security monitoring, identity controls and core digital operations support",
        "DETAILED_DESCRIPTION_KEYWORDS": "cybersecurity, security operations, identity, compliance tooling, core operations",
        "INCREASED_CSR_RISK": "MEDIUM",
        "ENVIRONMENTAL_IMPACT": "LOW",
        "HIGH_LEVEL_REFERENCE": "RC11",
        "GLOBAL_SCOPE": "GLOBAL",
        "GLOBAL_LEAD": "Security and Risk",
        "SPEND_TYPE": "Service",
        "CATEGORY_CLUSTER": "Corporate and Digital",
        "MATCH_TERMS": ["cyber", "security", "identity", "access management", "it4it", "security operations"],
    },
    {
        "CODE": "RC1104",
        "PROCUREMENT_GROUP": "DIGITAL SERVICES",
        "DESCRIPTION_HIGH_LEVEL": "Data Integration and Analytics Services",
        "DESCRIPTION_LOW_LEVEL": "Data engineering, integration and analytics services for reporting and insights",
        "DETAILED_DESCRIPTION_KEYWORDS": "data platform, analytics, reporting, integration, data engineering, insights",
        "INCREASED_CSR_RISK": "LOW",
        "ENVIRONMENTAL_IMPACT": "LOW",
        "HIGH_LEVEL_REFERENCE": "RC11",
        "GLOBAL_SCOPE": "GLOBAL",
        "GLOBAL_LEAD": "Data and Insights",
        "SPEND_TYPE": "Service",
        "CATEGORY_CLUSTER": "Corporate and Digital",
        "MATCH_TERMS": ["analytics", "data", "integration", "reporting", "bi", "insight"],
    },
    {
        "CODE": "RC1105",
        "PROCUREMENT_GROUP": "DIGITAL SERVICES",
        "DESCRIPTION_HIGH_LEVEL": "Connected Product and Online Services",
        "DESCRIPTION_LOW_LEVEL": "Customer-facing applications, online services and connected product capabilities",
        "DETAILED_DESCRIPTION_KEYWORDS": "online service, connected product, ecommerce, customer app, web platform",
        "INCREASED_CSR_RISK": "LOW",
        "ENVIRONMENTAL_IMPACT": "LOW",
        "HIGH_LEVEL_REFERENCE": "RC11",
        "GLOBAL_SCOPE": "GLOBAL",
        "GLOBAL_LEAD": "Digital Product Management",
        "SPEND_TYPE": "Service",
        "CATEGORY_CLUSTER": "Corporate and Digital",
        "MATCH_TERMS": ["connected", "online", "web", "commercial digital", "customer app", "experience services"],
    },
    {
        "CODE": "RC1106",
        "PROCUREMENT_GROUP": "DIGITAL SERVICES",
        "DESCRIPTION_HIGH_LEVEL": "Digital Delivery Capacity and Agile Teams",
        "DESCRIPTION_LOW_LEVEL": "External capacity used to extend product, delivery and operational digital teams",
        "DETAILED_DESCRIPTION_KEYWORDS": "managed capacity, agile team, delivery squad, specialist capacity, team extension",
        "INCREASED_CSR_RISK": "MEDIUM",
        "ENVIRONMENTAL_IMPACT": "LOW",
        "HIGH_LEVEL_REFERENCE": "RC11",
        "GLOBAL_SCOPE": "GLOBAL",
        "GLOBAL_LEAD": "Delivery Management",
        "SPEND_TYPE": "Service",
        "CATEGORY_CLUSTER": "Corporate and Digital",
        "MATCH_TERMS": ["managed capacity", "team", "time and material", "capacity", "agile", "delivery squad"],
    },
    {
        "CODE": "RC1201",
        "PROCUREMENT_GROUP": "ENGINEERING SERVICES",
        "DESCRIPTION_HIGH_LEVEL": "Engineering Design and Technical Consulting",
        "DESCRIPTION_LOW_LEVEL": "Engineering studies, technical design support and product development consulting",
        "DETAILED_DESCRIPTION_KEYWORDS": "engineering service, design support, technical consulting, prototype engineering",
        "INCREASED_CSR_RISK": "MEDIUM",
        "ENVIRONMENTAL_IMPACT": "LOW",
        "HIGH_LEVEL_REFERENCE": "RC12",
        "GLOBAL_SCOPE": "GLOBAL",
        "GLOBAL_LEAD": "Engineering Services",
        "SPEND_TYPE": "Service",
        "CATEGORY_CLUSTER": "Engineering and Manufacturing",
        "MATCH_TERMS": ["engineering", "design", "technical", "prototype", "consulting", "r d"],
    },
    {
        "CODE": "RC1202",
        "PROCUREMENT_GROUP": "INDUSTRIAL OPERATIONS",
        "DESCRIPTION_HIGH_LEVEL": "Laboratory and Test Equipment",
        "DESCRIPTION_LOW_LEVEL": "Measurement, validation and lab equipment used for quality and performance testing",
        "DETAILED_DESCRIPTION_KEYWORDS": "test bench, validation, calibration, measurement device, laboratory equipment",
        "INCREASED_CSR_RISK": "LOW",
        "ENVIRONMENTAL_IMPACT": "MEDIUM",
        "HIGH_LEVEL_REFERENCE": "RC12",
        "GLOBAL_SCOPE": "REGIONAL",
        "GLOBAL_LEAD": "Test and Validation",
        "SPEND_TYPE": "Equipment",
        "CATEGORY_CLUSTER": "Engineering and Manufacturing",
        "MATCH_TERMS": ["test equipment", "testing", "validation", "calibration", "measurement", "laboratory"],
    },
    {
        "CODE": "RC1203",
        "PROCUREMENT_GROUP": "INDUSTRIAL OPERATIONS",
        "DESCRIPTION_HIGH_LEVEL": "Industrial Automation and Control Systems",
        "DESCRIPTION_LOW_LEVEL": "Automation cells, control hardware and industrial mechanics for site operations",
        "DETAILED_DESCRIPTION_KEYWORDS": "automation, plc, robot cell, control system, industrial mechanic",
        "INCREASED_CSR_RISK": "MEDIUM",
        "ENVIRONMENTAL_IMPACT": "MEDIUM",
        "HIGH_LEVEL_REFERENCE": "RC12",
        "GLOBAL_SCOPE": "REGIONAL",
        "GLOBAL_LEAD": "Industrial Automation",
        "SPEND_TYPE": "Equipment",
        "CATEGORY_CLUSTER": "Engineering and Manufacturing",
        "MATCH_TERMS": ["automation", "mechanic", "control system", "plc", "robot", "automat"],
    },
    {
        "CODE": "RC1204",
        "PROCUREMENT_GROUP": "INDUSTRIAL TOOLING",
        "DESCRIPTION_HIGH_LEVEL": "Tooling, Dies and Prototype Fixtures",
        "DESCRIPTION_LOW_LEVEL": "Dies, fixtures, special tools and prototyping equipment used in industrial build phases",
        "DETAILED_DESCRIPTION_KEYWORDS": "die, fixture, jig, special tool, cutting tool, prototype printer",
        "INCREASED_CSR_RISK": "MEDIUM",
        "ENVIRONMENTAL_IMPACT": "MEDIUM",
        "HIGH_LEVEL_REFERENCE": "RC12",
        "GLOBAL_SCOPE": "REGIONAL",
        "GLOBAL_LEAD": "Tooling and Methods",
        "SPEND_TYPE": "Equipment",
        "CATEGORY_CLUSTER": "Engineering and Manufacturing",
        "MATCH_TERMS": ["press", "dies", "die", "fixture", "cutting tools", "special tools", "3d printers", "power tools"],
    },
    {
        "CODE": "RC1205",
        "PROCUREMENT_GROUP": "INDUSTRIAL OPERATIONS",
        "DESCRIPTION_HIGH_LEVEL": "Production Line and Plant Equipment",
        "DESCRIPTION_LOW_LEVEL": "Assembly, process and plant equipment used to operate manufacturing sites",
        "DETAILED_DESCRIPTION_KEYWORDS": "assembly line, plant equipment, process equipment, casting line, service machinery",
        "INCREASED_CSR_RISK": "MEDIUM",
        "ENVIRONMENTAL_IMPACT": "HIGH",
        "HIGH_LEVEL_REFERENCE": "RC12",
        "GLOBAL_SCOPE": "REGIONAL",
        "GLOBAL_LEAD": "Manufacturing Engineering",
        "SPEND_TYPE": "Equipment",
        "CATEGORY_CLUSTER": "Engineering and Manufacturing",
        "MATCH_TERMS": ["assembly shop", "plant process", "casting equipment", "service mach", "propulsion shop", "production line"],
    },
    {
        "CODE": "RC1206",
        "PROCUREMENT_GROUP": "INDUSTRIAL MATERIALS",
        "DESCRIPTION_HIGH_LEVEL": "Fabricated Metal and Machined Parts",
        "DESCRIPTION_LOW_LEVEL": "Cast, machined and fabricated components supplied for industrial applications",
        "DETAILED_DESCRIPTION_KEYWORDS": "machined part, fabricated component, cast component, structural metal part",
        "INCREASED_CSR_RISK": "MEDIUM",
        "ENVIRONMENTAL_IMPACT": "HIGH",
        "HIGH_LEVEL_REFERENCE": "RC12",
        "GLOBAL_SCOPE": "GLOBAL",
        "GLOBAL_LEAD": "Industrial Materials",
        "SPEND_TYPE": "Material",
        "CATEGORY_CLUSTER": "Engineering and Manufacturing",
        "MATCH_TERMS": ["casting", "machined", "fabricated", "metal part", "structural metal"],
    },
    {
        "CODE": "RC1207",
        "PROCUREMENT_GROUP": "INDUSTRIAL MATERIALS",
        "DESCRIPTION_HIGH_LEVEL": "Electrical Components and Signal Hardware",
        "DESCRIPTION_LOW_LEVEL": "Electrical hardware, controls, harnesses and signal distribution components",
        "DETAILED_DESCRIPTION_KEYWORDS": "electrical component, harness, connector, sensor, control hardware, signal device",
        "INCREASED_CSR_RISK": "MEDIUM",
        "ENVIRONMENTAL_IMPACT": "MEDIUM",
        "HIGH_LEVEL_REFERENCE": "RC12",
        "GLOBAL_SCOPE": "GLOBAL",
        "GLOBAL_LEAD": "Electrical Systems",
        "SPEND_TYPE": "Material",
        "CATEGORY_CLUSTER": "Engineering and Manufacturing",
        "MATCH_TERMS": ["electrical", "harness", "connector", "sensor", "signal", "control hardware"],
    },
    {
        "CODE": "RC1208",
        "PROCUREMENT_GROUP": "ENERGY SYSTEMS",
        "DESCRIPTION_HIGH_LEVEL": "Energy Storage and Charging Systems",
        "DESCRIPTION_LOW_LEVEL": "Charging equipment, power electronics and battery-adjacent energy systems",
        "DETAILED_DESCRIPTION_KEYWORDS": "charger, charging unit, battery system, inverter, power electronics, energy storage",
        "INCREASED_CSR_RISK": "MEDIUM",
        "ENVIRONMENTAL_IMPACT": "HIGH",
        "HIGH_LEVEL_REFERENCE": "RC12",
        "GLOBAL_SCOPE": "GLOBAL",
        "GLOBAL_LEAD": "Energy Systems",
        "SPEND_TYPE": "Equipment",
        "CATEGORY_CLUSTER": "Engineering and Manufacturing",
        "MATCH_TERMS": ["charger", "charging", "battery", "ac dc", "power electronics", "energy storage"],
    },
    {
        "CODE": "RC1209",
        "PROCUREMENT_GROUP": "INDUSTRIAL MATERIALS",
        "DESCRIPTION_HIGH_LEVEL": "Structural, Body and Interior Modules",
        "DESCRIPTION_LOW_LEVEL": "Body, trim and interior module assemblies for complex industrial products",
        "DETAILED_DESCRIPTION_KEYWORDS": "body module, trim component, interior assembly, exterior panel, structural module",
        "INCREASED_CSR_RISK": "MEDIUM",
        "ENVIRONMENTAL_IMPACT": "HIGH",
        "HIGH_LEVEL_REFERENCE": "RC12",
        "GLOBAL_SCOPE": "GLOBAL",
        "GLOBAL_LEAD": "Module Sourcing",
        "SPEND_TYPE": "Material",
        "CATEGORY_CLUSTER": "Engineering and Manufacturing",
        "MATCH_TERMS": ["body", "interior", "trim", "exterior", "module"],
    },
    {
        "CODE": "RC1210",
        "PROCUREMENT_GROUP": "INDUSTRIAL MATERIALS",
        "DESCRIPTION_HIGH_LEVEL": "Paint, Coatings and Surface Treatment",
        "DESCRIPTION_LOW_LEVEL": "Paint, coating and surface treatment materials used in finishing processes",
        "DETAILED_DESCRIPTION_KEYWORDS": "paint, coating, sealant, finishing chemistry, surface treatment",
        "INCREASED_CSR_RISK": "MEDIUM",
        "ENVIRONMENTAL_IMPACT": "HIGH",
        "HIGH_LEVEL_REFERENCE": "RC12",
        "GLOBAL_SCOPE": "REGIONAL",
        "GLOBAL_LEAD": "Surface Engineering",
        "SPEND_TYPE": "Material",
        "CATEGORY_CLUSTER": "Engineering and Manufacturing",
        "MATCH_TERMS": ["paint", "coating", "surface treatment", "finishing"],
    },
    {
        "CODE": "RC1211",
        "PROCUREMENT_GROUP": "INDUSTRIAL MATERIALS",
        "DESCRIPTION_HIGH_LEVEL": "Raw Materials, Chemicals and Industrial Gases",
        "DESCRIPTION_LOW_LEVEL": "Base materials, process chemicals and utility gases used in industrial operations",
        "DETAILED_DESCRIPTION_KEYWORDS": "raw material, industrial gas, chemical, hydrocarbon, process media",
        "INCREASED_CSR_RISK": "HIGH",
        "ENVIRONMENTAL_IMPACT": "HIGH",
        "HIGH_LEVEL_REFERENCE": "RC12",
        "GLOBAL_SCOPE": "GLOBAL",
        "GLOBAL_LEAD": "Materials Management",
        "SPEND_TYPE": "Material",
        "CATEGORY_CLUSTER": "Engineering and Manufacturing",
        "MATCH_TERMS": ["raw materials", "chemicals", "industrial gases", "hydrocarbons", "gas"],
    },
    {
        "CODE": "RC1212",
        "PROCUREMENT_GROUP": "SITE MAINTENANCE",
        "DESCRIPTION_HIGH_LEVEL": "Maintenance Supplies, Spare Parts and Power Tools",
        "DESCRIPTION_LOW_LEVEL": "Consumables, spare parts and portable tools used to maintain site operations",
        "DETAILED_DESCRIPTION_KEYWORDS": "consumable, spare part, repair item, power tool, maintenance supply",
        "INCREASED_CSR_RISK": "LOW",
        "ENVIRONMENTAL_IMPACT": "MEDIUM",
        "HIGH_LEVEL_REFERENCE": "RC12",
        "GLOBAL_SCOPE": "REGIONAL",
        "GLOBAL_LEAD": "Maintenance and Reliability",
        "SPEND_TYPE": "Material",
        "CATEGORY_CLUSTER": "Engineering and Manufacturing",
        "MATCH_TERMS": ["consumables", "oem spare parts", "repair", "power tools", "maintenance", "spare part"],
    },
    {
        "CODE": "RC1301",
        "PROCUREMENT_GROUP": "FACILITIES",
        "DESCRIPTION_HIGH_LEVEL": "Facilities Operations and Site Services",
        "DESCRIPTION_LOW_LEVEL": "Building operations, property services and site maintenance support",
        "DETAILED_DESCRIPTION_KEYWORDS": "facility management, property services, cleaning, building operations, site project",
        "INCREASED_CSR_RISK": "MEDIUM",
        "ENVIRONMENTAL_IMPACT": "MEDIUM",
        "HIGH_LEVEL_REFERENCE": "RC13",
        "GLOBAL_SCOPE": "REGIONAL",
        "GLOBAL_LEAD": "Facilities Management",
        "SPEND_TYPE": "Service",
        "CATEGORY_CLUSTER": "Facilities and Services",
        "MATCH_TERMS": ["facility management", "property", "construction", "building", "site services", "facilities"],
    },
    {
        "CODE": "RC1302",
        "PROCUREMENT_GROUP": "FACILITIES",
        "DESCRIPTION_HIGH_LEVEL": "Utilities and Environmental Services",
        "DESCRIPTION_LOW_LEVEL": "Utilities, energy, waste and environmental compliance services for operations",
        "DETAILED_DESCRIPTION_KEYWORDS": "utilities, energy supply, waste management, water service, environmental service",
        "INCREASED_CSR_RISK": "MEDIUM",
        "ENVIRONMENTAL_IMPACT": "HIGH",
        "HIGH_LEVEL_REFERENCE": "RC13",
        "GLOBAL_SCOPE": "REGIONAL",
        "GLOBAL_LEAD": "Energy and Environment",
        "SPEND_TYPE": "Service",
        "CATEGORY_CLUSTER": "Facilities and Services",
        "MATCH_TERMS": ["utilities", "environmental", "waste", "water", "energy supply"],
    },
    {
        "CODE": "RC1303",
        "PROCUREMENT_GROUP": "WORKFORCE SERVICES",
        "DESCRIPTION_HIGH_LEVEL": "Temporary Labor and HR Services",
        "DESCRIPTION_LOW_LEVEL": "Temporary staffing, recruiting and workforce support services",
        "DETAILED_DESCRIPTION_KEYWORDS": "temporary staffing, recruiting, workforce services, hr support, talent sourcing",
        "INCREASED_CSR_RISK": "HIGH",
        "ENVIRONMENTAL_IMPACT": "LOW",
        "HIGH_LEVEL_REFERENCE": "RC13",
        "GLOBAL_SCOPE": "REGIONAL",
        "GLOBAL_LEAD": "People Operations",
        "SPEND_TYPE": "Labor",
        "CATEGORY_CLUSTER": "Facilities and Services",
        "MATCH_TERMS": ["temporary staffing", "hr services", "recruiting", "background check", "drug screening", "it staff"],
    },
    {
        "CODE": "RC1304",
        "PROCUREMENT_GROUP": "BUSINESS SERVICES",
        "DESCRIPTION_HIGH_LEVEL": "Advisory, Program and Corporate Services",
        "DESCRIPTION_LOW_LEVEL": "Corporate advisory, program support and professional services for business operations",
        "DETAILED_DESCRIPTION_KEYWORDS": "consulting, strategic partner, program support, corporate service, finance support",
        "INCREASED_CSR_RISK": "MEDIUM",
        "ENVIRONMENTAL_IMPACT": "LOW",
        "HIGH_LEVEL_REFERENCE": "RC13",
        "GLOBAL_SCOPE": "GLOBAL",
        "GLOBAL_LEAD": "Business Operations",
        "SPEND_TYPE": "Service",
        "CATEGORY_CLUSTER": "Facilities and Services",
        "MATCH_TERMS": ["consultants", "consulting", "strategic partners", "program", "corporate services", "finance"],
    },
    {
        "CODE": "RC1305",
        "PROCUREMENT_GROUP": "MARKETING SERVICES",
        "DESCRIPTION_HIGH_LEVEL": "Marketing, Events and Creative Services",
        "DESCRIPTION_LOW_LEVEL": "Advertising, events and creative production services used for market-facing activity",
        "DETAILED_DESCRIPTION_KEYWORDS": "advertising, campaign, event, creative production, experiential marketing",
        "INCREASED_CSR_RISK": "MEDIUM",
        "ENVIRONMENTAL_IMPACT": "MEDIUM",
        "HIGH_LEVEL_REFERENCE": "RC13",
        "GLOBAL_SCOPE": "REGIONAL",
        "GLOBAL_LEAD": "Brand and Demand",
        "SPEND_TYPE": "Service",
        "CATEGORY_CLUSTER": "Facilities and Services",
        "MATCH_TERMS": ["advertising", "experiential", "music rights", "talents", "marketing", "creative"],
    },
    {
        "CODE": "RC1306",
        "PROCUREMENT_GROUP": "BUSINESS SERVICES",
        "DESCRIPTION_HIGH_LEVEL": "Print, Translation and Workplace Support",
        "DESCRIPTION_LOW_LEVEL": "Print production, translation and office support services for internal operations",
        "DETAILED_DESCRIPTION_KEYWORDS": "print, translation, office support, document production, point of sale material",
        "INCREASED_CSR_RISK": "LOW",
        "ENVIRONMENTAL_IMPACT": "MEDIUM",
        "HIGH_LEVEL_REFERENCE": "RC13",
        "GLOBAL_SCOPE": "REGIONAL",
        "GLOBAL_LEAD": "Business Support",
        "SPEND_TYPE": "Service",
        "CATEGORY_CLUSTER": "Facilities and Services",
        "MATCH_TERMS": ["print", "translation", "pos", "office support", "document"],
    },
    {
        "CODE": "RC1401",
        "PROCUREMENT_GROUP": "TRANSPORT",
        "DESCRIPTION_HIGH_LEVEL": "Regional Transport and Fleet Movement",
        "DESCRIPTION_LOW_LEVEL": "Regional road, rail and fleet transport services used for recurring movements",
        "DETAILED_DESCRIPTION_KEYWORDS": "road transport, rail transport, fleet movement, finished goods delivery, regional transport",
        "INCREASED_CSR_RISK": "MEDIUM",
        "ENVIRONMENTAL_IMPACT": "HIGH",
        "HIGH_LEVEL_REFERENCE": "RC14",
        "GLOBAL_SCOPE": "REGIONAL",
        "GLOBAL_LEAD": "Regional Logistics",
        "SPEND_TYPE": "Logistics",
        "CATEGORY_CLUSTER": "Logistics and Distribution",
        "MATCH_TERMS": ["road premium transp", "road transport", "rail transport", "regional transport", "finished vehicle", "fleet"],
    },
    {
        "CODE": "RC1402",
        "PROCUREMENT_GROUP": "TRANSPORT",
        "DESCRIPTION_HIGH_LEVEL": "Global Freight and Express Logistics",
        "DESCRIPTION_LOW_LEVEL": "Ocean, air and time-critical freight services for cross-border movements",
        "DETAILED_DESCRIPTION_KEYWORDS": "ocean freight, air freight, express logistics, international shipment, customs handling",
        "INCREASED_CSR_RISK": "MEDIUM",
        "ENVIRONMENTAL_IMPACT": "HIGH",
        "HIGH_LEVEL_REFERENCE": "RC14",
        "GLOBAL_SCOPE": "GLOBAL",
        "GLOBAL_LEAD": "Global Logistics",
        "SPEND_TYPE": "Logistics",
        "CATEGORY_CLUSTER": "Logistics and Distribution",
        "MATCH_TERMS": ["ocean transport", "air transport", "ocean", "airplane", "air", "freight", "global transport"],
    },
    {
        "CODE": "RC1403",
        "PROCUREMENT_GROUP": "LOGISTICS SERVICES",
        "DESCRIPTION_HIGH_LEVEL": "Warehousing, Packaging and Material Handling",
        "DESCRIPTION_LOW_LEVEL": "Packaging, warehousing, terminal handling and internal material movement services",
        "DETAILED_DESCRIPTION_KEYWORDS": "packaging, warehouse, terminal, storage, handling, forklift service",
        "INCREASED_CSR_RISK": "MEDIUM",
        "ENVIRONMENTAL_IMPACT": "MEDIUM",
        "HIGH_LEVEL_REFERENCE": "RC14",
        "GLOBAL_SCOPE": "REGIONAL",
        "GLOBAL_LEAD": "Distribution Operations",
        "SPEND_TYPE": "Logistics",
        "CATEGORY_CLUSTER": "Logistics and Distribution",
        "MATCH_TERMS": ["packaging", "storage", "handling", "warehouse", "terminal", "forklifts", "other logistics"],
    },
    {
        "CODE": "RC1499",
        "PROCUREMENT_GROUP": "SPECIALTY SPEND",
        "DESCRIPTION_HIGH_LEVEL": "Miscellaneous Specialized Purchases",
        "DESCRIPTION_LOW_LEVEL": "Specialized or infrequent purchases that do not fit the core synthetic taxonomy",
        "DETAILED_DESCRIPTION_KEYWORDS": "specialized purchase, one-off requirement, miscellaneous service, exceptional demand",
        "INCREASED_CSR_RISK": "MEDIUM",
        "ENVIRONMENTAL_IMPACT": "MEDIUM",
        "HIGH_LEVEL_REFERENCE": "RC14",
        "GLOBAL_SCOPE": "HYBRID",
        "GLOBAL_LEAD": "Category Management",
        "SPEND_TYPE": "Mixed",
        "CATEGORY_CLUSTER": "Specialty Spend",
        "MATCH_TERMS": ["misc purchases", "non procurement", "miscellaneous", "other"],
    },
]

SUPPLIER_NAME_PREFIXES = [
    "Atlas",
    "Beacon",
    "Crest",
    "Delta",
    "Element",
    "Forge",
    "Granite",
    "Harbor",
    "Ion",
    "Juniper",
    "Keystone",
    "Lumen",
]

SUPPLIER_NAME_DOMAINS = [
    "Industrial",
    "Digital",
    "Logistics",
    "Energy",
    "Facilities",
    "Materials",
    "Systems",
    "Advisory",
    "Fabrication",
    "Network",
]

SUPPLIER_NAME_SUFFIX = "Partners"


def _clean_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _clean_unique(values: Iterable[object]) -> list[str]:
    ordered_unique: dict[str, None] = {}
    for value in values:
        text = _clean_text(value)
        if text is not None:
            ordered_unique[text] = None
    return list(ordered_unique.keys())


def _load_env() -> None:
    if load_dotenv is None:
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate synthetic reference data and optionally load it to HANA.")
    parser.add_argument(
        "--supplier-source",
        default=str(base_dir / "data" / "supplier_material_groups.csv"),
        help="Path to the legacy supplier mapping CSV.",
    )
    parser.add_argument(
        "--catalog-source",
        default=str(base_dir / "data" / "embedding" / "Copy of Commodity codes list Jan 2021.xlsx"),
        help="Path to the legacy commodity catalog workbook.",
    )
    parser.add_argument(
        "--unspsc-source",
        default=str(base_dir / "data" / "embedding" / "UNSPSC_COMM CODE_BUYER MAPPING - tree structure - updated.xlsx"),
        help="Path to the legacy UNSPSC mapping workbook.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(base_dir.parent / "generated_reference_data"),
        help="Directory where synthetic files will be written.",
    )
    parser.add_argument(
        "--data-version",
        default=os.getenv("HANA_REFERENCE_DATA_VERSION", "synthetic-2026-04-01"),
        help="Version tag written into every generated reference table.",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Generate synthetic files only; do not upload to HANA.",
    )
    parser.add_argument(
        "--schema",
        default=os.getenv("HANA_SCHEMA", ""),
        help="Optional HANA schema to target.",
    )
    return parser.parse_args()


def _build_supplier_alias_map(values: pd.Series) -> Dict[str, str]:
    ordered = sorted(_clean_unique(values))
    aliases: Dict[str, str] = {}
    prefix_count = len(SUPPLIER_NAME_PREFIXES)
    domain_count = len(SUPPLIER_NAME_DOMAINS)
    for index, value in enumerate(ordered, start=1):
        prefix = SUPPLIER_NAME_PREFIXES[(index - 1) % prefix_count]
        domain = SUPPLIER_NAME_DOMAINS[((index - 1) // prefix_count) % domain_count]
        batch = ((index - 1) // (prefix_count * domain_count)) + 1
        aliases[value] = f"{prefix} {domain} {SUPPLIER_NAME_SUFFIX} {batch:03d}"
    return aliases


def _build_bpid_alias_map(values: pd.Series) -> Dict[str, str]:
    ordered = sorted(_clean_unique(values))
    return {value: f"BP-{index:06d}" for index, value in enumerate(ordered, start=1)}


def _normalize_text_for_matching(*parts: object) -> str:
    text = " ".join(part for part in (_clean_text(value) for value in parts) if part)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _find_first_matching_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _require_matching_column(df: pd.DataFrame, candidates: Iterable[str], dataset_name: str) -> str:
    column = _find_first_matching_column(df.columns, candidates)
    if column is None:
        expected = ", ".join(candidates)
        raise KeyError(f"{dataset_name} is missing required columns. Expected one of: {expected}")
    return column


def _match_score(normalized_text: str, match_terms: list[str]) -> int:
    if not normalized_text:
        return 0
    haystack = f" {normalized_text} "
    score = 0
    for term in match_terms:
        needle = _normalize_text_for_matching(term)
        if needle and f" {needle} " in haystack:
            score += 2 if " " in needle else 1
    return score


def _choose_taxonomy_code(normalized_text: str) -> str | None:
    best_code = None
    best_score = 0
    for entry in SYNTHETIC_TAXONOMY:
        score = _match_score(normalized_text, entry["MATCH_TERMS"])
        if score > best_score:
            best_score = score
            best_code = entry["CODE"]
    return best_code


def _build_catalog_reference_df(data_version: str, used_codes: set[str]) -> pd.DataFrame:
    ordered_columns = [
        "PROCUREMENT_GROUP",
        "DESCRIPTION_HIGH_LEVEL",
        "CODE",
        "DESCRIPTION_LOW_LEVEL",
        "DETAILED_DESCRIPTION_KEYWORDS",
        "INCREASED_CSR_RISK",
        "ENVIRONMENTAL_IMPACT",
        "HIGH_LEVEL_REFERENCE",
        "GLOBAL_SCOPE",
        "GLOBAL_LEAD",
        "SPEND_TYPE",
        "CATEGORY_CLUSTER",
        "DATA_VERSION",
    ]
    rows = []
    for entry in SYNTHETIC_TAXONOMY:
        if entry["CODE"] not in used_codes:
            continue
        row = {key: value for key, value in entry.items() if key != "MATCH_TERMS"}
        row["DATA_VERSION"] = data_version
        rows.append(row)
    catalog_df = pd.DataFrame(rows)
    if catalog_df.empty:
        raise ValueError("No synthetic taxonomy categories were selected.")
    return catalog_df[ordered_columns].sort_values("CODE").reset_index(drop=True)


def _build_legacy_metadata(catalog_source: Path, unspsc_source: Path) -> dict[str, str]:
    metadata_parts: defaultdict[str, list[str]] = defaultdict(list)

    catalog_df = pd.read_excel(catalog_source, sheet_name="Product Categories - Structure")
    for row in catalog_df.to_dict(orient="records"):
        code = _clean_text(row.get("Code"))
        if code is None:
            continue
        metadata_parts[code].append(
            _normalize_text_for_matching(
                row.get("Procurement Group"),
                row.get("Description - high level"),
                row.get("Description - low level"),
                row.get("Detailed description. Keywords"),
                row.get("Cluster"),
            )
        )

    unspsc_df = pd.read_excel(unspsc_source, sheet_name="Sheet1", header=1)
    reference_code_column = _require_matching_column(unspsc_df, UNSPSC_REFERENCE_CODE_COLUMNS, "UNSPSC mapping")
    description_column = _find_first_matching_column(unspsc_df.columns, UNSPSC_REFERENCE_DESCRIPTION_COLUMNS)
    for row in unspsc_df.to_dict(orient="records"):
        code = _clean_text(row.get(reference_code_column))
        if code is None:
            continue
        metadata_parts[code].append(
            _normalize_text_for_matching(
                row.get(description_column) if description_column else None,
                row.get("UNSPSC Code description"),
            )
        )

    return {code: " ".join(part for part in parts if part).strip() for code, parts in metadata_parts.items()}


def _build_code_map(supplier_source: Path, catalog_source: Path, unspsc_source: Path) -> Dict[str, str]:
    supplier_df = pd.read_csv(supplier_source)
    catalog_df = pd.read_excel(catalog_source, sheet_name="Product Categories - Structure")
    unspsc_df = pd.read_excel(unspsc_source, sheet_name="Sheet1", header=1)
    reference_code_column = _require_matching_column(unspsc_df, UNSPSC_REFERENCE_CODE_COLUMNS, "UNSPSC mapping")

    all_codes = sorted(
        set(_clean_unique(supplier_df["Material Group"]))
        | set(_clean_unique(catalog_df["Code"]))
        | set(_clean_unique(unspsc_df[reference_code_column]))
    )
    legacy_metadata = _build_legacy_metadata(catalog_source, unspsc_source)

    text_assignments: Dict[str, str] = {}
    prefix_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for code in all_codes:
        assigned = _choose_taxonomy_code(legacy_metadata.get(code, ""))
        if assigned:
            text_assignments[code] = assigned
            prefix_counts[code[:2]][assigned] += 1

    code_map: Dict[str, str] = {}
    for code in all_codes:
        assigned = text_assignments.get(code)
        if assigned is None:
            prefix_counter = prefix_counts.get(code[:2])
            assigned = prefix_counter.most_common(1)[0][0] if prefix_counter else "RC1499"
        code_map[code] = assigned
    return code_map


def _build_supplier_df(source_path: Path, code_map: Dict[str, str], data_version: str) -> pd.DataFrame:
    source_df = pd.read_csv(source_path)
    supplier_map = _build_supplier_alias_map(source_df["Supplier name"])
    bpid_map = _build_bpid_alias_map(source_df["Business partner ID"])
    material_groups = source_df["Material Group"].map(_clean_text)

    supplier_df = pd.DataFrame(
        {
            "SUPPLIER_NAME": source_df["Supplier name"].map(lambda value: supplier_map[_clean_text(value)]),
            "BUSINESS_PARTNER_ID": source_df["Business partner ID"].map(lambda value: bpid_map[_clean_text(value)]),
            "MATERIAL_GROUP": material_groups.map(code_map),
            "DATA_VERSION": data_version,
        }
    )
    if supplier_df["MATERIAL_GROUP"].isna().any():
        missing = sorted(_clean_unique(material_groups[supplier_df["MATERIAL_GROUP"].isna()]))
        raise ValueError(f"Unable to map supplier material groups to synthetic codes: {missing}")
    return supplier_df


def _build_unspsc_df(
    source_path: Path,
    catalog_df: pd.DataFrame,
    code_map: Dict[str, str],
    data_version: str,
) -> pd.DataFrame:
    source_df = pd.read_excel(source_path, sheet_name="Sheet1", header=1)
    description_by_code = dict(zip(catalog_df["CODE"], catalog_df["DESCRIPTION_LOW_LEVEL"]))
    reference_code_column = _require_matching_column(source_df, UNSPSC_REFERENCE_CODE_COLUMNS, "UNSPSC mapping")
    mapped_codes = source_df[reference_code_column].map(_clean_text)
    reference_codes = mapped_codes.map(code_map)
    missing_codes = sorted(_clean_unique(mapped_codes[reference_codes.isna()]))
    if missing_codes:
        raise ValueError(f"Unable to map UNSPSC reference codes to synthetic codes: {missing_codes}")

    unspsc_df = pd.DataFrame(
        {
            "LEVEL": source_df["Level"],
            "UNSPSC_CODE": source_df["UNSPSC Code"].astype(str),
            "SORT": source_df["Sort"].astype(str),
            "UNSPSC_CODE_DESCRIPTION": source_df["UNSPSC Code description"].astype(str),
            "ACTIVE_STATUS_IN_ARIBA": source_df["Active status in Ariba"].astype(str),
            "REFERENCE_CODE": reference_codes,
            "REFERENCE_CODE_DESCRIPTION": reference_codes.map(description_by_code),
            "DATA_VERSION": data_version,
        }
    )
    unspsc_df = unspsc_df.replace({"nan": None, "NaN": None})
    return unspsc_df


def _export_dataframes(output_dir: Path, supplier_df: pd.DataFrame, catalog_df: pd.DataFrame, unspsc_df: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    supplier_df.to_csv(output_dir / "supplier_material_groups.synthetic.csv", index=False)
    with pd.ExcelWriter(output_dir / "commodity_catalog.synthetic.xlsx", engine="openpyxl") as writer:
        catalog_df.to_excel(writer, index=False, sheet_name="REFERENCE_COMMODITY_CATALOG")
    with pd.ExcelWriter(output_dir / "unspsc_mapping.synthetic.xlsx", engine="openpyxl") as writer:
        unspsc_df.to_excel(writer, index=False, sheet_name="REFERENCE_UNSPSC_MAPPING")


def _quote_identifier(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def _qualified_name(schema: str, table: str) -> str:
    if schema:
        return f"{_quote_identifier(schema)}.{_quote_identifier(table)}"
    return _quote_identifier(table)


def _nvarchar_length(series: pd.Series) -> int:
    max_len = int(series.fillna("").astype(str).str.len().max())
    return min(max(max_len, 32), 5000)


def _column_sql_type(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "DOUBLE"
    return f"NVARCHAR({_nvarchar_length(series)})"


def _connect_to_hana():
    try:
        from hdbcli import dbapi
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("hdbcli must be installed to load synthetic data into HANA.") from exc

    address = os.getenv("hana_address", "").strip()
    port = int(os.getenv("hana_port", "").strip())
    user = os.getenv("hana_user", "").strip()
    password = os.getenv("hana_password", "").strip()
    encrypt = "true" if os.getenv("hana_encrypt", "true").strip().lower() in {"1", "true", "yes", "on"} else "false"
    validate_cert = "false" if os.getenv("hana_ssl_validate_certificate", "false").strip().lower() in {"0", "false", "no", "off"} else "true"

    if not all([address, port, user, password]):
        raise RuntimeError("Missing HANA credentials. Populate hana_address, hana_port, hana_user, and hana_password.")

    return dbapi.connect(
        address=address,
        port=port,
        user=user,
        password=password,
        encrypt=encrypt,
        sslValidateCertificate=validate_cert,
    )


def _load_table(connection, schema: str, table_name: str, df: pd.DataFrame) -> None:
    qualified_name = _qualified_name(schema, table_name)
    columns_sql = ", ".join(
        f'{_quote_identifier(column)} {_column_sql_type(df[column])}' for column in df.columns
    )
    create_sql = f"CREATE COLUMN TABLE {qualified_name} ({columns_sql})"
    truncate_sql = f"TRUNCATE TABLE {qualified_name}"
    insert_sql = (
        f"INSERT INTO {qualified_name} ({', '.join(_quote_identifier(column) for column in df.columns)}) "
        f"VALUES ({', '.join('?' for _ in df.columns)})"
    )

    cursor = connection.cursor()
    try:
        try:
            cursor.execute(create_sql)
        except Exception:
            pass
        cursor.execute(truncate_sql)
        rows = [tuple(None if pd.isna(value) else value for value in row) for row in df.itertuples(index=False, name=None)]
        cursor.executemany(insert_sql, rows)
    finally:
        cursor.close()


def _load_to_hana(schema: str, supplier_df: pd.DataFrame, catalog_df: pd.DataFrame, unspsc_df: pd.DataFrame) -> None:
    connection = _connect_to_hana()
    try:
        _load_table(connection, schema, os.getenv("HANA_SUPPLIER_GROUPS_TABLE", "REFERENCE_SUPPLIER_GROUPS"), supplier_df)
        _load_table(connection, schema, os.getenv("HANA_COMMODITY_CATALOG_TABLE", "REFERENCE_COMMODITY_CATALOG"), catalog_df)
        _load_table(connection, schema, os.getenv("HANA_UNSPSC_MAPPING_TABLE", "REFERENCE_UNSPSC_MAPPING"), unspsc_df)
        connection.commit()
    finally:
        connection.close()


def main() -> int:
    _load_env()
    args = _parse_args()

    supplier_source = Path(args.supplier_source).expanduser().resolve()
    catalog_source = Path(args.catalog_source).expanduser().resolve()
    unspsc_source = Path(args.unspsc_source).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    code_map = _build_code_map(supplier_source, catalog_source, unspsc_source)
    catalog_df = _build_catalog_reference_df(args.data_version, set(code_map.values()))
    supplier_df = _build_supplier_df(supplier_source, code_map, args.data_version)
    unspsc_df = _build_unspsc_df(unspsc_source, catalog_df, code_map, args.data_version)

    _export_dataframes(output_dir, supplier_df, catalog_df, unspsc_df)
    print(f"Wrote synthetic reference data to {output_dir}")

    if args.skip_load:
        return 0

    _load_to_hana(args.schema.strip(), supplier_df, catalog_df, unspsc_df)
    print("Loaded synthetic reference data into HANA.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
