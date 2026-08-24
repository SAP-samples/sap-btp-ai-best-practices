"""
evaluator.py
------------
Orquestador principal del modulo de evaluacion.
"""

import json
import logging
from pathlib import Path
from typing import Any

from modules.evaluation.field_analyzer import analyze_fields
from modules.evaluation.generate_report import build_executive_summary, save_all
from modules.evaluation.llm_evaluator import evaluate_with_llm
from modules.evaluation.score_calculator import calculate_scores

logger = logging.getLogger(__name__)

GENAI_OUTPUT_DIR: Path = Path(__file__).parent.parent.parent / "output" / "genai"

INPUT_FILES = {
    "sap":            GENAI_OUTPUT_DIR / "sap_result.json",
    "llm_prompting":  GENAI_OUTPUT_DIR / "llm_multimodal_prompting.json",
    "llm_structured": GENAI_OUTPUT_DIR / "llm_multimodal_structured.json",
}


class EvaluationError(Exception):
    pass


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise EvaluationError(
            f"Archivo no encontrado: {path}\n"
            "Ejecute primero la opcion 4 (Process Invoice with GenAI)."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_evaluation() -> dict[str, Any]:
    """
    Pipeline completo de evaluacion:
    1. Carga resultados de output/genai/
    2. Analiza campos (completitud, missing, conflictos)
    3. Calcula scores por metodo
    4. Evaluacion inteligente con LLM
    5. Genera reportes y guarda outputs
    """
    # 1. Cargar resultados
    logger.info("Loading results from output/genai/...")
    sap_raw   = _load_json(INPUT_FILES["sap"])
    llm_p1    = _load_json(INPUT_FILES["llm_prompting"])
    llm_p2    = _load_json(INPUT_FILES["llm_structured"])
    logger.info("Results loaded successfully.")

    # 2. Analyze fields
    logger.info("Analyzing fields...")
    analysis = analyze_fields(sap_raw, llm_p1, llm_p2)

    # 3. Calculate scores
    logger.info("Calculating scores...")
    scores_result = calculate_scores(analysis)

    # 4. LLM evaluation
    logger.info("Running intelligent LLM evaluation...")
    llm_eval = evaluate_with_llm(analysis, scores_result)

    # 5. Generate summary and save
    logger.info("Generating reports...")
    summary_text = build_executive_summary(scores_result, analysis, llm_eval)
    paths = save_all(analysis, scores_result, llm_eval, summary_text)

    return {
        "analysis": analysis,
        "scores": scores_result,
        "llm_evaluation": llm_eval,
        "summary": summary_text,
        "output_paths": paths,
    }