"""
Token simulation — mide tokens reales de cada llamada IA del flujo de scheduling.
Corre sin conexión a HANA ni SAP GenAI Hub; usa los CSVs locales.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from datetime import date

# ── 1. Carga de datos ────────────────────────────────────────────────────────
from backend.data import load_orders, load_capacity
from backend.scheduling import build_weekly_schedule, schedule_to_json
from backend.ai import build_explanation_prompt, build_conflict_prompt, build_query_prompt

orders = load_orders()
cap    = load_capacity()

# ── 2. Parámetros representativos ────────────────────────────────────────────
plant = "Sierrita"

work_centers = [
    "C/C DUST AND LUBRICATION CREW (SICCJ)",
    "C/C CRUSHER OVERHAUL & MAINT. CREW (SICCK)",
    "C/C PRIMARY CRUSHING MAINTENANCE (SICCH)",
]

# Semana con el mayor número de órdenes candidatas que también tiene capacity
week_start = date(2026, 8, 24)

print(f"\n{'='*60}")
print(f"  SIMULACIÓN DE TOKENS — FMI AI SCHEDULING")
print(f"{'='*60}")
print(f"  Planta       : {plant}")
print(f"  Work Centers : {work_centers}")
print(f"  Semana       : {week_start}")

# ── 3. Generar schedule (sin IA) ─────────────────────────────────────────────
sched_dict   = build_weekly_schedule(orders, cap, week_start, work_centers, plant)
schedule_json = schedule_to_json(sched_dict)

total_scheduled = sum(len(w["scheduled"]) for w in schedule_json)
total_deferred  = sum(len(w["unscheduled"]) for w in schedule_json)
print(f"\n  Órdenes agendadas  : {total_scheduled}")
print(f"  Órdenes diferidas  : {total_deferred}")

# ── 4. Construir los 3 prompts ───────────────────────────────────────────────
opp_added = []   # sin ordenes oportunistas para el caso base

prompt_explain  = build_explanation_prompt(schedule_json, str(week_start), opp_added or None)
prompt_conflict = build_conflict_prompt(schedule_json)

# Contexto para query: resumen compacto del schedule
context_lines = []
for wc in schedule_json:
    context_lines.append(
        f"{wc['work_center']}: {wc['capacity_used']:.0f}h/{wc['capacity_available']:.0f}h "
        f"({wc['load_pct']:.0f}%) — {len(wc['scheduled'])} scheduled, {len(wc['unscheduled'])} deferred"
    )
context = "\n".join(context_lines)
sample_question = "Which work center has the highest overload risk next week?"
prompt_query    = build_query_prompt(sample_question, context)

prompts = {
    "explain  (POST /api/ai/explain)":  prompt_explain,
    "conflict (POST /api/ai/conflict)": prompt_conflict,
    "query    (POST /api/ai/query)":    prompt_query,
}

# ── 5. Contar tokens (tiktoken — mismo encoding GPT-4/4o/5) ─────────────────
try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")   # gpt-4 / gpt-5 family
    method = "tiktoken cl100k_base (exact)"
except ImportError:
    # Fallback: estimación por palabras (~1.3 tokens/palabra para inglés técnico)
    class _FakeEnc:
        def encode(self, text):
            return text.split()
    enc = _FakeEnc()
    method = "estimación palabras×1.3 (install tiktoken para exacto)"

# GPT-5.4 pricing via SAP GenAI Hub (referencia OpenAI gpt-4o equivalente)
# Input: $2.50 / 1M tokens   Output: $10.00 / 1M tokens
INPUT_COST_PER_1M  = 2.50
OUTPUT_COST_PER_1M = 10.00

# Estimación de output: ~400 tokens por respuesta (típico para estas explicaciones)
EST_OUTPUT_TOKENS = 400

print(f"\n  Método de conteo : {method}")
print(f"  Modelo objetivo  : gpt-5.4 (via SAP GenAI Hub)")
print(f"\n{'─'*60}")
print(f"  {'LLAMADA':<42} {'INPUT':>7}  {'OUTPUT':>7}  {'TOTAL':>7}")
print(f"{'─'*60}")

grand_input  = 0
grand_output = 0

for name, prompt in prompts.items():
    raw_tokens = len(enc.encode(prompt))
    # tiktoken solo cuenta palabras en fallback — ajustar ×1.3
    if method.startswith("estimación"):
        input_tokens = int(raw_tokens * 1.3)
    else:
        input_tokens = raw_tokens

    output_tokens = EST_OUTPUT_TOKENS
    total = input_tokens + output_tokens
    grand_input  += input_tokens
    grand_output += output_tokens

    print(f"  {name:<42} {input_tokens:>7,}  {output_tokens:>7,}  {total:>7,}")

grand_total = grand_input + grand_output
print(f"{'─'*60}")
print(f"  {'TOTAL (las 3 llamadas juntas)':<42} {grand_input:>7,}  {grand_output:>7,}  {grand_total:>7,}")

# ── 6. Costo estimado ─────────────────────────────────────────────────────────
cost_input  = grand_input  / 1_000_000 * INPUT_COST_PER_1M
cost_output = grand_output / 1_000_000 * OUTPUT_COST_PER_1M
cost_total  = cost_input + cost_output

print(f"\n{'─'*60}")
print(f"  Costo estimado por sesión completa de scheduling:")
print(f"    Input  {grand_input:>6,} tokens × $2.50/1M  = ${cost_input:.5f}")
print(f"    Output {grand_output:>6,} tokens × $10.00/1M = ${cost_output:.5f}")
print(f"    TOTAL                             = ${cost_total:.5f}")
print(f"{'─'*60}")

# ── 7. Detalle de los prompts ────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  DETALLE DE PROMPTS")
print(f"{'='*60}")
for name, prompt in prompts.items():
    raw = len(enc.encode(prompt))
    tok = int(raw * 1.3) if method.startswith("estimación") else raw
    print(f"\n── {name} ({tok:,} tokens input) ──")
    print(prompt[:600] + (" [... truncado]" if len(prompt) > 600 else ""))

print(f"\n{'='*60}")
print(f"  RESUMEN EJECUTIVO")
print(f"{'='*60}")
print(f"  • Una sola sesión de scheduling (explain + conflict + query)")
print(f"    consume aprox. {grand_total:,} tokens en total.")
print(f"  • El prompt más pesado es 'explain' porque incluye hasta 5")
print(f"    órdenes por WC (scheduled) + hasta 3 diferidas.")
print(f"  • Costo por scheduling: ~${cost_total:.4f} USD")
print(f"  • A 100 schedules/día → ${cost_total*100:.2f} USD/día")
print(f"  • A 100 schedules/día → ${cost_total*100*30:.2f} USD/mes")
print()
