"""Flask application factory."""
import io
import os
import smtplib
import ssl
from datetime import date, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import certifi

import pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from flasgger import Swagger

from .data import load_capacity, load_orders, row_to_dict
from .scheduling import (
    build_weekly_schedule,
    find_opportunity_orders,
    schedule_to_json,
)
from .ai import ask_llm, build_conflict_prompt, build_explanation_prompt, build_query_prompt

_STATIC_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")

# The Angular application builder emits browser assets into a `browser/`
# subfolder unless `outputPath.browser` is set to "". Resolve to whichever
# layout actually contains index.html so the SPA is served correctly no matter
# how the frontend was built (local `ng build` vs. CF deployment artifacts).
def _resolve_static_dir() -> str:
    browser_dir = os.path.join(_STATIC_ROOT, "browser")
    if os.path.exists(os.path.join(browser_dir, "index.html")):
        return browser_dir
    return _STATIC_ROOT


STATIC_DIR = _resolve_static_dir()


def create_app() -> Flask:
    # Load .env so S/4 (and other) credentials are available locally. In CF the
    # variables come from the platform, where this call is a harmless no-op.
    from dotenv import load_dotenv
    load_dotenv()

    app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/")

    # Pre-load S/4 OData data synchronously so the first user request is instant.
    # This adds ~10 s to gunicorn worker startup, well within CF health-check limits.
    try:
        from .data import load_orders, load_capacity
        load_orders()
        load_capacity()
        print("[startup] Data cache warm.")
    except Exception as e:
        print(f"[startup] Cache warm failed (will retry on first request): {e}")

    Swagger(app, template={
        "info": {
            "title": "FMI Scheduling API",
            "description": "Maintenance scheduling and AI explainability API for Freeport-McMoRan.",
            "version": "1.0.0",
        },
        "tags": [
            {"name": "Meta"},
            {"name": "Orders"},
            {"name": "Capacity"},
            {"name": "Schedule"},
            {"name": "AI"},
        ],
    })

    # Allow the Angular dev server (localhost:4200) during development
    CORS(app, resources={r"/api/*": {"origins": [
        "http://localhost:4200",
        "http://localhost:5000",
    ]}})

    # ── Serve Angular SPA ─────────────────────────────────────────────────
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_spa(path: str):
        if path.startswith("api/"):
            return jsonify({"error": "not found"}), 404
        full = os.path.join(STATIC_DIR, path)
        if path and os.path.exists(full):
            return send_from_directory(STATIC_DIR, path)
        return send_from_directory(STATIC_DIR, "index.html")

    # ── GET /api/meta ─────────────────────────────────────────────────────
    @app.get("/api/meta")
    def meta():
        """Return plants, work centers, equipment, and capacity date range.
        ---
        tags: [Meta]
        responses:
          200:
            description: Metadata for the scheduling UI.
            schema:
              type: object
              properties:
                plants:
                  type: array
                  items: {type: string}
                work_centers_by_plant:
                  type: object
                equipment_by_plant:
                  type: object
                date_range:
                  type: object
                  properties:
                    min: {type: string, example: "2025-01-01"}
                    max: {type: string, example: "2026-11-30"}
        """
        orders = load_orders()
        cap = load_capacity()
        plants = sorted(orders["PLANT_NAME"].dropna().unique().tolist())
        wcs_by_plant = {}
        for plant in plants:
            wcs_by_plant[plant] = sorted(
                cap[cap["PLANT_NAME"] == plant]["OPER_WORK_CENTER"].dropna().unique().tolist()
            )
        equipment_by_plant = {}
        for plant in plants:
            eq = (
                orders[orders["PLANT_NAME"] == plant][["EQUIPMENT_NO", "EQUIPMENT_DESC"]]
                .drop_duplicates()
                .dropna(subset=["EQUIPMENT_NO"])
                .sort_values("EQUIPMENT_DESC")
            )
            equipment_by_plant[plant] = [
                {
                    "id": str(r["EQUIPMENT_NO"]).split(".")[0],
                    "label": f"{str(r['EQUIPMENT_NO']).split('.')[0]} — {str(r['EQUIPMENT_DESC']).strip()}",
                }
                for _, r in eq.iterrows()
            ]
        return jsonify({
            "plants": plants,
            "work_centers_by_plant": wcs_by_plant,
            "equipment_by_plant": equipment_by_plant,
            "date_range": {
                "min": cap["DATE"].min().strftime("%Y-%m-%d"),
                "max": cap["DATE"].max().strftime("%Y-%m-%d"),
            },
        })

    # ── GET /api/orders ───────────────────────────────────────────────────
    @app.get("/api/orders")
    def get_orders():
        """List maintenance orders, optionally filtered by plant.
        ---
        tags: [Orders]
        parameters:
          - name: plant
            in: query
            type: string
            required: false
            description: Filter by plant name (e.g. "Sierrita")
          - name: limit
            in: query
            type: integer
            required: false
            default: 500
        responses:
          200:
            description: Paginated order rows.
            schema:
              type: object
              properties:
                total: {type: integer}
                rows:
                  type: array
                  items: {type: object}
        """
        orders = load_orders()
        plant = request.args.get("plant")
        if plant:
            orders = orders[orders["PLANT_NAME"] == plant]
        limit = int(request.args.get("limit", 500))
        total = len(orders)
        cols = ["ORDER_NO", "OPER_NO", "OPER_SHORT_TEXT", "PRODUCTION_ORDER_HDR_DESC",
                "ORDER_TYPE_CODE", "EQUIPMENT_DESC", "EQUIPMENT_CRITICALITY",
                "OPER_WORK_CENTER", "PRIORITY", "MAINT_ACTIVITY_TYPE", "WO_PHASE",
                "ORDER_SUBPHASE", "SYSTEM_STATUS", "ACTIVITY_WORK_INVOLVE",
                "BASIC_START_DATE", "BASIC_FINISH_DATE", "LATEST_EXECTN_FINISH_DATE",
                "RELEASE_DATE", "FUNC_LOC_CODE"]
        cols = [c for c in cols if c in orders.columns]
        rows = orders[cols].head(limit).apply(row_to_dict, axis=1).tolist()
        return jsonify({"rows": rows, "total": total})

    # ── POST /api/orders/update-basic-start ──────────────────────────────
    @app.post("/api/orders/update-basic-start")
    def update_basic_start():
        """Update the BASIC_START_DATE of a work order in S/4HANA (OData).
        ---
        tags: [Orders]
        parameters:
          - in: body
            name: body
            required: true
            schema:
              type: object
              required: [order_no, basic_start_date]
              properties:
                order_no:
                  type: string
                  example: "400018927139"
                oper_no:
                  type: number
                  example: 10
                basic_start_date:
                  type: string
                  example: "2026-09-01"
        responses:
          200:
            description: Update confirmed.
          400:
            description: Missing or invalid parameters.
          500:
            description: S/4 update failed.
        """
        body = request.get_json(force=True)
        order_no = str(body.get("order_no", "")).strip()
        oper_no  = body.get("oper_no", "")
        new_date = str(body.get("basic_start_date", "")).strip()

        if not order_no or new_date == "":
            return jsonify({"error": "order_no and basic_start_date are required"}), 400

        try:
            date.fromisoformat(new_date)
        except ValueError:
            return jsonify({"error": f"Invalid date: {new_date}"}), 400

        # Write back to S/4HANA via OData (best-effort — skipped if S4 not configured)
        from . import s4_client
        if s4_client.is_configured():
            try:
                s4_client.update_order_basic_start(order_no, new_date)
            except Exception as e:
                return jsonify({"error": f"S/4 update failed: {e}"}), 500

        # Bust the in-process cache so the next schedule generation uses fresh data
        from .data import load_orders
        load_orders.cache_clear()

        return jsonify({"success": True, "order_no": order_no, "basic_start_date": new_date})

    # ── GET /api/capacity ─────────────────────────────────────────────────
    @app.get("/api/capacity")
    def get_capacity():
        """Return work center capacity records (shifts × days).
        ---
        tags: [Capacity]
        parameters:
          - name: plant
            in: query
            type: string
            required: false
            description: Filter by plant name
        responses:
          200:
            description: Up to 200 capacity rows.
            schema:
              type: array
              items: {type: object}
        """
        cap = load_capacity()
        plant = request.args.get("plant")
        if plant:
            cap = cap[cap["PLANT_NAME"] == plant]
        cols = ["WORK_CENTER", "OPER_WORK_CENTER", "DATE", "SHIFTNAME",
                "AVAILABLETIME", "MANPOWER", "CAPACITY"]
        return jsonify(cap[cols].head(200).apply(row_to_dict, axis=1).tolist())

    # ── POST /api/schedule ────────────────────────────────────────────────
    @app.post("/api/schedule")
    def post_schedule():
        """Generate a greedy weekly maintenance schedule (no AI).
        ---
        tags: [Schedule]
        parameters:
          - in: body
            name: body
            required: true
            schema:
              type: object
              required: [plant, work_centers, week_start]
              properties:
                plant:
                  type: string
                  example: "Sierrita"
                work_centers:
                  type: array
                  items: {type: string}
                  example: ["C/C DUST AND LUBRICATION CREW (SICCJ)"]
                week_start:
                  type: string
                  example: "2026-08-24"
                priority_filter:
                  type: array
                  items: {type: string}
                  example: ["Medium", "Low"]
                released_only:
                  type: boolean
                  default: true
        responses:
          200:
            description: Schedule grouped by work center.
          400:
            description: Missing required fields.
        """
        body = request.get_json(force=True)
        plant = body.get("plant", "")
        work_centers = body.get("work_centers", [])
        week_start_str = body.get("week_start", "")
        priority_filter = body.get("priority_filter", ["Medium", "Low"])
        released_only = body.get("released_only", True)

        if not plant or not work_centers or not week_start_str:
            return jsonify({"error": "plant, work_centers and week_start are required"}), 400

        week_start = date.fromisoformat(week_start_str)
        orders = load_orders()
        cap = load_capacity()

        sched = build_weekly_schedule(
            orders, cap, week_start, work_centers, plant,
            priority_filter=priority_filter,
            released_only=released_only,
        )
        return jsonify({
            "week_start": week_start_str,
            "plant": plant,
            "schedule": schedule_to_json(sched),
        })

    # ── POST /api/schedule/opportunity ───────────────────────────────────
    @app.post("/api/schedule/opportunity")
    def post_opportunity():
        """Find future RTS P3/P4 orders on the same equipment (opportunistic scheduling).
        ---
        tags: [Schedule]
        parameters:
          - in: body
            name: body
            required: true
            schema:
              type: object
              required: [equipment_no, week_start, plant]
              properties:
                equipment_no:
                  type: string
                  example: "100000011044"
                week_start:
                  type: string
                  example: "2026-08-24"
                plant:
                  type: string
                  example: "Sierrita"
        responses:
          200:
            description: List of opportunistic orders and week buckets.
          400:
            description: Missing required fields.
        """
        body = request.get_json(force=True)
        equipment_no = body.get("equipment_no", "")
        week_start_str = body.get("week_start", "")
        plant = body.get("plant", "")

        if not equipment_no or not week_start_str or not plant:
            return jsonify({"error": "equipment_no, week_start and plant are required"}), 400

        week_start = date.fromisoformat(week_start_str)
        orders = load_orders()
        result = find_opportunity_orders(orders, equipment_no, week_start, plant)

        if result.empty:
            return jsonify({"orders": [], "week_buckets": []})

        rows = result.apply(row_to_dict, axis=1).tolist()
        buckets = result["WEEK_BUCKET"].unique().tolist() if "WEEK_BUCKET" in result.columns else []
        return jsonify({"orders": rows, "week_buckets": buckets})

    # ── POST /api/schedule/opportunity/batch ─────────────────────────────
    @app.post("/api/schedule/opportunity/batch")
    def post_opportunity_batch():
        """Find opportunistic orders for multiple equipment numbers in one call.
        ---
        tags: [Schedule]
        parameters:
          - in: body
            name: body
            required: true
            schema:
              type: object
              required: [equipment_nos, week_start, plant]
              properties:
                equipment_nos:
                  type: array
                  items: {type: string}
                  example: ["100000011044", "100000011045"]
                week_start:
                  type: string
                  example: "2026-08-24"
                plant:
                  type: string
                  example: "Sierrita"
        responses:
          200:
            description: Opportunistic orders grouped by equipment.
          400:
            description: Missing required fields.
        """
        body = request.get_json(force=True)
        equipment_nos = body.get("equipment_nos", [])
        week_start_str = body.get("week_start", "")
        plant = body.get("plant", "")

        if not equipment_nos or not week_start_str or not plant:
            return jsonify({"error": "equipment_nos, week_start and plant are required"}), 400

        week_start = date.fromisoformat(week_start_str)
        orders = load_orders()

        results = []
        for eq_no in equipment_nos:
            result = find_opportunity_orders(orders, eq_no, week_start, plant)
            if result.empty:
                continue
            eq_desc = str(result["EQUIPMENT_DESC"].iloc[0]) if "EQUIPMENT_DESC" in result.columns else str(eq_no)
            rows = result.apply(row_to_dict, axis=1).tolist()
            results.append({
                "equipment_no": str(eq_no),
                "equipment_desc": eq_desc,
                "orders": rows,
            })

        return jsonify({"equipment": results})

    # ── POST /api/ai/explain ─────────────────────────────────────────────
    @app.post("/api/ai/explain")
    def ai_explain():
        """Generate an LLM explanation for why each order was scheduled or deferred.
        ---
        tags: [AI]
        parameters:
          - in: body
            name: body
            required: true
            schema:
              type: object
              required: [schedule, week_start]
              properties:
                schedule:
                  type: array
                  description: schedule_json from POST /api/schedule
                  items: {type: object}
                week_start:
                  type: string
                  example: "2026-08-24"
                opp_added:
                  type: array
                  items: {type: object}
                  description: Opportunistic orders added to the schedule
        responses:
          200:
            description: LLM narrative explanation per work center.
            schema:
              type: object
              properties:
                text: {type: string}
        """
        body = request.get_json(force=True)
        schedule_json = body.get("schedule", [])
        week_start = body.get("week_start", "")
        opp_added = body.get("opp_added", [])
        prompt = build_explanation_prompt(schedule_json, week_start, opp_added or None)
        return jsonify({"text": ask_llm(prompt)})

    # ── POST /api/ai/conflict ─────────────────────────────────────────────
    @app.post("/api/ai/conflict")
    def ai_conflict():
        """Generate an LLM analysis of overloaded, underused, and deferred orders.
        ---
        tags: [AI]
        parameters:
          - in: body
            name: body
            required: true
            schema:
              type: object
              required: [schedule]
              properties:
                schedule:
                  type: array
                  description: schedule_json from POST /api/schedule
                  items: {type: object}
        responses:
          200:
            description: Numbered action list from the LLM.
            schema:
              type: object
              properties:
                text: {type: string}
        """
        body = request.get_json(force=True)
        schedule_json = body.get("schedule", [])
        prompt = build_conflict_prompt(schedule_json)
        return jsonify({"text": ask_llm(prompt)})

    # ── POST /api/ai/query ────────────────────────────────────────────────
    @app.post("/api/ai/query")
    def ai_query():
        """Ask the LLM a free-text question about the current schedule.
        ---
        tags: [AI]
        parameters:
          - in: body
            name: body
            required: true
            schema:
              type: object
              required: [question]
              properties:
                question:
                  type: string
                  example: "Which work center has the highest overload risk?"
                context:
                  type: string
                  description: Schedule summary text to ground the answer
        responses:
          200:
            description: LLM answer.
            schema:
              type: object
              properties:
                text: {type: string}
          400:
            description: question is required.
        """
        body = request.get_json(force=True)
        question = body.get("question", "").strip()
        context = body.get("context", "")
        if not question:
            return jsonify({"error": "question is required"}), 400
        prompt = build_query_prompt(question, context)
        return jsonify({"text": ask_llm(prompt)})

    # ── POST /api/schedule/confirm ────────────────────────────────────────
    @app.post("/api/schedule/confirm")
    def confirm_schedule():
        """Write the scheduled dates back to S/4HANA (pin each operation's constraint date).
        ---
        tags: [Schedule]
        parameters:
          - in: body
            name: body
            required: true
            schema:
              type: object
              required: [schedule]
              properties:
                schedule:
                  type: array
                  description: schedule_json (each work center's scheduled ops carry SCHED_DATE)
                  items: {type: object}
        responses:
          200:
            description: Per-operation write results.
            schema:
              type: object
              properties:
                total: {type: integer}
                updated: {type: integer}
                failed:
                  type: array
                  items: {type: object}
          400:
            description: No scheduled operations, or S/4 not configured.
          500:
            description: S/4 write failed.
        """
        from . import s4_client
        if not s4_client.is_configured():
            return jsonify({"error": "S/4HANA connection is not configured"}), 400

        body = request.get_json(force=True)
        schedule_json = body.get("schedule", [])

        items = []
        for wc in schedule_json:
            for op in wc.get("scheduled", []):
                sched_date = op.get("SCHED_DATE") or op.get("BASIC_START_DATE")
                if not sched_date:
                    continue
                items.append({
                    "order_no": op.get("ORDER_NO"),
                    "oper_no": op.get("OPER_NO"),
                    "date_iso": sched_date,
                })

        if not items:
            return jsonify({"error": "no scheduled operations to confirm"}), 400

        try:
            results = s4_client.confirm_operations(items)
        except Exception as e:
            return jsonify({"error": f"S/4 write failed: {e}"}), 500

        failed = [r for r in results if not r["ok"]]
        return jsonify({
            "total": len(results),
            "updated": len(results) - len(failed),
            "failed": failed,
        })

    # ── POST /api/schedule/export ─────────────────────────────────────────
    @app.post("/api/schedule/export")
    def export_schedule():
        """Export the schedule as a downloadable CSV file.
        ---
        tags: [Schedule]
        parameters:
          - in: body
            name: body
            required: true
            schema:
              type: object
              required: [schedule]
              properties:
                schedule:
                  type: array
                  items: {type: object}
                week_start:
                  type: string
                  example: "2026-08-24"
        produces:
          - text/csv
        responses:
          200:
            description: CSV file download.
        """
        body = request.get_json(force=True)
        schedule_json = body.get("schedule", [])
        week_start = body.get("week_start", "unknown")
        rows = []
        for wc in schedule_json:
            for op in wc["scheduled"]:
                op["SCHEDULE_STATUS"] = op.pop("_OPPORTUNISTIC", None) and "Opportunistic" or "Scheduled"
                op["RECOMMENDED_WEEK"] = week_start
                rows.append(op)
            for op in wc["unscheduled"]:
                op["SCHEDULE_STATUS"] = "Deferred"
                op["RECOMMENDED_WEEK"] = week_start
                rows.append(op)
        df = pd.DataFrame(rows)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return send_file(
            io.BytesIO(buf.getvalue().encode()),
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"fmi_schedule_{week_start}.csv",
        )

    # ── POST /api/schedule/email ──────────────────────────────────────────
    @app.post("/api/schedule/email")
    def send_schedule_email():
        """Send the schedule as an HTML email via Gmail.
        ---
        tags: [Schedule]
        parameters:
          - in: body
            name: body
            required: true
            schema:
              type: object
              required: [schedule, to]
              properties:
                schedule:
                  type: array
                  items: {type: object}
                week_start:
                  type: string
                  example: "2026-08-24"
                plant:
                  type: string
                  example: "Sierrita"
                to:
                  type: string
                  example: "maintenance.manager@fmi.com"
        responses:
          200:
            description: Email sent successfully.
          400:
            description: Recipient address is required.
          500:
            description: Gmail credentials not configured.
        """
        body = request.get_json(force=True)
        schedule_json = body.get("schedule", [])
        week_start = body.get("week_start", "")
        plant = body.get("plant", "")
        to_addr = body.get("to", "").strip()

        if not to_addr:
            return jsonify({"error": "to (recipient email) is required"}), 400

        gmail_user = os.environ.get("GMAIL_USER", "")
        gmail_pass = os.environ.get("GMAIL_APP_PASSWORD", "")
        if not gmail_user or not gmail_pass:
            return jsonify({"error": "Gmail credentials not configured"}), 500

        html = _build_schedule_html(schedule_json, week_start, plant)
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Maintenance Schedule – {plant} – Week {week_start}"
        msg["From"] = gmail_user
        msg["To"] = to_addr
        msg.attach(MIMEText(html, "html"))

        ctx = ssl.create_default_context(cafile=certifi.where())
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
            server.login(gmail_user, gmail_pass)
            server.sendmail(gmail_user, to_addr, msg.as_string())

        return jsonify({"success": True})

    return app


_APP_URL = "https://fmi-scheduling.cfapps.eu10-005.hana.ondemand.com/"


def _build_schedule_html(schedule: list, week_start: str, plant: str) -> str:
    """Build an HTML email body mirroring the frontend schedule table."""
    STATUS_COLOR = {"positive": "#107e3e", "critical": "#e9730c", "negative": "#bb0000"}

    def load_color(pct: float) -> str:
        if pct <= 80:
            return STATUS_COLOR["positive"]
        if pct <= 100:
            return STATUS_COLOR["critical"]
        return STATUS_COLOR["negative"]

    header_style = (
        "font-family:Arial,sans-serif;background:#0a6ed1;color:#fff;"
        "padding:16px 24px;margin:0;"
    )
    body_style = "font-family:Arial,sans-serif;color:#32363a;padding:24px;"
    table_style = (
        "border-collapse:collapse;width:100%;font-size:13px;"
        "margin-top:12px;margin-bottom:24px;"
    )
    th_style = (
        "background:#f2f2f2;border:1px solid #d1d1d6;"
        "padding:6px 10px;text-align:left;white-space:nowrap;"
    )
    td_style = "border:1px solid #d1d1d6;padding:5px 10px;white-space:nowrap;"

    cols = [
        ("Work Center", "_work_center"),
        ("Rec. Date", "_sched_week"),
        ("Order No", "ORDER_NO"),
        ("Op", "OPER_NO"),
        ("Operation", "OPER_SHORT_TEXT"),
        ("Type", "ORDER_TYPE_CODE"),
        ("Equipment", "EQUIPMENT_DESC"),
        ("Priority", "PRIORITY"),
        ("Criticality", "EQUIPMENT_CRITICALITY"),
        ("Work (h)", "ACTIVITY_WORK_INVOLVE"),
        ("Order Date", "RELEASE_DATE"),
        ("Basic Start", "BASIC_START_DATE"),
        ("Basic Finish", "BASIC_FINISH_DATE"),
    ]

    sections_html = ""
    for wc in schedule:
        wc_name = wc.get("work_center", "")
        load_pct = wc.get("load_pct", 0)
        cap_used = wc.get("capacity_used", 0)
        cap_avail = wc.get("capacity_available", 0)
        n_sched = len(wc.get("scheduled", []))
        n_deferred = len(wc.get("unscheduled", []))
        color = load_color(load_pct)

        kpi_html = (
            f"<p style='margin:4px 0;font-size:14px;font-weight:bold;'>{wc_name}</p>"
            f"<p style='margin:2px 0;color:{color};font-weight:bold;'>{load_pct:.0f}% load</p>"
            f"<p style='margin:2px 0;font-size:12px;'>"
            f"{n_sched} scheduled &nbsp;·&nbsp; {n_deferred} deferred"
            f"&nbsp;·&nbsp; {cap_used:.1f} / {cap_avail:.1f} h</p>"
        )

        rows_html = ""
        n_cols = len(cols)
        for op in wc.get("scheduled", []):
            op_aug = dict(op)
            op_aug["_work_center"] = wc_name
            op_aug["_sched_week"] = op_aug.get("SCHED_DATE") or week_start

            def cell(key, _op=op_aug):
                val = _op.get(key, "")
                if val is None:
                    return "—"
                if key in ("BASIC_START_DATE", "BASIC_FINISH_DATE", "RELEASE_DATE") and val:
                    return str(val)[:10]
                if key == "ACTIVITY_WORK_INVOLVE":
                    try:
                        return f"{float(val):.1f}"
                    except (TypeError, ValueError):
                        return str(val)
                if key == "OPER_NO" and val:
                    return str(val).rstrip("0").rstrip(".")
                return str(val) if val != "" else "—"

            tds = "".join(
                f"<td style='{td_style}'>{cell(k)}</td>"
                for _, k in cols
            )
            rows_html += f"<tr>{tds}</tr>"

        ths = "".join(f"<th style='{th_style}'>{label}</th>" for label, _ in cols)
        empty_row = '<tr><td colspan="' + str(n_cols) + '" style="' + td_style + ';color:#888;">No scheduled operations</td></tr>'
        table_html = (
            f"<table style='{table_style}'>"
            f"<thead><tr>{ths}</tr></thead>"
            f"<tbody>{rows_html if rows_html else empty_row}</tbody>"
            f"</table>"
        )

        sections_html += (
            f"<div style='margin-bottom:32px;'>"
            f"<div style='background:#f7f7f7;padding:10px 16px;border-left:4px solid {color};margin-bottom:8px;'>"
            f"{kpi_html}</div>"
            f"{table_html}</div>"
        )

    confirm_block = (
        "<div style='text-align:center;margin-bottom:28px;'>"
        "<a href='" + _APP_URL + "'"
        " style='display:inline-block;padding:12px 28px;background:#0a6ed1;color:#fff;"
        "font-family:Arial,sans-serif;font-size:15px;font-weight:600;text-decoration:none;"
        "border-radius:4px;'>Confirm Schedule Here</a>"
        "</div>"
    )

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style='margin:0;padding:0;background:#fafafa;'>
  <div style='{header_style}'>
    <h2 style='margin:0;font-size:20px;'>Maintenance Schedule Recommendation</h2>
    <p style='margin:4px 0 0;font-size:14px;opacity:.85;'>{plant} &nbsp;·&nbsp; Week of {week_start}</p>
  </div>
  <div style='{body_style}'>
    {confirm_block}{sections_html}
  </div>
</body>
</html>"""
