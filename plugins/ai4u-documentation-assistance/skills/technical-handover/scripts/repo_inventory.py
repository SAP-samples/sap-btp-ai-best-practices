#!/usr/bin/env python3
"""
Build a factual inventory of a code repository to ground a technical handover.

Usage:
    python repo_inventory.py <repo_path> --out inventory.json [--md inventory.md]

Collects (heuristically, never values of secrets):
  directory map · languages · dependency manifests · deployment descriptors
  HTTP routes (FastAPI/Starlette, Flask, Express/Nest-ish, SAP CAP .cds services)
  environment variable NAMES referenced in code / .env.example / manifests
  AI signals: GenAI/LLM/agent SDKs, classical ML, optimisation solvers, SAP AI services
  prompt-like assets, tests, UI routes
The output is a map for reading the code, not a substitute for reading it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "env", "__pycache__", "dist", "build", ".next", ".pytest_cache",
             ".ruff_cache", ".mypy_cache", ".idea", ".vscode", "coverage", ".tox", "site-packages", ".cache", "job_runs",
             "outputs", ".DS_Store", "target", "gen", ".turbo", ".parcel-cache"}
CODE_EXT = {".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".java", ".cds", ".go", ".cs", ".kt", ".rb", ".sql", ".hdbtable"}
MAX_FILE = 1_500_000

AI_SIGNALS = {
    # GenAI / LLM / agents
    "openai": ("genai", r"\bfrom openai\b|\bimport openai\b|['\"]openai['\"]"),
    "anthropic": ("genai", r"\banthropic\b"),
    "sap-genai-hub-sdk": ("genai", r"gen_ai_hub|generative[-_]ai[-_]hub|sap[-_]ai[-_]sdk|@sap-ai-sdk|ai_core_sdk|AICORE_"),
    "langchain": ("genai", r"\blangchain"),
    "langgraph": ("agent", r"\blanggraph"),
    "deepagents": ("agent", r"\bdeepagents\b"),
    "llama-index": ("genai", r"llama_index|llama-index"),
    "litellm": ("genai", r"\blitellm\b"),
    "crewai/autogen": ("agent", r"\bcrewai\b|\bautogen\b"),
    "mcp": ("agent", r"\bmcp\b\.|from mcp\b|modelcontextprotocol|FastMCP"),
    "embeddings/vector": ("genai", r"embedding|REAL_VECTOR|COSINE_SIMILARITY|vectorstore|faiss|chromadb|pgvector"),
    "structured-output": ("genai", r"response_format|with_structured_output|json_schema"),
    "tool-calling": ("agent", r"@tool\b|bind_tools|tools\s*=\s*\[|function_call|tool_choice"),
    # Classical ML
    "scikit-learn": ("ml", r"\bsklearn\b|scikit-learn"),
    "xgboost": ("ml", r"\bxgboost\b"),
    "lightgbm": ("ml", r"\blightgbm\b"),
    "catboost": ("ml", r"\bcatboost\b"),
    "pytorch": ("ml", r"\bimport torch\b|\bfrom torch\b"),
    "tensorflow/keras": ("ml", r"\btensorflow\b|\bkeras\b"),
    "statsmodels": ("ml", r"\bstatsmodels\b"),
    "prophet": ("ml", r"\bprophet\b"),
    "hana-ml/PAL/APL": ("ml", r"hana_ml|\bPAL\b|\bAPL\b"),
    "sap-rpt-1": ("ml", r"\bRPT[-_]?1\b|rpt1|RPTClient"),
    "lifelines/survival": ("ml", r"\blifelines\b|sksurv"),
    "joblib/pickle models": ("ml", r"joblib\.(dump|load)|pickle\.(dump|load)"),
    # Optimisation
    "pyomo": ("optimization", r"\bpyomo\b"),
    "highs": ("optimization", r"\bhighspy\b|\bHiGHS\b|appsi_highs"),
    "or-tools": ("optimization", r"ortools"),
    "pulp": ("optimization", r"\bpulp\b"),
    "gurobi": ("optimization", r"gurobipy|\bGurobi\b"),
    "cplex/docplex": ("optimization", r"docplex|\bcplex\b"),
    "cvxpy": ("optimization", r"\bcvxpy\b"),
    "scipy.optimize": ("optimization", r"scipy\.optimize|from scipy import optimize|linprog|milp\("),
    "pymoo/pareto": ("optimization", r"\bpymoo\b|pareto|epsilon[-_ ]constraint"),
}

ENV_PATTERNS = [
    r"os\.getenv\(\s*['\"]([A-Z][A-Z0-9_]{2,})['\"]",
    r"os\.environ\.get\(\s*['\"]([A-Z][A-Z0-9_]{2,})['\"]",
    r"os\.environ\[\s*['\"]([A-Z][A-Z0-9_]{2,})['\"]\s*\]",
    r"getenv\(\s*['\"]([A-Z][A-Z0-9_]{2,})['\"]",
    r"process\.env\.([A-Z][A-Z0-9_]{2,})",
    r"process\.env\[\s*['\"]([A-Z][A-Z0-9_]{2,})['\"]\s*\]",
    r"import\.meta\.env\.([A-Z][A-Z0-9_]{2,})",
    r"Field\([^)]*env=['\"]([A-Z][A-Z0-9_]{2,})['\"]",
    r"\$\{([A-Z][A-Z0-9_]{2,})(?::-[^}]*)?\}",        # ${VAR} in yaml/json/sh
    r"env\(\s*['\"]([A-Z][A-Z0-9_]{2,})['\"]",           # env("VAR") helpers
]

DEPLOY_FILES = ["manifest.yml", "manifest.yaml", "mta.yaml", "mta.yml", "xs-security.json", "xs-app.json", "Dockerfile",
                "docker-compose.yml", "docker-compose.yaml", "Procfile", "runtime.txt", ".cfignore", "deploy.sh",
                "helm", "k8s", "kustomization.yaml", "Chart.yaml", "default-env.json", "package.json", "vercel.json",
                ".github/workflows", "azure-pipelines.yml", "Jenkinsfile", ".gitlab-ci.yml", "cds-plugin"]
DEP_FILES = ["requirements.txt", "pyproject.toml", "Pipfile", "setup.py", "setup.cfg", "package.json", "pom.xml",
             "build.gradle", "go.mod", "environment.yml", "poetry.lock", "uv.lock"]
SECRET_KEY_RE = re.compile(r"(?i)(secret|password|passwd|pwd|token|api[_-]?key|private[_-]?key|client[_-]?secret|credential)")


def walk(repo: Path):
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".") or d in (".github",)]
        for f in files:
            yield Path(root) / f


def read(p: Path):
    try:
        if p.stat().st_size > MAX_FILE:
            return ""
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def fastapi_routes(files):
    """Heuristic FastAPI/Starlette route extraction incl. router prefixes and include_router prefixes."""
    routes = []
    router_prefix = {}   # (module_stem, var) -> prefix
    include_prefix = defaultdict(list)  # module_stem or var -> [prefix]
    texts = {}
    for p in files:
        if p.suffix != ".py":
            continue
        t = read(p)
        if "APIRouter" not in t and "FastAPI" not in t and ".route(" not in t:
            continue
        texts[p] = t
        for m in re.finditer(r"(\w+)\s*=\s*APIRouter\(([^)]*)\)", t, re.S):
            pm = re.search(r"prefix\s*=\s*['\"]([^'\"]*)['\"]", m.group(2))
            router_prefix[(p.stem, m.group(1))] = pm.group(1) if pm else ""
        for m in re.finditer(r"include_router\(\s*([\w\.]+)([^)]*)\)", t, re.S):
            target = m.group(1)
            pm = re.search(r"prefix\s*=\s*['\"]([^'\"]*)['\"]", m.group(2))
            key = target.split(".")[0] if "." in target else target
            include_prefix[key].append(pm.group(1) if pm else "")
    # Matched on the whole file, so "@router.post(" with the path (or path="...") on the next line is found too.
    dec = re.compile(r"@(\w+)\.(get|post|put|patch|delete|options|head|websocket|api_route|route)\(\s*(?:path\s*=\s*)?['\"]([^'\"]*)['\"]")
    for p, t in texts.items():
        lines = t.splitlines()
        for m in dec.finditer(t):
            i = t.count("\n", 0, m.start())  # line of the decorator
            var, method, path = m.group(1), m.group(2).upper(), m.group(3)
            # decorator arguments and the handler: up to the first def after the decorator
            handler, summary, j = "", "", i + 1
            for j in range(i + 1, min(i + 20, len(lines))):
                hm = re.match(r"\s*(async\s+)?def\s+(\w+)", lines[j])
                if hm:
                    handler = hm.group(2)
                    for k in range(j + 1, min(j + 6, len(lines))):
                        dm = re.match(r'\s*[ru]?["\']{3}(.*?)(["\']{3})?\s*$', lines[k])
                        if dm:
                            summary = dm.group(1).strip()
                            break
                    break
            args = "\n".join(lines[i:j])
            if method in ("ROUTE", "API_ROUTE"):
                mm = re.search(r"methods\s*=\s*\[([^\]]*)\]", args)
                method = ",".join(re.findall(r"['\"](\w+)['\"]", mm.group(1))).upper() if mm else "GET"
            rp = router_prefix.get((p.stem, var), "")
            ip = include_prefix.get(p.stem, []) or include_prefix.get(var, [])
            prefix = (ip[0] if ip else "") + rp
            sm = re.search(r"summary\s*=\s*['\"]([^'\"]+)", args)
            if sm:
                summary = sm.group(1)
            deps = "auth?" if re.search(r"Depends\(|Security\(", "\n".join(lines[i:j + 8])) else ""
            routes.append({"framework": "fastapi", "method": method, "path": (prefix + path) or "/", "raw_path": path,
                           "prefix_resolution": "heuristic" if prefix else "none", "handler": handler,
                           "summary": summary[:160], "dependencies": deps, "file": str(p), "line": i + 1})
    return routes


def other_routes(files):
    routes = []
    flask = re.compile(r"@(\w+)\.route\(\s*['\"]([^'\"]+)['\"](?:[^)]*methods\s*=\s*\[([^\]]*)\])?")
    express = re.compile(r"\b(app|router|server)\.(get|post|put|patch|delete|use|all)\(\s*['\"`](/[^'\"`]*)['\"`]")
    nest = re.compile(r"@(Get|Post|Put|Patch|Delete)\(\s*['\"]?([^'\")]*)['\"]?\s*\)")
    cds = re.compile(r"^\s*service\s+(\w+)(?:\s*@\(path:\s*['\"]([^'\"]+)['\"]\))?", re.M)
    cds_entity = re.compile(r"^\s*(entity|action|function)\s+(\w+)", re.M)
    for p in files:
        if p.suffix not in (".py", ".js", ".mjs", ".ts", ".cds"):
            continue
        t = read(p)
        if p.suffix == ".py" and "FastAPI" not in t and "APIRouter" not in t:
            for i, line in enumerate(t.splitlines()):
                m = flask.search(line)
                if m:
                    meth = ",".join(re.findall(r"['\"](\w+)['\"]", m.group(3) or "")) or "GET"
                    routes.append({"framework": "flask", "method": meth.upper(), "path": m.group(2), "file": str(p), "line": i + 1})
        elif p.suffix in (".js", ".mjs", ".ts"):
            for i, line in enumerate(t.splitlines()):
                m = express.search(line)
                if m and m.group(2) != "use":
                    routes.append({"framework": "express", "method": m.group(2).upper(), "path": m.group(3), "file": str(p), "line": i + 1})
                n = nest.search(line)
                if n:
                    routes.append({"framework": "nest", "method": n.group(1).upper(), "path": "/" + n.group(2).lstrip("/"), "file": str(p), "line": i + 1})
        elif p.suffix == ".cds":
            for m in cds.finditer(t):
                routes.append({"framework": "cap", "method": "ODATA", "path": m.group(2) or f"/odata/v4/{m.group(1)}",
                               "service": m.group(1), "members": [x[1] for x in cds_entity.findall(t)][:40], "file": str(p)})
    return routes


def ui_routes(files):
    out = []
    for p in files:
        name = p.name.lower()
        if p.suffix in (".js", ".ts", ".jsx", ".tsx", ".vue", ".svelte") and ("route" in name or "router" in name or "navigation" in name or name in ("main.js", "main.ts", "app.js", "app.tsx", "app.jsx")):
            t = read(p)
            t = re.sub(r"/\*.*?\*/", "", t, flags=re.S)           # block comments
            t = "\n".join(l for l in t.splitlines() if not l.strip().startswith("//"))
            for m in re.finditer(r"path\s*[:=]\s*['\"`]([^'\"`]+)['\"`]", t):
                out.append({"path": m.group(1), "file": str(p)})
        if p.suffix in (".js", ".ts", ".jsx", ".tsx", ".vue", ".svelte", ".html"):
            t = read(p)
            for m in re.finditer(r"location\.hash\s*=\s*['\"`]#?([\w\-/]+)['\"`]", t):   # hash routing
                out.append({"path": "#" + m.group(1), "file": str(p), "kind": "hash"})
            for m in re.finditer(r"href=['\"]#/([\w\-/]+)['\"]", t):
                out.append({"path": "#/" + m.group(1), "file": str(p), "kind": "hash"})
        if name == "manifest.json" and "sap.ui5" in read(p):
            t = read(p)
            for m in re.finditer(r'"pattern"\s*:\s*"([^"]*)"', t):
                out.append({"path": m.group(1) or "/", "file": str(p), "kind": "ui5"})
    seen, uniq = set(), []
    for r in out:
        if r["path"] not in seen:
            seen.add(r["path"]); uniq.append(r)
    return uniq


def env_vars(files, repo):
    found = defaultdict(set)
    for p in files:
        if p.suffix in CODE_EXT or p.name.endswith((".yml", ".yaml", ".json", ".toml", ".sh", ".example", ".sample", ".template")) or p.name.startswith(".env"):
            t = read(p)
            rel = str(p.relative_to(repo))
            for pat in ENV_PATTERNS:
                for m in re.finditer(pat, t):
                    found[m.group(1)].add(rel)
            if p.name.startswith(".env") or p.name.endswith((".example", ".sample", ".template")):
                for m in re.finditer(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]{2,})\s*=", t, re.M):
                    found[m.group(1)].add(rel)
            if p.name in ("manifest.yml", "manifest.yaml", "mta.yaml", "mta.yml"):
                for m in re.finditer(r"^\s+([A-Z][A-Z0-9_]{2,})\s*:", t, re.M):
                    found[m.group(1)].add(rel)
    out = []
    placeholder = re.compile(r"^(ENV_VAR|VAR|VAR_NAME|NAME|KEY|VALUE|YOUR_\w+|EXAMPLE\w*|FOO|BAR)$")
    for k in sorted(found):
        if placeholder.match(k):
            continue
        out.append({"name": k, "sensitive": bool(SECRET_KEY_RE.search(k)), "files": sorted(found[k])[:6]})
    return out


def secret_hygiene(files, repo):
    """Report files that appear to hold literal secret values (names only; values are never read out)."""
    issues = []
    tracked = set()
    try:
        tracked = set(subprocess.run(["git", "-C", str(repo), "ls-files"], capture_output=True, text=True, timeout=20).stdout.split())
    except Exception:
        pass
    for p in files:
        rel = str(p.relative_to(repo))
        if not (p.name.startswith(".env") or p.name in ("manifest.yml", "manifest.yaml", "mta.yaml", "default-env.json", "vars.yml")
                or p.suffix in (".yml", ".yaml") and "manifest" in p.name):
            continue
        if p.name.endswith((".example", ".sample", ".template")):
            continue
        t = read(p)
        names = []
        for m in re.finditer(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(.+)$", t, re.M):
            k, v = m.group(1), m.group(2).strip().strip("\"'")
            if SECRET_KEY_RE.search(k) and v and not v.startswith(("((", "${", "<", "$")) and len(v) >= 8 and v.lower() not in ("changeme", "null", "none", "true", "false"):
                names.append(k)
        if names:
            issues.append({"file": rel, "literal_secret_keys": sorted(set(names)), "git_tracked": rel in tracked})
    return issues


def ai_signals(files, repo):
    hits = defaultdict(lambda: {"category": "", "files": Counter()})
    for p in files:
        if p.suffix not in (".py", ".js", ".ts", ".mjs", ".txt", ".toml", ".json", ".ipynb") or p.name in ("package-lock.json",):
            continue
        t = read(p)
        if not t:
            continue
        rel = str(p.relative_to(repo))
        for name, (cat, pat) in AI_SIGNALS.items():
            n = len(re.findall(pat, t))
            if n:
                hits[name]["category"] = cat
                hits[name]["files"][rel] += n
    out = {}
    for name, h in hits.items():
        out[name] = {"category": h["category"], "total_hits": sum(h["files"].values()),
                     "top_files": [f for f, _ in h["files"].most_common(8)]}
    return dict(sorted(out.items(), key=lambda kv: (kv[1]["category"], -kv[1]["total_hits"])))


def model_names(files, repo):
    pat = re.compile(r"['\"]((?:gpt|o[134]|claude|anthropic--claude|gemini|mistral|llama|text-embedding|sap-rpt)[\w\.\-:/]*)['\"]", re.I)
    c = Counter()
    for p in files:
        if p.suffix in (".py", ".js", ".ts", ".json", ".yaml", ".yml", ".env", ".example"):
            for m in pat.finditer(read(p)):
                s = m.group(1)
                if 3 < len(s) < 60:
                    c[s] += 1
    return [k for k, _ in c.most_common(25)]


def prompt_assets(files, repo):
    out = []
    for p in files:
        rel = str(p.relative_to(repo))
        low = rel.lower()
        if p.name == "SKILL.md" or "prompt" in low and p.suffix in (".md", ".txt", ".py", ".j2", ".jinja", ".yaml", ".json") or p.suffix in (".prompt", ".prompty"):
            out.append(rel)
        elif p.suffix == ".py":
            t = read(p)
            if re.search(r"(SYSTEM_PROMPT|system_prompt|SystemMessage\(|\"role\"\s*:\s*\"system\")", t):
                out.append(rel)
    return sorted(set(out))[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    files = list(walk(repo))
    rel = lambda p: str(p.relative_to(repo))  # noqa: E731

    ext = Counter(p.suffix.lower() for p in files if p.suffix)
    dirs = Counter()
    for p in files:
        parts = p.relative_to(repo).parts
        if len(parts) > 1:
            dirs["/".join(parts[:2]) if len(parts) > 2 else parts[0]] += 1
    top_dirs = defaultdict(int)
    for p in files:
        parts = p.relative_to(repo).parts
        top_dirs[parts[0] if len(parts) > 1 else "(root)"] += 1

    deploy = [rel(p) for p in files if p.name in DEPLOY_FILES or any(d in rel(p) for d in (".github/workflows", "helm/", "k8s/"))]
    deps = {}
    for p in files:
        if p.name in DEP_FILES and "node_modules" not in str(p):
            t = read(p)
            if p.name == "package.json":
                try:
                    j = json.loads(t)
                    deps[rel(p)] = {"dependencies": j.get("dependencies", {}), "devDependencies": list(j.get("devDependencies", {}).keys()),
                                    "scripts": j.get("scripts", {})}
                except Exception:
                    pass
            elif p.name == "requirements.txt":
                deps[rel(p)] = [l.strip() for l in t.splitlines() if l.strip() and not l.startswith("#")]
            else:
                deps[rel(p)] = t[:3000]

    fr = fastapi_routes(files)
    for r in fr:
        r["file"] = str(Path(r["file"]).relative_to(repo))
    orr = other_routes(files)
    for r in orr:
        r["file"] = str(Path(r["file"]).relative_to(repo))
    uir = ui_routes(files)
    for r in uir:
        r["file"] = str(Path(r["file"]).relative_to(repo))

    tests = [rel(p) for p in files if re.search(r"(^|/)(tests?|__tests__|spec)(/|$)|(_test|\.test|\.spec|^test_)", rel(p)) and p.suffix in CODE_EXT]
    docs = [rel(p) for p in files if p.suffix.lower() in (".md", ".rst", ".adoc") and "node_modules" not in rel(p)]

    inv = {
        "repo": str(repo), "name": repo.name,
        "file_count": len(files), "languages": dict(ext.most_common(15)), "top_dirs": dict(sorted(top_dirs.items(), key=lambda kv: -kv[1])),
        "subdirs": dict(dirs.most_common(40)),
        "deployment_files": sorted(deploy), "dependency_manifests": deps,
        "http_routes": fr + orr, "ui_routes": uir,
        "env_vars": env_vars(files, repo), "secret_hygiene": secret_hygiene(files, repo),
        "ai_signals": ai_signals(files, repo), "model_identifiers": model_names(files, repo),
        "prompt_assets": prompt_assets(files, repo),
        "tests": {"count": len(tests), "sample": tests[:30]}, "docs": docs[:120],
    }
    Path(a.out).write_text(json.dumps(inv, indent=2))
    if a.md:
        L = [f"# Inventory: {repo.name}", "",
             f"- Files scanned: {len(files)}; tests: {len(tests)}; docs: {len(docs)}", "", "## Top-level layout"]
        L += [f"- `{k}` — {v} files" for k, v in inv["top_dirs"].items()]
        L += ["", "## Deployment files"] + [f"- `{d}`" for d in inv["deployment_files"]]
        L += ["", f"## HTTP routes ({len(inv['http_routes'])})", "", "| Method | Path | Handler | File |", "|---|---|---|---|"]
        for r in inv["http_routes"]:
            L.append(f"| {r['method']} | `{r['path']}` | {r.get('handler', r.get('service', ''))} | `{r['file']}:{r.get('line', '')}` |")
        L += ["", f"## UI routes ({len(uir)})"] + [f"- `{r['path']}` ({r['file']})" for r in uir[:60]]
        L += ["", "## AI / ML / optimisation signals", "", "| Signal | Category | Hits | Top files |", "|---|---|---|---|"]
        for k, v in inv["ai_signals"].items():
            L.append(f"| {k} | {v['category']} | {v['total_hits']} | {', '.join('`'+f+'`' for f in v['top_files'][:4])} |")
        L += ["", "## Model identifiers seen", ", ".join(f"`{m}`" for m in inv["model_identifiers"]) or "none"]
        L += ["", "## Prompt / skill assets"] + [f"- `{p}`" for p in inv["prompt_assets"]]
        L += ["", f"## Environment variables ({len(inv['env_vars'])})", "", "| Name | Sensitive | Referenced in |", "|---|---|---|"]
        for e in inv["env_vars"]:
            L.append(f"| `{e['name']}` | {'yes' if e['sensitive'] else ''} | {', '.join(e['files'][:3])} |")
        L += ["", "## Secret hygiene (literal secret values found — names only)"]
        L += [f"- `{s['file']}` (git-tracked: {s['git_tracked']}): {', '.join(s['literal_secret_keys'])}" for s in inv["secret_hygiene"]] or ["- none detected"]
        Path(a.md).write_text("\n".join(L) + "\n")
    print(json.dumps({"routes": len(inv["http_routes"]), "ui_routes": len(uir), "env_vars": len(inv["env_vars"]),
                      "ai_signals": list(inv["ai_signals"].keys()), "secret_hygiene_files": len(inv["secret_hygiene"])}, indent=2))


if __name__ == "__main__":
    main()
