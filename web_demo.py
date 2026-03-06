"""
BigSMILES Web Demo — End-to-End Pipeline
BigSMILES Web 演示 — 端到端流水线

A lightweight HTTP server providing:
    1. Syntax checking (via bigsmiles_checker)
    2. Parsing & topology analysis (via bigsmiles_parser)
    3. Structural fingerprint extraction (via bigsmiles_fingerprint)
    4. Tg prediction (via ml_models + pre-trained ridge model)

Architecture:
    - Pure stdlib http.server — no Flask/Django dependency
    - JSON API endpoints: /api/check, /api/parse, /api/fingerprint, /api/predict, /api/pipeline
    - Single-page frontend served at /

Public API / 公共 API:
    handle_check(bigsmiles)      → dict
    handle_parse(bigsmiles)      → dict
    handle_fingerprint(bigsmiles) → dict
    handle_predict(bigsmiles)    → dict
    handle_pipeline(bigsmiles)   → dict  (runs all above)
    get_index_html()             → str
"""

import io
import json
import sys
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List, Optional, Tuple
from contextlib import redirect_stdout

# ---------------------------------------------------------------------------
# 1. Handler functions / 处理函数
# ---------------------------------------------------------------------------


def handle_check(bigsmiles: str) -> Dict[str, Any]:
    """Check BigSMILES syntax validity.
    检查 BigSMILES 语法有效性。

    Returns:
        {"valid": bool, "bigsmiles": str, "errors": list[str]}
    """
    if not bigsmiles.strip():
        return {"valid": False, "bigsmiles": bigsmiles, "errors": ["Empty input / 输入为空"]}

    from bigsmiles_checker import check_bigsmiles

    # Capture printed output as the checker prints diagnostics
    buf = io.StringIO()
    with redirect_stdout(buf):
        valid = check_bigsmiles(bigsmiles)

    output = buf.getvalue()
    errors: List[str] = []
    if not valid:
        for line in output.strip().split("\n"):
            line = line.strip()
            if line and ("error" in line.lower() or "错误" in line or "✗" in line
                         or "invalid" in line.lower() or "无效" in line):
                errors.append(line)
        if not errors:
            errors.append("Syntax check failed / 语法检查失败")

    return {"valid": valid, "bigsmiles": bigsmiles, "errors": errors}


def handle_parse(bigsmiles: str) -> Dict[str, Any]:
    """Parse BigSMILES and extract structural info.
    解析 BigSMILES 并提取结构信息。

    Returns:
        {"topology": str, "repeat_units": list, "bonding_descriptors": list,
         "round_trip": str} or {"error": str}
    """
    try:
        from bigsmiles_parser import BigSMILESParser
        parser = BigSMILESParser()

        ast = parser.parse(bigsmiles)
        topo_info = parser.get_topology(ast)
        topology_type = topo_info.get("topology", "unknown") if isinstance(topo_info, dict) else str(topo_info)
        repeat_units = parser.get_repeat_units(ast)
        bonding_descs = parser.get_bonding_descriptors(ast)
        round_trip = parser.round_trip(bigsmiles)

        return {
            "topology": topology_type,
            "repeat_units": repeat_units,
            "bonding_descriptors": bonding_descs,
            "round_trip": round_trip,
            "topology_details": topo_info if isinstance(topo_info, dict) else {},
        }
    except Exception as e:
        return {"error": str(e), "topology": "unknown", "repeat_units": [], "bonding_descriptors": []}


def _extract_smiles_from_bigsmiles(bigsmiles: str) -> str:
    """Extract repeat-unit SMILES from BigSMILES for fingerprinting.
    从 BigSMILES 提取重复单元 SMILES 用于指纹计算。

    Strategy: remove stochastic markers {, }, [$], [>], [<] → treat as SMILES.
    """
    s = bigsmiles.strip()
    # Remove stochastic object brackets
    s = s.replace("{", "").replace("}", "")
    # Remove bonding descriptors [$], [<], [>], [$1], etc.
    s = re.sub(r'\[[<>$]\d*\]', '', s)
    # Replace * with [H] for RDKit
    s = s.replace("*", "[H]")
    # Clean up empty parentheses or leading/trailing dots
    s = s.replace("()", "").strip(".")
    return s if s else "CC"


def handle_fingerprint(bigsmiles: str) -> Dict[str, Any]:
    """Extract structural fingerprints from BigSMILES.
    从 BigSMILES 提取结构指纹。

    Returns:
        {"morgan": list|None, "morgan_bits": int, "fragments": list,
         "fragment_names": list, "descriptors": list, "descriptor_names": list,
         "total_features": int}
    """
    smiles = _extract_smiles_from_bigsmiles(bigsmiles)

    from bigsmiles_fingerprint import (
        morgan_fingerprint, fragment_vector, fragment_names,
        descriptor_vector, descriptor_names,
    )

    # Morgan fingerprint (may fail without RDKit)
    morgan = None
    morgan_bits = 256
    try:
        morgan = morgan_fingerprint(smiles, radius=2, n_bits=morgan_bits)
    except Exception:
        pass

    # Fragment counts (always works)
    frags = list(fragment_vector(smiles))
    frag_names = list(fragment_names())

    # Polymer descriptors
    descs = descriptor_vector(smiles, bigsmiles)
    desc_names = list(descriptor_names())

    total = (len(morgan) if morgan else 0) + len(frags) + len(descs)

    return {
        "morgan": morgan,
        "morgan_bits": morgan_bits if morgan else 0,
        "fragments": frags,
        "fragment_names": frag_names,
        "descriptors": descs,
        "descriptor_names": desc_names,
        "total_features": total,
        "smiles_used": smiles,
    }


def _get_trained_model() -> Tuple[Any, List[float], List[float]]:
    """Train a ridge regression model on the Bicerano dataset.
    在 Bicerano 数据集上训练 Ridge 回归模型。

    Returns:
        (model, feature_means, feature_stds) for normalization.
    """
    from ml_experiment import build_dataset
    from ml_models import get_model, normalize

    buf = io.StringIO()
    with redirect_stdout(buf):
        X, y, _, _ = build_dataset(morgan_bits=256)

    # normalize() expects (X_train, X_test) → (X_train_norm, X_test_norm, means, stds)
    # Use X as both train and test to get means/stds
    X_norm, _, means, stds = normalize(X, X)

    model = get_model("ridge", alpha=0.1, lr=0.01, n_iter=2000)
    model.fit(X_norm, y)
    return model, means, stds


# Module-level cache for trained model
_MODEL_CACHE: Dict[str, Any] = {}


def _ensure_model() -> Tuple[Any, List[float], List[float]]:
    """Lazy-load and cache the trained model.
    懒加载并缓存训练好的模型。
    """
    if "model" not in _MODEL_CACHE:
        model, means, stds = _get_trained_model()
        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["means"] = means
        _MODEL_CACHE["stds"] = stds
    return _MODEL_CACHE["model"], _MODEL_CACHE["means"], _MODEL_CACHE["stds"]


def handle_predict(bigsmiles: str) -> Dict[str, Any]:
    """Predict Tg from BigSMILES string.
    从 BigSMILES 字符串预测 Tg。

    Returns:
        {"predicted_tg_k": float, "model": str, "smiles_used": str}
    """
    smiles = _extract_smiles_from_bigsmiles(bigsmiles)

    from bigsmiles_fingerprint import (
        morgan_fingerprint, fragment_vector, descriptor_vector,
    )
    # Build feature vector (same as build_dataset)
    features: List[float] = []
    try:
        fp = morgan_fingerprint(smiles, radius=2, n_bits=256)
        features.extend(float(x) for x in fp)
    except Exception:
        features.extend([0.0] * 256)

    features.extend(float(x) for x in fragment_vector(smiles))
    features.extend(descriptor_vector(smiles, bigsmiles))

    # Get model and predict — manually normalize using stored means/stds
    model, means, stds = _ensure_model()
    row_norm = []
    for j, val in enumerate(features):
        if j < len(stds) and stds[j] > 1e-12:
            row_norm.append((val - means[j]) / stds[j])
        else:
            row_norm.append(0.0)
    y_pred = model.predict([row_norm])

    tg = round(y_pred[0], 1)
    # Clamp to reasonable range
    tg = max(50.0, min(tg, 900.0))

    return {
        "predicted_tg_k": tg,
        "predicted_tg_c": round(tg - 273.15, 1),
        "model": "Ridge Regression (alpha=0.1)",
        "smiles_used": smiles,
        "num_features": len(features),
    }


def handle_pipeline(bigsmiles: str) -> Dict[str, Any]:
    """Run the full pipeline: check → parse → fingerprint → predict.
    运行完整流水线：检查 → 解析 → 指纹 → 预测。
    """
    result: Dict[str, Any] = {"input": bigsmiles}

    # Step 1: Check
    result["check"] = handle_check(bigsmiles)

    # Step 2: Parse (attempt even if check fails — for partial results)
    result["parse"] = handle_parse(bigsmiles)

    # Step 3: Fingerprint
    result["fingerprint"] = handle_fingerprint(bigsmiles)

    # Step 4: Predict
    result["predict"] = handle_predict(bigsmiles)

    return result


# ---------------------------------------------------------------------------
# 2. HTML Frontend / HTML 前端
# ---------------------------------------------------------------------------

def get_index_html() -> str:
    """Return the single-page HTML frontend.
    返回单页 HTML 前端。
    """
    return r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BigSMILES Demo - AI-Assisted Polymer Design</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f7fa; color: #333; line-height: 1.6;
        }
        .container { max-width: 960px; margin: 0 auto; padding: 20px; }
        header {
            text-align: center; padding: 30px 0 20px;
            border-bottom: 2px solid #e0e6ed;
        }
        header h1 { font-size: 1.8em; color: #2c3e50; }
        header p { color: #7f8c8d; margin-top: 5px; }

        .input-section {
            background: #fff; border-radius: 8px; padding: 20px;
            margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .input-section label { font-weight: 600; display: block; margin-bottom: 8px; }
        .input-row { display: flex; gap: 10px; }
        .input-row input {
            flex: 1; padding: 10px 14px; font-size: 1em;
            border: 2px solid #dce1e8; border-radius: 6px;
            font-family: "Fira Code", "Cascadia Code", monospace;
            outline: none; transition: border-color 0.2s;
        }
        .input-row input:focus { border-color: #3498db; }
        .btn {
            padding: 10px 24px; font-size: 1em; font-weight: 600;
            background: #3498db; color: #fff; border: none; border-radius: 6px;
            cursor: pointer; transition: background 0.2s;
        }
        .btn:hover { background: #2980b9; }
        .btn:disabled { background: #bdc3c7; cursor: not-allowed; }

        .examples {
            margin: 8px 0 0; font-size: 0.85em; color: #7f8c8d;
        }
        .examples span {
            cursor: pointer; color: #3498db; text-decoration: underline;
            margin-right: 12px;
        }
        .examples span:hover { color: #2980b9; }

        .results { margin-top: 20px; }
        .result-card {
            background: #fff; border-radius: 8px; padding: 16px 20px;
            margin-bottom: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        }
        .result-card h3 {
            font-size: 1em; color: #2c3e50; margin-bottom: 10px;
            padding-bottom: 6px; border-bottom: 1px solid #eee;
        }
        .result-card .status {
            display: inline-block; padding: 2px 10px; border-radius: 12px;
            font-size: 0.85em; font-weight: 600;
        }
        .status-pass { background: #d4edda; color: #155724; }
        .status-fail { background: #f8d7da; color: #721c24; }

        .kv-table { width: 100%; border-collapse: collapse; }
        .kv-table td {
            padding: 4px 8px; font-size: 0.9em; border-bottom: 1px solid #f0f0f0;
        }
        .kv-table td:first-child {
            font-weight: 600; color: #555; width: 180px;
        }

        .predict-highlight {
            text-align: center; padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 8px; color: #fff; margin-top: 12px;
        }
        .predict-highlight .tg-value { font-size: 2.2em; font-weight: 700; }
        .predict-highlight .tg-label { font-size: 0.9em; opacity: 0.85; }

        .loading { display: none; text-align: center; padding: 20px; color: #7f8c8d; }
        .error-msg { color: #e74c3c; font-size: 0.9em; margin-top: 8px; }
        .fp-bar { display: flex; flex-wrap: wrap; gap: 2px; margin-top: 6px; }
        .fp-bit {
            width: 6px; height: 16px; border-radius: 2px;
        }
        .fp-bit-1 { background: #3498db; }
        .fp-bit-0 { background: #ecf0f1; }

        footer {
            text-align: center; padding: 20px; color: #95a5a6;
            font-size: 0.85em; margin-top: 30px;
        }
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>BigSMILES Demo</h1>
        <p>AI-Assisted Polymer Material Design | Tongji University SITP</p>
    </header>

    <div class="input-section">
        <label for="bigsmiles-input">BigSMILES Input</label>
        <div class="input-row">
            <input type="text" id="bigsmiles-input" placeholder="{[$]CC[$]}"
                   value="{[$]CC(c1ccccc1)[$]}" />
            <button class="btn" id="run-btn" onclick="runPipeline()">Analyze</button>
        </div>
        <div class="examples">
            Examples:
            <span onclick="setInput('{[$]CC[$]}')">PE</span>
            <span onclick="setInput('{[$]CC(c1ccccc1)[$]}')">PS</span>
            <span onclick="setInput('{[$]CC(C)(C(=O)OC)[$]}')">PMMA</span>
            <span onclick="setInput('{[$]CC(C#N)[$]}')">PAN</span>
            <span onclick="setInput('{[$]c1ccc(Oc2ccc(C(=O)c3ccc([$])cc3)cc2)cc1[$]}')">PEEK</span>
        </div>
    </div>

    <div class="loading" id="loading">Analyzing...</div>

    <div class="results" id="results" style="display:none;">
        <!-- Step 1: Check -->
        <div class="result-card" id="card-check">
            <h3>1. Syntax Check</h3>
            <div id="check-result"></div>
        </div>

        <!-- Step 2: Parse -->
        <div class="result-card" id="card-parse">
            <h3>2. Parse & Topology</h3>
            <div id="parse-result"></div>
        </div>

        <!-- Step 3: Fingerprint -->
        <div class="result-card" id="card-fp">
            <h3>3. Structural Fingerprint</h3>
            <div id="fp-result"></div>
        </div>

        <!-- Step 4: Predict -->
        <div class="result-card" id="card-predict">
            <h3>4. Tg Prediction</h3>
            <div id="predict-result"></div>
        </div>
    </div>

    <footer>
        BigSMILES Demo &mdash; SITP Project, Tongji University
    </footer>
</div>

<script>
function setInput(val) {
    document.getElementById('bigsmiles-input').value = val;
}

async function runPipeline() {
    const input = document.getElementById('bigsmiles-input').value.trim();
    if (!input) return;

    const btn = document.getElementById('run-btn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');

    btn.disabled = true;
    loading.style.display = 'block';
    results.style.display = 'none';

    try {
        const resp = await fetch('/api/pipeline', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({bigsmiles: input}),
        });
        const data = await resp.json();
        renderResults(data);
        results.style.display = 'block';
    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        btn.disabled = false;
        loading.style.display = 'none';
    }
}

function renderResults(data) {
    // Check
    const check = data.check || {};
    const checkEl = document.getElementById('check-result');
    const statusClass = check.valid ? 'status-pass' : 'status-fail';
    const statusText = check.valid ? 'VALID' : 'INVALID';
    let checkHtml = `<span class="status ${statusClass}">${statusText}</span>`;
    if (check.errors && check.errors.length > 0) {
        checkHtml += '<div class="error-msg">' + check.errors.join('<br>') + '</div>';
    }
    checkEl.innerHTML = checkHtml;

    // Parse
    const parse = data.parse || {};
    const parseEl = document.getElementById('parse-result');
    if (parse.error) {
        parseEl.innerHTML = '<div class="error-msg">' + parse.error + '</div>';
    } else {
        parseEl.innerHTML = `
            <table class="kv-table">
                <tr><td>Topology</td><td>${parse.topology || 'N/A'}</td></tr>
                <tr><td>Repeat Units</td><td>${(parse.repeat_units || []).join(', ') || 'None'}</td></tr>
                <tr><td>Bonding Descriptors</td><td>${(parse.bonding_descriptors || []).join(', ') || 'None'}</td></tr>
                <tr><td>Round-trip</td><td><code>${parse.round_trip || 'N/A'}</code></td></tr>
            </table>`;
    }

    // Fingerprint
    const fp = data.fingerprint || {};
    const fpEl = document.getElementById('fp-result');
    let fpHtml = `
        <table class="kv-table">
            <tr><td>SMILES Used</td><td><code>${fp.smiles_used || 'N/A'}</code></td></tr>
            <tr><td>Total Features</td><td>${fp.total_features || 0}</td></tr>
            <tr><td>Morgan Bits</td><td>${fp.morgan_bits || 'N/A (no RDKit)'}</td></tr>
            <tr><td>Fragments (${(fp.fragment_names || []).length})</td><td>${formatFragments(fp)}</td></tr>
        </table>`;
    if (fp.morgan && fp.morgan.length > 0) {
        fpHtml += '<div style="margin-top:8px;font-size:0.85em;color:#777;">Morgan Fingerprint Bits:</div>';
        fpHtml += '<div class="fp-bar">';
        for (const bit of fp.morgan) {
            fpHtml += `<div class="fp-bit fp-bit-${bit}"></div>`;
        }
        fpHtml += '</div>';
    }
    fpEl.innerHTML = fpHtml;

    // Predict
    const pred = data.predict || {};
    const predEl = document.getElementById('predict-result');
    predEl.innerHTML = `
        <div class="predict-highlight">
            <div class="tg-label">Predicted Glass Transition Temperature (Tg)</div>
            <div class="tg-value">${pred.predicted_tg_k || '?'} K</div>
            <div class="tg-label">${pred.predicted_tg_c || '?'} &deg;C</div>
        </div>
        <table class="kv-table" style="margin-top:12px;">
            <tr><td>Model</td><td>${pred.model || 'N/A'}</td></tr>
            <tr><td>Features Used</td><td>${pred.num_features || 0}</td></tr>
        </table>`;
}

function formatFragments(fp) {
    if (!fp.fragment_names || !fp.fragments) return 'N/A';
    const parts = [];
    for (let i = 0; i < fp.fragment_names.length; i++) {
        if (fp.fragments[i] > 0) {
            parts.push(fp.fragment_names[i] + '=' + fp.fragments[i]);
        }
    }
    return parts.length > 0 ? parts.join(', ') : 'None detected';
}

// Run on Enter key
document.getElementById('bigsmiles-input').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') runPipeline();
});
</script>
</body>
</html>'''


# ---------------------------------------------------------------------------
# 3. HTTP Server / HTTP 服务器
# ---------------------------------------------------------------------------

class BigSMILESDemoHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the BigSMILES demo.
    BigSMILES 演示 HTTP 请求处理器。
    """

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html = get_index_html()
            self._respond(200, "text/html", html.encode("utf-8"))
        else:
            self._respond(404, "text/plain", b"Not Found")

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._respond_json(400, {"error": "Invalid JSON"})
            return

        bigsmiles = data.get("bigsmiles", "").strip()

        handlers = {
            "/api/check": handle_check,
            "/api/parse": handle_parse,
            "/api/fingerprint": handle_fingerprint,
            "/api/predict": handle_predict,
            "/api/pipeline": handle_pipeline,
        }

        handler = handlers.get(self.path)
        if handler:
            try:
                result = handler(bigsmiles)
                self._respond_json(200, result)
            except Exception as e:
                self._respond_json(500, {"error": str(e)})
        else:
            self._respond_json(404, {"error": f"Unknown endpoint: {self.path}"})

    def _respond(self, code: int, content_type: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _respond_json(self, code: int, data: Any):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self._respond(code, "application/json", body)

    def log_message(self, format, *args):
        """Suppress default logging for cleaner output."""
        pass


def run_server(host: str = "127.0.0.1", port: int = 8765):
    """Start the BigSMILES demo server.
    启动 BigSMILES 演示服务器。
    """
    server = HTTPServer((host, port), BigSMILESDemoHandler)
    print(f"BigSMILES Demo Server running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


# ---------------------------------------------------------------------------
# CLI entry / 命令行入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = 8765
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
    if "--help" in sys.argv:
        print("Usage: python web_demo.py [--port PORT]")
        print(f"  Default port: {port}")
        sys.exit(0)

    # Pre-train model on startup
    print("Pre-training model on Bicerano dataset...")
    _ensure_model()
    print("Model ready.")

    run_server(port=port)
