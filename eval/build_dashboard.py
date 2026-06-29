"""Build a lightweight HTML dashboard summarizing pairwise benchmark results.

Reads eval/results/{type}/results_pairwise.json for all 4 query types and
produces a single self-contained HTML page with:
  - Win count per mode per type (grouped bar chart)
  - Mean rank per mode per type (grouped bar chart, lower=better)
  - Overall summary table
  - Latency chart + table (mean/p50/p95 per mode)

Output: eval/dashboard.html  (open in browser)
"""

import json
from collections import defaultdict
from pathlib import Path

TYPES = ["factoid", "relational", "broad", "aggregate", "2hop"]
MODES = ["bm25", "naive", "hybrid", "mix", "graph"]
MODE_COLORS = {
    "bm25":   "#9CA3AF",
    "naive":  "#60A5FA",
    "hybrid": "#34D399",
    "mix":    "#FBBF24",
    "graph":  "#F87171",
}

HERE = Path(__file__).parent


def load_pairwise(t: str) -> dict:
    p = HERE / "results" / t / "results_pairwise.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_latency() -> dict:
    """Compute mean/p50/p95 latency per mode across all types."""
    lat = defaultdict(list)
    for t in TYPES:
        p = HERE / "results" / t / "results_raw.json"
        if not p.exists():
            continue
        data = json.load(open(p, encoding="utf-8"))
        for e in data["results"]:
            if not e.get("error") and e.get("latency_sec", 0) > 0:
                lat[e["mode"]].append(e["latency_sec"])
    stats = {}
    for m in MODES:
        vals = sorted(lat[m])
        if not vals:
            stats[m] = {"mean": None, "p50": None, "p95": None, "n": 0}
            continue
        stats[m] = {
            "mean": round(sum(vals) / len(vals), 2),
            "p50":  round(vals[len(vals) // 2], 2),
            "p95":  round(vals[int(len(vals) * 0.95)], 2),
            "n":    len(vals),
        }
    return stats


def main():
    data_per_type = {t: load_pairwise(t) for t in TYPES}

    # Collect win + mean_rank per (type, mode)
    win_data = {m: [] for m in MODES}      # win_data[mode] = [factoid, relational, broad, aggregate]
    rank_data = {m: [] for m in MODES}

    for t in TYPES:
        agg = data_per_type[t]["aggregate"]["overall"]
        wins = agg.get("win_count", {})
        borda = agg.get("borda", {})
        for m in MODES:
            win_data[m].append(wins.get(m, 0))
            rank_data[m].append(borda.get(m, {}).get("mean_rank"))

    # Overall totals
    overall = {m: {"wins": sum(win_data[m]), "mean_rank": None} for m in MODES}
    for m in MODES:
        ranks = [r for r in rank_data[m] if r is not None]
        overall[m]["mean_rank"] = round(sum(ranks) / len(ranks), 3) if ranks else None

    total_queries = sum(data_per_type[t]["aggregate"].get("total_queries", 0) for t in TYPES)
    latency = load_latency()

    # Build HTML
    html = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>TLU-Chatbot Pairwise Benchmark Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 1200px; margin: 24px auto; padding: 0 20px; color: #1f2937; }
  h1 { margin-bottom: 4px; }
  .subtitle { color: #6b7280; margin-bottom: 24px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }
  .card { background: #f9fafb; border-radius: 8px; padding: 16px; }
  .card h2 { margin: 0 0 12px; font-size: 16px; }
  .full { grid-column: 1 / -1; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th, td { padding: 8px 12px; text-align: right; border-bottom: 1px solid #e5e7eb; }
  th:first-child, td:first-child { text-align: left; }
  th { background: #f3f4f6; font-weight: 600; }
  tr:hover { background: #fafafa; }
  .winner { color: #059669; font-weight: 600; }
  .canvas-wrap { height: 320px; position: relative; }
  .legend-note { font-size: 12px; color: #6b7280; margin-top: 8px; }
</style>
</head>
<body>
<h1>TLU-Chatbot — Pairwise Benchmark Dashboard</h1>
<p class="subtitle">"""
    html += f"{total_queries} queries × 5 modes = {total_queries*5} responses · LLM-as-judge Option F · Borda count"
    html += """</p>

<div class="grid">
  <div class="card">
    <h2>Win Count per Mode per Type</h2>
    <div class="canvas-wrap"><canvas id="winChart"></canvas></div>
    <p class="legend-note">Số query mode đó được judge xếp rank 1 (cao = tốt)</p>
  </div>
  <div class="card">
    <h2>Mean Rank per Mode per Type</h2>
    <div class="canvas-wrap"><canvas id="rankChart"></canvas></div>
    <p class="legend-note">Mean rank (1=best, 5=worst). THẤP hơn = tốt hơn</p>
  </div>
</div>

<div class="card full">
  <h2>Overall Summary (across """ + str(len(TYPES)) + """ query types)</h2>
  <table>
    <thead><tr><th>Mode</th><th>Total Wins</th><th>Win %</th><th>Mean Rank (avg of """ + str(len(TYPES)) + """ types)</th></tr></thead>
    <tbody>
"""
    sorted_modes = sorted(MODES, key=lambda m: -overall[m]["wins"])
    for i, m in enumerate(sorted_modes):
        cls = ' class="winner"' if i == 0 else ''
        win_pct = overall[m]["wins"] / total_queries * 100 if total_queries else 0
        mr = overall[m]["mean_rank"]
        mr_str = f"{mr}" if mr is not None else "—"
        html += f"      <tr{cls}><td>{m}</td><td>{overall[m]['wins']}</td><td>{win_pct:.1f}%</td><td>{mr_str}</td></tr>\n"
    html += """    </tbody>
  </table>
</div>

<div class="card full" style="margin-top: 24px;">
  <h2>Per-Type Detail</h2>
  <table>
    <thead><tr><th>Type</th>"""
    for m in MODES:
        html += f"<th>{m}<br><small>wins / rank</small></th>"
    html += """</tr></thead>
    <tbody>
"""
    for ti, t in enumerate(TYPES):
        html += f"      <tr><td><b>{t}</b></td>"
        # Find winner per type
        wins_t = [(m, win_data[m][ti]) for m in MODES]
        wmax = max(w for _, w in wins_t)
        for m in MODES:
            w = win_data[m][ti]
            r = rank_data[m][ti]
            r_str = f"{r:.2f}" if r is not None else "—"
            cls = ' class="winner"' if w == wmax else ''
            html += f'<td{cls}>{w} / {r_str}</td>'
        html += "</tr>\n"
    html += """    </tbody>
  </table>
  <p class="legend-note">Highlight = mode thắng nhiều nhất per type. Format: wins / mean_rank</p>
</div>

<div class="grid" style="margin-top: 24px;">
  <div class="card">
    <h2>Mean Latency per Mode (seconds)</h2>
    <div class="canvas-wrap"><canvas id="latChart"></canvas></div>
    <p class="legend-note">Mean response time tính trên tất cả queries × types. THẤP hơn = nhanh hơn</p>
  </div>
  <div class="card">
    <h2>Latency Summary</h2>
    <table>
      <thead><tr><th>Mode</th><th>Mean</th><th>P50</th><th>P95</th><th>n</th></tr></thead>
      <tbody>
"""
    lat_sorted = sorted(MODES, key=lambda m: latency[m]["mean"] or 999)
    for i, m in enumerate(lat_sorted):
        s = latency[m]
        cls = ' class="winner"' if i == 0 else ''
        mean_s = f"{s['mean']}s" if s['mean'] else "—"
        p50_s  = f"{s['p50']}s"  if s['p50']  else "—"
        p95_s  = f"{s['p95']}s"  if s['p95']  else "—"
        html += f"      <tr{cls}><td>{m}</td><td>{mean_s}</td><td>{p50_s}</td><td>{p95_s}</td><td>{s['n']}</td></tr>\n"
    html += """      </tbody>
    </table>
    <p class="legend-note">Highlight = mode nhanh nhất. Tính trên tất cả types.</p>
  </div>
</div>

<script>
const labels = """ + json.dumps(TYPES) + """;
const winDatasets = [
"""
    for m in MODES:
        html += f"""  {{label: "{m}", data: {win_data[m]}, backgroundColor: "{MODE_COLORS[m]}"}},
"""
    html += """];
const rankDatasets = [
"""
    for m in MODES:
        html += f"""  {{label: "{m}", data: {[r if r is not None else None for r in rank_data[m]]}, backgroundColor: "{MODE_COLORS[m]}"}},
"""
    html += """];
new Chart(document.getElementById('winChart'), {
  type: 'bar',
  data: {labels, datasets: winDatasets},
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {legend: {position: 'bottom'}},
    scales: {y: {beginAtZero: true, title: {display: true, text: 'Wins (rank 1)'}}}
  }
});
new Chart(document.getElementById('rankChart'), {
  type: 'bar',
  data: {labels, datasets: rankDatasets},
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {legend: {position: 'bottom'}},
    scales: {y: {beginAtZero: false, min: 2, max: 5, title: {display: true, text: 'Mean Rank (lower=better)'}}}
  }
});
</script>"""

    # Latency chart data
    lat_means = [latency[m]["mean"] or 0 for m in MODES]
    lat_p50   = [latency[m]["p50"]  or 0 for m in MODES]
    lat_p95   = [latency[m]["p95"]  or 0 for m in MODES]
    lat_colors = [MODE_COLORS[m] for m in MODES]
    lat_labels = json.dumps(MODES)

    html += f"""
<script>
new Chart(document.getElementById('latChart'), {{
  type: 'bar',
  data: {{
    labels: {lat_labels},
    datasets: [
      {{label: 'Mean', data: {lat_means}, backgroundColor: {json.dumps(lat_colors)}}},
      {{label: 'P50',  data: {lat_p50},  backgroundColor: {json.dumps([c + '99' for c in lat_colors])}}},
      {{label: 'P95',  data: {lat_p95},  backgroundColor: {json.dumps([c + '44' for c in lat_colors])}}},
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{legend: {{position: 'bottom'}}}},
    scales: {{y: {{beginAtZero: true, title: {{display: true, text: 'Seconds'}}}}}}
  }}
}});
</script>
</body>
</html>
"""

    out = HERE / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"[OK] Dashboard written to: {out}")
    print(f"     Open in browser: file://{out}")


if __name__ == "__main__":
    main()
