#!/usr/bin/env python3
"""Build a filterable prediction explanation graph for soccer model outputs."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd


try:
    from betboard_soccer_extension.modeling.draw_policy import load_draw_policy
except ModuleNotFoundError:
    if "__file__" in globals():
        candidate = Path(__file__).resolve().parents[2] / "src"
        if candidate.exists():
            sys.path.insert(0, str(candidate))
    from betboard_soccer_extension.modeling.draw_policy import load_draw_policy  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build filterable prediction explanation graph.")
    parser.add_argument("--predictions", required=True, help="Prediction CSV from predict_future_fixtures.py or export script.")
    parser.add_argument("--model-input", required=True, help="Parquet model-input rows used for predictions.")
    parser.add_argument("--preprocessing", required=True, help="preprocessing.joblib from model training.")
    parser.add_argument("--model", default="", help="Optional trained Keras model for perturbation-based feature contributions.")
    parser.add_argument("--draw-policy", default="configs/draw_policy_eng1.json")
    parser.add_argument("--output-dir", default="outputs/eng1_soccer_nn/future_2026/explanations")
    parser.add_argument("--top-features", type=int, default=12)
    parser.add_argument("--candidate-features", type=int, default=80)
    return parser.parse_args()


def slug(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def read_table(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def feature_group(feature: str) -> str:
    if feature.startswith("home_"):
        return "home"
    if feature.startswith("away_"):
        return "away"
    if feature.startswith("odds_"):
        return "odds"
    if "rest" in feature or "travel" in feature:
        return "schedule"
    if "injury" in feature:
        return "injury"
    return "model_feature"


def clean_feature_name(feature: str) -> str:
    return feature.replace("_", " ")


def baseline_delta_contribution_frame(model_input: pd.DataFrame, preprocessing: dict[str, Any], top_n: int) -> pd.DataFrame:
    feature_names = list(preprocessing["feature_names"])
    imputer = preprocessing["imputer"]
    scaler = preprocessing["scaler"]
    baseline = pd.Series(imputer.statistics_, index=feature_names)
    scale = pd.Series(getattr(scaler, "scale_", np.ones(len(feature_names))), index=feature_names).replace(0, 1.0)
    rows = []
    for _, match in model_input.iterrows():
        matchup_key = match["matchup_key"]
        values = pd.to_numeric(match.reindex(feature_names), errors="coerce")
        filled = values.fillna(baseline)
        z_delta = ((filled - baseline) / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        top = z_delta.abs().sort_values(ascending=False).head(top_n)
        for feature, magnitude in top.items():
            signed = float(z_delta.loc[feature])
            rows.append(
                {
                    "matchup_key": matchup_key,
                    "feature": feature,
                    "feature_label": clean_feature_name(feature),
                    "feature_group": feature_group(feature),
                    "feature_value": None if pd.isna(values.loc[feature]) else float(values.loc[feature]),
                    "baseline_value": None if pd.isna(baseline.loc[feature]) else float(baseline.loc[feature]),
                    "contribution_score": signed,
                    "abs_contribution_score": float(magnitude),
                    "direction": "above_baseline" if signed >= 0 else "below_baseline",
                    "method": "baseline_z_delta",
                    "explained_target": None,
                }
            )
    return pd.DataFrame(rows)


def model_sensitivity_contribution_frame(
    predictions: pd.DataFrame,
    model_input: pd.DataFrame,
    preprocessing: dict[str, Any],
    model_path: str,
    top_n: int,
    candidate_features: int,
) -> pd.DataFrame:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise SystemExit("TensorFlow is missing. In Colab run: !pip install -r requirements-colab.txt") from exc

    feature_names = list(preprocessing["feature_names"])
    imputer = preprocessing["imputer"]
    scaler = preprocessing["scaler"]
    baseline = pd.Series(imputer.statistics_, index=feature_names)
    scale = pd.Series(getattr(scaler, "scale_", np.ones(len(feature_names))), index=feature_names).replace(0, 1.0)
    model = tf.keras.models.load_model(model_path)
    label_to_idx = {"home": 0, "draw": 1, "away": 2}

    raw = model_input[feature_names].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    filled = raw.fillna(baseline)
    x = scaler.transform(imputer.transform(raw))
    base_probs = model.predict(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), verbose=0)
    pred_lookup = predictions.set_index("matchup_key")

    rows = []
    feature_index = {feature: idx for idx, feature in enumerate(feature_names)}
    for row_idx, row in model_input.reset_index(drop=True).iterrows():
        matchup_key = row["matchup_key"]
        if matchup_key not in pred_lookup.index:
            continue
        pred_row = pred_lookup.loc[matchup_key]
        target_label = pred_row.get("raw_model_pick", pred_row.get("prediction", "home"))
        target_idx = label_to_idx.get(str(target_label), 0)
        z_delta = ((filled.iloc[row_idx] - baseline) / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        candidates = z_delta.abs().sort_values(ascending=False).head(candidate_features).index.tolist()
        contribs = []
        for feature in candidates:
            perturbed = x[row_idx : row_idx + 1].copy()
            perturbed[0, feature_index[feature]] = 0.0
            perturbed_prob = model.predict(np.nan_to_num(perturbed, nan=0.0, posinf=0.0, neginf=0.0), verbose=0)[0, target_idx]
            contribution = float(base_probs[row_idx, target_idx] - perturbed_prob)
            contribs.append((feature, contribution))
        for feature, contribution in sorted(contribs, key=lambda item: abs(item[1]), reverse=True)[:top_n]:
            value = raw.iloc[row_idx][feature]
            rows.append(
                {
                    "matchup_key": matchup_key,
                    "feature": feature,
                    "feature_label": clean_feature_name(feature),
                    "feature_group": feature_group(feature),
                    "feature_value": None if pd.isna(value) else float(value),
                    "baseline_value": None if pd.isna(baseline.loc[feature]) else float(baseline.loc[feature]),
                    "contribution_score": contribution,
                    "abs_contribution_score": abs(contribution),
                    "direction": "supports_prediction" if contribution >= 0 else "opposes_prediction",
                    "method": "model_probability_perturbation",
                    "explained_target": target_label,
                }
            )
    return pd.DataFrame(rows)


def probability_nodes(row: pd.Series) -> list[dict[str, Any]]:
    return [
        {"id": f"{row.matchup_key}:prob_home", "label": f"Home {row.prob_home:.3f}", "type": "probability", "value": float(row.prob_home)},
        {"id": f"{row.matchup_key}:prob_draw", "label": f"Draw {row.prob_draw:.3f}", "type": "probability", "value": float(row.prob_draw)},
        {"id": f"{row.matchup_key}:prob_away", "label": f"Away {row.prob_away:.3f}", "type": "probability", "value": float(row.prob_away)},
    ]


def build_graph(predictions: pd.DataFrame, contributions: pd.DataFrame, draw_policy: dict[str, Any] | None) -> dict[str, Any]:
    matchups = []
    for row in predictions.itertuples(index=False):
        matchup_nodes: list[dict[str, Any]] = []
        matchup_edges: list[dict[str, Any]] = []
        match_id = f"{row.matchup_key}:match"
        home_id = f"{row.matchup_key}:home"
        away_id = f"{row.matchup_key}:away"
        pick = getattr(row, "recommended_pick", getattr(row, "prediction", ""))
        raw_pick = getattr(row, "raw_model_pick", getattr(row, "prediction", ""))
        draw_risk = bool(getattr(row, "draw_risk", False))
        matchup_nodes.extend(
            [
                {
                    "id": match_id,
                    "label": row.matchup_key,
                    "type": "match",
                    "kickoff_utc": str(row.kickoff_utc),
                    "matchweek": getattr(row, "matchweek", None),
                },
                {"id": home_id, "label": str(row.home_team_name), "type": "team", "side": "home"},
                {"id": away_id, "label": str(row.away_team_name), "type": "team", "side": "away"},
                {"id": f"{row.matchup_key}:recommended", "label": f"Recommended: {pick}", "type": "decision", "pick": pick},
                {"id": f"{row.matchup_key}:raw", "label": f"Raw model: {raw_pick}", "type": "model_pick", "pick": raw_pick},
            ]
        )
        matchup_nodes.extend(probability_nodes(row))
        matchup_edges.extend(
            [
                {"source": home_id, "target": match_id, "label": "home_team"},
                {"source": away_id, "target": match_id, "label": "away_team"},
                {"source": match_id, "target": f"{row.matchup_key}:raw", "label": "model_outputs"},
                {"source": f"{row.matchup_key}:raw", "target": f"{row.matchup_key}:recommended", "label": "decision_layer"},
                {"source": f"{row.matchup_key}:prob_home", "target": f"{row.matchup_key}:recommended", "label": "probability_input"},
                {"source": f"{row.matchup_key}:prob_draw", "target": f"{row.matchup_key}:recommended", "label": "probability_input"},
                {"source": f"{row.matchup_key}:prob_away", "target": f"{row.matchup_key}:recommended", "label": "probability_input"},
            ]
        )
        if draw_policy is not None:
            policy_id = f"{row.matchup_key}:draw_policy"
            matchup_nodes.append(
                {
                    "id": policy_id,
                    "label": f"Draw policy: {draw_policy.get('version')}",
                    "type": "draw_policy",
                    "draw_risk": draw_risk,
                    "draw_gap": getattr(row, "draw_gap", None),
                    "home_away_gap": getattr(row, "home_away_gap", None),
                }
            )
            matchup_edges.append({"source": policy_id, "target": f"{row.matchup_key}:recommended", "label": "adjusts_pick"})

        match_contrib = contributions[contributions["matchup_key"] == row.matchup_key]
        for item in match_contrib.itertuples(index=False):
            node_id = f"{row.matchup_key}:feature:{slug(item.feature)}"
            matchup_nodes.append(
                {
                    "id": node_id,
                    "label": item.feature_label,
                    "type": "feature",
                    "group": item.feature_group,
                    "value": item.feature_value,
                    "baseline": item.baseline_value,
                    "score": item.contribution_score,
                    "direction": item.direction,
                }
            )
            matchup_edges.append(
                {
                    "source": node_id,
                    "target": f"{row.matchup_key}:raw",
                    "label": item.direction,
                    "weight": item.abs_contribution_score,
                }
            )
        matchups.append(
            {
                "matchup_key": row.matchup_key,
                "label": f"{row.home_team_name} vs {row.away_team_name}",
                "recommended_pick": pick,
                "raw_model_pick": raw_pick,
                "draw_risk": draw_risk,
                "prob_home": float(row.prob_home),
                "prob_draw": float(row.prob_draw),
                "prob_away": float(row.prob_away),
                "nodes": matchup_nodes,
                "edges": matchup_edges,
            }
        )
    return {"matchups": matchups, "draw_policy": draw_policy}


def html_doc(graph: dict[str, Any]) -> str:
    payload = json.dumps(graph, default=str)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Prediction Explanation Graph</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f6f7f9; color: #20242a; }}
    header {{ padding: 16px 20px; background: #101820; color: white; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }}
    select {{ min-width: 320px; padding: 8px; }}
    main {{ display: grid; grid-template-columns: 340px 1fr; min-height: calc(100vh - 65px); }}
    aside {{ padding: 16px; background: white; border-right: 1px solid #d9dde3; overflow: auto; }}
    #graph {{ position: relative; min-height: 720px; overflow: auto; }}
    .metric {{ display: grid; grid-template-columns: 1fr auto; padding: 6px 0; border-bottom: 1px solid #edf0f2; }}
    .node {{ position: absolute; border: 1px solid #b9c1ca; background: white; border-radius: 8px; padding: 10px; min-width: 145px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    .match {{ background: #eaf2ff; border-color: #7ba7e8; }}
    .team {{ background: #f8f4e7; border-color: #d8be70; }}
    .decision {{ background: #e8f7ee; border-color: #6bb985; }}
    .model_pick {{ background: #f1edff; border-color: #9b88d8; }}
    .draw_policy {{ background: #fff0f0; border-color: #dd8888; }}
    .feature {{ background: #ffffff; }}
    .feature.above_baseline {{ border-left: 5px solid #267a3e; }}
    .feature.below_baseline {{ border-left: 5px solid #b64242; }}
    .feature.supports_prediction {{ border-left: 5px solid #267a3e; }}
    .feature.opposes_prediction {{ border-left: 5px solid #b64242; }}
    svg {{ position: absolute; inset: 0; width: 1300px; height: 880px; pointer-events: none; }}
    line {{ stroke: #8a94a3; stroke-width: 1.4; }}
    .small {{ color: #58616d; font-size: 12px; margin-top: 4px; }}
    @media (max-width: 900px) {{ main {{ grid-template-columns: 1fr; }} aside {{ border-right: 0; border-bottom: 1px solid #d9dde3; }} }}
  </style>
</head>
<body>
  <header>
    <strong>Prediction Explanation Graph</strong>
    <select id="matchup"></select>
  </header>
  <main>
    <aside>
      <h3 id="title"></h3>
      <div id="metrics"></div>
      <h4>Top Feature Signals</h4>
      <div id="features"></div>
    </aside>
    <section id="graph"><svg id="edges"></svg></section>
  </main>
  <script>
    const graph = {payload};
    const positions = {{
      match: [520, 40], team_home: [260, 150], team_away: [780, 150],
      prob_home: [250, 300], prob_draw: [520, 300], prob_away: [790, 300],
      raw: [390, 460], recommended: [650, 460], draw_policy: [910, 460]
    }};
    const select = document.getElementById('matchup');
    graph.matchups.forEach((m, i) => {{
      const opt = document.createElement('option');
      opt.value = i;
      opt.textContent = `${{m.matchup_key}} | ${{m.label}}`;
      select.appendChild(opt);
    }});
    select.addEventListener('change', () => render(Number(select.value)));
    function posFor(node, index) {{
      if (node.type === 'team') return node.side === 'home' ? positions.team_home : positions.team_away;
      if (node.type === 'match') return positions.match;
      if (node.type === 'decision') return positions.recommended;
      if (node.type === 'model_pick') return positions.raw;
      if (node.type === 'draw_policy') return positions.draw_policy;
      if (node.id.endsWith(':prob_home')) return positions.prob_home;
      if (node.id.endsWith(':prob_draw')) return positions.prob_draw;
      if (node.id.endsWith(':prob_away')) return positions.prob_away;
      const featureIndex = index;
      return [70 + (featureIndex % 4) * 290, 610 + Math.floor(featureIndex / 4) * 95];
    }}
    function render(index) {{
      const m = graph.matchups[index];
      document.getElementById('title').textContent = m.label;
      document.getElementById('metrics').innerHTML = [
        ['Recommended', m.recommended_pick], ['Raw model', m.raw_model_pick], ['Draw risk', m.draw_risk],
        ['Home', m.prob_home.toFixed(3)], ['Draw', m.prob_draw.toFixed(3)], ['Away', m.prob_away.toFixed(3)]
      ].map(([k,v]) => `<div class="metric"><span>${{k}}</span><strong>${{v}}</strong></div>`).join('');
      const g = document.getElementById('graph');
      g.querySelectorAll('.node').forEach(n => n.remove());
      const featureNodes = m.nodes.filter(n => n.type === 'feature');
      document.getElementById('features').innerHTML = featureNodes.map(n => `<div class="metric"><span>${{n.label}}</span><strong>${{Number(n.score).toFixed(2)}}</strong></div>`).join('');
      const coords = {{}};
      let featureIndex = 0;
      m.nodes.forEach(n => {{
        const idx = n.type === 'feature' ? featureIndex++ : 0;
        const [x,y] = posFor(n, idx);
        coords[n.id] = [x,y];
        const div = document.createElement('div');
        div.className = `node ${{n.type}} ${{n.direction || ''}}`;
        div.style.left = `${{x}}px`; div.style.top = `${{y}}px`;
        div.innerHTML = `<strong>${{htmlEscape(n.label)}}</strong><div class="small">${{n.type}}</div>`;
        g.appendChild(div);
      }});
      const svg = document.getElementById('edges');
      svg.innerHTML = '';
      m.edges.forEach(e => {{
        if (!coords[e.source] || !coords[e.target]) return;
        const [x1,y1] = coords[e.source], [x2,y2] = coords[e.target];
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1 + 75); line.setAttribute('y1', y1 + 25);
        line.setAttribute('x2', x2 + 75); line.setAttribute('y2', y2 + 25);
        svg.appendChild(line);
      }});
    }}
    function htmlEscape(s) {{ return String(s).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
    render(0);
  </script>
</body>
</html>"""


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = read_table(args.predictions)
    model_input = read_table(args.model_input)
    preprocessing = joblib.load(args.preprocessing)
    draw_policy = load_draw_policy(args.draw_policy).to_dict() if args.draw_policy else None
    required = {"matchup_key", "prob_home", "prob_draw", "prob_away", "home_team_name", "away_team_name"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise RuntimeError(f"Prediction file is missing required columns: {missing}")
    if "matchup_key" not in model_input.columns:
        raise RuntimeError("Model input file must include matchup_key.")

    if args.model:
        contributions = model_sensitivity_contribution_frame(
            predictions=predictions,
            model_input=model_input,
            preprocessing=preprocessing,
            model_path=args.model,
            top_n=args.top_features,
            candidate_features=args.candidate_features,
        )
    else:
        contributions = baseline_delta_contribution_frame(model_input, preprocessing, args.top_features)
    graph = build_graph(predictions, contributions, draw_policy)
    graph_path = output_dir / "prediction_explanation_graph.json"
    html_path = output_dir / "prediction_explanation_graph.html"
    contributions_path = output_dir / "feature_contributions.csv"
    graph_path.write_text(json.dumps(graph, indent=2, default=str) + "\n")
    html_path.write_text(html_doc(graph))
    contributions.to_csv(contributions_path, index=False)
    summary = {
        "predictions": args.predictions,
        "model_input": args.model_input,
        "model": args.model,
        "contribution_method": "model_probability_perturbation" if args.model else "baseline_z_delta",
        "matchups": len(graph["matchups"]),
        "outputs": {
            "graph_json": str(graph_path),
            "graph_html": str(html_path),
            "feature_contributions": str(contributions_path),
        },
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
