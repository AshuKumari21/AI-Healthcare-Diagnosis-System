"""
Explainable AI (XAI) Visualizer: Interactive SHAP Plotly Generator
Aegis AI Healthcare Diagnosis System

Creates interactive Plotly/Dash charts illustrating feature-level SHAP attributions,
positive/negative contributions, and baseline thresholds.
"""

from typing import Any, Dict, Optional
import plotly.graph_objects as go


def create_shap_figure(explanation: Dict[str, Any], max_features: int = 7) -> go.Figure:
    """
    Builds an interactive Plotly horizontal diverging bar chart representing
    SHAP feature contributions for the 'Why This Prediction?' dashboard section.

    - Coral Red (#ff3366): Positive contribution (Increases predicted risk)
    - Emerald Green (#00e676): Negative contribution (Decreases predicted risk / Protective)
    """
    if not explanation or not explanation.get("all_attributions"):
        fig = go.Figure()
        fig.add_annotation(
            text="SHAP explainability data awaiting inference.<br><sup>Run diagnostic inference to visualize feature attribution.</sup>",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color="#64748b", family="Outfit, sans-serif")
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=280
        )
        return fig

    # Extract all features and pick top N by absolute SHAP impact
    all_attribs = explanation.get("all_attributions", [])
    sorted_features = sorted(all_attribs, key=lambda x: x["abs_shap"], reverse=True)[:max_features]

    # Reverse so largest impact appears at the top of horizontal chart
    sorted_features.reverse()

    labels = []
    shap_vals = []
    colors = []
    hover_texts = []
    text_labels = []

    for item in sorted_features:
        name = item["display_name"]
        val = item["recorded_value"]
        shap = item["shap_value"]
        direction = item["direction"]

        is_positive = shap >= 0
        color = "#ff3366" if is_positive else "#00e676"
        effect_text = "Increases Risk (+)" if is_positive else "Decreases Risk / Protective (-)"
        sign = "+" if is_positive else ""

        labels.append(name)
        shap_vals.append(shap)
        colors.append(color)
        text_labels.append(f"{sign}{shap:.2f}")

        hover_texts.append(
            f"<b>{name}</b><br>"
            f"Recorded Patient Value: <b>{val}</b><br>"
            f"SHAP Impact Score: <b>{sign}{shap:.3f}</b><br>"
            f"Clinical Effect: <b style='color:{color}'>{effect_text}</b>"
        )

    fig = go.Figure()

    # Add diverging bar trace
    fig.add_trace(go.Bar(
        y=labels,
        x=shap_vals,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255, 255, 255, 0.25)', width=1)
        ),
        text=text_labels,
        textposition=['outside' if s >= 0 else 'inside' for s in shap_vals],
        textfont=dict(color='#f1f5f9', size=11, family="Outfit, sans-serif"),
        hovertext=hover_texts,
        hoverinfo='text'
    ))

    # Add reference baseline zero-line
    fig.add_vline(
        x=0,
        line_color="rgba(255, 255, 255, 0.4)",
        line_width=1.5,
        annotation_text="Baseline (0)",
        annotation_position="top right",
        annotation_font=dict(color="#94a3b8", size=9, family="Outfit, sans-serif")
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255, 255, 255, 0.02)',
        margin=dict(l=15, r=35, t=25, b=35),
        height=max(280, len(labels) * 40),
        showlegend=False,
        xaxis=dict(
            title="SHAP Attribution Value (Impact on Risk)",
            title_font=dict(size=11, color="#94a3b8", family="Outfit, sans-serif"),
            gridcolor='rgba(255, 255, 255, 0.06)',
            zerolinecolor='rgba(255, 255, 255, 0.2)'
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="#e2e8f0", family="Outfit, sans-serif")
        )
    )

    return fig


def build_why_prediction_ui(
    explanation: Optional[Dict[str, Any]],
    disease_id: str = "general",
    prob: float = 0.0,
    level: str = "NORMAL"
) -> Any:
    """
    Renders the rich Dash HTML breakdown for the 'Why This Prediction?' section:
    - Predicted disease / risk status badge
    - Risk probability score
    - Top 5 contributing factors with positive/negative tags
    - Dynamic natural language clinical explanations
    """
    from dash import html
    import dash_bootstrap_components as dbc

    if not explanation or not explanation.get("all_attributions"):
        return html.Div(className="text-center py-4 text-muted", children=[
            html.I(className="fa-solid fa-microscope text-info mb-2 fs-3 d-block"),
            html.P("Awaiting patient evaluation.", className="fw-semibold mb-1 text-white"),
            html.Small("Enter patient vitals or clinical notes and click 'Run Diagnostic Inference' to compute feature-level SHAP attributions and clinical rationale.", className="text-muted")
        ])

    top_5 = explanation.get("top_5_contributing", [])
    positive = explanation.get("positive_factors", [])
    negative = explanation.get("negative_factors", [])
    narrative = explanation.get("clinical_explanation", [])
    explainer_name = explanation.get("explainer_type", "SHAP Engine")
    primary_driver = explanation.get("primary_driver", "Primary Parameter")

    # Risk badge styling
    level_lower = level.lower()
    if "crit" in level_lower:
        badge_color, badge_bg = "#ff3366", "rgba(255, 51, 102, 0.15)"
    elif "high" in level_lower:
        badge_color, badge_bg = "#ff851b", "rgba(255, 133, 27, 0.15)"
    elif "mod" in level_lower:
        badge_color, badge_bg = "#ffb300", "rgba(255, 179, 0, 0.15)"
    else:
        badge_color, badge_bg = "#00e676", "rgba(0, 230, 118, 0.15)"

    # Build Top 5 Factor Cards
    factor_items = []
    for i, factor in enumerate(top_5, 1):
        is_pos = factor["direction"] == "positive"
        tag_color = "#ff3366" if is_pos else "#00e676"
        tag_bg = "rgba(255, 51, 102, 0.12)" if is_pos else "rgba(0, 230, 118, 0.12)"
        dir_icon = "fa-arrow-trend-up" if is_pos else "fa-arrow-trend-down"
        dir_text = "Increases Risk" if is_pos else "Protective / Decreases"

        factor_items.append(
            html.Div(
                className="d-flex justify-content-between align-items-center p-2 mb-2 rounded-2",
                style={"background": "rgba(255, 255, 255, 0.03)", "border": "1px solid rgba(255, 255, 255, 0.06)"},
                children=[
                    html.Div([
                        html.Span(f"#{i} ", className="text-muted small fw-bold me-1"),
                        html.Strong(factor["display_name"], className="text-white small me-2"),
                        html.Span(f"({factor['recorded_value']})", className="text-muted", style={"fontSize": "0.75rem"})
                    ]),
                    html.Div(className="d-flex align-items-center", children=[
                        html.Span(
                            [html.I(className=f"fa-solid {dir_icon} me-1"), factor["impact_formatted"]],
                            className="badge px-2 py-1 fw-bold me-2",
                            style={"background": tag_bg, "color": tag_color, "fontSize": "0.78rem"}
                        ),
                        html.Small(dir_text, className="text-muted d-none d-md-inline", style={"fontSize": "0.72rem"})
                    ])
                ]
            )
        )

    # Build Dynamic Narrative Bullet Points
    narrative_items = []
    for line in narrative:
        narrative_items.append(
            html.Div(className="d-flex align-items-start mb-2", children=[
                html.I(className="fa-solid fa-circle-check text-info me-2 mt-1", style={"fontSize": "0.8rem"}),
                html.Span(line, className="text-light small", style={"lineHeight": "1.4"})
            ])
        )

    return html.Div(children=[
        # 1. Summary Header Card
        dbc.Row(className="g-3 mb-3", children=[
            dbc.Col([
                html.Div(className="p-3 rounded-3", style={"background": "rgba(255,255,255,0.02)", "border": "1px solid rgba(255,255,255,0.08)"}, children=[
                    html.Small("Predicted Diagnostic Status", className="text-muted d-block mb-1"),
                    html.Span(
                        level,
                        className="badge px-3 py-2 fw-bold",
                        style={"background": badge_bg, "color": badge_color, "fontSize": "0.88rem", "border": f"1px solid {badge_color}44"}
                    )
                ])
            ], xs=6, md=3),
            dbc.Col([
                html.Div(className="p-3 rounded-3", style={"background": "rgba(255,255,255,0.02)", "border": "1px solid rgba(255,255,255,0.08)"}, children=[
                    html.Small("Risk Probability / Score", className="text-muted d-block mb-1"),
                    html.H5(f"{prob:.1f}%", className="text-white fw-bold mb-0")
                ])
            ], xs=6, md=3),
            dbc.Col([
                html.Div(className="p-3 rounded-3", style={"background": "rgba(255,255,255,0.02)", "border": "1px solid rgba(255,255,255,0.08)"}, children=[
                    html.Small("Primary Risk Driver", className="text-muted d-block mb-1"),
                    html.H6(primary_driver, className="text-info fw-bold mb-0 text-truncate")
                ])
            ], xs=6, md=3),
            dbc.Col([
                html.Div(className="p-3 rounded-3", style={"background": "rgba(255,255,255,0.02)", "border": "1px solid rgba(255,255,255,0.08)"}, children=[
                    html.Small("SHAP Explainer Engine", className="text-muted d-block mb-1"),
                    html.Span([
                        html.I(className="fa-solid fa-code-branch text-info me-1"),
                        explainer_name
                    ], className="text-light fw-semibold small")
                ])
            ], xs=6, md=3),
        ]),

        # 2. Top 5 Contributing Factors List
        html.Div(className="mb-3", children=[
            html.Div(className="d-flex justify-content-between align-items-center mb-2", children=[
                html.Small("Top 5 Contributing Factors (SHAP Feature Importance)", className="text-white fw-semibold"),
                html.Small([
                    html.Span(f"{len(positive)} Risk-Increasing (+)", className="text-danger me-2", style={"fontSize": "0.72rem"}),
                    html.Span(f"{len(negative)} Protective (-)", className="text-success", style={"fontSize": "0.72rem"})
                ])
            ]),
            html.Div(factor_items)
        ]),

        # 3. Dynamic Natural Language Clinical Explanations Callout
        html.Div(
            className="p-3 rounded-3",
            style={
                "background": "rgba(0, 242, 255, 0.04)",
                "border": "1px solid rgba(0, 242, 255, 0.2)",
                "borderRadius": "8px"
            },
            children=[
                html.Div(className="d-flex align-items-center mb-2", children=[
                    html.I(className="fa-solid fa-lightbulb text-info me-2"),
                    html.Strong("Clinical Interpretability Rationale", className="text-info small")
                ]),
                html.Div(narrative_items)
            ]
        )
    ])
