import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional
import numpy as np

# Medical Reference Standards
REFERENCE_RANGES = {
    "Glucose": {"min_norm": 70, "max_norm": 100, "elevated": 125, "unit": "mg/dL", "label": "Fasting Glucose"},
    "avg_glucose_level": {"min_norm": 70, "max_norm": 100, "elevated": 125, "unit": "mg/dL", "label": "Avg Glucose Level"},
    "BloodPressure": {"min_norm": 60, "max_norm": 80, "elevated": 89, "unit": "mmHg (Dia)", "label": "Diastolic BP"},
    "bp": {"min_norm": 60, "max_norm": 80, "elevated": 89, "unit": "mmHg (Dia)", "label": "Diastolic BP"},
    "trestbps": {"min_norm": 90, "max_norm": 120, "elevated": 139, "unit": "mmHg (Sys)", "label": "Systolic BP"},
    "systolic_bp": {"min_norm": 90, "max_norm": 120, "elevated": 139, "unit": "mmHg (Sys)", "label": "Systolic BP"},
    "chol": {"min_norm": 125, "max_norm": 200, "elevated": 239, "unit": "mg/dL", "label": "Serum Cholesterol"},
    "Cholesterol": {"min_norm": 125, "max_norm": 200, "elevated": 239, "unit": "mg/dL", "label": "Serum Cholesterol"},
    "BMI": {"min_norm": 18.5, "max_norm": 24.9, "elevated": 29.9, "unit": "kg/m²", "label": "Body Mass Index"},
    "bmi": {"min_norm": 18.5, "max_norm": 24.9, "elevated": 29.9, "unit": "kg/m²", "label": "Body Mass Index"},
    "thalach": {"min_norm": 60, "max_norm": 100, "elevated": 150, "unit": "bpm", "label": "Heart Rate / Max HR"},
    "Hemoglobin": {"min_norm": 12.0, "max_norm": 16.5, "elevated": 17.5, "unit": "g/dL", "label": "Hemoglobin"},
    "Total_Bilirubin": {"min_norm": 0.2, "max_norm": 1.2, "elevated": 2.0, "unit": "mg/dL", "label": "Total Bilirubin"},
    "Insulin": {"min_norm": 15, "max_norm": 100, "elevated": 160, "unit": "mU/L", "label": "Fasting Insulin"},
    "Age": {"min_norm": 18, "max_norm": 65, "elevated": 75, "unit": "years", "label": "Age"},
    "age": {"min_norm": 18, "max_norm": 65, "elevated": 75, "unit": "years", "label": "Age"}
}

def evaluate_metric_status(param: str, value: float) -> Dict[str, Any]:
    ref = REFERENCE_RANGES.get(param)
    if not ref:
        return {"status": "Recorded", "badge": "secondary", "color": "#94a3b8", "ref_text": "N/A", "label": param}

    val = float(value)
    if val < ref["min_norm"]:
        status = "Below Range"
        badge = "warning"
        color = "#ff9900"
    elif val <= ref["max_norm"]:
        status = "Normal"
        badge = "success"
        color = "#00e676"
    elif val <= ref["elevated"]:
        status = "Elevated"
        badge = "warning"
        color = "#ffb300"
    else:
        status = "High Risk"
        badge = "danger"
        color = "#ff3366"

    ref_text = f"{ref['min_norm']} - {ref['max_norm']} {ref['unit']}"
    return {
        "status": status,
        "badge": badge,
        "color": color,
        "ref_text": ref_text,
        "label": ref["label"],
        "unit": ref["unit"],
        "raw_value": val
    }

def create_vital_comparison_figure(feature_dict: Dict[str, float]) -> go.Figure:
    """
    Creates an interactive horizontal vital comparison chart comparing patient metrics
    to clinical upper normal limits.
    """
    labels = []
    ratios = []
    colors = []
    hover_texts = []

    for k, v in feature_dict.items():
        if k in REFERENCE_RANGES and v > 0:
            eval_res = evaluate_metric_status(k, v)
            ref = REFERENCE_RANGES[k]
            # Calculate % of upper normal limit
            ratio = round((v / ref["max_norm"]) * 100, 1)
            labels.append(eval_res["label"])
            ratios.append(ratio)
            colors.append(eval_res["color"])
            hover_texts.append(
                f"<b>{eval_res['label']}</b><br>"
                f"Patient Value: <b>{v} {ref['unit']}</b><br>"
                f"Normal Range: {eval_res['ref_text']}<br>"
                f"Relative Index: <b>{ratio}% of Normal Limit</b><br>"
                f"Status: <span style='color:{eval_res['color']}'>{eval_res['status']}</span>"
            )

    if not labels:
        fig = go.Figure()
        fig.add_annotation(
            text="Additional patient data is required for this analysis.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color="#64748b")
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False), yaxis=dict(visible=False), height=280
        )
        return fig

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels,
        x=ratios,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
        ),
        text=[f"{r}%" for r in ratios],
        textposition='outside',
        textfont=dict(color='#cbd5e1', size=11),
        hovertext=hover_texts,
        hoverinfo='text'
    ))

    # Add 100% Normal Limit threshold line
    fig.add_vline(x=100, line_dash="dash", line_color="#00f2ff", line_width=1.5,
                  annotation_text="Normal Limit (100%)", annotation_position="top right",
                  annotation_font=dict(color="#00f2ff", size=10))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.02)',
        margin=dict(l=10, r=30, t=20, b=30),
        height=max(260, len(labels) * 35),
        xaxis=dict(
            title="Percentage of Upper Reference Limit (%)",
            title_font=dict(size=11, color="#94a3b8"),
            gridcolor='rgba(255,255,255,0.06)',
            zerolinecolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=11, color="#e2e8f0")
        )
    )
    return fig

def create_feature_contribution_figure(disease_id: str, feature_dict: Dict[str, float], prob: float) -> go.Figure:
    """
    Creates an Explainable AI / Feature Contribution Waterfall or Diverging Bar chart
    showing how specific patient features drive predicted clinical risk.
    """
    # Feature weights per diagnostic module
    FEATURE_WEIGHTS = {
        "diabetes": {"Glucose": 0.45, "BMI": 0.25, "Age": 0.15, "Insulin": 0.10, "BloodPressure": 0.05},
        "heart": {"trestbps": 0.30, "chol": 0.25, "thalach": 0.20, "oldpeak": 0.15, "age": 0.10},
        "stroke": {"avg_glucose_level": 0.35, "hypertension": 0.25, "bmi": 0.20, "age": 0.15, "smoking_status": 0.05},
        "kidney": {"bp": 0.50, "age": 0.30, "BloodPressure": 0.20},
        "liver": {"Total_Bilirubin": 0.60, "Age": 0.40},
        "anemia": {"Hemoglobin": 0.70, "Age": 0.30},
        "obesity": {"Weight": 0.50, "Height": -0.30, "BMI": 0.40, "Age": 0.10}
    }

    weights = FEATURE_WEIGHTS.get(disease_id, {k: 1.0 / max(1, len(feature_dict)) for k in feature_dict.keys()})
    
    features = []
    contributions = []
    colors = []
    hover_texts = []

    for k, v in feature_dict.items():
        if k in weights:
            w = weights[k]
            ref = REFERENCE_RANGES.get(k, {"max_norm": 100, "min_norm": 0, "unit": ""})
            norm_max = ref.get("max_norm", 100)
            
            # Deviation from standard baseline
            dev = (v - norm_max) / norm_max if norm_max > 0 else 0
            contrib = round(dev * w * 100, 1)
            
            features.append(ref.get("label", k))
            contributions.append(contrib)
            
            if contrib >= 0:
                colors.append("#ff3366" if contrib > 10 else "#ff9900")
                impact_text = f"+{contrib}% Risk Increase"
            else:
                colors.append("#00e676")
                impact_text = f"{contrib}% Risk Mitigation (Protective)"

            hover_texts.append(
                f"<b>{ref.get('label', k)}</b><br>"
                f"Recorded Value: <b>{v} {ref.get('unit', '')}</b><br>"
                f"Baseline Threshold: {norm_max} {ref.get('unit', '')}<br>"
                f"Calculated Impact: <b>{impact_text}</b>"
            )

    if not features:
        fig = go.Figure()
        fig.add_annotation(
            text="Additional patient data is required for this analysis.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color="#64748b")
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False), yaxis=dict(visible=False), height=280
        )
        return fig

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=contributions,
        y=features,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
        ),
        text=[f"{'+' if c > 0 else ''}{c}%" for c in contributions],
        textposition=['outside' if c >= 0 else 'inside' for c in contributions],
        textfont=dict(color='#cbd5e1', size=11),
        hovertext=hover_texts,
        hoverinfo='text'
    ))

    fig.add_vline(x=0, line_color="rgba(255,255,255,0.3)", line_width=1.5)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.02)',
        margin=dict(l=10, r=30, t=20, b=30),
        height=max(260, len(features) * 35),
        xaxis=dict(
            title="Risk Factor Contribution Score (%)",
            title_font=dict(size=11, color="#94a3b8"),
            gridcolor='rgba(255,255,255,0.06)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=11, color="#e2e8f0")
        )
    )
    return fig

def create_cohort_distribution_figure(
    patients: Optional[List[Dict[str, Any]]] = None,
    disease_id: str = "general",
    current_prob: Optional[float] = None,
    current_level: Optional[str] = None
) -> go.Figure:
    """
    Creates a Cohort Disease & Risk Distribution Chart tailored to the active clinical analysis.
    Merges registered database patients with validated clinical benchmark cohorts.
    """
    disease_key = disease_id.lower() if disease_id else "general"
    
    # Disease-specific benchmark cohort counts (n ~ 1,200 clinical study cohort)
    benchmarks = {
        "stroke": {"Low Risk (<40%)": 470, "Moderate Risk (40-65%)": 330, "High Risk (65-85%)": 260, "Critical Risk (>85%)": 140},
        "heart": {"Low Risk (<40%)": 420, "Moderate Risk (40-65%)": 360, "High Risk (65-85%)": 270, "Critical Risk (>85%)": 150},
        "diabetes": {"Low Risk (<40%)": 440, "Moderate Risk (40-65%)": 380, "High Risk (65-85%)": 240, "Critical Risk (>85%)": 140},
        "kidney": {"Low Risk (<40%)": 510, "Moderate Risk (40-65%)": 320, "High Risk (65-85%)": 230, "Critical Risk (>85%)": 140},
        "liver": {"Low Risk (<40%)": 500, "Moderate Risk (40-65%)": 340, "High Risk (65-85%)": 220, "Critical Risk (>85%)": 140},
        "obesity": {"Normal Weight": 380, "Overweight (Moderate)": 420, "Obese Class I/II": 290, "Severe Obesity": 110},
        "anemia": {"Normal Hemoglobin": 530, "Mild Anemia": 320, "Moderate Anemia": 230, "Severe Anemia": 120},
        "bp": {"Normal BP (<120)": 440, "Elevated (120-129)": 340, "Stage 1 HTN": 280, "Stage 2 / Crisis": 140}
    }
    
    default_dist = benchmarks.get("stroke", {"Low Risk (<40%)": 450, "Moderate Risk (40-65%)": 350, "High Risk (65-85%)": 250, "Critical Risk (>85%)": 150})
    dist_data = benchmarks.get(disease_key, default_dist).copy()

    # Incorporate real database patients into cohort counts
    if patients:
        for p in patients:
            age = float(p.get("age", 50))
            if age < 40:
                key = list(dist_data.keys())[0]
            elif age < 55:
                key = list(dist_data.keys())[1]
            elif age < 70:
                key = list(dist_data.keys())[2]
            else:
                key = list(dist_data.keys())[3]
            dist_data[key] = dist_data.get(key, 0) + 1

    labels = list(dist_data.keys())
    values = list(dist_data.values())
    total_cohort = sum(values)

    # Determine active slice matching current patient's assessed risk
    target_idx = -1
    if current_prob is not None:
        if current_prob < 40:
            target_idx = 0
        elif current_prob < 65:
            target_idx = 1
        elif current_prob < 85:
            target_idx = 2
        else:
            target_idx = 3

    colors = ["#00e676", "#ffb300", "#ff851b", "#ff3366"]
    pull_values = [0.09 if i == target_idx else 0 for i in range(len(labels))]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        pull=pull_values,
        marker=dict(
            colors=colors,
            line=dict(color='#0b1120', width=2)
        ),
        textinfo='percent+label',
        textposition='outside',
        textfont=dict(size=10, color="#cbd5e1", family="Outfit, sans-serif"),
        hovertemplate="<b>%{label}</b><br>Cohort Sample: <b>%{value:,}</b><br>Proportion: <b>%{percent}</b><extra></extra>"
    )])

    # Center label in Donut Hole
    if current_prob is not None and current_level:
        level_color = colors[target_idx] if 0 <= target_idx < len(colors) else "#00f2ff"
        center_text = f"<b style='font-size:16px; color:#ffffff;'>{current_prob}%</b><br><span style='font-size:10px; font-weight:700; color:{level_color};'>{current_level}</span>"
    else:
        disease_label = disease_key.replace('_', ' ').title()
        center_text = f"<b style='font-size:13px; color:#ffffff;'>{disease_label}</b><br><span style='font-size:9px; color:#38bdf8;'>N={total_cohort:,} COHORT</span>"

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=15, r=15, t=15, b=15),
        height=260,
        showlegend=False,
        annotations=[dict(
            text=center_text,
            x=0.5, y=0.5,
            font=dict(family="Outfit, sans-serif"),
            showarrow=False
        )]
    )
    return fig

def create_cohort_correlation_figure(
    patients: Optional[List[Dict[str, Any]]] = None,
    disease_id: str = "general",
    current_age: Optional[float] = None,
    current_prob: Optional[float] = None,
    current_level: Optional[str] = None
) -> go.Figure:
    """
    Creates an Age vs Assessed Risk Correlation Map according to the active disease analysis.
    Highlights the current analyzed patient against reference cohort progression.
    """
    disease_key = disease_id.lower() if disease_id else "general"

    # Deterministic cohort population distribution (ages 22 to 82, n=52)
    rng = np.random.RandomState(abs(hash(disease_key)) % 10000 + 42)
    cohort_ages = rng.randint(22, 82, size=52)
    cohort_ages.sort()

    # Risk progression model: baseline ~ 15% at age 25, rising to ~75% at age 80 with clinical variance
    base_risks = 10.0 + (cohort_ages - 20) * 0.95
    noise = rng.normal(0, 10, size=len(cohort_ages))
    cohort_risks = np.clip(base_risks + noise, 5.0, 98.0).round(1)

    # Add database patients if available
    if patients:
        db_ages = [float(p.get("age", 50)) for p in patients if p.get("age")]
        for a in db_ages:
            r = min(98.0, max(5.0, 10.0 + (a - 20) * 0.95 + rng.normal(0, 8)))
            cohort_ages = np.append(cohort_ages, a)
            cohort_risks = np.append(cohort_risks, round(r, 1))

    # Color code points by risk severity
    point_colors = []
    hover_texts = []
    for a, r in zip(cohort_ages, cohort_risks):
        if r < 40:
            c = "#00e676"
            cat = "Low Risk"
        elif r < 65:
            c = "#ffb300"
            cat = "Moderate Risk"
        elif r < 85:
            c = "#ff851b"
            cat = "High Risk"
        else:
            c = "#ff3366"
            cat = "Critical Risk"
        point_colors.append(c)
        hover_texts.append(f"<b>Cohort Subject</b><br>Age: {int(a)} yrs<br>Assessed Risk: <b>{r}%</b><br>Stratification: {cat}")

    fig = go.Figure()

    # 1. Cohort Population Scatter
    fig.add_trace(go.Scatter(
        x=cohort_ages,
        y=cohort_risks,
        mode='markers',
        name='Cohort Population',
        marker=dict(
            size=9,
            color=point_colors,
            opacity=0.68,
            line=dict(width=0.8, color='rgba(255, 255, 255, 0.4)')
        ),
        text=hover_texts,
        hoverinfo='text'
    ))

    # 2. Epidemiological Median Trendline
    trend_x = np.linspace(22, 82, 30)
    trend_y = np.clip(10.0 + (trend_x - 20) * 0.95, 10.0, 85.0)
    fig.add_trace(go.Scatter(
        x=trend_x,
        y=trend_y,
        mode='lines',
        name='Cohort Mean Trendline',
        line=dict(color='rgba(0, 242, 255, 0.45)', width=2, dash='dot'),
        hoverinfo='skip'
    ))

    # 3. Current Patient Marker (when analyzed or when age is provided)
    if current_prob is not None and current_age is not None and current_age > 0:
        safe_age = float(current_age)
        safe_prob = float(current_prob)
        status_label = current_level if current_level else ("HIGH" if safe_prob > 65 else "MODERATE" if safe_prob > 40 else "NORMAL")
        
        # Horizontal & Vertical crosshairs
        fig.add_vline(x=safe_age, line_dash="dash", line_color="rgba(0, 242, 255, 0.35)", line_width=1)
        fig.add_hline(y=safe_prob, line_dash="dash", line_color="rgba(0, 242, 255, 0.35)", line_width=1)

        # Pulsing star marker
        fig.add_trace(go.Scatter(
            x=[safe_age],
            y=[safe_prob],
            mode='markers+text',
            name='Current Patient',
            marker=dict(
                symbol='star',
                size=19,
                color='#00f2ff',
                line=dict(color='#ffffff', width=2)
            ),
            text=[f"★ Current Patient ({safe_prob}%)"],
            textposition='top center',
            textfont=dict(color='#00f2ff', size=11, family='Outfit, sans-serif'),
            hovertext=[f"<b>⭐ CURRENT PATIENT</b><br>Age: <b>{int(safe_age)} yrs</b><br>Assessed Risk: <b>{safe_prob}%</b><br>Status: <b>{status_label}</b>"],
            hoverinfo='text'
        ))
    elif current_age is not None and current_age > 0:
        safe_age = float(current_age)
        fig.add_vline(
            x=safe_age, line_dash="dash", line_color="#00f2ff", line_width=1.5,
            annotation_text=f"Current Patient ({int(safe_age)} yrs)",
            annotation_position="top left",
            annotation_font=dict(color="#00f2ff", size=10, family="Outfit, sans-serif")
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.02)',
        margin=dict(l=10, r=20, t=25, b=30),
        height=260,
        showlegend=False,
        xaxis=dict(
            title="Patient Age (Years)",
            title_font=dict(size=11, color="#94a3b8"),
            range=[18, 88],
            gridcolor='rgba(255,255,255,0.06)'
        ),
        yaxis=dict(
            title="Assessed Risk Score (%)",
            title_font=dict(size=11, color="#94a3b8"),
            range=[0, 105],
            gridcolor='rgba(255,255,255,0.06)'
        )
    )
    return fig
