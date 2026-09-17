"""
Streamlit application for the TCO Material Recommendation System.
Uses a pre-trained RandomForestRegressor to support suitability predictions
and ranking of Transparent Conducting Oxide materials for fluorescent lamp coatings.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "tco_materials.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "random_forest.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

FEATURE_COLS = [
    "Electrical_Conductivity",
    "Optical_Transparency",
    "Thermal_Stability",
    "Chemical_Stability",
    "Mercury_Corrosion_Resistance",
    "Adhesion_Strength",
    "Durability",
    "Cost",
]

# Qualitative → numerical mapping (1–10 scale used in the demo dataset)
# Higher is better for all properties EXCEPT Cost (higher Cost = more expensive)
LEVEL_MAP = {
    "Very Low": 2.0,
    "Low": 4.0,
    "Medium": 6.0,
    "High": 8.0,
    "Very High": 9.5,
}

# For Cost the user selects “desired cost level”; we invert so that
# “Very Low” cost becomes a high numerical preference (good).
COST_LEVEL_MAP = {
    "Very Low": 9.5,   # user wants cheap → high “goodness”
    "Low": 8.0,
    "Medium": 6.0,
    "High": 4.0,
    "Very High": 2.0,  # user accepts expensive → low “goodness”
}


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def map_user_requirements(user_inputs: dict) -> np.ndarray:
    """Convert qualitative user choices into the numerical feature vector
    expected by the model / distance calculations."""
    vec = []
    for col in FEATURE_COLS:
        if col == "Cost":
            vec.append(COST_LEVEL_MAP[user_inputs[col]])
        else:
            vec.append(LEVEL_MAP[user_inputs[col]])
    return np.array(vec, dtype=float)


def compute_match_scores(df: pd.DataFrame, user_vec: np.ndarray, importances: np.ndarray):
    """
    Rank materials by a weighted match score that incorporates:
    - Absolute difference on each property (normalized)
    - Feature importances learned by the Random Forest
    Cost is already inverted in user_vec so higher = better preference.
    """
    props = df[FEATURE_COLS].values.astype(float)
    # For Cost column in data, higher = more expensive = worse.
    # Invert the material Cost so that lower real cost → higher “goodness”.
    props_inv = props.copy()
    cost_idx = FEATURE_COLS.index("Cost")
    props_inv[:, cost_idx] = 10.0 - props_inv[:, cost_idx]  # invert

    # Absolute difference (user preference vs material “goodness”)
    diffs = np.abs(props_inv - user_vec)
    # Normalize by a typical range (~10)
    norm_diffs = diffs / 10.0
    # Weighted match: 1 - weighted average difference
    weighted_diff = np.average(norm_diffs, axis=1, weights=importances)
    match_scores = (1.0 - weighted_diff) * 100.0  # percentage-like
    return match_scores


def recommendation_label(score: float) -> str:
    if score >= 90:
        return "Highly Recommended"
    elif score >= 80:
        return "Recommended"
    elif score >= 70:
        return "Moderately Recommended"
    else:
        return "Less Suitable"


def generate_explanation(best_row: pd.Series, user_inputs: dict) -> str:
    """Generate a short natural-language explanation based on actual values."""
    strengths = []
    # Look at properties the user requested as High / Very High
    positive_props = [
        "Electrical_Conductivity",
        "Optical_Transparency",
        "Thermal_Stability",
        "Chemical_Stability",
        "Mercury_Corrosion_Resistance",
        "Adhesion_Strength",
        "Durability",
    ]
    for prop in positive_props:
        user_level = user_inputs[prop]
        mat_val = best_row[prop]
        if user_level in ("High", "Very High") and mat_val >= 7.5:
            nice_name = prop.replace("_", " ")
            strengths.append(nice_name.lower())
    # Cost
    if user_inputs["Cost"] in ("Very Low", "Low") and best_row["Cost"] <= 5.0:
        strengths.append("relatively low cost")

    if not strengths:
        strengths = ["a balanced set of properties"]

    strength_str = ", ".join(strengths[:-1]) + (" and " + strengths[-1] if len(strengths) > 1 else strengths[0])
    return (
        f"{best_row['Material']} is recommended because it provides a strong combination of "
        f"{strength_str} that aligns well with the specified requirements."
    )


def main():
    st.set_page_config(
        page_title="TCO Material Recommendation System",
        page_icon="💡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for a cleaner look
    st.markdown(
        """
        <style>
        .main-header {font-size: 2.2rem; font-weight: 700; color: #1f4e79;}
        .sub-header {font-size: 1.3rem; color: #2e75b6; margin-top: 1rem;}
        .metric-card {background-color: #f0f7ff; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #2e75b6;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load resources
    df = load_data()
    model, scaler = load_model_and_scaler()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "🏠 Home",
            "📋 User Requirements",
            "🏆 Recommendation",
            "📊 Material Comparison",
            "🔍 Model Insights",
            "📈 Visualizations",
            "ℹ️ About & Limitations",
        ],
    )

    # ------------------------------------------------------------------
    # HOME
    # ------------------------------------------------------------------
    if page == "🏠 Home":
        st.markdown('<p class="main-header">TCO Material Recommendation System for Fluorescent Lamp Coatings</p>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
            """
            ### What is a Transparent Conducting Oxide (TCO)?
            Transparent Conducting Oxides are a class of materials that combine high optical transparency
            in the visible spectrum with good electrical conductivity. Common examples include
            Indium Tin Oxide (ITO), Fluorine-doped Tin Oxide (FTO), Aluminium-doped Zinc Oxide (AZO)
            and related compounds.

            ### Why TCO coatings on fluorescent lamps?
            Fluorescent lamps contain mercury vapour. Coatings on the glass envelope or electrode regions
            can:
            - Improve electrical contact or charge dissipation
            - Reduce mercury consumption / corrosion of glass or metal parts
            - Enhance optical transmission of useful light
            - Increase thermal and chemical durability of the lamp surfaces

            Selecting the most suitable TCO for a given set of operating conditions is therefore important
            for lamp lifetime, efficiency and environmental performance.

            ### Purpose of this ML system
            This application uses a **Random Forest Regressor** trained on a demonstration dataset of
            TCO property profiles. The user specifies desired levels for eight key properties; the system
            then ranks the available materials according to how well they match those requirements and
            provides a suitability-based recommendation together with model insights.
            """
        )
        st.info(
            "**Important:** The numerical property values and suitability scores in the current dataset "
            "are **synthetic / demonstration data** created only to illustrate the machine-learning pipeline. "
            "They are **not** experimentally measured values. The CSV can be replaced later with "
            "literature-derived experimental data without changing the rest of the code."
        )

    # ------------------------------------------------------------------
    # USER REQUIREMENTS
    # ------------------------------------------------------------------
    elif page == "📋 User Requirements":
        st.markdown('<p class="main-header">Specify Desired TCO Properties</p>', unsafe_allow_html=True)
        st.markdown("Select the performance level you need for each property, then click the button.")

        col1, col2 = st.columns(2)
        with col1:
            elec = st.selectbox("Electrical Conductivity", ["Low", "Medium", "High", "Very High"], index=3)
            opt = st.selectbox("Optical Transparency", ["Low", "Medium", "High", "Very High"], index=3)
            therm = st.selectbox("Thermal Stability", ["Low", "Medium", "High", "Very High"], index=2)
            chem = st.selectbox("Chemical Stability", ["Low", "Medium", "High", "Very High"], index=2)
        with col2:
            merc = st.selectbox("Mercury Corrosion Resistance", ["Low", "Medium", "High", "Very High"], index=3)
            adh = st.selectbox("Adhesion Strength", ["Low", "Medium", "High", "Very High"], index=2)
            dur = st.selectbox("Durability", ["Low", "Medium", "High", "Very High"], index=2)
            cost = st.selectbox("Cost (desired level)", ["Very Low", "Low", "Medium", "High", "Very High"], index=1)

        user_inputs = {
            "Electrical_Conductivity": elec,
            "Optical_Transparency": opt,
            "Thermal_Stability": therm,
            "Chemical_Stability": chem,
            "Mercury_Corrosion_Resistance": merc,
            "Adhesion_Strength": adh,
            "Durability": dur,
            "Cost": cost,
        }

        if st.button("🔍 Find Best TCO Material", type="primary"):
            st.session_state["user_inputs"] = user_inputs
            st.session_state["run_recommendation"] = True
            st.success("Requirements recorded. Go to the **Recommendation** page to see results.")
        else:
            if "user_inputs" not in st.session_state:
                st.session_state["user_inputs"] = user_inputs

    # ------------------------------------------------------------------
    # RECOMMENDATION
    # ------------------------------------------------------------------
    elif page == "🏆 Recommendation":
        st.markdown('<p class="main-header">Material Recommendation</p>', unsafe_allow_html=True)

        if "user_inputs" not in st.session_state:
            st.warning("Please first set your requirements on the **User Requirements** page.")
            return

        user_inputs = st.session_state["user_inputs"]
        user_vec = map_user_requirements(user_inputs)
        importances = model.feature_importances_

        # Match scores (percentage-like) that incorporate RF feature importance
        match_scores = compute_match_scores(df, user_vec, importances)

        # Also get the model’s predicted suitability for each material (using scaled features)
        X_scaled = scaler.transform(df[FEATURE_COLS])
        X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS)
        pred_suit = model.predict(X_scaled_df)

        # Build ranking dataframe
        rank_df = df[["Material"]].copy()
        rank_df["Predicted_Suitability"] = pred_suit
        rank_df["Match_Score_%"] = match_scores
        rank_df["Recommendation"] = rank_df["Match_Score_%"].apply(recommendation_label)
        rank_df = rank_df.sort_values("Match_Score_%", ascending=False).reset_index(drop=True)
        rank_df.insert(0, "Rank", range(1, len(rank_df) + 1))

        best = rank_df.iloc[0]
        best_material_row = df[df["Material"] == best["Material"]].iloc[0]

        # Highlight best
        st.markdown("### 🥇 Best Recommended Material")
        c1, c2, c3 = st.columns(3)
        c1.metric("Material", best["Material"])
        c2.metric("Match Score", f"{best['Match_Score_%']:.1f}%")
        c3.metric("Model Predicted Suitability", f"{best['Predicted_Suitability']:.1f}")

        st.success(generate_explanation(best_material_row, user_inputs))

        st.markdown("### Full Ranking")
        st.dataframe(
            rank_df.style.format({
                "Predicted_Suitability": "{:.1f}",
                "Match_Score_%": "{:.1f}",
            }),
            use_container_width=True,
        )

        # Simple bar chart of match scores
        fig = px.bar(
            rank_df,
            x="Material",
            y="Match_Score_%",
            color="Match_Score_%",
            color_continuous_scale="Blues",
            title="Material Match Scores for Your Requirements",
            labels={"Match_Score_%": "Match Score (%)"},
        )
        fig.update_layout(xaxis_title="TCO Material", yaxis_title="Match Score (%)")
        st.plotly_chart(fig, use_container_width=True)

        st.session_state["rank_df"] = rank_df  # for other pages if needed

    # ------------------------------------------------------------------
    # MATERIAL COMPARISON
    # ------------------------------------------------------------------
    elif page == "📊 Material Comparison":
        st.markdown('<p class="main-header">Material Property Comparison</p>', unsafe_allow_html=True)
        st.markdown(
            "All values are **relative demonstration scores on a 1–10 scale** "
            "(higher is better for every property except Cost, where higher means more expensive)."
        )
        st.dataframe(df.set_index("Material"), use_container_width=True)

        # Radar / parallel coordinates style comparison
        st.markdown("### Property Profiles (Interactive)")
        fig = px.line_polar(
            df.melt(id_vars=["Material"], value_vars=FEATURE_COLS),
            r="value",
            theta="variable",
            color="Material",
            line_close=True,
            title="Radar-style Property Comparison",
        )
        # Note: plotly polar line may need adjustment; alternative bar comparison
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap alternative that is always reliable
        heat_df = df.set_index("Material")[FEATURE_COLS]
        fig2 = px.imshow(
            heat_df,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale="YlGnBu",
            title="Property Heatmap (Demonstration Values)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ------------------------------------------------------------------
    # MODEL INSIGHTS
    # ------------------------------------------------------------------
    elif page == "🔍 Model Insights":
        st.markdown('<p class="main-header">Model Insights – Random Forest</p>', unsafe_allow_html=True)

        # Metrics on the full demo set (tiny data)
        X_scaled = scaler.transform(df[FEATURE_COLS])
        y_true = df["Suitability_Score"].values
        X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS)
        y_pred = model.predict(X_scaled_df)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        st.markdown("### Evaluation Metrics (Demonstration Dataset)")
        m1, m2, m3 = st.columns(3)
        m1.metric("R² Score", f"{r2:.3f}")
        m2.metric("MAE", f"{mae:.2f}")
        m3.metric("RMSE", f"{rmse:.2f}")
        st.caption(
            "Note: With only 8 samples the absolute metric values are limited in meaning; "
            "they serve mainly to verify that the pipeline runs correctly."
        )

        st.markdown("### Feature Importance")
        imp_df = pd.DataFrame({
            "Property": FEATURE_COLS,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)

        fig = px.bar(
            imp_df,
            x="Importance",
            y="Property",
            orientation="h",
            title="Random Forest Feature Importances",
            color="Importance",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            **Interpretation:** Properties with higher importance have a stronger influence
            on the predicted suitability score according to the trained Random Forest.
            These importances are also used as weights when ranking materials against
            the user’s stated requirements.
            """
        )

        # Actual vs Predicted
        st.markdown("### Actual vs Predicted Suitability")
        avp = pd.DataFrame({"Actual": y_true, "Predicted": y_pred, "Material": df["Material"]})
        fig2 = px.scatter(
            avp,
            x="Actual",
            y="Predicted",
            text="Material",
            title="Actual vs Predicted Suitability Scores",
        )
        fig2.add_shape(
            type="line",
            x0=avp["Actual"].min() - 2,
            y0=avp["Actual"].min() - 2,
            x1=avp["Actual"].max() + 2,
            y1=avp["Actual"].max() + 2,
            line=dict(dash="dash", color="gray"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ------------------------------------------------------------------
    # VISUALIZATIONS
    # ------------------------------------------------------------------
    elif page == "📈 Visualizations":
        st.markdown('<p class="main-header">Additional Visualizations</p>', unsafe_allow_html=True)

        # Suitability comparison
        st.markdown("### Demonstration Suitability Scores by Material")
        fig = px.bar(
            df.sort_values("Suitability_Score", ascending=False),
            x="Material",
            y="Suitability_Score",
            color="Suitability_Score",
            color_continuous_scale="Teal",
            title="Synthetic Suitability Scores (for pipeline demonstration only)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Grouped property bars for selected materials
        st.markdown("### Side-by-side Property Comparison")
        selected = st.multiselect(
            "Select materials to compare",
            options=df["Material"].tolist(),
            default=df["Material"].tolist()[:4],
        )
        if selected:
            sub = df[df["Material"].isin(selected)].melt(
                id_vars="Material", value_vars=FEATURE_COLS, var_name="Property", value_name="Score"
            )
            fig2 = px.bar(
                sub,
                x="Property",
                y="Score",
                color="Material",
                barmode="group",
                title="Property Scores for Selected Materials",
            )
            fig2.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig2, use_container_width=True)

    # ------------------------------------------------------------------
    # ABOUT & LIMITATIONS
    # ------------------------------------------------------------------
    elif page == "ℹ️ About & Limitations":
        st.markdown('<p class="main-header">About this Project & Limitations</p>', unsafe_allow_html=True)
        st.markdown(
            """
            ### Dataset Status
            The file `data/tco_materials.csv` contains **synthetic demonstration values**.
            Relative ordering of materials roughly follows trends reported in the open literature
            (e.g., ITO tends to show high conductivity, FTO and SnO₂-based materials tend to show
            superior thermal/chemical stability and lower cost, ZnO-based materials are low-cost, etc.).
            **None of the numbers should be treated as experimentally measured data.**

            ### How to replace with real data
            1. Collect literature or experimental values for the eight properties (use consistent units or
               convert everything to a common relative scale).
            2. Overwrite or extend `data/tco_materials.csv` while keeping the same column names.
            3. Re-run `python model/train_model.py` to retrain the Random Forest and regenerate the
               scaler and plots.
            4. Restart the Streamlit app.

            ### Model choice
            Only **RandomForestRegressor** (scikit-learn) is used, as required. No other algorithms
            (XGBoost, SVM, neural nets, etc.) are employed.

            ### Limitations
            - Extremely small sample size (8 materials) → metrics and importances are illustrative only.
            - No experimental validation of the recommendations has been performed.
            - Property scales are arbitrary relative scores; real engineering decisions require
              absolute measured values, process compatibility, deposition method, thickness, etc.
            - Mercury-corrosion behaviour is highly application-specific and depends on lamp chemistry,
              temperature and coating microstructure – literature values must be used carefully.

            ### Future improvements
            - Expand the dataset with measured or carefully curated literature data.
            - Add uncertainty estimates (e.g., quantile regression forests).
            - Include deposition-process parameters and thickness as additional features.
            - Perform experimental validation on selected candidate coatings inside real fluorescent lamps.
            """
        )


if __name__ == "__main__":
    main()
