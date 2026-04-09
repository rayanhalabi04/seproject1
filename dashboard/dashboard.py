import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client


# =========================
# APP SETUP
# =========================
st.set_page_config(
    page_title="Salary Storyboard",
    page_icon="💼",
    layout="wide",
)

load_dotenv()

API_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000/predict")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")

EXPERIENCE_BANDS = {
    "EN": {"low": 70000, "median": 95000, "high": 120000, "label": "Entry Level"},
    "MI": {"low": 110000, "median": 135000, "high": 160000, "label": "Mid Level"},
    "SE": {"low": 150000, "median": 180000, "high": 220000, "label": "Senior Level"},
    "EX": {"low": 190000, "median": 235000, "high": 300000, "label": "Executive Level"},
}

EXPERIENCE_ORDER = ["EN", "MI", "SE", "EX"]
EMPLOYMENT_LABELS = {
    "FT": "Full-time",
    "PT": "Part-time",
    "CT": "Contract",
    "FL": "Freelance",
}
COMPANY_SIZE_LABELS = {"S": "Small", "M": "Medium", "L": "Large"}
REMOTE_LABELS = {0: "On-site", 50: "Hybrid", 100: "Remote"}


# =========================
# STYLING
# =========================
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .main {
            background: radial-gradient(circle at top right, #edf3fb 0%, #f8fafc 45%, #f9fbfd 100%);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.3rem;
            padding-bottom: 2rem;
        }

        .hero {
            background: linear-gradient(130deg, #0f172a 0%, #19406a 55%, #29597f 100%);
            border-radius: 22px;
            padding: 1.45rem 1.65rem;
            color: #f8fafc;
            margin-bottom: 1rem;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.22);
            border: 1px solid rgba(226, 232, 240, 0.12);
        }

        .hero h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.01em;
            font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
        }

        .hero p {
            margin: 0.55rem 0 0 0;
            color: #e2e8f0;
            font-size: 1rem;
            line-height: 1.55;
            max-width: 880px;
        }

        .section-card {
            background: #ffffff;
            border: 1px solid #e4edf7;
            border-radius: 18px;
            padding: 1.05rem 1.15rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.07);
            margin-bottom: 1rem;
        }

        .section-title {
            margin: 0 0 0.33rem 0;
            font-size: 1.12rem;
            color: #0f172a;
            font-weight: 700;
            font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
        }

        .section-subtitle {
            margin: 0;
            color: #475569;
            font-size: 0.92rem;
            line-height: 1.5;
        }

        .form-card {
            background: #ffffff;
            border: 1px solid #e4edf7;
            border-radius: 18px;
            padding: 1rem 1.1rem 0.85rem 1.1rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.07);
            margin-bottom: 1rem;
        }

        .spotlight-card {
            background: linear-gradient(130deg, #0b3a62 0%, #1d5c8f 100%);
            border-radius: 18px;
            border: 1px solid #2e6796;
            color: white;
            padding: 1.05rem 1.12rem;
            box-shadow: 0 12px 28px rgba(14, 57, 90, 0.24);
        }

        .spotlight-label {
            color: #dbeafe;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .spotlight-value {
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.12;
            letter-spacing: -0.01em;
            font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
        }

        .spotlight-sub {
            margin-top: 0.4rem;
            font-size: 0.9rem;
            color: #dbeafe;
            line-height: 1.45;
        }

        .kpi-card {
            background: linear-gradient(180deg, #ffffff 0%, #f9fbfe 100%);
            border: 1px solid #e4ecf7;
            border-radius: 16px;
            padding: 0.92rem 1rem;
            min-height: 118px;
            box-shadow: 0 7px 18px rgba(15, 23, 42, 0.06);
        }

        .kpi-label {
            color: #64748b;
            font-size: 0.79rem;
            margin-bottom: 0.34rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 700;
        }

        .kpi-value {
            color: #0f172a;
            font-size: 1.33rem;
            font-weight: 800;
            line-height: 1.2;
            letter-spacing: -0.01em;
            font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
        }

        .kpi-sub {
            color: #475569;
            margin-top: 0.28rem;
            font-size: 0.85rem;
            line-height: 1.42;
        }

        .insight-box {
            background: linear-gradient(160deg, #eef5ff 0%, #f8fbff 100%);
            border: 1px solid #d4e4fb;
            border-radius: 14px;
            padding: 0.83rem 0.95rem;
            color: #0f172a;
            line-height: 1.62;
            font-size: 0.95rem;
        }

        .takeaway-box {
            background: #f8fbff;
            border: 1px solid #d8e7fb;
            border-left: 5px solid #1f5f96;
            border-radius: 12px;
            padding: 0.7rem 0.9rem;
            color: #0f172a;
            font-size: 0.92rem;
            line-height: 1.5;
            margin-top: 0.56rem;
        }

        .stButton button {
            border-radius: 10px;
            border: 0;
            background: linear-gradient(130deg, #0b3a62 0%, #1d5b8f 100%);
            color: white;
            font-weight: 700;
            padding: 0.58rem 1rem;
        }

        .history-title {
            margin: 0;
            font-size: 1rem;
            color: #0f172a;
            font-weight: 700;
        }

        .muted {
            color: #64748b;
            font-size: 0.86rem;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# HELPERS
# =========================
def call_prediction_api(payload: dict) -> dict:
    response = requests.post(API_URL, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def get_experience_band(experience_level: str) -> dict:
    return EXPERIENCE_BANDS.get(experience_level, EXPERIENCE_BANDS["EN"])


def classify_job_family(job_title: str) -> str:
    title = str(job_title).lower().strip()

    if any(token in title for token in ["manager", "director", "head", "lead"]):
        return "leadership"
    if any(token in title for token in ["machine learning", "ml", "ai", "nlp", "vision"]):
        return "ML/AI"
    if any(token in title for token in ["engineer", "etl", "architect"]):
        return "data engineering"
    if any(token in title for token in ["analyst", "analytics"]):
        return "analytics"
    if "scientist" in title:
        return "data science"
    return "data"


def assess_band_position(predicted_salary: float, band: dict) -> dict:
    low, high = float(band["low"]), float(band["high"])

    if predicted_salary < low:
        return {
            "status": "below",
            "label": "Below expected range",
            "gap": float(low - predicted_salary),
            "within_ratio": None,
        }

    if predicted_salary > high:
        return {
            "status": "above",
            "label": "Above expected range",
            "gap": float(predicted_salary - high),
            "within_ratio": None,
        }

    ratio = 0.5 if high == low else float((predicted_salary - low) / (high - low))
    return {
        "status": "within",
        "label": "Within expected range",
        "gap": 0.0,
        "within_ratio": ratio,
    }


def clean_llm_output(text: str) -> str:
    if not text:
        return ""

    disallowed_starts = (
        "thinking",
        "analysis",
        "reasoning",
        "1.",
        "2.",
        "3.",
        "-",
        "*",
        "<think",
    )

    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith(disallowed_starts):
            continue
        cleaned_lines.append(line)

    cleaned = " ".join(cleaned_lines).strip().replace("\n", " ")
    cleaned = " ".join(cleaned.split())

    if not cleaned:
        return ""

    parts = [p.strip() for p in cleaned.split(".") if p.strip()]
    if len(parts) > 2:
        cleaned = ". ".join(parts[:2]) + "."
    elif not cleaned.endswith((".", "!", "?")):
        cleaned += "."

    return cleaned


def load_history() -> pd.DataFrame:
    if st.session_state.get("supabase_client") is None:
        return pd.DataFrame()

    try:
        response = (
            st.session_state["supabase_client"]
            .table("salary_predictions")
            .select("*")
            .order("created_at", desc=True)
            .limit(120)
            .execute()
        )
        rows = response.data if response.data else []
        df = pd.DataFrame(rows)
        if df.empty:
            return df

        if "predicted_salary_usd" in df.columns:
            df["predicted_salary_usd"] = pd.to_numeric(df["predicted_salary_usd"], errors="coerce")
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

        return df.dropna(subset=["predicted_salary_usd"], how="any")
    except Exception:
        return pd.DataFrame()


def compute_history_stats(predicted_salary: float, history_df: pd.DataFrame) -> dict:
    if history_df.empty or "predicted_salary_usd" not in history_df.columns:
        return {}

    salaries = history_df["predicted_salary_usd"].dropna().astype(float)
    if salaries.empty:
        return {}

    percentile = float((salaries <= predicted_salary).mean() * 100)

    return {
        "count": int(len(salaries)),
        "median": float(salaries.median()),
        "mean": float(salaries.mean()),
        "min": float(salaries.min()),
        "max": float(salaries.max()),
        "percentile": percentile,
        "delta_vs_median": float(predicted_salary - salaries.median()),
    }


def generate_fallback_insight(payload: dict, predicted_salary: float, history_stats: dict) -> str:
    band = get_experience_band(payload["experience_level"])
    band_position = assess_band_position(predicted_salary, band)
    family = classify_job_family(payload["job_title"])

    sentence_one = (
        f"The estimate for this {band['label'].lower()} role is ${predicted_salary:,.0f} and is {band_position['label'].lower()}."
    )

    if history_stats and history_stats.get("count", 0) >= 8:
        delta = history_stats["delta_vs_median"]
        direction = "above" if delta >= 0 else "below"
        sentence_two = (
            f"Against recent saved scenarios, it is ${abs(delta):,.0f} {direction} the median, consistent with a {family} profile in {payload['company_location']}."
        )
    else:
        sentence_two = (
            f"The combination of {EMPLOYMENT_LABELS.get(payload['employment_type'], payload['employment_type']).lower()}, "
            f"{COMPANY_SIZE_LABELS.get(payload['company_size'], payload['company_size']).lower()} company scale, and "
            f"{REMOTE_LABELS.get(payload['remote_ratio'], 'flexible').lower()} setup shapes this outcome."
        )

    return f"{sentence_one} {sentence_two}"


def generate_llm_insight(payload: dict, predicted_salary: float, history_stats: dict) -> str:
    band = get_experience_band(payload["experience_level"])
    band_position = assess_band_position(predicted_salary, band)

    if history_stats and history_stats.get("count", 0) > 0:
        history_context = (
            f"Recent median: ${history_stats['median']:,.0f}; "
            f"Percentile among recent predictions: {history_stats['percentile']:.0f}."
        )
    else:
        history_context = "Recent history not available."

    prompt = (
        "You are a senior compensation analyst writing for a product dashboard.\n"
        "Write 1 to 2 concise sentences, no bullet points.\n"
        "No reasoning text, no process notes, no markdown, no AI mention.\n"
        "Tone: professional, clear, executive-friendly.\n"
        "Maximum 50 words total.\n\n"
        f"Predicted salary: ${predicted_salary:,.0f}\n"
        f"Experience level: {band['label']} ({payload['experience_level']})\n"
        f"Expected range: ${band['low']:,.0f} to ${band['high']:,.0f}\n"
        f"Range position: {band_position['label']}\n"
        f"Employment type: {payload['employment_type']}\n"
        f"Company size: {payload['company_size']}\n"
        f"Remote ratio: {payload['remote_ratio']}\n"
        f"Location: {payload['company_location']}\n"
        f"Job title: {payload['job_title']}\n"
        f"{history_context}"
    )

    body = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 90,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=body, timeout=80)
        response.raise_for_status()
        response_json = response.json()
        raw_text = response_json.get("response", "")

        cleaned = clean_llm_output(raw_text)
        if cleaned:
            return cleaned
        return generate_fallback_insight(payload, predicted_salary, history_stats)
    except Exception:
        return generate_fallback_insight(payload, predicted_salary, history_stats)


def save_to_supabase(payload: dict, predicted_salary: float, insight: str) -> None:
    if st.session_state.get("supabase_client") is None:
        return

    record = {
        "experience_level": payload["experience_level"],
        "employment_type": payload["employment_type"],
        "company_size": payload["company_size"],
        "remote_ratio": payload["remote_ratio"],
        "company_location": payload["company_location"],
        "job_title": payload["job_title"],
        "predicted_salary_usd": predicted_salary,
        "llm_insight": insight,
        "created_at": datetime.utcnow().isoformat(),
    }

    try:
        st.session_state["supabase_client"].table("salary_predictions").insert(record).execute()
    except Exception:
        pass


def get_takeaway_text(
    predicted_salary: float,
    band: dict,
    band_position: dict,
    history_stats: dict,
) -> str:
    if history_stats and history_stats.get("count", 0) >= 8:
        direction = "above" if history_stats["delta_vs_median"] >= 0 else "below"
        return (
            f"Takeaway: The prediction is {band_position['label'].lower()} and sits ${abs(history_stats['delta_vs_median']):,.0f} "
            f"{direction} the recent median across saved scenarios."
        )

    if band_position["status"] == "within":
        return (
            f"Takeaway: The estimate is within the expected {band['label'].lower()} band, "
            "which indicates a market-consistent outcome for the selected profile."
        )

    if band_position["status"] == "above":
        return (
            f"Takeaway: The estimate is above the expected {band['label'].lower()} band by ${band_position['gap']:,.0f}, "
            "suggesting an above-market scenario for the current configuration."
        )

    return (
        f"Takeaway: The estimate is below the expected {band['label'].lower()} band by ${band_position['gap']:,.0f}, "
        "indicating a more conservative compensation scenario."
    )


# =========================
# VISUALS
# =========================
def plot_salary_benchmark(predicted_salary: float, experience_level: str) -> None:
    band = get_experience_band(experience_level)
    low, median, high = band["low"], band["median"], band["high"]
    position = assess_band_position(predicted_salary, band)

    spread = max(high - low, 1)
    outlier_gap = max(abs(predicted_salary - high), abs(low - predicted_salary), 0)
    margin = max(12000, 0.25 * spread, 0.25 * outlier_gap)

    x_min = min(low, predicted_salary) - margin
    x_max = max(high, predicted_salary) + margin

    fig, ax = plt.subplots(figsize=(10, 2.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot([low, high], [0, 0], color="#d7e5f7", linewidth=18, solid_capstyle="round")
    ax.plot([low, median], [0, 0], color="#a9c9ec", linewidth=18, solid_capstyle="round")
    ax.axvline(median, color="#64748b", linestyle="--", linewidth=2)

    marker_color = {
        "below": "#6b7280",
        "within": "#0f4c81",
        "above": "#0f4c81",
    }[position["status"]]

    ax.scatter(predicted_salary, 0, s=240, color=marker_color, zorder=5)

    ax.text(low, -0.24, f"Low\n${low:,.0f}", ha="center", va="top", fontsize=9, color="#475569")
    ax.text(median, 0.24, f"Median\n${median:,.0f}", ha="center", va="bottom", fontsize=9, color="#475569")
    ax.text(high, -0.24, f"High\n${high:,.0f}", ha="center", va="top", fontsize=9, color="#475569")

    label = f"Prediction\n${predicted_salary:,.0f}"
    ax.text(predicted_salary, 0.26, label, ha="center", va="bottom", fontsize=9, color="#0f4c81", fontweight="bold")

    if position["status"] == "above":
        ax.annotate(
            f"Above range by ${position['gap']:,.0f}",
            xy=(predicted_salary, 0),
            xytext=(22, -18),
            textcoords="offset points",
            fontsize=9,
            color="#0f4c81",
            arrowprops={"arrowstyle": "-", "color": "#0f4c81", "lw": 1.2},
        )
    elif position["status"] == "below":
        ax.annotate(
            f"Below range by ${position['gap']:,.0f}",
            xy=(predicted_salary, 0),
            xytext=(-125, -18),
            textcoords="offset points",
            fontsize=9,
            color="#6b7280",
            arrowprops={"arrowstyle": "-", "color": "#6b7280", "lw": 1.2},
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.48, 0.48)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(
        "1) Benchmark Position: Predicted Salary vs Expected Band",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    st.pyplot(fig, use_container_width=True)
    st.caption("This chart shows whether the estimate is below, within, or above the expected range for the selected experience level.")


def plot_career_context(predicted_salary: float, current_experience: str) -> None:
    levels = EXPERIENCE_ORDER
    low_values = [EXPERIENCE_BANDS[level]["low"] for level in levels]
    med_values = [EXPERIENCE_BANDS[level]["median"] for level in levels]
    high_values = [EXPERIENCE_BANDS[level]["high"] for level in levels]

    x = np.arange(len(levels))
    current_idx = levels.index(current_experience) if current_experience in levels else 0

    y_min = min(low_values + [predicted_salary]) - 18000
    y_max = max(high_values + [predicted_salary]) + 18000

    fig, ax = plt.subplots(figsize=(10, 4.35))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.fill_between(x, low_values, high_values, color="#dbe7f5", alpha=0.92, label="Expected range by level")
    ax.plot(x, med_values, color="#3c6e99", linewidth=2.5, marker="o", markersize=6, label="Median by level")

    ax.axvspan(current_idx - 0.35, current_idx + 0.35, color="#eef5ff", alpha=1.0)
    ax.scatter(current_idx, predicted_salary, s=180, color="#0f4c81", zorder=6)
    ax.annotate(
        f"Prediction: ${predicted_salary:,.0f}",
        (current_idx, predicted_salary),
        xytext=(10, 13),
        textcoords="offset points",
        fontsize=9,
        color="#0f4c81",
        fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([EXPERIENCE_BANDS[level]["label"] for level in levels], fontsize=9)
    ax.set_ylabel("Salary (USD)", fontsize=10, color="#334155")
    ax.set_ylim(y_min, y_max)
    ax.set_title(
        "2) Career Context: Where This Estimate Sits Across Levels",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.22)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(frameon=False, loc="upper left")

    st.pyplot(fig, use_container_width=True)
    st.caption("This view places the estimate in the broader compensation progression from entry to executive levels.")


def plot_context_comparison(predicted_salary: float, selected_level: str, history_df: pd.DataFrame) -> None:
    has_history = not history_df.empty and "predicted_salary_usd" in history_df.columns
    salaries = (
        history_df["predicted_salary_usd"].dropna().astype(float)
        if has_history
        else pd.Series(dtype=float)
    )

    fig, ax = plt.subplots(figsize=(10, 4.3))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Case 1: Enough grouped history for boxplot by experience level
    grouped_ready = False
    if has_history and "experience_level" in history_df.columns and len(salaries) >= 12:
        groups = []
        labels = []
        positions = []
        selected_position = None

        for idx, level in enumerate(EXPERIENCE_ORDER, start=1):
            values = (
                history_df.loc[history_df["experience_level"] == level, "predicted_salary_usd"]
                .dropna()
                .astype(float)
            )
            if len(values) >= 3:
                groups.append(values.values)
                labels.append(f"{level}\n(n={len(values)})")
                positions.append(idx)
                if level == selected_level:
                    selected_position = idx

        if len(groups) >= 2:
            grouped_ready = True
            box = ax.boxplot(
                groups,
                positions=positions,
                widths=0.58,
                patch_artist=True,
                showfliers=False,
            )

            for patch in box["boxes"]:
                patch.set_facecolor("#d9e9f8")
                patch.set_edgecolor("#5a86ad")
                patch.set_linewidth(1.2)

            for item in box["medians"]:
                item.set_color("#0f4c81")
                item.set_linewidth(2)

            for whisker in box["whiskers"]:
                whisker.set_color("#5a86ad")

            for cap in box["caps"]:
                cap.set_color("#5a86ad")

            if selected_position is not None:
                ax.scatter(selected_position, predicted_salary, s=165, color="#0f4c81", zorder=6)
                ax.annotate(
                    f"Current\n${predicted_salary:,.0f}",
                    (selected_position, predicted_salary),
                    xytext=(9, 10),
                    textcoords="offset points",
                    fontsize=8.8,
                    color="#0f4c81",
                    fontweight="bold",
                )

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel("Predicted Salary (USD)", fontsize=10, color="#334155")
            ax.set_title(
                "3) Real Context: Recent Saved Predictions by Experience Level",
                loc="left",
                fontsize=13,
                fontweight="bold",
                color="#0f172a",
            )
            ax.grid(axis="y", linestyle="--", alpha=0.2)

    # Case 2: History exists but grouped split is too sparse
    if not grouped_ready and len(salaries) >= 8:
        bins = min(12, max(5, int(np.sqrt(len(salaries)))))
        median_val = float(salaries.median())
        q1, q3 = float(salaries.quantile(0.25)), float(salaries.quantile(0.75))

        ax.hist(salaries, bins=bins, color="#bfd7ed", edgecolor="#5c84ad", alpha=0.95)
        ax.axvspan(q1, q3, color="#eaf3fd", alpha=0.9, label="Middle 50%")
        ax.axvline(median_val, color="#64748b", linestyle="--", linewidth=2, label="Recent median")
        ax.axvline(predicted_salary, color="#0f4c81", linewidth=2.6, label="Current prediction")

        percentile = float((salaries <= predicted_salary).mean() * 100)
        ax.text(
            0.02,
            0.95,
            f"Percentile: {percentile:.0f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#1e3a5f",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#eef5ff", "edgecolor": "#c7dcf5"},
        )

        ax.set_title(
            "3) Real Context: Current Estimate vs Recent Prediction Distribution",
            loc="left",
            fontsize=13,
            fontweight="bold",
            color="#0f172a",
        )
        ax.set_xlabel("Predicted Salary (USD)", fontsize=10, color="#334155")
        ax.set_ylabel("Count", fontsize=10, color="#334155")
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.2)

    # Case 3: History is limited -> defensible checkpoint deltas
    if len(salaries) < 8:
        band = get_experience_band(selected_level)
        checkpoints = {
            "Low bound": band["low"],
            "Band median": band["median"],
            "High bound": band["high"],
        }

        labels = list(checkpoints.keys())
        deltas = [predicted_salary - checkpoints[label] for label in labels]
        y_pos = np.arange(len(labels))

        colors = ["#0f4c81" if val >= 0 else "#94a3b8" for val in deltas]
        ax.barh(y_pos, deltas, color=colors, height=0.56)
        ax.axvline(0, color="#475569", linewidth=1.5)

        for idx, val in enumerate(deltas):
            sign = "+" if val >= 0 else "-"
            text_x = val + (2500 if val >= 0 else -9500)
            ax.text(text_x, idx, f"{sign}${abs(val):,.0f}", va="center", fontsize=9, color="#334155")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Difference from benchmark", fontsize=10, color="#334155")
        ax.set_title(
            "3) Context Check: Prediction vs Key Benchmarks",
            loc="left",
            fontsize=13,
            fontweight="bold",
            color="#0f172a",
        )
        ax.grid(axis="x", linestyle="--", alpha=0.2)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.set_axisbelow(True)
    st.pyplot(fig, use_container_width=True)

    if len(salaries) < 8:
        st.caption("Saved history is currently limited, so this view compares the estimate against transparent benchmark checkpoints.")
    else:
        st.caption("This chart uses recent saved predictions to provide real context around the current estimate.")


# =========================
# APP STRUCTURE
# =========================
inject_css()

if "supabase_client" not in st.session_state:
    st.session_state["supabase_client"] = None

if SUPABASE_URL and SUPABASE_KEY and st.session_state["supabase_client"] is None:
    try:
        st.session_state["supabase_client"] = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        st.session_state["supabase_client"] = None

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None


# =========================
# HEADER
# =========================
st.markdown(
    """
    <div class="hero">
        <h1>Salary Prediction Dashboard</h1>
        <p>
            Build a role scenario, generate a salary estimate, and review a concise analytical story that explains
            where the result sits in market context.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
# SCENARIO BUILDER
# =========================
st.markdown('<div class="form-card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">Scenario Builder</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-subtitle">Define the role attributes below to generate a salary estimate.</p>',
    unsafe_allow_html=True,
)

with st.form("salary_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        experience_level = st.selectbox(
            "Experience Level",
            EXPERIENCE_ORDER,
            help="EN: Entry, MI: Mid, SE: Senior, EX: Executive",
        )
        employment_type = st.selectbox("Employment Type", ["FT", "CT", "FL", "PT"])

    with c2:
        company_size = st.selectbox("Company Size", ["S", "M", "L"])
        remote_ratio = st.selectbox("Remote Ratio", [0, 50, 100])

    with c3:
        company_location = st.text_input("Company Location", value="US")
        job_title = st.text_input("Job Title", value="Data Scientist")

    submitted = st.form_submit_button("Generate Salary Estimate")

st.markdown("</div>", unsafe_allow_html=True)


# =========================
# PREDICTION FLOW
# =========================
history_df = load_history()

if submitted:
    payload = {
        "experience_level": experience_level,
        "employment_type": employment_type,
        "company_size": company_size,
        "remote_ratio": remote_ratio,
        "company_location": company_location.strip().upper(),
        "job_title": job_title.strip(),
    }

    try:
        api_result = call_prediction_api(payload)
        predicted_salary = float(api_result["predicted_salary_usd"])

        history_stats = compute_history_stats(predicted_salary, history_df)

        with st.spinner("Generating insight..."):
            insight = generate_llm_insight(payload, predicted_salary, history_stats)

        save_to_supabase(payload, predicted_salary, insight)

        st.session_state["last_result"] = {
            "payload": payload,
            "predicted_salary": predicted_salary,
            "insight": insight,
        }

        history_df = load_history()

    except requests.exceptions.RequestException:
        st.error("Prediction service is currently unavailable. Please try again shortly.")
    except (KeyError, ValueError):
        st.error("The prediction response could not be processed. Please try again.")
    except Exception:
        st.error("An unexpected issue occurred while generating the prediction.")


# =========================
# RESULTS
# =========================
result = st.session_state.get("last_result")

if result:
    payload = result["payload"]
    predicted_salary = float(result["predicted_salary"])
    insight = result["insight"]

    band = get_experience_band(payload["experience_level"])
    band_position = assess_band_position(predicted_salary, band)
    history_stats = compute_history_stats(predicted_salary, history_df)

    delta_vs_median = predicted_salary - band["median"]

    if band_position["status"] == "within" and band_position["within_ratio"] is not None:
        range_detail = f"{band_position['within_ratio'] * 100:.0f}% through the expected range"
    elif band_position["status"] == "above":
        range_detail = f"${band_position['gap']:,.0f} above the upper bound"
    else:
        range_detail = f"${band_position['gap']:,.0f} below the lower bound"

    if history_stats and history_stats.get("count", 0) >= 8:
        history_direction = "above" if history_stats["delta_vs_median"] >= 0 else "below"
        history_kpi_value = f"{history_stats['percentile']:.0f}"
        history_kpi_sub = (
            f"Percentile in recent history, ${abs(history_stats['delta_vs_median']):,.0f} "
            f"{history_direction} recent median"
        )
    else:
        history_kpi_value = "N/A"
        history_kpi_sub = "More saved scenarios are needed for robust historical percentile context"

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Prediction Summary</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">A concise view of the estimate and what it means.</p>', unsafe_allow_html=True)

    highlight_col, kpi_col = st.columns([1.05, 2])

    with highlight_col:
        st.markdown(
            f"""
            <div class="spotlight-card">
                <div class="spotlight-label">Predicted Salary</div>
                <div class="spotlight-value">${predicted_salary:,.0f}</div>
                <div class="spotlight-sub">{band['label']} • {payload['job_title']} • {payload['company_location']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col:
        k1, k2, k3 = st.columns(3)

        with k1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Range Status</div>
                    <div class="kpi-value">{band_position['label']}</div>
                    <div class="kpi-sub">Expected {band['label'].lower()} band: ${band['low']:,.0f} to ${band['high']:,.0f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k2:
            direction = "above" if delta_vs_median >= 0 else "below"
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Distance to Benchmarks</div>
                    <div class="kpi-value">{range_detail}</div>
                    <div class="kpi-sub">${abs(delta_vs_median):,.0f} {direction} the level median (${band['median']:,.0f})</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">History Context</div>
                    <div class="kpi-value">{history_kpi_value}</div>
                    <div class="kpi-sub">{history_kpi_sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f'<div class="takeaway-box">{get_takeaway_text(predicted_salary, band, band_position, history_stats)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    insight_col, summary_col = st.columns([1.35, 1])

    with insight_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">AI Insight</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-subtitle">Short narrative interpretation of the current result.</p>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with summary_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Scenario Details</p>', unsafe_allow_html=True)
        summary_df = pd.DataFrame(
            {
                "Field": [
                    "Experience",
                    "Employment",
                    "Company Size",
                    "Remote",
                    "Location",
                    "Job Title",
                ],
                "Value": [
                    f"{payload['experience_level']} ({band['label']})",
                    EMPLOYMENT_LABELS.get(payload["employment_type"], payload["employment_type"]),
                    COMPANY_SIZE_LABELS.get(payload["company_size"], payload["company_size"]),
                    REMOTE_LABELS.get(payload["remote_ratio"], f"{payload['remote_ratio']}%"),
                    payload["company_location"],
                    payload["job_title"],
                ],
            }
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Visual Story</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-subtitle">Three focused visuals explain benchmark fit, career context, and real comparison context.</p>',
        unsafe_allow_html=True,
    )

    plot_salary_benchmark(predicted_salary, payload["experience_level"])
    plot_career_context(predicted_salary, payload["experience_level"])
    plot_context_comparison(predicted_salary, payload["experience_level"], history_df)

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Awaiting Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-subtitle">Submit a scenario to generate the salary estimate, interpretation, and visual context.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# HISTORY (SECONDARY)
# =========================
with st.expander("Recent Prediction History", expanded=False):
    if not history_df.empty:
        display_columns = [
            col
            for col in [
                "created_at",
                "job_title",
                "experience_level",
                "employment_type",
                "company_size",
                "remote_ratio",
                "company_location",
                "predicted_salary_usd",
            ]
            if col in history_df.columns
        ]

        history_display = history_df[display_columns].copy()

        if "created_at" in history_display.columns:
            history_display["created_at"] = history_display["created_at"].dt.strftime("%Y-%m-%d %H:%M")

        st.dataframe(history_display.head(40), use_container_width=True, hide_index=True)
    else:
        st.markdown('<p class="muted">No saved history available yet.</p>', unsafe_allow_html=True)
