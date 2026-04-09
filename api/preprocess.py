import pandas as pd


def group_job(title: str) -> str:
    title = str(title).lower().strip()

    # 1. Management / Leadership
    if "manager" in title or "director" in title or "head" in title:
        return "Management"

    # 2. Machine Learning / AI
    elif (
        "machine learning" in title
        or title == "ml engineer"
        or "ai" in title
        or "computer vision" in title
        or "nlp" in title
    ):
        return "ML_AI"

    # 3. Data Engineering
    elif (
        "data engineer" in title
        or "big data engineer" in title
        or "etl" in title
        or "architect" in title
        or "analytics engineer" in title
        or "data analytics engineer" in title
        or "cloud data engineer" in title
    ):
        return "Data_Engineering"

    # 4. Data Analysis
    elif "analyst" in title or "analytics lead" in title:
        return "Data_Analysis"

    # 5. Data Science
    elif (
        "scientist" in title
        or "data science consultant" in title
        or "data science engineer" in title
    ):
        return "Data_Science"

    # 6. Other
    else:
        return "Other"


def preprocess_input(
    input_dict: dict,
    model_columns: list,
    top_locations: list,
    experience_map: dict,
    size_map: dict
) -> pd.DataFrame:
    df_input = pd.DataFrame([input_dict])

    # Encode ordinal features
    df_input["experience_level"] = df_input["experience_level"].map(experience_map)
    df_input["company_size"] = df_input["company_size"].map(size_map)

    # Group job title
    df_input["job_group"] = df_input["job_title"].apply(group_job)
    df_input = df_input.drop(columns=["job_title"], errors="ignore")

    # Group company location
    df_input["company_location_grouped"] = df_input["company_location"].apply(
        lambda x: x if x in top_locations else "Other"
    )
    df_input = df_input.drop(columns=["company_location"], errors="ignore")

    # Drop column not used by model
    df_input = df_input.drop(columns=["employee_residence"], errors="ignore")

    # One-hot encode categorical columns
    df_input = pd.get_dummies(
        df_input,
        columns=["employment_type", "company_location_grouped", "job_group"],
        drop_first=False,
        dtype=int
    )

    # Add missing columns
    for col in model_columns:
        if col not in df_input.columns:
            df_input[col] = 0

    # Keep exact same column order as training
    df_input = df_input[model_columns]

    return df_input