import os
import pandas as pd
import streamlit as st

from backend.config import CSV_FILE


if CSV_FILE not in os.listdir():
    pd.DataFrame(columns=["Title", "Source", "Content"]).to_csv(CSV_FILE, index=False)


@st.cache_data
def load_data():
    data = pd.read_csv(CSV_FILE)
    # Ensure all required columns exist
    required_cols = ["Title", "Source", "Content"]
    for col in required_cols:
        if col not in data.columns:
            data[col] = ""
    return data


def save_data(new_entry):
    """Save new entry to CSV and clear cache"""
    # Load current data
    current_data = pd.read_csv(CSV_FILE)

    # Add new entry
    new_row = pd.DataFrame([new_entry])
    updated_data = pd.concat([current_data, new_row], ignore_index=True)

    # Save to CSV
    updated_data.to_csv(CSV_FILE, index=False)

    # Clear the cache so data reloads
    load_data.clear()

    return True
