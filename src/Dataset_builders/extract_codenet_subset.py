import os
import pandas as pd
from tqdm import tqdm

# Paths inside the container (mounted from host)
METADATA_DIR = "/app/data/Project_CodeNet/metadata"
DATA_DIR = "/app/data/Project_CodeNet/data"

# Languages to filter
LANGUAGES = {"Python", "Python3", "C", "C++", "c", "c++"}
MAX_SAMPLES = 500_000

def extract_multilang_subset():
    records = []
    csv_files = sorted([f for f in os.listdir(METADATA_DIR) if f.endswith(".csv")])

    for csv_file in tqdm(csv_files, desc="Scanning metadata files"):
        df = pd.read_csv(os.path.join(METADATA_DIR, csv_file))

        # Keep only the desired languages
        df = df[df["language"].isin(LANGUAGES)]

        for _, row in df.iterrows():
            problem_id = str(row["problem_id"])
            submission_id = str(row["submission_id"])
            ext = row["filename_ext"]
            language = row["language"]

            # Check if the file exists in the data folder
            code_path = os.path.join(DATA_DIR, problem_id, language, f"{submission_id}.{ext}")
            if os.path.exists(code_path):
                records.append({
                    "submission_id": submission_id,
                    "problem_id": problem_id,
                    "user_id": row.get("user_id", -1),
                    "status": row.get("status", "Unknown"),
                    "date": row.get("date", None),
                    "language": language,
                    "code_path": code_path
                })

            if len(records) >= MAX_SAMPLES:
                break
        if len(records) >= MAX_SAMPLES:
            break

    # Save subset
    df_out = pd.DataFrame(records)
    out_path = "/app/codenet_subset.csv"
    df_out.to_csv(out_path, index=False)
    print(f"✅ Saved {len(df_out)} rows to {out_path}")

if __name__ == "__main__":
    extract_multilang_subset()
