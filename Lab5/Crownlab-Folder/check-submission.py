# This script runs some checks on the submission.csv file to ensure that it is valid and can be used for evaluation.
# It checks that the file:
# - exists
# - is a valid CSV file
# - has the correct number of rows
# - has the correct columns (index, value)
# - the columns have the correct data types
# - the predicted column ("value") contain valid values
# - all expected indices are found in the submission file

import pandas as pd
import os

if __name__ == "__main__":
    submission_file = "./submission.csv"
    n_entries = 4000
    dtypes = {
        "index": int,
        "value": str
    }

    assert os.path.exists(submission_file), f"File {submission_file} not found. Please make sure to run the pipeline code in main.py and generate the submission.csv file."

    df = pd.read_csv(submission_file, dtype=dtypes)
    assert len(df) == n_entries, f"Submission file should contain {n_entries} rows, but found {len(df)} rows."

    assert "index" in df.columns, "Submission file should contain 'index' column."
    assert "value" in df.columns, "Submission file should contain 'value' column."
    assert df["index"].is_unique, "Submission file 'index' column should contain unique values."

    # check that all the indices are present in the submission file
    expected_indices = set(range(n_entries))
    actual_indices = set(df["index"].values)
    assert not (expected_indices - actual_indices), f"Submission file 'index' column is missing some indices. Missing indices: {expected_indices - actual_indices}"
    assert not (actual_indices - expected_indices), f"Submission file 'index' column contains unexpected indices. Unexpected indices: {actual_indices - expected_indices}"

    print("Submission file is valid.")