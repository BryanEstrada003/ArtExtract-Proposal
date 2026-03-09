import pandas as pd
from pathlib import Path

# Define paths
csv_dir = Path("datasets/wikiart/wikiart_csv")
images_root = Path("datasets/wikiart")

# Load all CSV files
print("Loading CSV files...")
artist_train = pd.read_csv(csv_dir / "artist_train.csv")
artist_val = pd.read_csv(csv_dir / "artist_val.csv")
genre_train = pd.read_csv(csv_dir / "genre_train.csv")
genre_val = pd.read_csv(csv_dir / "genre_val.csv")
style_train = pd.read_csv(csv_dir / "style_train.csv")
style_val = pd.read_csv(csv_dir / "style_val.csv")

print(f"Loaded {len(artist_train)} artist_train records")
print(f"Loaded {len(artist_val)} artist_val records")
print(f"Loaded {len(genre_train)} genre_train records")
print(f"Loaded {len(genre_val)} genre_val records")
print(f"Loaded {len(style_train)} style_train records")
print(f"Loaded {len(style_val)} style_val records")


def standardize_attribute_df(df, expected_id_col, attribute_name):
    """Ensure dataframe has path_images and expected id column."""
    df = df.copy()
    if expected_id_col not in df.columns:
        id_cols = [col for col in df.columns if "id" in col.lower()]
        if not id_cols:
            raise ValueError(
                f"No ID column found for {attribute_name}. Columns: {list(df.columns)}"
            )
        df = df.rename(columns={id_cols[0]: expected_id_col})

    required_cols = {"path_images", expected_id_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns for {attribute_name}: {sorted(missing)}"
        )

    return df[["path_images", expected_id_col]]


def union_train_val(train_df, val_df, expected_id_col, attribute_name):
    """Union train and val for one attribute, resolving duplicates deterministically."""
    train_df = standardize_attribute_df(train_df, expected_id_col, attribute_name)
    val_df = standardize_attribute_df(val_df, expected_id_col, attribute_name)

    combined = pd.concat([train_df, val_df], ignore_index=True)
    print(
        f"Union {attribute_name}: {len(train_df)} train + {len(val_df)} val = {len(combined)}"
    )

    conflict_counts = (
        combined.groupby("path_images")[expected_id_col]
        .nunique()
        .sort_values(ascending=False)
    )
    conflicting_paths = conflict_counts[conflict_counts > 1]
    if len(conflicting_paths) > 0:
        print(
            f"WARNING: Found {len(conflicting_paths)} conflicting {attribute_name} labels for same path_images. "
            f"Keeping first occurrence (train preferred over val)."
        )
        conflict_examples = combined[
            combined["path_images"].isin(conflicting_paths.index)
        ][["path_images", expected_id_col]].head(10)
        print("Conflict examples:")
        print(conflict_examples)

    deduped = combined.drop_duplicates(subset=["path_images"], keep="first")
    duplicates_removed = len(combined) - len(deduped)
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows in {attribute_name} union")

    print(f"Final {attribute_name} union size: {len(deduped)}")
    return deduped


# Function to merge dataframes
def merge_attributes(artist_df, genre_df, style_df, set_name):
    """Merge artist, genre, and style dataframes"""
    artist_subset = standardize_attribute_df(artist_df, "artist_id", "artist")
    genre_subset = standardize_attribute_df(genre_df, "genre_id", "genre")
    style_subset = standardize_attribute_df(style_df, "style_id", "style")

    merged = artist_subset.merge(genre_subset, on="path_images", how="inner")
    merged = merged.merge(style_subset, on="path_images", how="inner")
    merged = merged[["path_images", "artist_id", "genre_id", "style_id"]]

    print(f"Merged {set_name}: {len(merged)} records")
    return merged


# Function to validate merge integrity
def validate_merge(
    original_artist, original_genre, original_style, merged_df, set_name
):
    """Validate that path_images in merged data correspond to correct labels"""
    print(f"\n--- VALIDATION CHECK FOR {set_name} ---")

    # Check 1: Verify all merged path_images exist in original data
    print(f"Checking {len(merged_df)} records...")

    merged_paths = set(merged_df["path_images"])
    artist_paths = set(original_artist["path_images"])
    genre_paths = set(original_genre["path_images"])
    style_paths = set(original_style["path_images"])

    # Paths in merged that exist in all originals
    all_paths_valid = (
        merged_paths.issubset(artist_paths)
        and merged_paths.issubset(genre_paths)
        and merged_paths.issubset(style_paths)
    )

    if all_paths_valid:
        print(f"✓ All {len(merged_paths)} path_images exist in original datasets")
    else:
        print(f"✗ WARNING: Some paths not in original data!")
        print(f"  Missing in artist: {len(merged_paths - artist_paths)}")
        print(f"  Missing in genre: {len(merged_paths - genre_paths)}")
        print(f"  Missing in style: {len(merged_paths - style_paths)}")

    # Check 2: Verify label correspondence for sample records
    print(f"\nSpot-checking label correspondence...")
    sample_size = min(5, len(merged_df))

    # Create lookup dictionaries from originals
    artist_lookup = dict(
        zip(original_artist["path_images"], original_artist["artist_id"])
    )
    genre_lookup = dict(zip(original_genre["path_images"], original_genre["genre_id"]))
    style_lookup = dict(zip(original_style["path_images"], original_style["style_id"]))

    mismatches = 0
    for idx, row in merged_df.head(sample_size).iterrows():
        path = row["path_images"]
        artist_match = path in artist_lookup and artist_lookup[path] == row["artist_id"]
        genre_match = path in genre_lookup and genre_lookup[path] == row["genre_id"]
        style_match = path in style_lookup and style_lookup[path] == row["style_id"]

        if not (artist_match and genre_match and style_match):
            print(f"✗ MISMATCH at {path}")
            mismatches += 1
        else:
            print(f"✓ {path}")

    if mismatches == 0:
        print(f"✓ All {sample_size} sample records have correct label correspondence")
    else:
        print(f"✗ Found {mismatches} mismatches in sample!")

    return mismatches == 0


def get_required_labels(df, label_cols):
    """Labels with at least 3 samples can, in principle, appear in all 3 splits."""
    required = {}
    for col in label_cols:
        counts = df[col].value_counts()
        required[col] = set(counts[counts >= 3].index.tolist())

        impossible = counts[counts < 3]
        if len(impossible) > 0:
            print(
                f"WARNING: {len(impossible)} labels in {col} have <3 samples and cannot appear in all train/val/test splits."
            )
    return required


def get_missing_required_labels(df_split, required_labels, label_cols):
    """Return missing required labels per attribute for one split."""
    missing = {}
    for col in label_cols:
        present = set(df_split[col].unique())
        missing[col] = required_labels[col] - present
    return missing


def count_total_missing(missing_by_split):
    total = 0
    for split_missing in missing_by_split.values():
        for missing_set in split_missing.values():
            total += len(missing_set)
    return total


def evaluate_split_coverage(splits, required_labels, label_cols):
    """Compute coverage diagnostics for all splits."""
    missing_by_split = {}
    for split_name, split_df in splits.items():
        missing_by_split[split_name] = get_missing_required_labels(
            split_df, required_labels, label_cols
        )

    total_missing = count_total_missing(missing_by_split)
    return total_missing == 0, total_missing, missing_by_split


def try_random_coverage_split(
    df,
    label_cols,
    train_ratio=0.8,
    val_ratio=0.1,
    attempts=400,
    base_seed=42,
):
    """Try many random splits; keep the first perfect one or the best available."""
    required_labels = get_required_labels(df, label_cols)
    total_size = len(df)
    train_end = int(total_size * train_ratio)
    val_end = int(total_size * (train_ratio + val_ratio))

    best_splits = None
    best_missing = None
    best_missing_map = None

    for attempt in range(attempts):
        shuffled = df.sample(frac=1, random_state=base_seed + attempt).reset_index(
            drop=True
        )

        splits = {
            "train": shuffled.iloc[:train_end].copy(),
            "val": shuffled.iloc[train_end:val_end].copy(),
            "test": shuffled.iloc[val_end:].copy(),
        }

        ok, total_missing, missing_map = evaluate_split_coverage(
            splits, required_labels, label_cols
        )
        if ok:
            print(f"Found perfect coverage split at attempt {attempt + 1}")
            return splits, required_labels, True

        if best_missing is None or total_missing < best_missing:
            best_missing = total_missing
            best_splits = splits
            best_missing_map = missing_map

    print(
        f"No perfect random split found after {attempts} attempts. "
        f"Best split has {best_missing} missing required-label occurrences."
    )
    if best_missing_map is not None:
        for split_name in ["train", "val", "test"]:
            summary = {
                col: len(best_missing_map[split_name][col]) for col in label_cols
            }
            print(f"  Best split missing in {split_name}: {summary}")

    return best_splits, required_labels, False


def repair_split_coverage(splits, required_labels, label_cols, max_iterations=20000):
    """Move rows across splits to satisfy required-label coverage when possible."""
    split_order = ["train", "val", "test"]

    for _ in range(max_iterations):
        ok, _, missing_map = evaluate_split_coverage(
            splits, required_labels, label_cols
        )
        if ok:
            return splits, True

        moved = False
        for target_name in split_order:
            target_df = splits[target_name]
            for col in label_cols:
                missing_labels = list(missing_map[target_name][col])
                for missing_label in missing_labels:
                    donor_names = [name for name in split_order if name != target_name]
                    donor_names = sorted(
                        donor_names,
                        key=lambda name: len(
                            splits[name][splits[name][col] == missing_label]
                        ),
                        reverse=True,
                    )

                    transferred = False
                    for donor_name in donor_names:
                        donor_df = splits[donor_name]
                        candidates = donor_df[donor_df[col] == missing_label]
                        if len(candidates) == 0:
                            continue

                        valid_idx = None
                        for idx, row in candidates.iterrows():
                            removable = True
                            for c in label_cols:
                                if (donor_df[c] == row[c]).sum() <= 1:
                                    removable = False
                                    break
                            if removable:
                                valid_idx = idx
                                break

                        if valid_idx is None:
                            continue

                        moved_row = donor_df.loc[[valid_idx]]
                        splits[donor_name] = donor_df.drop(index=valid_idx).reset_index(
                            drop=True
                        )
                        splits[target_name] = pd.concat(
                            [target_df, moved_row], ignore_index=True
                        )
                        moved = True
                        transferred = True
                        break

                    if transferred:
                        target_df = splits[target_name]

        if not moved:
            break

    final_ok, _, _ = evaluate_split_coverage(splits, required_labels, label_cols)
    return splits, final_ok


def print_split_coverage_report(splits, required_labels, label_cols):
    """Print final coverage diagnostics."""
    print("\n--- LABEL COVERAGE REPORT (required labels only) ---")
    ok, total_missing, missing_map = evaluate_split_coverage(
        splits, required_labels, label_cols
    )

    for split_name in ["train", "val", "test"]:
        print(f"{split_name.upper()} ({len(splits[split_name])} samples)")
        for col in label_cols:
            missing_count = len(missing_map[split_name][col])
            total_required = len(required_labels[col])
            print(
                f"  {col}: present {total_required - missing_count}/{total_required}, missing {missing_count}"
            )

    if ok:
        print("✓ All required labels are present in train/val/test")
    else:
        print(f"✗ Remaining missing required-label occurrences: {total_missing}")
    return ok


def filter_existing_image_paths(df, image_root, path_col="path_images"):
    """Keep only rows whose image files exist under image_root/path_images."""
    if path_col not in df.columns:
        raise ValueError(f"Missing required path column: {path_col}")

    exists_mask = df[path_col].map(lambda rel_path: (image_root / rel_path).is_file())
    filtered = df[exists_mask].reset_index(drop=True)
    removed_count = len(df) - len(filtered)

    print("\n--- IMAGE FILE EXISTENCE CHECK ---")
    print(f"Image root: {image_root}")
    print(f"Rows before filter: {len(df)}")
    print(f"Rows after filter : {len(filtered)}")
    print(f"Rows removed      : {removed_count}")

    if removed_count > 0:
        print("Sample missing paths:")
        print(df.loc[~exists_mask, path_col].head(10).to_string(index=False))

    return filtered


# Union train and val first for each attribute
artist_all = union_train_val(artist_train, artist_val, "artist_id", "artist")
genre_all = union_train_val(genre_train, genre_val, "genre_id", "genre")
style_all = union_train_val(style_train, style_val, "style_id", "style")

# Merge all attributes on full dataset
merged_all = merge_attributes(artist_all, genre_all, style_all, "ALL")
validate_merge(artist_all, genre_all, style_all, merged_all, "ALL")

# Remove records whose image files are missing from datasets/wikiart
merged_all = filter_existing_image_paths(
    merged_all, images_root, path_col="path_images"
)

# Split into new train/val/test with label coverage constraints
label_columns = ["artist_id", "genre_id", "style_id"]
splits, required_labels, perfect_random = try_random_coverage_split(
    merged_all,
    label_columns,
    train_ratio=0.8,
    val_ratio=0.1,
    attempts=400,
    base_seed=42,
)

if not perfect_random:
    print("Attempting coverage repair by moving samples across splits...")
    splits, repaired_ok = repair_split_coverage(
        splits, required_labels, label_columns, max_iterations=20000
    )
    if repaired_ok:
        print("Coverage repair successful.")
    else:
        print(
            "WARNING: Full coverage could not be achieved for all required labels. "
            "Saving best-effort split."
        )

merged_train = splits["train"]
merged_val = splits["val"]
merged_test = splits["test"]

print(
    f"\nFinal split sizes: "
    f"{len(merged_train)} train / {len(merged_val)} val / {len(merged_test)} test"
)
print_split_coverage_report(splits, required_labels, label_columns)

# Save merged CSV files
output_train_path = csv_dir / "train.csv"
output_val_path = csv_dir / "val.csv"
output_test_path = csv_dir / "test.csv"

print(f"\nSaving merged files...")
merged_train.to_csv(output_train_path, index=False)
merged_val.to_csv(output_val_path, index=False)
merged_test.to_csv(output_test_path, index=False)

print(f"✓ Saved {output_train_path}")
print(f"✓ Saved {output_val_path}")
print(f"✓ Saved {output_test_path}")

# Display sample rows
print("\n--- TRAINING SET SAMPLE ---")
print(merged_train.head())
print(f"\nTotal training records: {len(merged_train)}")

print("\n--- VALIDATION SET SAMPLE ---")
print(merged_val.head())
print(f"\nTotal validation records: {len(merged_val)}")

print("\n--- TEST SET SAMPLE ---")
print(merged_test.head())
print(f"\nTotal test records: {len(merged_test)}")

# Verify no data loss
total_expected = len(merged_all)
total_saved = len(merged_train) + len(merged_val) + len(merged_test)
print(f"\n--- DATA INTEGRITY CHECK ---")
print(f"Expected merged records: {total_expected}")
print(f"Saved merged records: {total_saved}")
kept_pct = (total_saved / total_expected) * 100 if total_expected > 0 else 0.0
print(f"Records kept: {kept_pct:.2f}%")
