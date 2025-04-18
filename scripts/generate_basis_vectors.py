# --- START OF FILE generate_basis_vectors.py ---

import argparse
import datetime
import traceback
from pathlib import Path
import numpy as np
from collections import defaultdict
import re # Keep re import if needed by other helpers here

# Local application imports
import utils # Import our new utility functions

# --- Constants ---
DEFAULT_EXPERIMENT_BASE_DIR = "experiments"
DEFAULT_BASIS_KEY_ENSEMBLE = 'basis_vectors'
MIN_VECTORS_DEFAULT = 1

# --- Helper Functions ---

# REMOVED sanitize_label definition from here

# --- Helper Functions (Keep _concise_filter_repr and generate_default_basis_label here as they are only used locally) ---
def _concise_filter_repr(filter_dict: dict | None) -> str:
    """Generates a short string representation of a filter dictionary."""
    if not filter_dict: return "all"
    parts = []
    key_map = {'type': 't', 'level': 'l', 'sweep': 's', 'core_id': 'cid'}
    sorted_keys = sorted(filter_dict.keys(), key=lambda k: str(key_map.get(k, k)))
    for key in sorted_keys:
        prefix = key_map.get(key, str(key)[:3])
        value = str(filter_dict[key])
        value_short = value.replace('declarative', 'decl').replace('rhetorical', 'rhet')
        value_short = value_short.replace('observational', 'obs').replace('authoritative', 'auth')
        value_short = value_short.replace('baseline','base')[:8]
        parts.append(f"{prefix}_{value_short}")
    return "_".join(parts) if parts else "all"

def generate_default_basis_label(args: argparse.Namespace) -> str:
    """Generates a default filename label based on mode and filters."""
    if args.mode == 'single_plane':
        f1 = utils.parse_filter_string(args.filter_set_1)
        f2 = utils.parse_filter_string(args.filter_set_2)
        f1_repr = _concise_filter_repr(f1)
        f2_repr = _concise_filter_repr(f2)
        return f"single_plane_{f1_repr}_vs_{f2_repr}"
    elif args.mode == 'ensemble':
        group_key = args.ensemble_group_key or "nogroup"
        fixed_f = utils.parse_filter_string(args.fixed_filters)
        fixed_repr = _concise_filter_repr(fixed_f)
        output_key = args.output_key or DEFAULT_BASIS_KEY_ENSEMBLE
        label_parts = ["ensemble", f"grp_{group_key}"]
        if fixed_repr != "all": label_parts.append(f"fixed_{fixed_repr}")
        label_parts.append(f"key_{output_key}")
        return "_".join(label_parts)
    else:
        return f"unknown_mode_{datetime.datetime.now().strftime('%Y%m%d')}"


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Generate basis vectors (.npz) and metadata (.json) from captured activation vectors. "
                    f"Selects a baseline run, finds vectors in '{utils.CAPTURE_SUBFOLDER}/{utils.VECTORS_SUBFOLDER}/', "
                    f"and saves output to the selected run's '{utils.BASIS_SUBFOLDER}/'."
    )
    # --- Arguments ---
    parser.add_argument("--experiment_base_dir", type=str, default=DEFAULT_EXPERIMENT_BASE_DIR, help="Base directory containing experiment run folders.")
    parser.add_argument("--output_basis_label", type=str, default=None, help="[Optional] Custom descriptive label for the output basis file (e.g., 'declarative1_vs_rhetorical5'). If None, a default is auto-generated based on mode/filters.")
    parser.add_argument("--mode", type=str, choices=['single_plane', 'ensemble'], required=True, help="Operation mode: 'single_plane' (generate basis_1, basis_2) or 'ensemble' (generate multiple basis vectors grouped by a key).")
    parser.add_argument("--filter_set_1", type=str, default=None, help="[Single Plane] Filters for basis_1 (e.g., 'type=declarative,level=1'). Required if mode=single_plane.")
    parser.add_argument("--filter_set_2", type=str, default=None, help="[Single Plane] Filters for basis_2 (e.g., 'type=rhetorical,level=5'). Required if mode=single_plane.")
    parser.add_argument("--ensemble_group_key", type=str, choices=utils.VALID_GROUP_KEYS, default=None, help="[Ensemble] Metadata key to group by for basis generation (e.g., 'type'). Required if mode=ensemble.")
    parser.add_argument("--fixed_filters", type=str, default=None, help="[Ensemble] Additional fixed filters applied to ALL groups before grouping (e.g., 'level=5').")
    parser.add_argument("--output_key", type=str, default=DEFAULT_BASIS_KEY_ENSEMBLE, help=f"[Ensemble] Key name for the basis array in the output NPZ (default: {DEFAULT_BASIS_KEY_ENSEMBLE}).")
    parser.add_argument("--min_vectors_per_group", type=int, default=MIN_VECTORS_DEFAULT, help=f"Minimum number of vectors required to calculate a mean for a group/set (default: {MIN_VECTORS_DEFAULT}).")
    args = parser.parse_args()

    # --- Input Validation ---
    if args.mode == 'single_plane' and (not args.filter_set_1 or not args.filter_set_2): parser.error("--filter_set_1 and --filter_set_2 are required for mode 'single_plane'.")
    if args.mode == 'ensemble' and not args.ensemble_group_key: parser.error("--ensemble_group_key is required for mode 'ensemble'.")
    if args.min_vectors_per_group < 1: print(f"Warning: --min_vectors_per_group must be >= 1. Setting to {MIN_VECTORS_DEFAULT}."); args.min_vectors_per_group = MIN_VECTORS_DEFAULT

    # --- Select Source Run Directory (Interactive) ---
    base_dir = Path(args.experiment_base_dir)
    print(f"\nLooking for *baseline* runs to generate basis from...")
    selected_run_dir = utils.select_experiment_folder(base_dir, prompt_text="Select a BASELINE run directory containing vectors:", pattern="run_baseline_*")
    if not selected_run_dir: print("No baseline run directory selected. Exiting."); exit(1)
    print(f"Using baseline run directory: {selected_run_dir.name}")

    # --- Locate Input Vector File (Automatic/Interactive within selected run) ---
    input_vector_path = utils.find_vector_file(selected_run_dir, expected_suffix="_baseline")
    if not input_vector_path: print(f"Could not find a suitable baseline vector file in {selected_run_dir / utils.CAPTURE_SUBFOLDER / utils.VECTORS_SUBFOLDER}. Exiting."); exit(1)
    print(f"Using input vector file: {input_vector_path.name}")

    # --- Prepare Output Paths and Determine Label ---
    basis_dir = selected_run_dir / utils.BASIS_SUBFOLDER
    basis_dir.mkdir(parents=True, exist_ok=True)
    generation_timestamp = datetime.datetime.now()
    timestamp_str = generation_timestamp.strftime("%Y%m%d_%H%M%S")

    final_output_label = None
    if args.output_basis_label:
        sanitized_user_label = utils.sanitize_label(args.output_basis_label) # Now uses utils.sanitize_label
        if sanitized_user_label and sanitized_user_label != "unlabeled":
            final_output_label = sanitized_user_label
            print(f"Using provided (and sanitized) output label: '{final_output_label}'")
        else:
            print("Warning: Provided label was empty or invalid after sanitization. Generating default label.")
    if not final_output_label:
         print("Generating default basis label based on mode and filters.")
         # generate_default_basis_label calls _concise_filter_repr locally
         default_label_raw = generate_default_basis_label(args)
         final_output_label = utils.sanitize_label(default_label_raw) # Sanitize the generated label
         print(f"Generated default label: '{final_output_label}'")

    base_filename = f"basis_{final_output_label}"
    output_npz_path = basis_dir / f"{base_filename}.npz"
    output_json_path = basis_dir / f"{base_filename}.json"
    print(f"Output basis NPZ will be saved to: {output_npz_path}")
    print(f"Output metadata JSON will be saved to: {output_json_path}")

    # --- Load Data using utils ---
    structured_data, source_metadata = utils.load_vector_data(input_vector_path, expected_dim=utils.DIMENSION)
    if structured_data is None: print("Exiting: Failed to load input vectors."); exit(1)
    if not structured_data: print("Warning: No valid vectors were loaded from the input file. Cannot generate basis."); exit(0)

    # --- Prepare Basis Metadata ---
    basis_metadata = {
        "script_name": Path(__file__).name,
        "generation_timestamp": generation_timestamp.isoformat(),
        "basis_generation_mode": args.mode,
        "source_run_directory": str(selected_run_dir),
        "source_run_directory_name": selected_run_dir.name,
        "source_vector_file_relative": str(input_vector_path.relative_to(selected_run_dir)),
        "source_vector_metadata": source_metadata,
        "output_basis_directory_relative": str(basis_dir.relative_to(selected_run_dir)),
        "output_basis_file_relative": str(output_npz_path.relative_to(selected_run_dir)),
        "output_metadata_file_relative": str(output_json_path.relative_to(selected_run_dir)),
        "user_provided_output_basis_label": args.output_basis_label,
        "generated_filename_label": final_output_label,
        "dimension": utils.DIMENSION,
        "min_vectors_per_group": args.min_vectors_per_group,
        "cli_args": vars(args),
        "single_plane_details": None,
        "ensemble_details": None,
    }

    # --- Generate Basis Vectors ---
    basis_saved = False
    try:
        if args.mode == 'single_plane':
            print("\n--- Generating Single Plane Basis ---")
            filters1 = utils.parse_filter_string(args.filter_set_1)
            filters2 = utils.parse_filter_string(args.filter_set_2)
            if filters1 is None or filters2 is None: raise ValueError("Error parsing filter strings.")
            print(f"Filtering for Basis 1 with: {filters1}")
            vectors1 = utils.filter_data(structured_data, filters1); num_vec1 = len(vectors1)
            print(f"Found {num_vec1} vectors for Basis 1.")
            print(f"Filtering for Basis 2 with: {filters2}")
            vectors2 = utils.filter_data(structured_data, filters2); num_vec2 = len(vectors2)
            print(f"Found {num_vec2} vectors for Basis 2.")
            basis_metadata["single_plane_details"] = {
                "filter_set_1_str": args.filter_set_1, "filter_set_2_str": args.filter_set_2,
                "filter_set_1_parsed": filters1, "filter_set_2_parsed": filters2,
                "num_vectors_basis_1": num_vec1, "num_vectors_basis_2": num_vec2
            }
            if num_vec1 < args.min_vectors_per_group or num_vec2 < args.min_vectors_per_group:
                 print(f"Error: Insufficient vectors found (Min required: {args.min_vectors_per_group}). Basis file NOT saved.")
                 basis_1 = None; basis_2 = None
            else:
                basis_1 = utils.calculate_mean_vector(vectors1)
                basis_2 = utils.calculate_mean_vector(vectors2)
            if basis_1 is not None and basis_2 is not None:
                print(f"Calculated basis_1 (shape {basis_1.shape}) and basis_2 (shape {basis_2.shape}).")
                basis_saved = utils.save_basis_vectors(output_npz_path, basis_1=basis_1, basis_2=basis_2, metadata=basis_metadata)
            else: print("Basis vectors could not be calculated. No files saved.")

        elif args.mode == 'ensemble':
            print("\n--- Generating Ensemble Basis ---")
            fixed_filters = utils.parse_filter_string(args.fixed_filters)
            if fixed_filters is None: raise ValueError(f"Error parsing fixed filters string: '{args.fixed_filters}'")
            if fixed_filters: print(f"Applying fixed filters to all groups: {fixed_filters}")
            else: print("No fixed filters applied.")
            group_key = args.ensemble_group_key
            print(f"Grouping by key: '{group_key}'")
            pre_filtered_data = utils.filter_data(structured_data, fixed_filters)
            print(f"Found {len(pre_filtered_data)} vectors matching fixed filters.")
            if not pre_filtered_data: print(f"Error: No data remaining after applying fixed filters. Cannot generate ensemble basis."); exit(0)

            grouped_data = defaultdict(list); missing_group_key_count = 0
            for item in pre_filtered_data:
                if group_key in item['key_components']: grouped_data[str(item['key_components'][group_key])].append(item['vector'])
                else: missing_group_key_count += 1
            if not grouped_data: print(f"Error: No vectors found with the grouping key '{group_key}' after applying fixed filters."); exit(0)
            if missing_group_key_count > 0: print(f"Warning: {missing_group_key_count} vectors lacked the group key '{group_key}'.")

            ensemble_vectors = []; group_labels = []; group_details_meta = {}; skipped_groups_count = 0; valid_groups_found = 0
            sorted_group_keys = sorted(grouped_data.keys())
            print(f"Found {len(sorted_group_keys)} potential groups for key '{group_key}' (after fixed filters).")
            for group_value in sorted_group_keys:
                vectors = grouped_data[group_value]; count = len(vectors)
                if count >= args.min_vectors_per_group:
                    mean_vec = utils.calculate_mean_vector(vectors)
                    if mean_vec is not None:
                        ensemble_vectors.append(mean_vec); group_labels.append(str(group_value))
                        group_details_meta[str(group_value)] = count; valid_groups_found += 1
                    else: print(f"Warning: Mean calculation failed for group '{group_value}'. Skipping."); skipped_groups_count += 1
                else: skipped_groups_count += 1

            num_generated = len(ensemble_vectors)
            print(f"\nGenerated {num_generated} basis vectors for {valid_groups_found} valid groups.")
            if skipped_groups_count > 0: print(f"Skipped {skipped_groups_count} groups due to insufficient vectors or calculation error.")
            basis_metadata["ensemble_details"] = {
                 "ensemble_group_key": args.ensemble_group_key, "fixed_filters_str": args.fixed_filters,
                 "fixed_filters_parsed": fixed_filters, "output_key_in_npz": args.output_key,
                 "group_vector_counts": group_details_meta, "generated_group_labels": group_labels,
                 "num_potential_groups_found": len(grouped_data), "num_groups_skipped": skipped_groups_count,
                 "num_basis_vectors_generated": num_generated,
            }
            if num_generated < 2: print(f"Error: Need at least 2 basis vectors for ensemble. Basis file NOT saved.")
            else:
                ensemble_array = np.array(ensemble_vectors)
                print(f"Final ensemble basis shape: {ensemble_array.shape}")
                basis_saved = utils.save_basis_vectors(output_npz_path, ensemble_basis=ensemble_array, ensemble_key=args.output_key, group_labels=group_labels, metadata=basis_metadata)

        if basis_saved:
            print(f"\nSaving basis generation metadata to: {output_json_path}")
            utils.save_json_metadata(output_json_path, basis_metadata)
        else:
            print("\nBasis NPZ file was not saved. Metadata JSON file will also not be saved.")

    except Exception as e:
        print(f"\n--- An error occurred during basis generation ---"); traceback.print_exc(); print("Basis generation failed.")

    print("\nScript finished.")

# --- END OF FILE generate_basis_vectors.py ---