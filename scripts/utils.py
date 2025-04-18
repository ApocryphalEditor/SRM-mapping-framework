# --- START OF FILE utils.py ---

import os
import numpy as np
import json
import re # <--- Make sure re is imported
import datetime
import traceback
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import random
import itertools

# --- Constants ---
DIMENSION = 3072 # Assuming GPT-2 small MLP dimension
VALID_GROUP_KEYS = ['core_id', 'type', 'level', 'sweep']
CAPTURE_SUBFOLDER = "capture"
ANALYSES_SUBFOLDER = "analyses"
BASIS_SUBFOLDER = "basis"
VECTORS_SUBFOLDER = "vectors"
LOGS_SUBFOLDER = "logs"
METADATA_SUBFOLDER = "metadata"
PLOTS_SUBFOLDER = "plots" # Added constant for clarity in plotting

# === Directory and File Handling ===

def list_experiment_folders(base_dir: Path, pattern: str = "run_*") -> list[Path]:
    """Lists directories matching a pattern within the base directory."""
    if not base_dir.is_dir():
        return []
    # Ensure pattern matching is consistent (e.g., handle trailing '*' if present)
    prefix = pattern.replace('*', '')
    return sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(prefix) and not d.name.startswith('.')])

def select_experiment_folder(base_dir: Path, prompt_text: str = "Select an experiment run directory:", pattern: str = "run_*") -> Path | None:
    """Interactively prompts the user to select a directory from a list."""
    run_dirs = list_experiment_folders(base_dir, pattern)
    if not run_dirs:
        print(f"Error: No run directories matching '{pattern}' found in {base_dir}.")
        return None

    print(f"\n{prompt_text}")
    for i, dir_path in enumerate(run_dirs):
        print(f"  {i+1}: {dir_path.name}")

    selected_dir = None
    while selected_dir is None:
        try:
            choice = input(f"Enter the number of the directory (1-{len(run_dirs)}): ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(run_dirs):
                selected_dir = run_dirs[choice_idx]
                print(f"Selected: {selected_dir.name}")
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except (EOFError, KeyboardInterrupt):
            print("\nSelection cancelled.")
            return None
    return selected_dir.resolve()

def find_latest_file(directory: Path, pattern: str = "*.npz") -> Path | None:
    """Finds the most recently modified file matching the pattern in a directory."""
    if not directory.is_dir():
        return None
    try:
        files = sorted(directory.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
        return files[0] if files else None
    except PermissionError:
        print(f"Warning: Permission denied when accessing directory {directory}")
        return None
    except Exception as e:
        print(f"Warning: Error finding latest file in {directory}: {e}")
        return None


def find_file_interactive(directory: Path, pattern: str = "*.npz", file_type_desc: str = "file") -> Path | None:
    """Finds files matching a pattern and prompts the user if multiple are found."""
    if not directory.is_dir():
        print(f"Error: Directory for interactive search not found: {directory}")
        return None

    try:
        files = sorted(list(directory.glob(pattern)))
    except PermissionError:
        print(f"Error: Permission denied when searching for files in {directory}")
        return None
    except Exception as e:
        print(f"Error searching for files in {directory}: {e}")
        return None


    if not files:
        print(f"Info: No {file_type_desc} files matching '{pattern}' found in {directory}")
        return None
    elif len(files) == 1:
        print(f"Auto-selected {file_type_desc} file: {files[0].name}")
        return files[0]
    else:
        print(f"Multiple {file_type_desc} files found matching '{pattern}' in {directory.name}:")
        for i, fp in enumerate(files):
            print(f"  {i+1}: {fp.name}")

        selected_file = None
        while selected_file is None:
            try:
                choice = input(f"Enter the number of the {file_type_desc} file to use (1-{len(files)}): ")
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(files):
                    selected_file = files[choice_idx]
                    print(f"Selected {file_type_desc} file: {selected_file.name}")
                else:
                    print("Invalid choice.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except (EOFError, KeyboardInterrupt):
                print("\nSelection cancelled.")
                return None
        return selected_file

def find_vector_file(run_dir: Path, expected_suffix: str | None = None) -> Path | None:
    """
    Finds the vector NPZ file within the standard capture/vectors subdirectory.
    Prompts if multiple files exist.
    """
    vector_dir = run_dir / CAPTURE_SUBFOLDER / VECTORS_SUBFOLDER
    pattern = f"*{expected_suffix}.npz" if expected_suffix else "*.npz"
    return find_file_interactive(vector_dir, pattern, "vector")

def find_basis_file(basis_search_dir: Path, specific_filename: str | None = None) -> Path | None:
    """
    Finds a basis NPZ file within the given search directory (expected to be a 'basis' subfolder).
    If specific_filename is given, looks for that. Otherwise, prompts if multiple are found.
    """
    basis_dir = basis_search_dir
    if not basis_dir.is_dir():
         potential_basis_dir = basis_search_dir / BASIS_SUBFOLDER
         if potential_basis_dir.is_dir():
              basis_dir = potential_basis_dir
         else:
             print(f"Info: Basis directory not found at expected location: {basis_search_dir}")
             return None

    if specific_filename:
        target_file = basis_dir / specific_filename
        if target_file.is_file():
            print(f"Found specified basis file: {target_file.name}")
            return target_file
        else:
            print(f"Info: Specified basis file '{specific_filename}' not found in {basis_dir}.")
            print("Searching for other basis files...")
            return find_file_interactive(basis_dir, "*.npz", "basis")
    else:
        return find_file_interactive(basis_dir, "*.npz", "basis")


# === Data Loading and Parsing ===

def sanitize_label(label: str | None) -> str:
    """ Replaces potentially problematic characters in a label for filenames. """
    if not label: return "unlabeled"
    sanitized = re.sub(r'[\\/:*?"<>|,=\s]+', '_', label)
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    return sanitized[:100]


def parse_vector_key(key_str: str) -> dict:
    """Parses key=value pairs from vector filenames."""
    components = {}
    remaining_str = key_str
    if remaining_str.startswith("core_id="):
        val_start_idx = len("core_id=")
        possible_ends = [idx for key in VALID_GROUP_KEYS + ['sweep'] if key != 'core_id' and (idx := remaining_str.find(f"_{key}=")) != -1]
        end_idx = min(possible_ends) if possible_ends else -1
        if end_idx != -1:
            components['core_id'] = remaining_str[val_start_idx:end_idx]
            remaining_str = remaining_str[end_idx:]
        else:
            components['core_id'] = remaining_str[val_start_idx:]
            remaining_str = ""

    keys_to_parse = ['type', 'level', 'sweep']
    for key in keys_to_parse:
        prefix = f"_{key}="
        if remaining_str.startswith(prefix):
            val_start_idx = len(prefix)
            possible_ends = [idx for k in keys_to_parse if k != key and (idx := remaining_str.find(f"_{k}=")) != -1]
            end_idx = min(possible_ends) if possible_ends else -1
            value_str = ""
            if end_idx != -1:
                value_str = remaining_str[val_start_idx:end_idx]
                remaining_str = remaining_str[end_idx:]
            else:
                value_str = remaining_str[val_start_idx:]
                remaining_str = ""
            if key == 'level':
                try:
                    components[key] = int(value_str)
                except ValueError:
                    components[key] = value_str
            else:
                components[key] = value_str
    if not any(k in components for k in VALID_GROUP_KEYS + ['sweep']):
        return {}
    return components


def load_vector_data(npz_path: Path, expected_dim: int = DIMENSION) -> tuple[list[dict], dict] | tuple[None, None]:
    """Loads vectors from NPZ, parses keys, and returns structured list and metadata."""
    structured_vectors = []
    metadata = {}

    if not npz_path.is_file():
        print(f"Error: Input vector file not found: '{npz_path}'")
        return None, None

    print(f"\nLoading vectors and keys from: {npz_path.name}")
    try:
        with np.load(npz_path, allow_pickle=True) as loaded_data:
            if '__metadata__' in loaded_data:
                try:
                    metadata = loaded_data['__metadata__'].item()
                    if isinstance(metadata, dict):
                        print("Loaded metadata embedded in NPZ file.")
                    else:
                        print("Warning: Embedded '__metadata__' is not a dictionary. Ignoring.")
                        metadata = {}
                except Exception as meta_e:
                    print(f"Warning: Could not load embedded '__metadata__': {meta_e}")
                    metadata = {}

            vector_keys = [k for k in loaded_data.files if k != '__metadata__']
            print(f"Found {len(vector_keys)} potential vector keys in the input file.")
            valid_count = 0
            skipped_count = 0

            for key in tqdm(vector_keys, desc="Loading vectors", leave=False, unit="key"):
                try:
                    vec = loaded_data[key]
                    if not isinstance(vec, np.ndarray) or vec.shape != (expected_dim,):
                        skipped_count += 1
                        continue
                    key_components = parse_vector_key(key)
                    if not key_components:
                        skipped_count += 1
                        continue
                    structured_vectors.append({
                        'key': key,
                        'key_components': key_components,
                        'vector': vec
                    })
                    valid_count += 1
                except Exception as e:
                    print(f"\nError processing key '{key}': {e}")
                    traceback.print_exc()
                    skipped_count += 1
                    continue
    except FileNotFoundError:
        print(f"Error: Input vector file not found: '{npz_path}'")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred loading {npz_path}: {e}")
        traceback.print_exc()
        return None, None

    if skipped_count > 0:
        print(f"Skipped {skipped_count} invalid or unparsable entries during loading.")
    if not structured_vectors:
        print(f"Warning: No valid vectors loaded from {npz_path}.")
        return [], metadata
    print(f"Successfully loaded {valid_count} vectors with keys.")
    return structured_vectors, metadata


def load_metadata(metadata_path: Path) -> dict | None:
    """Loads JSON metadata from a file."""
    if not metadata_path.is_file():
        print(f"Warning: Metadata file not found: {metadata_path}")
        return None
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in metadata file: {metadata_path}")
        return None
    except Exception as e:
        print(f"Error loading metadata file {metadata_path}: {e}")
        return None

# === Basis Generation Helpers ===

def parse_filter_string(filter_str: str | None) -> dict | None:
    """ Parses 'key1=value1,key2=value2' into {'key1': 'value1', 'key2': 'value2'} """
    filters = {}
    if not filter_str:
        return filters
    try:
        pairs = filter_str.split(',')
        for pair in pairs:
            pair = pair.strip()
            if not pair: continue
            if '=' not in pair:
                raise ValueError(f"Invalid filter format in pair: '{pair}' (missing '=')")
            key, value = pair.split('=', 1)
            key = key.strip()
            value = value.strip()
            if not key:
                raise ValueError("Filter key cannot be empty.")
            filters[key] = value
    except Exception as e:
        print(f"Error parsing filter string '{filter_str}': {e}")
        return None
    return filters


def filter_data(structured_data: list[dict], filters: dict) -> list[np.ndarray]:
    """ Filters the loaded data based on the provided filter dictionary. """
    if not filters:
        return [item['vector'] for item in structured_data]

    matching_vectors = []
    for item in structured_data:
        components = item['key_components']
        match = True
        for key, filter_value_str in filters.items():
            if key not in components:
                match = False
                break
            component_value = components[key]
            if str(component_value).lower() != str(filter_value_str).lower():
                match = False
                break
        if match:
            matching_vectors.append(item['vector'])
    return matching_vectors


def calculate_mean_vector(vectors: list[np.ndarray], expected_dim: int = DIMENSION) -> np.ndarray | None:
    """ Calculates the mean vector from a list of vectors. """
    if not vectors:
        return None
    try:
        vector_array = np.array(vectors)
        if vector_array.ndim != 2 or vector_array.shape[1] != expected_dim:
            print(f"Error: Invalid shape for mean calculation. Expected [N, {expected_dim}], got {vector_array.shape}")
            return None
        mean_vec = np.mean(vector_array, axis=0)
        return mean_vec
    except Exception as e:
        print(f"Error calculating mean vector: {e}")
        return None

def save_basis_vectors(
    path: Path,
    basis_1: np.ndarray | None = None,
    basis_2: np.ndarray | None = None,
    ensemble_basis: np.ndarray | None = None,
    ensemble_key: str = 'basis_vectors',
    group_labels: list[str] | np.ndarray | None = None,
    metadata: dict | None = None
    ):
    """Saves basis vectors (single plane or ensemble) and optional metadata to NPZ."""
    save_dict = {}
    if basis_1 is not None and basis_2 is not None:
        save_dict['basis_1'] = basis_1
        save_dict['basis_2'] = basis_2
        print(f"Preparing to save single plane basis (basis_1, basis_2) to {path.name}")
    elif ensemble_basis is not None:
        save_dict[ensemble_key] = ensemble_basis
        if group_labels is not None:
            save_dict['group_labels'] = np.array(group_labels)
            print(f"Preparing to save ensemble basis (key: '{ensemble_key}') and {len(group_labels)} labels to {path.name}")
        else:
             print(f"Preparing to save ensemble basis (key: '{ensemble_key}') without labels to {path.name}")
    else:
        print("Error: No valid basis vectors provided for saving.")
        return False

    if metadata:
        save_dict['__metadata__'] = np.array(metadata, dtype=object)
        print("Including metadata in NPZ file.")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **save_dict)
        print(f"Successfully saved basis file: {path}")
        return True
    except Exception as e:
        print(f"Error saving basis file {path}: {e}")
        traceback.print_exc()
        return False


def save_json_metadata(path: Path, metadata: dict):
    """Saves a dictionary as a JSON file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
        print(f"Successfully saved JSON metadata: {path}")
        return True
    except Exception as e:
        print(f"Error saving JSON metadata file {path}: {e}")
        traceback.print_exc()
        return False

# === Analysis Helpers (Moved/Adapted from analyze_srm_sweep.py) ===

def normalise(array: np.ndarray, axis: int) -> np.ndarray:
    """Normalizes vectors along a specified axis."""
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    zero_norm_mask = (norms == 0)
    norms[zero_norm_mask] = 1.0
    normalized_array = array / norms
    if isinstance(axis, int):
        mask_shape_expected = list(array.shape)
        mask_shape_expected.pop(axis)
        squeezed_mask = zero_norm_mask.reshape(mask_shape_expected)
        if axis == 0:
             normalized_array[:, squeezed_mask] = 0.0
        elif axis == 1 or (axis == -1 and array.ndim > 1):
             normalized_array[squeezed_mask, :] = 0.0
    return normalized_array

def vectors_to_bivectors(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    """Computes the bivector (rotation generator) from two normalized vectors."""
    v1_norm = normalise(vector1, axis=0)
    v2_norm = normalise(vector2, axis=0)
    outer_product = np.outer(v1_norm, v2_norm)
    rotation_generator = outer_product - outer_product.T
    return rotation_generator

def hermitian_conjugate(array: np.ndarray) -> np.ndarray:
    """Computes the Hermitian conjugate (conjugate transpose)."""
    return np.conj(array).T

def generate_special_orthogonal_matrices(generator: np.ndarray, angles_rad: np.ndarray, debug_log_file=None) -> np.ndarray:
    """Generates SO(D) rotation matrices using the generator and angles (Rodrigues' formula generalization)."""
    if debug_log_file: debug_log_file.write("\n--- [DEBUG] generate_special_orthogonal_matrices ---\n")
    D = generator.shape[0]
    num_angles = len(angles_rad)
    identity_matrix = np.identity(D)

    generator_norm = np.linalg.norm(generator)
    if debug_log_file: debug_log_file.write(f"[DEBUG] Generator Frobenius norm: {generator_norm:.4e}\n")
    if generator_norm < 1e-9:
        if debug_log_file: debug_log_file.write("[DEBUG] Generator is near zero. Returning identity matrices.\n")
        return np.array([identity_matrix] * num_angles)

    antisymmetry_check = np.max(np.abs(generator + generator.T))
    if debug_log_file: debug_log_file.write(f"[DEBUG] Antisymmetry check (max|G + G.T|): {antisymmetry_check:.4e}\n")

    try:
        vals, vecs = np.linalg.eig(generator)
        sort_indices = np.argsort(np.abs(vals.imag))[::-1]
        vals = vals[sort_indices]
        vecs = vecs[:, sort_indices]
        if debug_log_file: debug_log_file.write(f"[DEBUG] Eigenvalues calculated (sorted, first 5): {vals[:5]}\n")
        non_zero_real_part = np.max(np.abs(vals.real))
        if debug_log_file: debug_log_file.write(f"[DEBUG] Max abs real part of eigenvalues: {non_zero_real_part:.4e}\n")
        if non_zero_real_part > 1e-6:
             if debug_log_file: debug_log_file.write("[DEBUG] Warning: Eigenvalues have significant real parts. Matrix might not be perfectly anti-symmetric.\n")
    except np.linalg.LinAlgError as e:
        if debug_log_file: debug_log_file.write(f"[DEBUG] Eigendecomposition failed: {e}. Returning identity matrices.\n")
        return np.array([identity_matrix] * num_angles)

    imag_parts = vals.imag
    non_zero_mask = np.abs(imag_parts) > 1e-9
    num_non_zero_imag = np.sum(non_zero_mask)
    if debug_log_file: debug_log_file.write(f"[DEBUG] Number of non-zero imaginary parts found: {num_non_zero_imag}\n")
    if num_non_zero_imag == 0:
        if debug_log_file: debug_log_file.write("[DEBUG] No non-zero imaginary eigenvalues. Returning identity matrices.\n")
        return np.array([identity_matrix] * num_angles)

    normalisation_factor = np.abs(imag_parts[0])
    if debug_log_file: debug_log_file.write(f"[DEBUG] Normalisation factor (abs(imag_lambda_max)): {normalisation_factor:.4e}\n")
    if normalisation_factor < 1e-9:
        if debug_log_file: debug_log_file.write("[DEBUG] Eigenvalue normalization factor near zero. Returning identity matrices.\n")
        return np.array([identity_matrix] * num_angles)

    normalized_eigenvalues = np.where(non_zero_mask, 1j * (imag_parts / normalisation_factor), 0+0j)
    normalized_eigenvalues = normalized_eigenvalues.imag * 1j
    if debug_log_file: debug_log_file.write(f"[DEBUG] Normalized eigenvalues (first 5): {normalized_eigenvalues[:5]}\n")
    vecs_h = hermitian_conjugate(vecs)
    exp_lambda_angled = np.exp(angles_rad[:, np.newaxis] * normalized_eigenvalues)
    complex_rotation_matrices = np.einsum('ik,ak,kj->aij', vecs, exp_lambda_angled, vecs_h, optimize='optimal')
    real_rotation_matrices = complex_rotation_matrices.real

    if debug_log_file:
        max_imag_residue_final = np.max(np.abs(real_rotation_matrices.imag))
        if debug_log_file: debug_log_file.write(f"[DEBUG] Max abs imaginary part in final matrix array: {max_imag_residue_final:.4e}\n")
        if debug_log_file: debug_log_file.write(f"[DEBUG] Shape of final real_rotation_matrices: {real_rotation_matrices.shape}\n")
        if len(real_rotation_matrices) > 0:
            matrix_0 = real_rotation_matrices[0]
            identity_check = np.max(np.abs(matrix_0 - identity_matrix))
            if debug_log_file: debug_log_file.write(f"[DEBUG] Max diff R(0) vs Identity: {identity_check:.4e}\n")
            ortho_check_0 = np.max(np.abs(matrix_0 @ matrix_0.T - identity_matrix))
            if debug_log_file: debug_log_file.write(f"[DEBUG] Orthogonality check R(0)@R(0).T vs I: {ortho_check_0:.4e}\n")
            det_0 = np.linalg.det(matrix_0)
            if debug_log_file: debug_log_file.write(f"[DEBUG] Determinant R(0): {det_0:.4f}\n")
            idx_90 = num_angles // 4
            if idx_90 < num_angles:
                 matrix_90 = real_rotation_matrices[idx_90]
                 diff_90_0 = np.max(np.abs(matrix_90 - matrix_0))
                 if debug_log_file: debug_log_file.write(f"[DEBUG] Max diff R(~90) vs R(0): {diff_90_0:.4e}\n")
                 ortho_check_90 = np.max(np.abs(matrix_90 @ matrix_90.T - identity_matrix))
                 if debug_log_file: debug_log_file.write(f"[DEBUG] Ortho check R(~90): {ortho_check_90:.4e}\n")
                 det_90 = np.linalg.det(matrix_90)
                 if debug_log_file: debug_log_file.write(f"[DEBUG] Determinant R(~90): {det_90:.4f}\n")
    if debug_log_file: debug_log_file.write("--- [DEBUG] generate_special_orthogonal_matrices END ---\n")

    return real_rotation_matrices


# --- Analysis Execution ---
def perform_srm_analysis(
    structured_data: list[dict],
    basis_details: dict,
    group_by: str | None,
    signed: bool,
    thresholds: list[float],
    num_angles: int = 72,
    ensemble_max_planes: int | None = None,
    ensemble_plane_selection: str = 'comb',
    debug_log_file=None
    ) -> dict[str, pd.DataFrame]:
    """
    Performs SRM analysis (single plane or ensemble) on grouped data.
    Returns: Dictionary mapping group names to pandas DataFrames with SRM results.
    """
    analysis_mode = basis_details.get('mode')
    if analysis_mode not in ['single_plane', 'ensemble']:
        print(f"Error: Invalid analysis mode '{analysis_mode}' in basis_details.")
        return {}

    print(f"\nGrouping vectors by: '{group_by if group_by else 'All Vectors'}'")
    grouped_vectors = defaultdict(list)
    if group_by:
        valid_vectors_in_grouping = 0; missing_key_count = 0
        for item in structured_data:
            group_val = item['key_components'].get(group_by)
            if group_val is not None:
                grouped_vectors[str(group_val)].append(item['vector'])
                valid_vectors_in_grouping += 1
            else: missing_key_count += 1
        if valid_vectors_in_grouping == 0:
            print(f"Warning: No vectors found with key '{group_by}'. Skipping analysis for this grouping.")
            return {}
        print(f"Found {len(grouped_vectors)} groups for key '{group_by}': {sorted(list(grouped_vectors.keys()))}")
        if missing_key_count > 0: print(f"  (Note: {missing_key_count} vectors lacked the '{group_by}' key)")
    else:
        grouped_vectors['all'] = [item['vector'] for item in structured_data]
        if not grouped_vectors['all']: print("Error: No vectors loaded to analyze."); return {}
        print(f"Analyzing all {len(grouped_vectors['all'])} vectors together.")

    results_by_group = {}
    angles_deg = np.linspace(0, 360, num_angles, endpoint=False)
    angles_rad = np.radians(angles_deg)

    if analysis_mode == 'single_plane':
        basis_vector_1 = basis_details.get('basis_1'); basis_vector_2 = basis_details.get('basis_2')
        rotation_mode = basis_details.get('rotation_mode', 'linear')
        if basis_vector_1 is None or basis_vector_2 is None: print("Error: basis_1 or basis_2 missing."); return {}
        norm_basis_1 = normalise(basis_vector_1, axis=0); norm_basis_2 = normalise(basis_vector_2, axis=0)
        dot_prod = np.abs(np.dot(norm_basis_1, norm_basis_2));
        if dot_prod > 0.999: print(f"Warning: Basis vectors nearly collinear (dot product: {dot_prod:.4f}).")

        print(f"Running Single Plane SRM (Rotation: {rotation_mode})...")
        group_iterator = tqdm(grouped_vectors.items(), desc=f"SRM for groups", leave=False)
        for group_name, vector_list in group_iterator:
             if not vector_list: print(f"Skipping empty group '{group_name}'."); continue
             data_vectors = np.array(vector_list)
             if data_vectors.ndim != 2 or data_vectors.shape[1] != DIMENSION: print(f"Skipping group '{group_name}': Invalid vector shape."); continue
             N, D = data_vectors.shape
             normalized_data_vectors = normalise(data_vectors, axis=1)
             results_list = []

             if rotation_mode == 'linear':
                 for i, angle_rad in enumerate(angles_rad):
                     spotlight_vec = np.cos(angle_rad) * norm_basis_1 + np.sin(angle_rad) * norm_basis_2
                     norm_spotlight = np.linalg.norm(spotlight_vec)
                     if norm_spotlight < 1e-9: similarities = np.zeros(N)
                     else: normalized_spotlight_vec = spotlight_vec / norm_spotlight; similarities = normalized_data_vectors @ normalized_spotlight_vec
                     angle_results = {"angle_deg": angles_deg[i], "mean_similarity": np.mean(similarities)}
                     for thresh in thresholds:
                         positive_count = np.sum(similarities >= thresh); angle_results[f"count_thresh_{thresh}"] = positive_count
                         if signed: negative_count = np.sum(similarities <= -thresh); angle_results[f"signed_count_thresh_{thresh}"] = positive_count - negative_count
                     results_list.append(angle_results)

             elif rotation_mode == 'matrix':
                 try:
                     rotation_generator = vectors_to_bivectors(norm_basis_1, norm_basis_2)
                     rotation_matrices = generate_special_orthogonal_matrices(rotation_generator, angles_rad, debug_log_file=debug_log_file)
                     probe_vector = norm_basis_1
                     rotated_probes = np.einsum('aji,j->ai', rotation_matrices, probe_vector, optimize='optimal')
                     all_similarities = np.einsum("bi,ai->ba", normalized_data_vectors, rotated_probes, optimize='optimal')
                     if debug_log_file: debug_log_file.write(f"[DEBUG] Matrix Mode Similarities shape for group {group_name}: {all_similarities.shape}\n")
                     for a in range(num_angles):
                         similarities_at_angle = all_similarities[:, a]
                         angle_results = {"angle_deg": angles_deg[a], "mean_similarity": np.mean(similarities_at_angle)}
                         for thresh in thresholds:
                             positive_count = np.sum(similarities_at_angle >= thresh); angle_results[f"count_thresh_{thresh}"] = positive_count
                             if signed: negative_count = np.sum(similarities_at_angle <= -thresh); angle_results[f"signed_count_thresh_{thresh}"] = positive_count - negative_count
                         results_list.append(angle_results)
                 except np.linalg.LinAlgError as e: print(f"\nError during matrix rotation for group {group_name} (LinAlgError): {e}. Skipping group."); results_list = None
                 except Exception as e: print(f"\nUnexpected error during matrix rotation for group {group_name}: {e}"); traceback.print_exc(); results_list = None

             if results_list: results_by_group[group_name] = pd.DataFrame(results_list)

    elif analysis_mode == 'ensemble':
        ensemble_basis = basis_details.get('ensemble_basis'); ensemble_labels = basis_details.get('ensemble_labels')
        if ensemble_basis is None: print("Error: ensemble_basis missing."); return {}
        m, d = ensemble_basis.shape
        if d != DIMENSION: print(f"Error: Ensemble basis dimension ({d}) != expected dimension ({DIMENSION})."); return {}
        if m < 2: print(f"Error: Ensemble basis must contain at least 2 vectors (found {m})."); return {}
        ensemble_basis = normalise(ensemble_basis, axis=1)

        basis_indices = list(range(m))
        if ensemble_plane_selection == 'comb': plane_index_pairs = list(itertools.combinations(basis_indices, 2))
        else: plane_index_pairs = list(itertools.permutations(basis_indices, 2))
        num_total_planes = len(plane_index_pairs); print(f"Generated {num_total_planes} total plane index pairs using '{ensemble_plane_selection}'.")
        if ensemble_max_planes is not None and ensemble_max_planes > 0 and ensemble_max_planes < num_total_planes:
            print(f"Sampling {ensemble_max_planes} random planes from {num_total_planes}."); random.seed(42); plane_index_pairs = random.sample(plane_index_pairs, ensemble_max_planes)
        num_planes_to_run = len(plane_index_pairs); print(f"Analyzing {num_planes_to_run} planes.")

        print("Running Ensemble SRM (Rotation: matrix)...")
        group_iterator_outer = tqdm(grouped_vectors.items(), desc=f"Groups", leave=True)
        aggregated_results_by_group = {}
        for group_name, vector_list in group_iterator_outer:
            group_iterator_outer.set_postfix_str(f"Group: {group_name}")
            if not vector_list: print(f"\nSkipping empty group '{group_name}'."); continue
            data_vectors = np.array(vector_list)
            if data_vectors.ndim != 2 or data_vectors.shape[1] != DIMENSION: print(f"\nSkipping group '{group_name}': Invalid vector shape."); continue
            normalized_data_vectors = normalise(data_vectors, axis=1); N = data_vectors.shape[0]

            results_for_this_group_all_planes = []
            plane_iterator_inner = tqdm(plane_index_pairs, desc=f"Planes", leave=False, total=num_planes_to_run)
            for i, (idx1, idx2) in enumerate(plane_iterator_inner):
                norm_basis_1 = ensemble_basis[idx1]; norm_basis_2 = ensemble_basis[idx2]
                plane_id_str = f"{idx1}-{idx2}"
                if ensemble_labels is not None and idx1 < len(ensemble_labels) and idx2 < len(ensemble_labels):
                    try: plane_id_str = f"{ensemble_labels[idx1]}-{ensemble_labels[idx2]}"
                    except IndexError: pass
                plane_iterator_inner.set_postfix_str(f"Plane: {plane_id_str}")
                plane_results_list = []
                try:
                    rotation_generator = vectors_to_bivectors(norm_basis_1, norm_basis_2)
                    rotation_matrices = generate_special_orthogonal_matrices(rotation_generator, angles_rad, debug_log_file=debug_log_file)
                    probe_vector = norm_basis_1
                    rotated_probes = np.einsum('aji,j->ai', rotation_matrices, probe_vector, optimize='optimal')
                    all_similarities = np.einsum("bi,ai->ba", normalized_data_vectors, rotated_probes, optimize='optimal')
                    for a in range(num_angles):
                        similarities_at_angle = all_similarities[:, a]
                        angle_results = {"angle_deg": angles_deg[a], "mean_similarity": np.mean(similarities_at_angle)}
                        for thresh in thresholds:
                            positive_count = np.sum(similarities_at_angle >= thresh); angle_results[f"count_thresh_{thresh}"] = positive_count
                            if signed: negative_count = np.sum(similarities_at_angle <= -thresh); angle_results[f"signed_count_thresh_{thresh}"] = positive_count - negative_count
                        plane_results_list.append(angle_results)
                except np.linalg.LinAlgError as e: print(f"\nLinAlgError for group '{group_name}' plane {plane_id_str}: {e}. Skipping plane."); plane_results_list = None
                except Exception as e: print(f"\nError for group '{group_name}' plane {plane_id_str}: {e}. Skipping plane."); traceback.print_exc(); plane_results_list = None
                if plane_results_list: plane_df = pd.DataFrame(plane_results_list); plane_df['plane_label'] = plane_id_str; results_for_this_group_all_planes.append(plane_df)

            if results_for_this_group_all_planes:
                group_ensemble_df = pd.concat(results_for_this_group_all_planes, ignore_index=True)
                agg_cols = ['mean_similarity'] + [f"count_thresh_{t}" for t in thresholds];
                if signed: agg_cols.extend([f"signed_count_thresh_{t}" for t in thresholds])
                agg_cols_present = [col for col in agg_cols if col in group_ensemble_df.columns]
                if agg_cols_present: aggregated_df = group_ensemble_df.groupby('angle_deg')[agg_cols_present].mean().reset_index(); aggregated_results_by_group[group_name] = aggregated_df
                else: print(f"\nWarning: No columns to aggregate for group '{group_name}'. Skipping aggregation.")
            else: print(f"\nWarning: SRM failed for ALL planes for group '{group_name}'. Skipping group.")
        results_by_group = aggregated_results_by_group

    return results_by_group


# === Plotting ===

def plot_srm_results_grouped(
    grouped_results_dfs: dict[str, pd.DataFrame],
    group_by_key: str | None,
    basis_id_str: str,
    analysis_mode: str,
    rotation_mode: str | None,
    signed_mode: bool,
    save_dir: Path,
    plot_threshold: float | None = None,
    self_srm_df: pd.DataFrame | None = None,
    ):
    """ Generates and saves a plot for grouped SRM results. """
    if not grouped_results_dfs: print("No grouped results provided to plot."); return

    fig, ax1 = plt.subplots(figsize=(14, 7)); num_groups = len(grouped_results_dfs)
    colors = cm.viridis(np.linspace(0, 0.95, num_groups)) if num_groups <= 10 else cm.tab20(np.linspace(0, 1, num_groups))
    group_by_title = str(group_by_key).replace('_',' ').title() if group_by_key else "All Vectors"
    mode_title = f"Mode: {rotation_mode.capitalize()}" if analysis_mode == 'single_plane' else "Mode: Ensemble (Matrix)"
    signed_title_comp = " (Signed)" if signed_mode and plot_threshold is not None else ""
    ensemble_title_comp = " (Ensemble Avg)" if analysis_mode == 'ensemble' else ""
    ax1.set_xlabel('Angle (degrees)'); ax1.grid(True, axis='x', linestyle=':')

    ax2 = ax1; count_lines = []; count_col = None
    if plot_threshold is not None:
        count_col_prefix = "signed_count" if signed_mode else "count"
        count_col = f"{count_col_prefix}_thresh_{plot_threshold}"
        first_df = next(iter(grouped_results_dfs.values()), pd.DataFrame())
        if count_col in first_df.columns:
            ax2 = ax1.twinx(); count_label = f"Signed Count (Thr: {plot_threshold})" if signed_mode else f"Count (Sim >= {plot_threshold})"
            count_legend_title = f"Threshold {plot_threshold}{signed_title_comp}"; plot_type_title = f"({count_label})"
            ax1.set_ylabel(count_label)
            sorted_items = sorted(grouped_results_dfs.items(), key=lambda item: str(item[0]))
            for i, (group_name, df) in enumerate(sorted_items):
                if count_col in df.columns:
                    line, = ax1.plot(df['angle_deg'], df[count_col], color=colors[i % len(colors)], marker='.', markersize=3, linestyle=':', label=f'{group_name}')
                    count_lines.append(line)
            if count_lines: ax1.legend(handles=count_lines, loc='upper left', title=count_legend_title, fontsize='small')
            ax1.tick_params(axis='y')
        else:
            print(f"Warning: Count column '{count_col}' for threshold {plot_threshold} not found. Plotting mean similarity only.")
            plot_threshold = None; plot_type_title = "(Mean Similarity Only)"; ax1.set_ylabel('Mean Cosine Similarity')
    else: plot_type_title = "(Mean Similarity Only)"; ax1.set_ylabel('Mean Cosine Similarity'); ax1.tick_params(axis='y')

    ax2.set_ylabel('Mean Cosine Similarity', color='black' if plot_threshold is None else 'tab:grey')
    mean_lines = []
    sorted_items = sorted(grouped_results_dfs.items(), key=lambda item: str(item[0]))
    for i, (group_name, df) in enumerate(sorted_items):
         if 'mean_similarity' in df.columns:
             line, = ax2.plot(df['angle_deg'], df['mean_similarity'], color=colors[i % len(colors)], marker=None, linestyle='-', linewidth=2, label=f'{group_name}')
             mean_lines.append(line)

    if analysis_mode == 'single_plane' and self_srm_df is not None and not self_srm_df.empty:
        if 'mean_similarity' in self_srm_df.columns and 'angle_deg' in self_srm_df.columns:
             line, = ax2.plot(self_srm_df['angle_deg'], self_srm_df['mean_similarity'], color='black', linestyle='--', linewidth=1.5, label='Self-SRM Ref.')
             mean_lines.append(line)

    if mean_lines:
        loc = 'center left' if plot_threshold is not None else ('center left' if (self_srm_df is not None and analysis_mode == 'single_plane') else 'upper left')
        legend_title = f"Mean Similarity ({group_by_title})"
        ax2.legend(handles=mean_lines, loc=loc, title=legend_title, fontsize='small')
    ax2.tick_params(axis='y', labelcolor='black' if plot_threshold is None else 'tab:grey')

    fig.suptitle(f'Grouped SRM Sweep Analysis: {group_by_title}', fontsize=16, y=1.02)
    ax1.set_title(f'Basis ({basis_id_str}){ensemble_title_comp}, {mode_title}{signed_title_comp} {plot_type_title}', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); plt.xticks(np.arange(0, 361, 45))

    # --- Save Plot ---
    # --- DRASTICALLY SIMPLIFIED FILENAME FOR DEBUGGING ---
    thresh_tag = f"thresh{plot_threshold}" if plot_threshold is not None else "meansim"
    # Use a very short base name + threshold tag ONLY
    plot_base_filename = f"srm_plot_{thresh_tag}"
    # Remove the extra sanitization of the base filename
    # plot_base_filename = sanitize_label(plot_base_filename) # REMOVED this line
    # --- END SIMPLIFIED FILENAME ---

    plot_filename = save_dir / (plot_base_filename + ".png") # save_dir should be analysis_plot_dir

    try:
        # --- Ensure directory exists AGAIN right before saving ---
        if not save_dir.exists():
             print(f"Plot save directory '{save_dir}' does not exist. Attempting creation...")
             save_dir.mkdir(parents=True, exist_ok=True)
        elif not save_dir.is_dir():
             print(f"Error: Expected plot save directory '{save_dir}' exists but is not a directory.")
             raise NotADirectoryError(f"Path exists but is not a directory: {save_dir}")

        # --- Now try saving, converting Path to string ---
        plot_filename_str = str(plot_filename.resolve()) # Convert to absolute string path
        print(f"Attempting to save plot to: {plot_filename_str}") # Log the string path
        plt.savefig(plot_filename_str) # <--- Pass the string path
        print(f"Saved grouped SRM plot: {plot_filename.name}") # Log original Path object's name
    except Exception as e:
        print(f"Error saving plot {plot_filename.name}: {e}")
        traceback.print_exc() # Print full traceback for this error
    finally:
        plt.close(fig) # Close the figure to free memory

# --- END OF FILE utils.py ---