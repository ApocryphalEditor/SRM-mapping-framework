# --- START OF FILE analyze_srm_sweep.py ---

import argparse
import traceback
import datetime
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Local application imports
import utils # Import our new utility functions

# --- Constants ---
DEFAULT_EXPERIMENT_BASE_DIR = "experiments"
DEFAULT_ANALYSIS_LABEL_PREFIX = "srm_analysis"
VALID_ANALYSIS_MODES = ['single_plane', 'ensemble']
VALID_PLANE_SELECTIONS = ['comb', 'perm'] # For ensemble mode
VALID_GROUP_KEYS = utils.VALID_GROUP_KEYS # Use keys defined in utils
DEBUG_LOG_FILENAME = "srm_debug_log.txt"

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Run grouped SRM sweep analysis. Selects a run directory, loads vectors from "
                    f"'../{utils.CAPTURE_SUBFOLDER}/', finds/loads basis vectors (interactively if needed), performs analysis, "
                    f"and saves results to a new subfolder within '{utils.ANALYSES_SUBFOLDER}/'."
    )
    # --- Run Selection Args ---
    parser.add_argument("--experiment_base_dir", type=str, default=DEFAULT_EXPERIMENT_BASE_DIR, help="Base directory containing timestamped run folders.")

    # --- Analysis Config Args ---
    parser.add_argument("--analysis_label", type=str, default=None, help="[Optional] Custom name for the analysis subfolder within 'analyses/'. If None, a default name with mode and timestamp is generated.")
    parser.add_argument("--enable_debug_logging", action='store_true', help="If set, enables detailed debug logging to a file within the analysis metadata folder.")

    # --- Basis Specification Args ---
    parser.add_argument("--basis_file", type=str, default=None, help=f"[Optional] Path to a specific basis file (.npz). Can be absolute or relative. If not specified or not found, interactive search is triggered.")
    parser.add_argument("--basis_run_directory", type=str, default=None, help=f"[Optional] Path to the run directory containing the '{utils.BASIS_SUBFOLDER}/' subfolder for the basis file. Prioritized if provided when searching.")
    # Ensemble Specific (Only relevant if basis_file is ensemble type)
    parser.add_argument("--ensemble_basis_key", type=str, default='basis_vectors', help="[Ensemble] Key for basis array within the ensemble .npz file.")

    # --- SRM Execution Args ---
    parser.add_argument("--analysis_mode", type=str, choices=VALID_ANALYSIS_MODES, required=True, help="Analysis mode: 'single_plane' (requires basis file with basis_1, basis_2) or 'ensemble' (requires basis file with an array of vectors).")
    parser.add_argument("--rotation_mode", type=str, choices=['linear', 'matrix'], default='linear', help="[Single Plane] SRM rotation mode (linear or matrix). Ignored for ensemble mode (always matrix).")
    parser.add_argument("--thresholds", type=float, nargs='+', required=True, help="List of similarity thresholds (epsilon) for counting.")
    parser.add_argument("--num_angles", type=int, default=72, help="Number of angles for the SRM sweep (e.g., 72 for 5-degree steps).")
    parser.add_argument("--signed", action='store_true', help="Calculate signed resonance counts (positive counts - negative counts).")

    # Ensemble Specific Execution Args
    parser.add_argument("--plane_selection", type=str, choices=VALID_PLANE_SELECTIONS, default='comb', help="[Ensemble] Use combinations ('comb') or permutations ('perm') of basis vectors to generate planes.")
    parser.add_argument("--max_planes", type=int, default=None, help="[Ensemble] Maximum number of planes to sample randomly from all possible pairs. If None, use all generated planes.")

    # --- Grouping/Plotting/Saving Args ---
    parser.add_argument("--group_by", type=str, choices=VALID_GROUP_KEYS + [None], default=None, help="Metadata key from vector filename to group results by (e.g., 'type', 'level'). If None or omitted, analyzes all vectors together.")
    parser.add_argument("--plot_threshold", type=float, default=None, help="Single epsilon threshold value for plotting count/signed_count data. If None, only mean similarity is plotted.")
    parser.add_argument("--plot_all_thresholds", action='store_true', help="Generate separate plots for count data at ALL calculated thresholds, plus one for mean similarity. Overrides --plot_threshold if set.")
    parser.add_argument("--save_csv", action='store_true', help="Save detailed SRM results to CSV files within the analysis data folder.")

    args = parser.parse_args()

    # --- Input Validation ---
    if args.analysis_mode == 'single_plane' and args.rotation_mode == 'matrix':
         print("Info: Single plane analysis requested with matrix rotation.")
    elif args.analysis_mode == 'ensemble':
         args.rotation_mode = 'matrix'
         print("Info: Ensemble analysis mode selected. Using matrix rotation.")

    if args.plot_threshold is not None and args.plot_threshold not in args.thresholds:
        print(f"Warning: Specified --plot_threshold {args.plot_threshold} is not in the list of calculated --thresholds {args.thresholds}. Count plot for this specific threshold will not be generated unless --plot_all_thresholds is also set.")
    if args.plot_all_thresholds and args.plot_threshold is not None:
        print("Info: --plot_all_thresholds is set, ignoring specific --plot_threshold value.")

    # --- Select Experiment Run Directory (Interactive) ---
    base_dir = Path(args.experiment_base_dir)
    selected_run_dir = utils.select_experiment_folder(
        base_dir,
        prompt_text="Select the experiment run directory to analyze:"
    )
    if not selected_run_dir:
        print("No run directory selected. Exiting.")
        exit(1)
    print(f"Analyzing run directory: {selected_run_dir.name}")
    run_identifier = selected_run_dir.name

    # --- Define Analysis Output Structure ---
    analysis_timestamp = datetime.datetime.now()
    analysis_timestamp_str = analysis_timestamp.strftime("%Y%m%d_%H%M%S")
    if args.analysis_label:
        analysis_folder_name = utils.sanitize_label(args.analysis_label)
        print(f"Using provided analysis label: '{analysis_folder_name}'")
    else:
        group_tag = f"by_{args.group_by}" if args.group_by else "all"
        analysis_folder_name = f"{DEFAULT_ANALYSIS_LABEL_PREFIX}_{args.analysis_mode}_{group_tag}_{analysis_timestamp_str}"
        print(f"Using generated analysis label: '{analysis_folder_name}'")

    analyses_base_path = selected_run_dir / utils.ANALYSES_SUBFOLDER
    analysis_path = analyses_base_path / analysis_folder_name
    analysis_plot_dir = analysis_path / "plots"
    analysis_data_dir = analysis_path / "data"
    analysis_metadata_dir = analysis_path / "metadata"
    try:
        analysis_metadata_dir.mkdir(parents=True, exist_ok=True)
        analysis_plot_dir.mkdir(parents=True, exist_ok=True)
        analysis_data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Analysis outputs will be saved in: {analysis_path}")
    except OSError as e:
        print(f"Error creating analysis output directories in '{analysis_path}': {e}")
        traceback.print_exc(); exit(1)

    # --- Setup Debug Logging (if enabled) ---
    debug_log_file = None
    debug_log_path = None
    debug_log_file_rel_path = None
    if args.enable_debug_logging:
        debug_log_path = analysis_metadata_dir / DEBUG_LOG_FILENAME
        debug_log_file_rel_path = str(debug_log_path.relative_to(analysis_path))
        try:
            debug_log_file = open(debug_log_path, 'w', encoding='utf-8')
            print(f"Debug logging enabled. Writing details to: {debug_log_path}")
            # Log initial info...
        except Exception as e:
            print(f"Warning: Could not open debug log file '{debug_log_path}': {e}. Debug logging disabled.")
            debug_log_file = None
            debug_log_file_rel_path = None

    # --- Locate and Load Input Vectors ---
    input_vector_path = utils.find_vector_file(selected_run_dir)
    if not input_vector_path:
        print(f"Could not find required vector input file in {selected_run_dir}. Exiting.")
        if debug_log_file: debug_log_file.close()
        exit(1)
    print(f"Using Input Vectors: {input_vector_path.name}")
    structured_data, source_metadata = utils.load_vector_data(input_vector_path, expected_dim=utils.DIMENSION)
    if structured_data is None:
        print("Failed to load vectors. Cannot proceed.")
        if debug_log_file: debug_log_file.close()
        exit(1)
    if not structured_data:
         print("Warning: No valid vectors found in the loaded file. Exiting.")
         if debug_log_file: debug_log_file.close()
         exit(0)
    print(f"Total vectors loaded: {len(structured_data)}")

    # --- Resolve Basis File Path (REVISED LOGIC) ---
    print("\nResolving basis file path...")
    basis_path = None
    basis_source_info = "Not resolved"
    basis_run_dir_used = None # Track which run dir the basis came from

    # 1. Check explicit --basis_file (absolute or relative to CWD)
    if args.basis_file:
        basis_file_p = Path(args.basis_file)
        try:
            resolved_direct = basis_file_p.resolve(strict=True)
            basis_path = resolved_direct
            basis_source_info = f"direct_path_arg (--basis_file): {basis_path}"
            print(f"Found basis file via direct path: {basis_path}")
        except FileNotFoundError:
             # 1b. If direct failed, AND --basis_run_directory exists, try relative to it
             if args.basis_run_directory:
                 basis_run_p = Path(args.basis_run_directory)
                 if basis_run_p.is_dir():
                     basis_search_dir = basis_run_p / utils.BASIS_SUBFOLDER
                     # Use find_basis_file utility, providing the specific filename to look for
                     basis_path_rel = utils.find_basis_file(basis_search_dir, specific_filename=basis_file_p.name)
                     if basis_path_rel:
                          basis_path = basis_path_rel # Assign if found
                          basis_source_info = f"relative_in_basis_run_dir (--basis_file {args.basis_file} in --basis_run_directory {args.basis_run_directory})"
                          print(f"Found basis file via relative path in specified basis run dir: {basis_path}")
                          basis_run_dir_used = basis_run_p
                     else:
                          print(f"Info: Basis file '{args.basis_file}' not found in specified --basis_run_directory's '{utils.BASIS_SUBFOLDER}' folder: {basis_search_dir}")
                 else:
                     print(f"Warning: Specified --basis_run_directory not found: {args.basis_run_directory}")
             else:
                  print(f"Info: Specified --basis_file '{args.basis_file}' not found directly or relative to CWD.")
        except Exception as e:
             print(f"Warning: Error resolving --basis_file '{args.basis_file}': {e}")

    # 2. If not found via --basis_file, check --basis_run_directory directly (without specific filename)
    if basis_path is None and args.basis_run_directory:
        basis_run_p = Path(args.basis_run_directory)
        if basis_run_p.is_dir():
             print(f"Attempting to find basis file automatically in specified --basis_run_directory: {args.basis_run_directory}")
             basis_search_dir = basis_run_p / utils.BASIS_SUBFOLDER
             basis_path_found = utils.find_basis_file(basis_search_dir) # Interactive if multiple
             if basis_path_found:
                  basis_path = basis_path_found
                  basis_source_info = f"auto_search_in_basis_run_dir (--basis_run_directory {args.basis_run_directory})"
                  basis_run_dir_used = basis_run_p
             else:
                  print(f"Info: No basis files found in specified --basis_run_directory's '{utils.BASIS_SUBFOLDER}' folder: {basis_search_dir}")
        else:
             # This case handled earlier if --basis_file was also given, but handle if only --basis_run_dir given and invalid
             print(f"Warning: Specified --basis_run_directory not found: {args.basis_run_directory}")

    # 3. If still not found, check the *currently selected* run directory's basis folder
    if basis_path is None:
        print(f"No basis specified or found via arguments. Checking locally in '{selected_run_dir.name}/{utils.BASIS_SUBFOLDER}'...")
        local_basis_search_dir = selected_run_dir / utils.BASIS_SUBFOLDER
        basis_path_local = utils.find_basis_file(local_basis_search_dir) # Interactive if multiple
        if basis_path_local:
             basis_path = basis_path_local
             basis_source_info = "auto_search_local (in analyzed run dir)"
             basis_run_dir_used = selected_run_dir # Basis came from the run being analyzed
        else:
             print(f"Info: No basis files found locally in '{local_basis_search_dir.relative_to(selected_run_dir.parent)}'.")

    # 4. *** NEW *** If STILL not found, trigger interactive selection of BASELINE run
    if basis_path is None:
        print("\nBasis file not found locally.")
        print("Attempting interactive selection of source baseline run...")
        selected_baseline_run = utils.select_experiment_folder(
            base_dir,
            prompt_text="Please select the BASELINE run directory containing the required basis file:",
            pattern="run_baseline_*" # Filter for baseline runs
        )
        if selected_baseline_run:
            baseline_basis_search_dir = selected_baseline_run / utils.BASIS_SUBFOLDER
            print(f"Searching for basis files in selected baseline run: '{baseline_basis_search_dir.relative_to(selected_run_dir.parent)}'")
            basis_path_interactive = utils.find_basis_file(baseline_basis_search_dir) # Interactive if multiple
            if basis_path_interactive:
                 basis_path = basis_path_interactive
                 basis_source_info = f"auto_search_interactive_baseline ({selected_baseline_run.name})"
                 basis_run_dir_used = selected_baseline_run # Track the interactively selected baseline run
            else:
                 print(f"Info: No basis files found in the interactively selected baseline run's basis folder: {baseline_basis_search_dir}")
        else:
             print("Interactive selection of baseline run cancelled.")

    # 5. Final check if we have a basis path
    if not basis_path or not basis_path.is_file(): # Double check it resolved to a file
        print("\nError: Could not find or select a suitable basis file after all checks. Cannot proceed with analysis.")
        if debug_log_file: debug_log_file.close()
        exit(1)

    # --- Load Basis Vectors ---
    # (Load logic remains the same, using the successfully resolved basis_path)
    print(f"\nLoading basis vectors from: {basis_path}")
    basis_details = {'mode': args.analysis_mode, 'source_path': str(basis_path)}
    basis_id_str = f"{basis_path.stem}"
    try:
        basis_data = np.load(basis_path, allow_pickle=True)
        # ... (rest of loading logic for single_plane and ensemble modes is unchanged) ...
        if args.analysis_mode == 'single_plane':
            if 'basis_1' not in basis_data or 'basis_2' not in basis_data:
                parser.error(f"Keys 'basis_1', 'basis_2' missing in basis file '{basis_path}' for single_plane mode.")
            basis_1 = basis_data['basis_1']; basis_2 = basis_data['basis_2']
            if basis_1.shape != (utils.DIMENSION,) or basis_2.shape != (utils.DIMENSION,):
                parser.error(f"Basis vector shape mismatch in {basis_path}. Expected ({utils.DIMENSION},).")
            basis_details['basis_1'] = basis_1
            basis_details['basis_2'] = basis_2
            basis_details['rotation_mode'] = args.rotation_mode
            basis_id_str = f"SinglePlane-{basis_path.stem}"
            print(f"Loaded single plane basis (basis_1, basis_2) from {basis_path.name}")
        elif args.analysis_mode == 'ensemble':
            # ... (ensemble loading logic unchanged) ...
            ensemble_key = args.ensemble_basis_key
            if ensemble_key not in basis_data:
                parser.error(f"Ensemble key '{ensemble_key}' not found in basis file '{basis_path}'. Use --ensemble_basis_key if needed.")
            ensemble_basis = basis_data[ensemble_key]
            m, d = ensemble_basis.shape
            if d != utils.DIMENSION: parser.error(f"Ensemble basis dimension mismatch in {basis_path}. Got {d}, expected {utils.DIMENSION}.")
            if m < 2: parser.error(f"Ensemble basis needs at least 2 vectors, found {m} in {basis_path}.")
            basis_details['ensemble_basis'] = ensemble_basis
            basis_id_str = f"Ensemble-{basis_path.stem}-{m}vec"
            if 'group_labels' in basis_data:
                basis_details['ensemble_labels'] = basis_data['group_labels']
                if len(basis_details['ensemble_labels']) == m:
                    print(f"Loaded ensemble basis (key: '{ensemble_key}', shape: {ensemble_basis.shape}) with {m} labels from {basis_path.name}")
                else:
                    print(f"Warning: Number of labels ({len(basis_details['ensemble_labels'])}) in basis file does not match number of vectors ({m}). Ignoring labels.")
                    basis_details['ensemble_labels'] = None
                    print(f"Loaded ensemble basis (key: '{ensemble_key}', shape: {ensemble_basis.shape}) WITHOUT matching labels from {basis_path.name}")
            else:
                 basis_details['ensemble_labels'] = None
                 print(f"Loaded ensemble basis (key: '{ensemble_key}', shape: {ensemble_basis.shape}) from {basis_path.name}")

    except Exception as e:
        print(f"Error loading basis file {basis_path}: {e}"); traceback.print_exc()
        if debug_log_file: debug_log_file.close();
        exit(1)

    # --- Prepare and Run Analysis ---
    print(f"\nStarting SRM Analysis (Mode: {args.analysis_mode}, Basis: {basis_path.name})")
    # --- Prepare Metadata ---
    all_run_metadata = {
        "script_name": Path(__file__).name,
        "analysis_timestamp": analysis_timestamp.isoformat(),
        "analysis_label": analysis_folder_name,
        "analyzed_run_directory": str(selected_run_dir),
        "analyzed_run_directory_name": selected_run_dir.name,
        "input_vector_file_relative": str(input_vector_path.relative_to(selected_run_dir)),
        "input_vector_source_metadata": source_metadata,
        # UPDATED Basis Info
        "basis_source_description": basis_source_info, # Reflects how it was found
        "basis_file_path_resolved": str(basis_path),
        "basis_file_relative_to_analyzed_run": str(basis_path.relative_to(selected_run_dir)) if basis_path.is_relative_to(selected_run_dir) else "external or from other run",
        "basis_file_source_run_dir": str(basis_run_dir_used) if basis_run_dir_used else None, # Track actual source run
        "basis_id_string": basis_id_str,
        # (rest of metadata setup is unchanged)
        "analysis_mode": args.analysis_mode,
        "dimension": utils.DIMENSION,
        "num_angles": args.num_angles,
        "signed_mode_enabled": args.signed,
        "tested_thresholds": sorted(args.thresholds),
        "grouping_key": args.group_by if args.group_by else "all_vectors",
        "single_plane_params": None,
        "ensemble_params": None,
        "output_analysis_directory_relative": str(analysis_path.relative_to(selected_run_dir)),
        "output_plot_dir_relative": str(analysis_plot_dir.relative_to(analysis_path)),
        "output_data_dir_relative": str(analysis_data_dir.relative_to(analysis_path)),
        "output_metadata_dir_relative": str(analysis_metadata_dir.relative_to(analysis_path)),
        "debug_logging_enabled": args.enable_debug_logging,
        "debug_log_file_relative": debug_log_file_rel_path,
        "cli_args": vars(args),
        "analysis_results_summary": {},
    }
    if args.analysis_mode == 'single_plane':
         all_run_metadata["single_plane_params"] = {"rotation_mode": args.rotation_mode}
    else:
         all_run_metadata["ensemble_params"] = {
             "ensemble_basis_key": args.ensemble_basis_key,
             "plane_selection_method": args.plane_selection,
             "max_planes_setting": args.max_planes,
             "num_planes_analyzed": None
         }

    # --- Calculate Self-SRM (unchanged) ---
    self_srm_df = None
    if args.analysis_mode == 'single_plane':
        # ... (self-SRM calculation logic unchanged) ...
        print("\nCalculating Self-SRM reference curve...")
        try:
            self_srm_data = np.array([basis_details['basis_1']])
            self_srm_results = utils.perform_srm_analysis(
                structured_data=[{'key': 'self', 'key_components': {}, 'vector': basis_details['basis_1']}],
                basis_details=basis_details,
                group_by=None,
                signed=False,
                thresholds=[],
                num_angles=args.num_angles,
                debug_log_file=debug_log_file
            )
            if self_srm_results and 'all' in self_srm_results:
                 self_srm_df = self_srm_results['all']
                 print("Self-SRM reference calculated.")
            else: print("Warning: Failed to calculate Self-SRM reference line.")
        except Exception as e: print(f"Warning: Error calculating Self-SRM reference: {e}")


    # --- Execute SRM Analysis (call to utils unchanged) ---
    analysis_results_by_group = {}
    try:
        # ... (call to utils.perform_srm_analysis unchanged) ...
        analysis_results_by_group = utils.perform_srm_analysis(
            structured_data=structured_data,
            basis_details=basis_details,
            group_by=args.group_by,
            signed=args.signed,
            thresholds=args.thresholds,
            num_angles=args.num_angles,
            ensemble_max_planes=args.max_planes if args.analysis_mode == 'ensemble' else None,
            ensemble_plane_selection=args.plane_selection if args.analysis_mode == 'ensemble' else None,
            debug_log_file=debug_log_file
        )
        if not analysis_results_by_group:
             print("\nWarning: SRM analysis returned no results. Check logs and input data.")
        else:
             print(f"\nSRM analysis completed for {len(analysis_results_by_group)} group(s).")
             all_run_metadata["analysis_results_summary"]["groups_analyzed"] = sorted(list(analysis_results_by_group.keys()))
             all_run_metadata["analysis_results_summary"]["num_groups_successful"] = len(analysis_results_by_group)

    except Exception as e:
        # ... (error handling unchanged) ...
        print("\n--- FATAL ERROR during SRM analysis execution ---")
        traceback.print_exc()
        if debug_log_file:
            debug_log_file.write("\n--- FATAL ERROR during SRM analysis execution ---\n")
            traceback.print_exc(file=debug_log_file)
        all_run_metadata["analysis_results_summary"]["status"] = "Failed"
        all_run_metadata["analysis_results_summary"]["error"] = str(e)
        metadata_path = analysis_metadata_dir / "analysis_metadata_FAILED.json"
        utils.save_json_metadata(metadata_path, all_run_metadata)
        print(f"Attempted to save failure metadata to {metadata_path}")
        if debug_log_file: debug_log_file.close()
        exit(1)

    # --- Save CSV Data (unchanged) ---
    saved_csv_files = {}
    if args.save_csv and analysis_results_by_group:
        # ... (CSV saving logic unchanged) ...
        print("\nSaving detailed SRM results to CSV...")
        csv_base_filename = f"srm_data_{analysis_folder_name}"
        for group_name, df in analysis_results_by_group.items():
            safe_group_name = utils.sanitize_label(group_name)
            csv_filename = f"{csv_base_filename}_group_{safe_group_name}.csv"
            csv_path = analysis_data_dir / csv_filename
            try:
                df.to_csv(csv_path, index=False)
                print(f"Saved data for group '{group_name}' to: {csv_path.name}")
                saved_csv_files[group_name] = str(csv_path.relative_to(analysis_path))
            except Exception as e:
                print(f"Error saving data for group {group_name} to {csv_path.name}: {e}")
        all_run_metadata["analysis_results_summary"]["output_data_files_relative"] = saved_csv_files
    elif not analysis_results_by_group:
         print("\nSkipping CSV saving: No analysis results were generated.")


    # --- Generate Plots (call to utils unchanged) ---
    generated_plot_files = []
    if analysis_results_by_group:
        # ... (plotting logic unchanged, calls utils.plot_srm_results_grouped) ...
        print("\nGenerating plots...")
        if args.plot_all_thresholds:
            thresholds_to_plot = sorted(args.thresholds) + [None]
        elif args.plot_threshold is not None:
            thresholds_to_plot = [args.plot_threshold, None]
        else:
            thresholds_to_plot = [None]
        print(f"Plotting for thresholds: {thresholds_to_plot} (None = Mean Sim Only)")

        plot_configs_run = []
        for current_plot_thresh in thresholds_to_plot:
             is_redundant_mean_plot = (current_plot_thresh is None and
                                       len(analysis_results_by_group) == 1 and
                                       'all' in analysis_results_by_group and
                                       (self_srm_df is None or self_srm_df.empty))
             if is_redundant_mean_plot:
                 print("  Skipping mean similarity only plot for 'all_vectors' group as it's redundant without comparison or SelfSRM.")
                 continue

             try:
                 utils.plot_srm_results_grouped(
                     grouped_results_dfs=analysis_results_by_group,
                     group_by_key=args.group_by,
                     basis_id_str=basis_id_str,
                     analysis_mode=args.analysis_mode,
                     rotation_mode=args.rotation_mode if args.analysis_mode == 'single_plane' else None,
                     signed_mode=args.signed,
                     save_dir=analysis_plot_dir,
                     plot_threshold=current_plot_thresh,
                     self_srm_df=self_srm_df if args.analysis_mode == 'single_plane' else None
                 )
                 thresh_tag = f"_thresh{current_plot_thresh}" if current_plot_thresh is not None else "_mean_sim_only"
                 group_tag = f"_grouped_by_{args.group_by}" if args.group_by else "_all_vectors"
                 basis_tag = f"_basis_{utils.sanitize_label(basis_id_str)}"
                 mode_tag = f"_mode_{args.rotation_mode}" if args.analysis_mode == 'single_plane' else "_mode_ensemble"
                 signed_tag = "_signed" if args.signed and current_plot_thresh is not None else ""
                 plot_base_filename = f"srm_plot{group_tag}{basis_tag}{mode_tag}{signed_tag}{thresh_tag}"
                 plot_rel_path = Path(utils.PLOTS_SUBFOLDER) / f"{plot_base_filename}.png"

                 generated_plot_files.append(str(plot_rel_path))
                 plot_configs_run.append({"threshold": current_plot_thresh, "filename_relative": str(plot_rel_path)})

             except Exception as plot_e:
                 print(f"Error generating plot for threshold {current_plot_thresh}: {plot_e}")
                 traceback.print_exc()

        all_run_metadata["analysis_results_summary"]["output_plot_files_relative"] = generated_plot_files
        all_run_metadata["analysis_results_summary"]["plot_configurations"] = plot_configs_run
    else:
        print("\nSkipping plotting: No analysis results were generated.")

    # --- Finalize and Save Metadata (unchanged) ---
    # ... (final metadata saving logic unchanged) ...
    print(f"\nSaving final analysis metadata...")
    analysis_metadata_file_path = analysis_metadata_dir / "analysis_metadata.json"
    all_run_metadata["analysis_results_summary"]["status"] = "Completed"
    if utils.save_json_metadata(analysis_metadata_file_path, all_run_metadata):
        print(f"Analysis metadata saved successfully to: {analysis_metadata_file_path}")
    else:
        print(f"--- ERROR saving final analysis metadata ---")


    # --- Close Debug Log (unchanged) ---
    # ... (debug log closing logic unchanged) ...
    if debug_log_file:
        try:
            debug_log_file.close()
            print(f"Closed debug log file: {debug_log_path.name}")
        except Exception as e:
            print(f"Warning: Error closing debug log file: {e}")


    print(f"\nScript finished. Analysis results are in directory: {analysis_path}")


# --- END OF FILE analyze_srm_sweep.py ---