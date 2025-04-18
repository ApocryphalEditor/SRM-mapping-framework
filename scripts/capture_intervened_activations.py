# --- START OF FILE capture_intervened_activations.py ---

import torch
import argparse
import warnings
import sys
import traceback
import datetime
import re
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Local application imports
from transformer_lens import HookedTransformer, utils as tl_utils # Use alias
import utils # Import our new utility functions

# --- Constants ---
DEFAULT_RUN_PREFIX = "run"

# --- Helper Functions ---

def parse_sweep_values(value_str: str) -> list[float | int | None] | None:
    """ Parses a comma-separated string of numbers and 'None' into a list. """
    if not value_str:
        return None
    values = []
    for item in value_str.split(','):
        item = item.strip()
        if item.lower() == 'none':
            values.append(None)
        else:
            try:
                val = int(item)
                values.append(val)
            except ValueError:
                try:
                    val = float(item)
                    values.append(val)
                except ValueError:
                    print(f"Warning: Could not parse sweep value '{item}'. Skipping.")
    if not values:
        return None
    return values

def parse_structured_prompts(filepath: Path) -> list[dict] | None:
    """Parses the structured prompt file based on the template format."""
    prompts_data = []
    current_core_id = None; current_level = None
    level_pattern = re.compile(r"\[LEVEL (\d+)\]"); core_id_pattern = re.compile(r">> CORE_ID:\s*(.*)")
    type_pattern = re.compile(r"^\s*([a-zA-Z_]+):\s*(.*)")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('---') or line.startswith(">> PROPOSITION:"): continue
                core_id_match = core_id_pattern.match(line)
                if core_id_match: current_core_id = core_id_match.group(1).strip().replace(" ", "_"); current_level = None; continue
                level_match = level_pattern.match(line)
                if level_match: current_level = int(level_match.group(1)); continue
                type_match = type_pattern.match(line)
                if type_match:
                    prompt_type = type_match.group(1).lower(); prompt_text = type_match.group(2).strip()
                    if current_core_id and current_level is not None: prompts_data.append({'prompt_text': prompt_text, 'core_id': current_core_id, 'type': prompt_type, 'level': current_level})
                    else: print(f"Warning: Found prompt '{prompt_text}' on line {line_num} but CORE_ID or LEVEL not set. Skipping."); continue
    except FileNotFoundError: print(f"Error: Prompt file not found at '{filepath}'"); return None
    except Exception as e: print(f"Error parsing prompt file '{filepath}': {e}"); traceback.print_exc(); return None
    if not prompts_data: print(f"Warning: No valid prompts parsed from '{filepath}'. Check format."); return None
    print(f"Successfully parsed {len(prompts_data)} prompts from {filepath.name}.")
    return prompts_data

# --- Main Script Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture MLP activations for structured prompts with neuron interventions. Saves results in a structured experiment run directory.")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the structured prompt file (e.g., promptsets/epistemic_certainty_grid.txt).")
    parser.add_argument("--experiment_base_dir", type=str, required=True, help="Path to the base directory where the unique run directory will be created (e.g., ./experiments).")
    parser.add_argument("--run_prefix", type=str, default=DEFAULT_RUN_PREFIX, help="Prefix for the timestamped run directory name (e.g., 'run').")
    parser.add_argument("--generate_length", type=int, default=50, help="Number of new tokens to generate.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling parameter. If None or 0, use greedy decoding.")
    parser.add_argument("--layer", type=int, required=True, help="MLP layer index for intervention and capture.")
    parser.add_argument("--target_neuron", type=int, required=True, help="Index of the neuron to intervene on in the MLP layer.")
    parser.add_argument("--sweep_values", type=str, required=True, help="Comma-separated list of values to clamp the target neuron to (e.g., 'None,10,-10'). 'None' performs a baseline run within the sweep.")
    args = parser.parse_args()

    intervention_values = parse_sweep_values(args.sweep_values)
    if not intervention_values:
        print("Error: No valid sweep values provided or parsed from --sweep_values argument. Exiting.")
        sys.exit(1)
    print(f"Intervention sweep values: {intervention_values}")

    # --- Setup Paths using pathlib and utils constants ---
    base_dir = Path(args.experiment_base_dir)
    prompt_path = Path(args.prompt_file)
    if not prompt_path.is_file():
         print(f"Error: Prompt file not found at {prompt_path}"); sys.exit(1)
    prompt_basename = prompt_path.stem
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir_name = f"{args.run_prefix}_intervened_L{args.layer}N{args.target_neuron}_{prompt_basename}_{timestamp}"
    run_path = base_dir / run_dir_name

    capture_path = run_path / utils.CAPTURE_SUBFOLDER
    vector_dir = capture_path / utils.VECTORS_SUBFOLDER
    log_dir = capture_path / utils.LOGS_SUBFOLDER
    metadata_dir = capture_path / utils.METADATA_SUBFOLDER

    try:
        metadata_dir.mkdir(parents=True, exist_ok=True)
        vector_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)
        print(f"Created unique run directory structure: {run_path}")
        print(f" -> Capture outputs will be saved in: {capture_path}")
    except OSError as e:
        print(f"Error creating output directories in '{run_path}': {e}", file=sys.stderr)
        traceback.print_exc(); sys.exit(1)

    vector_filename = f"captured_vectors_intervened.npz"
    log_filename = f"run_log_intervened.md"
    metadata_filename = f"run_metadata_intervened.json"
    vector_path = vector_dir / vector_filename
    log_path = log_dir / log_filename
    metadata_path = metadata_dir / metadata_filename
    # --- End Path Setup ---

    # --- Parse Prompts ---
    parsed_prompts = parse_structured_prompts(prompt_path)
    if not parsed_prompts: print("Exiting due to prompt parsing failure."); sys.exit(1)

    # --- Load Model ---
    print("\nLoading GPT-2 Small model...")
    try:
        model = HookedTransformer.from_pretrained("gpt2") # Removed Mlp=None argument
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device); model.eval()
        print(f"Using device: {device}")
        tokenizer = model.tokenizer
        if tokenizer.pad_token is None:
            print("Tokenizer does not have a pad token; setting to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        TARGET_HOOK_POINT = tl_utils.get_act_name("post", args.layer)
        capture_hook_point_name = TARGET_HOOK_POINT
        DIMENSION = model.cfg.d_mlp

        if not (0 <= args.target_neuron < DIMENSION):
            print(f"Error: Target neuron index {args.target_neuron} is out of bounds for MLP dimension {DIMENSION}.")
            sys.exit(1)
        print(f"Targeting layer {args.layer}, neuron {args.target_neuron} (hook: '{TARGET_HOOK_POINT}', dim: {DIMENSION})")
    except Exception as e: print(f"Error loading model: {e}", file=sys.stderr); traceback.print_exc(); sys.exit(1)

    # --- Prepare Metadata ---
    run_metadata = {
        "script_name": Path(__file__).name,
        "run_type": "intervention_capture",
        "model_name": "gpt2",
        "target_layer": args.layer,
        "target_neuron": args.target_neuron,
        "intervention_hook": TARGET_HOOK_POINT,
        "capture_hook": capture_hook_point_name,
        "activation_dimension": DIMENSION,
        "sweep_values": intervention_values,
        "prompt_file_path": str(prompt_path),
        "prompt_file_relative": str(prompt_path.relative_to(base_dir)) if prompt_path.is_relative_to(base_dir) else str(prompt_path),
        "prompt_file_basename": prompt_basename,
        "num_prompts_parsed": len(parsed_prompts),
        "generate_length": args.generate_length,
        "top_k_setting": args.top_k if args.top_k is not None and args.top_k > 0 else "greedy",
        "run_timestamp": timestamp,
        "run_directory": str(run_path),
        "run_directory_name": run_path.name,
        "capture_directory_relative": str(capture_path.relative_to(run_path)),
        "output_vector_file_relative": str(vector_path.relative_to(run_path)),
        "output_log_file_relative": str(log_path.relative_to(run_path)),
        "output_metadata_file_relative": str(metadata_path.relative_to(run_path)),
        "device": device,
        "cli_args": vars(args),
    }
    if utils.save_json_metadata(metadata_path, run_metadata):
         print(f"Initial metadata saved to: {metadata_path}")
    else:
         print(f"Warning: Could not save initial metadata file '{metadata_path}'. Continuing run.")

    # --- Main Processing ---
    print(f"\nStarting intervention runs for {len(parsed_prompts)} prompts across {len(intervention_values)} sweep values.")
    all_vectors = {}
    total_runs = len(parsed_prompts) * len(intervention_values)
    processed_prompts_count = 0
    processed_sweeps_count = 0
    skipped_capture_count = 0

    try:
        with open(log_path, 'w', encoding='utf-8') as logfile:
            logfile.write(f"# Intervention Activation Capture Log: {prompt_basename} (L{args.layer}N{args.target_neuron}) ({timestamp})\n\n")
            logfile.write("## Run Parameters\n")
            for key, value in run_metadata.items(): logfile.write(f"- **{key.replace('_', ' ').title()}**: `{value}`\n")
            logfile.write("\n---\n\n## Prompt Processing\n\n"); logfile.flush()

            with torch.no_grad(), tqdm(total=total_runs, desc="Prompt Interventions", unit="run") as pbar:
                for i, prompt_info in enumerate(parsed_prompts):
                    prompt_text = prompt_info['prompt_text']; core_id = prompt_info['core_id']
                    prompt_type = prompt_info['type']; level = prompt_info['level']
                    prompt_base_key = f"core_id={core_id}_type={prompt_type}_level={level}"
                    logfile.write(f"### Prompt {i+1}/{len(parsed_prompts)}: {prompt_base_key}\n")

                    prompt_sweep_success_flag = True

                    for sweep_idx, sweep_value in enumerate(intervention_values):

                        def intervention_hook(activation, hook):
                            if sweep_value is None:
                                return activation
                            if activation.ndim == 3:
                                activation[:, :, args.target_neuron] = sweep_value
                            elif activation.ndim == 2:
                                activation[:, args.target_neuron] = sweep_value
                            else:
                                warnings.warn(f"Unexpected activation shape in intervention hook: {activation.shape}. Neuron not clamped.")
                            return activation

                        sweep_tag = f"sweep={sweep_value if sweep_value is not None else 'baseline'}"
                        vector_key = f"{prompt_base_key}_{sweep_tag}"
                        pbar.set_description(f"{core_id[:10]}.. L{level} Swp:{sweep_value}")
                        logfile.write(f"\n#### Intervention: {sweep_tag}\n"); logfile.write(f"- **Full Key:** `{vector_key}`\n")

                        try:
                            input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
                            input_len = input_ids.shape[1]
                            if input_len == 0:
                                logfile.write("- **Result:** Error - Empty prompt after tokenization.\n\n")
                                logfile.flush(); pbar.update(1); skipped_capture_count += 1; prompt_sweep_success_flag = False; continue

                            # --- Generate Text (WITH intervention hook) ---
                            gen_hooks = [(TARGET_HOOK_POINT, intervention_hook)]
                            with model.hooks(fwd_hooks=gen_hooks):
                                output_ids = model.generate(
                                     input_ids,
                                     max_new_tokens=args.generate_length,
                                     do_sample=(args.top_k is not None and args.top_k > 0),
                                     top_k=args.top_k if (args.top_k is not None and args.top_k > 0) else None,
                                     eos_token_id=tokenizer.eos_token_id
                                     # pad_token_id removed
                                     )
                            # --- End Generate Text ---

                            generated_len = output_ids.shape[1] - input_len
                            if generated_len > 0:
                                 result_text = tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True)
                                 logfile.write(f"- **Generated Text:**\n```\n{result_text}\n```\n")
                            else:
                                logfile.write("- **Generated Text:** (No new tokens)\n- **Result:** Skipped capture.\n\n")
                                logfile.flush(); pbar.update(1); skipped_capture_count += 1; prompt_sweep_success_flag = False; continue

                            # --- Capture Activations (Re-run forward pass with BOTH hooks) ---
                            capture_container = [None]
                            def save_hook(activation_tensor, hook): capture_container[0] = activation_tensor.clone().detach().cpu()

                            fwd_pass_hooks = [(TARGET_HOOK_POINT, intervention_hook), (capture_hook_point_name, save_hook)]
                            try:
                                with model.hooks(fwd_hooks=fwd_pass_hooks):
                                    _ = model(output_ids, return_type=None)
                            except Exception as e_inner:
                                logfile.write(f"- **Result:** Error during capture forward pass: {e_inner}\n\n")
                                traceback.print_exc(file=sys.stderr); logfile.flush(); pbar.update(1); skipped_capture_count += 1; prompt_sweep_success_flag = False; continue

                            # --- Process and Store Vector ---
                            captured_mlp_post_activation = capture_container[0]
                            if captured_mlp_post_activation is not None:
                                generated_vectors_tensor = captured_mlp_post_activation[:, input_len:, :]
                                if generated_vectors_tensor.shape[1] > 0:
                                    mean_vector_np = np.mean(generated_vectors_tensor.squeeze(0).numpy(), axis=0)
                                    if mean_vector_np.shape == (DIMENSION,):
                                        all_vectors[vector_key] = mean_vector_np
                                        logfile.write(f"- **Result:** Vector captured successfully (shape: {mean_vector_np.shape}).\n\n")
                                        processed_sweeps_count += 1
                                    else:
                                        logfile.write(f"- **Result:** Error - Mean vector shape mismatch ({mean_vector_np.shape}). Expected ({DIMENSION},).\n\n")
                                        skipped_capture_count += 1; prompt_sweep_success_flag = False
                                else:
                                    logfile.write("- **Result:** Warning - Sliced activation tensor for generated tokens had 0 length.\n\n")
                                    skipped_capture_count += 1; prompt_sweep_success_flag = False
                            else:
                                logfile.write("- **Result:** Error - Failed to capture activation (container[0] is None).\n\n")
                                skipped_capture_count += 1; prompt_sweep_success_flag = False

                        except Exception as e:
                            logfile.write(f"- **Result:** ERROR processing this prompt/sweep: {str(e)}\n\n")
                            print(f"\n--- ERROR processing {vector_key} ---"); traceback.print_exc(); print(f"--- END ERROR ---")
                            skipped_capture_count += 1
                            prompt_sweep_success_flag = False
                        finally:
                             logfile.flush(); pbar.update(1)

                    if prompt_sweep_success_flag:
                        processed_prompts_count += 1
                    logfile.write("\n---\n"); logfile.flush()
            logfile.write("\nRun Complete.\n")

    except Exception as e:
        print(f"\n--- FATAL ERROR during main processing loop ---"); traceback.print_exc()
        if 'logfile' in locals() and not logfile.closed:
             try: logfile.write(f"\n\nFATAL ERROR occurred: {e}\n"); traceback.print_exc(file=logfile)
             except Exception: pass

    # --- Final Saving ---
    final_vector_count = len(all_vectors)
    run_metadata["final_vector_count"] = final_vector_count
    run_metadata["prompts_completed_all_sweeps"] = processed_prompts_count
    run_metadata["prompt_sweep_combinations_processed_successfully"] = processed_sweeps_count
    run_metadata["prompt_sweep_combinations_skipped_or_failed"] = skipped_capture_count

    if utils.save_json_metadata(metadata_path, run_metadata):
         print(f"\nFinal metadata updated and saved to: {metadata_path}")
    else:
         print(f"Warning: Failed to save final metadata update to {metadata_path}")

    if all_vectors:
        print(f"Saving {final_vector_count} collected mean vectors to {vector_path}...")
        try:
            np.savez_compressed(vector_path, **all_vectors, __metadata__=np.array(run_metadata, dtype=object))
            print("Vectors and embedded metadata saved successfully.")
        except Exception as e:
            print(f"\n--- ERROR saving final vectors ---"); traceback.print_exc()
            try:
                print("Attempting to save vectors without embedded metadata...")
                np.savez_compressed(vector_path, **all_vectors)
                print("Fallback save successful (vectors only).")
            except Exception as e_fallback:
                 print(f"Fallback vector save failed: {e_fallback}")
    else:
        print(f"\nNo vectors were successfully collected to save to {vector_path}.")

    print(f"\nScript finished. Results are in top-level directory: {run_path}")
    print(f"Capture outputs (vectors, logs, metadata) are within: {capture_path}")


# --- END OF FILE capture_intervened_activations.py ---