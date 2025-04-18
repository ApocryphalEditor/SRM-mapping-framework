
# SRM Resonance Mapper  
*A framework for mapping the internal geometry of transformer representations using angular projection, neuron-level modulation, and epistemically structured prompt sets.*

---

## 🌀 What Is This?

**SRM Resonance Mapper** is a fully modular Python framework for conducting geometric interpretability experiments in transformer models. It builds on the original [Spotlight Resonance Method (SRM)](https://github.com/GeorgeBird1/Spotlight-Resonance-Method) proposed by Bird, extending it into a flexible system for:

- Capturing and projecting internal activation vectors
- Constructing 2D basis planes from prompt-derived contrast sets
- Running angular sweeps to detect directional resonance
- Modulating individual neurons during generation
- Visualizing results as Compass Roses, drift maps, and more

This repo includes both baseline SRM workflows and **targeted experimental extensions** that support hypothesis-driven neuron interrogation.

---

## 🧠 Core Concepts

- **Spotlight Resonance**: Measures how many vectors fall within a directional "cone" as a probe vector sweeps around a plane.
- **Basis Planes**: Constructed from filtered prompt groups to define the semantic axes of projection.
- **SRM Analysis**: Projects all activation vectors into a basis plane and quantifies their angular alignment.
- **Intervention Capture**: Modifies specific neurons (e.g., clamps Neuron 373 to +10) during generation to observe representational shift.
- **Compass Rose Visualization**: Polar plots showing angular distribution of group alignments.

---

## 🛠 Features

- Interactive CLI for all core workflows  
- Modular file/folder structure (fully timestamped and traceable)  
- Dynamic prompt parsing with metadata embedding (`core_id`, `type`, `level`)  
- Single-plane or ensemble basis generation  
- Exportable `.csv` and `.png` outputs for all SRM runs  
- Fully documented in `/docs` with step-by-step walkthroughs

---

## 📦 Repository Structure

```
/scripts            → All capture, basis, and analysis scripts.
/promptsets         → Example structured prompt files
/docs               → Full PDF documentation of the framework
/examples           → Optional: Compass Roses and plots (to be added)
/experiments        → Your output folder after running any captures
```
---
## ⚠️ Path Handling Note
All scripts in this repository assume that they are run from the root directory of the repo, even if the script itself is located in /scripts.
For example, run scripts like this:
_python scripts/capture_baseline_activations.py --prompt_file promptsets/your_prompts.txt_
Avoid running them from inside /scripts directly unless you manually adjust relative paths. This ensures that file loading (e.g., promptsets/, experiments/, or docs/) works consistently across systems and collaborators.

---

## 📝 Documentation

You can find the complete v6 documentation, including installation, workflow steps, and theory notes here:

📄 [`/docs/SRM_v6_Documentation.pdf`](./docs/SRM_v6_Documentation.pdf)

---

## 📄 License

MIT License (see LICENSE file).  
SRM concept originally developed by [Bird](https://github.com/GeorgeBird1/Spotlight-Resonance-Method). This project extends the method with additional tools, structure, and visualization workflows.

---

## 🧭 Author

**Nick Blood**  
Experimental designer, interpretability naturalist, conceptual drifter.

---

## 📣 Feedback, Experiments, Forks

This is a working framework intended to be remixed, extended, and pushed into new territory. If you build something with it—or break it beautifully—please share. Open issues, fork, or reach out.
