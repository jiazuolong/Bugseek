# MLPHO
## Phonon Calculator and Phase Diagram Computation

This repository provides a set of tools for computing phonon properties and phase diagrams using various force field calculators and the Phonopy and Phono3py tools. It includes the following steps:

1. **Structural Optimization**: The structure can be optimized using either the `MatterSim` or `MACE` calculator.
2. **Force Constant Generation**: Both second-order (`fc2`) and third-order (`fc3`) force constants are computed.
3. **Phonon Spectrum**: The phonon band plot is generated using the `sumo` tool.
4. **Thermal Properties**: The thermal properties (e.g., thermal conductivity) are computed using Phono3py.
5. **Phase Diagram**: If requested, the phase diagram for the given structure is computed using the Materials Project API.

## Prerequisites

- Python 3.7 or higher
- Required Python packages:
  - `ase`
  - `phonopy`
  - `phono3py`
  - `matplotlib`
  - `numpy`
  - `click`
  - `pymatgen`
  - `plotly`
  - `h5py`
  
  You can install these dependencies using `pip`:

  ```bash
  pip install -r pyproject.toml
  ```
## Installation
  1.Clone the repository:
  ```
  git clone https://github.com/your-username/phonon-calculator.git
  cd phonon-calculator
  ```
  2.Install the necessary Python dependencies:
  ```
  pip install -r requirements.txt
  ```
## Usage
  You can run the main program via the command line using the following structure:
```
python main.py --supercell 3 3 3 --strain 1.0 --calculator mattersim --mesh 19 19 19 --do_compute_phase_diagram --mpr_api_key YOUR_MATERIALS_PROJECT_API_KEY
```


