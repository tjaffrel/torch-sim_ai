"""Calculate quasi-harmonic thermal properties for MOF5 using custom MACE model
with GPU memory-safe autobatching.
"""

# /// script
# dependencies = [
#     "mace-torch>=0.3.12",
#     "phonopy>=2.35",
#     "pymatviz==0.16",
#     "plotly!=6.2.0", # TODO remove pin pending https://github.com/plotly/plotly.py/issues/5253#issuecomment-3016615635
#     "kaleido>=0.2.1",  # Required for PDF export
#     "ase>=3.26",
# ]
# ///
import os

# Patch pynvml to fix cuEquivariance GPU power management errors (e.g., laptop on battery)
import pynvml
_original_nvml_get_power_limit = pynvml.nvmlDeviceGetPowerManagementLimit
_original_nvml_get_num_gpu_cores = getattr(pynvml, 'nvmlDeviceGetNumGpuCores', None)
_original_nvml_check_return = pynvml._nvmlCheckReturn

def _patched_nvml_get_power_limit(handle):
    try:
        return _original_nvml_get_power_limit(handle)
    except pynvml.NVMLError:
        return 0

def _patched_nvml_get_num_gpu_cores(handle):
    if _original_nvml_get_num_gpu_cores is None:
        return 0
    try:
        return _original_nvml_get_num_gpu_cores(handle)
    except pynvml.NVMLError:
        return 0

def _patched_nvml_check_return(ret):
    if ret == 8:  # NotSupported error code
        return
    return _original_nvml_check_return(ret)

pynvml.nvmlDeviceGetPowerManagementLimit = _patched_nvml_get_power_limit
if _original_nvml_get_num_gpu_cores is not None:
    pynvml.nvmlDeviceGetNumGpuCores = _patched_nvml_get_num_gpu_cores
pynvml._nvmlCheckReturn = _patched_nvml_check_return

import numpy as np
import plotly.graph_objects as go
import torch
from ase.io import read
from mace.calculators import MACECalculator
from phonopy import Phonopy
from phonopy.api_qha import PhonopyQHA
from phonopy.structure.atoms import PhonopyAtoms

import torch_sim as ts
from torch_sim.autobatching import BinningAutoBatcher, InFlightAutoBatcher
from torch_sim.models.interface import ModelInterface
from torch_sim.models.mace import MaceModel


def get_relaxed_structure(
    struct,
    model: ModelInterface,
    max_steps: int = 300,
    fmax: float = 1e-3,
    *,
    use_autobatcher: bool = False,
    autobatcher=None,
) -> ts.SimState:
    """Get relaxed structure.

    Args:
        struct: ASE structure
        model: MACE model
        max_steps: Maximum number of relaxation steps
        fmax: Force convergence criterion
        use_autobatcher: Whether to use automatic batching
        autobatcher: Autobatcher instance to use

    Returns:
        SimState: Relaxed structure
    """
    trajectory_file = "traj.h5"
    reporter = ts.TrajectoryReporter(
        trajectory_file,
        state_frequency=0,
        prop_calculators={
            1: {
                "potential_energy": lambda state: state.energy,
                "forces": lambda state: state.forces,
            },
        },
    )
    converge_max_force = ts.runners.generate_force_convergence_fn(force_tol=fmax)
    final_state = ts.optimize(
        system=struct,
        model=model,
        optimizer=ts.Optimizer.fire,
        max_steps=max_steps,
        convergence_fn=converge_max_force,
        trajectory_reporter=reporter,
        autobatcher=autobatcher if use_autobatcher else None,
        init_kwargs=dict(
            cell_filter=ts.CellFilter.frechet,
            constant_volume=False,  # Allow volume to change to find equilibrium
            hydrostatic_strain=False,  # Allow full cell relaxation
        ),
    )

    os.remove(trajectory_file)

    return final_state


def get_qha_structures(
    state: ts.SimState,
    length_factors: np.ndarray,
    model: ModelInterface,
    Nmax: int = 300,
    fmax: float = 1e-3,
    *,
    use_autobatcher: bool = False,
    autobatcher=None,
) -> list[PhonopyAtoms]:
    """Get relaxed structures at different volumes.

    Args:
        state: Initial state
        length_factors: Array of scaling factors
        model: Calculator model
        Nmax: Maximum number of relaxation steps
        fmax: Force convergence criterion
        use_autobatcher: Whether to use automatic batching
        autobatcher: Autobatcher instance to use

    Returns:
        list[PhonopyAtoms]: Relaxed PhonopyAtoms structures at different volumes
    """
    # Convert state to PhonopyAtoms
    relaxed_struct = ts.io.state_to_phonopy(state)[0]

    # Create scaled structures
    scaled_structs = [
        PhonopyAtoms(
            cell=relaxed_struct.cell * factor,
            scaled_positions=relaxed_struct.scaled_positions,
            symbols=relaxed_struct.symbols,
        )
        for factor in length_factors
    ]

    # Relax all structures
    scaled_state = ts.optimize(
        system=scaled_structs,
        model=model,
        optimizer=ts.Optimizer.fire,
        max_steps=Nmax,
        convergence_fn=ts.runners.generate_force_convergence_fn(force_tol=fmax),
        autobatcher=autobatcher if use_autobatcher else None,
        init_kwargs=dict(
            cell_filter=ts.CellFilter.frechet,
            constant_volume=True,
            hydrostatic_strain=True,
        ),
    )

    return scaled_state.to_phonopy()


def plot_and_save_property(
    temperatures: np.ndarray,
    values: np.ndarray,
    property_name: str,
    ylabel: str,
    results_dir: str,
    axis_style: dict,
    html_figures: list = None,
) -> None:
    """Plot and save a property vs temperature.
    
    Args:
        temperatures: Temperature array
        values: Property values array
        property_name: Name of the property (for filenames)
        ylabel: Y-axis label
        results_dir: Directory to save files
        axis_style: Plotly axis style dictionary
        html_figures: Optional list to append figure for HTML export
    """
    # Handle length mismatch - use minimum length
    min_len = min(len(temperatures), len(values))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=temperatures[:min_len], y=values[:min_len], mode="lines", line=dict(width=4))
    )
    fig.update_layout(
        xaxis_title="Temperature (K)",
        yaxis_title=ylabel,
        font=dict(size=24),
        xaxis=axis_style,
        yaxis=axis_style,
        width=800,
        height=600,
        plot_bgcolor="white",
    )
    # Skip fig.show() in headless environments (e.g., WSL2) to avoid gio errors
    # Plots are still saved to files and HTML
    
    # Save as PDF (fallback to PNG if Chrome/Kaleido not available)
    try:
        fig.write_image(os.path.join(results_dir, f"{property_name}.pdf"))
    except RuntimeError as e:
        if "Chrome" in str(e) or "kaleido" in str(e).lower():
            fig.write_image(os.path.join(results_dir, f"{property_name}.png"))
        else:
            raise
    
    # Save data to text file
    txt_file = os.path.join(results_dir, f"{property_name}_vs_temperature.txt")
    with open(txt_file, "w") as f:
        f.write(f"# Temperature (K)    {ylabel}\n")
        for i in range(min_len):
            T = temperatures[i]
            val = values[i]
            if "Expansion" in ylabel or "1/K" in ylabel:
                f.write(f"{T:12.2f}    {val:12.6e}\n")
            else:
                f.write(f"{T:12.2f}    {val:12.6f}\n")
    
    # Add to HTML figures list if provided
    if html_figures is not None:
        html_figures.append(fig)


def get_qha_phonons(
    scaled_structures: list[PhonopyAtoms],
    model: ModelInterface,
    supercell_matrix: np.ndarray | None,
    displ: float = 0.05,
    *,
    use_autobatcher: bool = False,
    autobatcher=None,
) -> tuple[list[Phonopy], list[list[np.ndarray]], np.ndarray]:
    """Get phonon objects for each scaled atom.

    Args:
        scaled_structures: List of PhonopyAtoms objects
        model: Calculator model
        supercell_matrix: Supercell matrix
        displ: Atomic displacement for phonons
        use_autobatcher: Whether to use automatic batching
        autobatcher: Autobatcher instance to use

    Returns:
        tuple[list[Phonopy], list[list[np.ndarray]], np.ndarray]: Contains:
            - List of Phonopy objects
            - List of force sets for each structure
            - Array of energies
    """
    # Generate phonon object for each scaled structure
    supercells_flat = []
    supercell_boundaries = [0]
    ph_sets = []
    if supercell_matrix is None:
        supercell_matrix = np.eye(3)
    for atoms in scaled_structures:
        ph = Phonopy(
            atoms,
            supercell_matrix=supercell_matrix,
            primitive_matrix="auto",
        )
        ph.generate_displacements(distance=displ)
        supercells = ph.supercells_with_displacements
        n_atoms = 0 if supercells is None else sum(len(cell) for cell in supercells)
        supercell_boundaries.append(supercell_boundaries[-1] + n_atoms)
        supercells_flat.extend([] if supercells is None else supercells)
        ph_sets.append(ph)

    # Run the model on flattened structure
    reporter = ts.TrajectoryReporter(
        None,
        state_frequency=0,
        prop_calculators={
            1: {
                "potential_energy": lambda state: state.energy,
                "forces": lambda state: state.forces,
            }
        },
    )
    results = ts.static(
        system=supercells_flat,
        model=model,
        autobatcher=autobatcher if use_autobatcher else None,
        trajectory_reporter=reporter,
    )

    # Reconstruct force sets and energies
    force_sets = []
    forces = torch.cat([r["forces"] for r in results]).detach().cpu().numpy()
    energies = (
        torch.tensor([r["potential_energy"] for r in results]).detach().cpu().numpy()
    )
    for sys_idx, ph in enumerate(ph_sets):
        start, end = supercell_boundaries[sys_idx], supercell_boundaries[sys_idx + 1]
        forces_i = forces[start:end]
        n_atoms = len(ph.supercell)
        n_displacements = len(ph.supercells_with_displacements)
        force_sets_i = []
        for disp_idx in range(n_displacements):
            start_j = disp_idx * n_atoms
            end_j = (disp_idx + 1) * n_atoms
            force_sets_i.append(forces_i[start_j:end_j])
        force_sets.append(force_sets_i)

    return ph_sets, force_sets, energies


# Set device and data type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

# Load custom MACE model from file
model_path = "/home/theoj/temp/mofs_v2.model"
head_name = "pbe_d3"
calculator = MACECalculator(
    model_path=model_path,
    device=str(device),
    head=head_name,
    default_dtype="float64",
)
raw_model = calculator.models[0]

model = MaceModel(
    model=raw_model,
    device=device,
    compute_forces=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=True,  # Enable cuEquivariance acceleration
)

# Ensure all model parameters are on the correct device after cuEquivariance conversion
# This fixes device mismatch issues where some parameters (e.g., atomic_energies) remain on CPU
if torch.cuda.is_available():
    # Move model to device and ensure all submodules are also on device
    model.model = model.model.to(device)
    # Recursively move all parameters and buffers to device
    for module in model.model.modules():
        for param in module.parameters(recurse=False):
            if param.device != device:
                param.data = param.data.to(device)
        for buffer in module.buffers(recurse=False):
            if buffer.device != device:
                buffer.data = buffer.data.to(device)
        # Also check for tensor attributes (like atomic_energies)
        for attr_name in dir(module):
            attr = getattr(module, attr_name, None)
            if isinstance(attr, torch.Tensor) and attr.device != device:
                setattr(module, attr_name, attr.to(device))

# Load MOF5 structure from CIF file
mof5_path = "/home/theoj/temp/MOF5.cif"
struct = read(mof5_path)

# Enable autobatcher with GPU memory limits
use_autobatcher = True

# Create autobatchers with limited memory estimation to reduce GPU usage
static_autobatcher = BinningAutoBatcher(
    model=model,
    max_memory_padding=0.7,  # Use 70% of estimated max to be conservative
    max_atoms_to_try=5000,  # Limit memory estimation to prevent OOM
)

opt_autobatcher = InFlightAutoBatcher(
    model=model,
    max_memory_padding=0.7,  # Use 70% of estimated max to be conservative
    max_atoms_to_try=5000,  # Limit memory estimation to prevent OOM
)

# Structure and input parameters
supercell_matrix = 1 * np.eye(3)  # supercell matrix for phonon calculation
mesh = [20, 20, 20]  # Phonon mesh
fmax = 1e-6  # force convergence
Nmax = 2000  # maximum number of relaxation steps (increased to ensure convergence)

# Atomic displacement for phonon calculations (displ)
# This is the distance atoms are displaced to calculate force constants (FC2)
# Typical values: 0.01-0.05 Å
# - Smaller values (0.01-0.02 Å): More accurate but requires higher precision
# - Larger values (0.03-0.05 Å): More robust, better for anharmonic systems
# - 0.05 Å is a good default for most systems (used in quasi-harmonic examples)
# - 0.01 Å is used in regular phonon DOS calculations
displ = 0.03  # atomic displacement for phonons (in Angstrom)

temperatures = np.arange(0, 1410, 10)  # temperature range for quasi-harmonic calculation

# Volume scaling factors for EOS calculation
# For 7 volumes with +/-3% volume changes:
# - Volume scale: 0.97 to 1.03 (97% to 103% of equilibrium volume)
# - Length factor = (volume_scale)^(1/3) since V ∝ L^3
# - For 0.97: length_factor = 0.97^(1/3) ≈ 0.9900
# - For 1.03: length_factor = 1.03^(1/3) ≈ 1.0099
# - So length factors range from ~0.990 to ~1.010
# Using np.linspace with 7 points gives evenly spaced volumes
min_volume_scale = 0.97  # 97% of equilibrium volume (-3%)
max_volume_scale = 1.03  # 103% of equilibrium volume (+3%)
min_length_factor = np.cbrt(min_volume_scale)  # ≈ 0.9900
max_length_factor = np.cbrt(max_volume_scale)  # ≈ 1.0099
n_volumes = 7
length_factors = np.linspace(min_length_factor, max_length_factor, n_volumes)

# Relax initial structure
state = get_relaxed_structure(
    struct=struct,
    model=model,
    max_steps=Nmax,
    fmax=fmax,
    use_autobatcher=use_autobatcher,
    autobatcher=opt_autobatcher,
)

# Get relaxed structures at different volumes
scaled_structures = get_qha_structures(
    state=state,
    length_factors=length_factors,
    model=model,
    Nmax=Nmax,
    fmax=fmax,
    use_autobatcher=use_autobatcher,
    autobatcher=opt_autobatcher,
)

# Get phonons, FC2 forces, and energies for all set of scaled structures
ph_sets, force_sets, energy_sets = get_qha_phonons(
    scaled_structures=scaled_structures,
    model=model,
    supercell_matrix=supercell_matrix,
    displ=displ,
    use_autobatcher=use_autobatcher,
    autobatcher=static_autobatcher,
)

# Calculate thermal properties for each supercells
volumes = []
energies = []
free_energies = []
entropies = []
heat_capacities = []
actual_temperatures = None  # Will be set from Phonopy's output
n_displacements = len(getattr(ph_sets[0], "supercells_with_displacements", []))
for i in range(len(ph_sets)):
    ph_sets[i].forces = force_sets[i]
    ph_sets[i].produce_force_constants()
    ph_sets[i].run_mesh(mesh)
    # Calculate t_step safely to avoid division by zero
    # Note: Phonopy's run_thermal_properties generates temperatures from t_min to t_max
    # with step t_step, but may not include t_max if (t_max - t_min) is not divisible by t_step
    # To get exactly len(temperatures) points, we need to ensure t_max is included
    t_step = int((temperatures[-1] - temperatures[0]) / max(len(temperatures) - 1, 1))
    # Adjust t_max to ensure we get the expected number of points
    # Phonopy calculates: n_points = int((t_max - t_min) / t_step) + 1
    # So: t_max = t_min + (n_points - 1) * t_step
    expected_n_points = len(temperatures)
    t_max_adjusted = temperatures[0] + (expected_n_points - 1) * t_step
    ph_sets[i].run_thermal_properties(
        t_min=temperatures[0],
        t_max=t_max_adjusted,
        t_step=t_step,
    )

    # Store volume, energy, entropies, heat capacities
    thermal_props = ph_sets[i].get_thermal_properties_dict()
    # Extract actual temperatures from Phonopy (first time only)
    if actual_temperatures is None and "temperatures" in thermal_props:
        actual_temperatures = thermal_props["temperatures"]
    n_unit_cells = np.prod(np.diag(supercell_matrix))
    cell = scaled_structures[i].cell
    volume = np.linalg.det(cell)
    volumes.append(volume)
    energies.append(energy_sets[i * n_displacements].item() / n_unit_cells)
    free_energies.append(thermal_props["free_energy"])
    entropies.append(thermal_props["entropy"])
    heat_capacities.append(thermal_props["heat_capacity"])

# Create results directory
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Use actual temperatures from Phonopy if available, otherwise use input temperatures
# This ensures consistency between temperatures and QHA properties
if actual_temperatures is not None:
    temperatures = actual_temperatures

# run QHA
# Ensure all arrays have compatible shapes
n_volumes = len(volumes)
n_temps = len(temperatures)

qha = PhonopyQHA(
    volumes=volumes,
    electronic_energies=np.tile(energies, (n_temps, 1)),
    temperatures=temperatures,
    free_energy=np.array(free_energies).T,
    cv=np.array(heat_capacities).T,
    entropy=np.array(entropies).T,
    eos="vinet",
)

# Print key values for comparison with experiment
print("\n" + "="*60)
print("Key Properties for Experimental Comparison:")
print("="*60)
# Find indices for 0K and 300K - use safe indexing with length checks
idx_0K = 0
idx_300K = np.argmin(np.abs(temperatures - 300))

# Safe access to QHA properties with length checking
bulk_mod_0K = None
bulk_mod_300K = None
thermal_exp_300K = None
if hasattr(qha, 'bulk_modulus_temperature') and qha.bulk_modulus_temperature is not None:
    bulk_mod_len = len(qha.bulk_modulus_temperature)
    if bulk_mod_len > idx_0K:
        bulk_mod_0K = qha.bulk_modulus_temperature[idx_0K]
    if bulk_mod_len > idx_300K:
        bulk_mod_300K = qha.bulk_modulus_temperature[idx_300K]

if hasattr(qha, 'thermal_expansion') and qha.thermal_expansion is not None:
    thermal_exp_len = len(qha.thermal_expansion)
    if thermal_exp_len > idx_300K:
        thermal_exp_300K = qha.thermal_expansion[idx_300K]

# Get heat capacities with safe indexing
cv_300K = None
if len(heat_capacities) > 0 and len(heat_capacities[0]) > idx_300K:
    cv_300K = heat_capacities[0][idx_300K]

cp_300K = None
if hasattr(qha, 'heat_capacity_P_numerical') and qha.heat_capacity_P_numerical is not None:
    cp_len = len(qha.heat_capacity_P_numerical)
    if cp_len > idx_300K:
        cp_300K = qha.heat_capacity_P_numerical[idx_300K]

# Print results
if bulk_mod_0K is not None:
    print(f"Bulk Modulus at 0 K:     {bulk_mod_0K:.4f} GPa")
if bulk_mod_300K is not None:
    print(f"Bulk Modulus at 300 K:   {bulk_mod_300K:.4f} GPa")
if thermal_exp_300K is not None:
    print(f"Thermal Expansion at 300 K: {thermal_exp_300K:.6e} 1/K")
if cv_300K is not None:
    print(f"Heat Capacity Cv at 300 K: {cv_300K:.4f} J/(mol·K)")
if cp_300K is not None:
    print(f"Heat Capacity Cp at 300 K: {cp_300K:.4f} J/(mol·K)")
print("="*60 + "\n")

# Axis style for all plots
axis_style = dict(
    showgrid=False, zeroline=False, linecolor="black", showline=True,
    ticks="inside", mirror=True, linewidth=3, tickwidth=3, ticklen=10,
)

# Collect figures for HTML export
html_figures = []

# Plot and save all properties (with safe length handling)
if hasattr(qha, 'thermal_expansion') and qha.thermal_expansion is not None:
    plot_and_save_property(
        temperatures, qha.thermal_expansion, "thermal_expansion",
        "Thermal Expansion (1/K)", results_dir, axis_style, html_figures
    )

if hasattr(qha, 'bulk_modulus_temperature') and qha.bulk_modulus_temperature is not None:
    plot_and_save_property(
        temperatures, qha.bulk_modulus_temperature, "bulk_modulus",
        "Bulk Modulus (GPa)", results_dir, axis_style, html_figures
    )

# Plot Cv if available (handle length mismatches automatically)
if len(heat_capacities) > 0 and len(heat_capacities[0]) > 0:
    plot_and_save_property(
        temperatures, heat_capacities[0], "heat_capacity_cv",
        "Heat Capacity Cv (J/(mol·K))", results_dir, axis_style, html_figures
    )

# Plot Cp if available (using numerical method, handle length mismatches automatically)
if (hasattr(qha, 'heat_capacity_P_numerical') and qha.heat_capacity_P_numerical is not None
    and len(qha.heat_capacity_P_numerical) > 0):
    plot_and_save_property(
        temperatures, qha.heat_capacity_P_numerical, "heat_capacity_cp",
        "Heat Capacity Cp (J/(mol·K))", results_dir, axis_style, html_figures
    )

# Save all plots in a single interactive HTML file
if html_figures:
    html_file = os.path.join(results_dir, "all_plots.html")
    with open(html_file, "w") as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>QHA Results - Interactive Plots</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 40px;
            margin-bottom: 20px;
            padding: 10px;
            background-color: #e8f5e9;
            border-left: 5px solid #4CAF50;
        }
        .plot-container {
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <h1>Quasi-Harmonic Analysis Results</h1>
""")
        for i, fig in enumerate(html_figures):
            plot_html = fig.to_html(include_plotlyjs='cdn' if i == 0 else False, div_id=f"plot{i}", full_html=False)
            f.write(f'    <div class="plot-container">\n')
            f.write(f'        <h2>{fig.layout.yaxis.title.text}</h2>\n')
            f.write(f'        {plot_html}\n')
            f.write(f'    </div>\n')
        f.write("""</body>
</html>
""")

