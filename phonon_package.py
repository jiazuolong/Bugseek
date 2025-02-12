import os
import shutil
import click
import subprocess
import numpy as np
from ase.io import read, write
from ase.atoms import Atoms
from mattersim.forcefield.potential import MatterSimCalculator
from mace.calculators.foundations_models import mace_mp
from ase.optimize import LBFGSLineSearch
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.file_IO import write_force_constants_to_hdf5
from phono3py import Phono3py
import h5py
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from ase.constraints import UnitCellFilter
import plotly.express as px

class PhononCalculator:
    """Read the POSCAR file;    
    Perform structural optimization using MACE/MatterSim;
    Calculate and generate the second-order and third-order force constants;
    Generate the phonon spectra (Lattice Thermal Conductivity)"""
    def __init__(self, poscar_path, volume_scaling_factor, supercell_matrix, calculator_type, mace_model_path=None):
        self.poscar_path = poscar_path  
        self.volume_scaling_factor = volume_scaling_factor  
        self.supercell_matrix = supercell_matrix  
        self.atoms = read(poscar_path, format='vasp')
        self.calculator_type = calculator_type  
        self.mace_model_path = mace_model_path
        self.calc = self._initialize_calculator()
        self.atoms.calc = self.calc

    """Initialize the calculator according to the type of the calculator"""
    def _initialize_calculator(self):
        if self.calculator_type == 'mattersim':
            return MatterSimCalculator()
        elif self.calculator_type == 'mace':
            if not self.mace_model_path:
                raise ValueError("The path to the MACE model was not provided")
            return mace_mp(model=self.mace_model_path, default_dtype='float64')
        else:
            raise ValueError(f"Unknown calculator")

    """Apply volumetric strain to the structure"""
    def apply_strain(self):
        original_cell = self.atoms.get_cell()
        scaled_cell = original_cell * (self.volume_scaling_factor ** (1 / 3))
        self.atoms.set_cell(scaled_cell, scale_atoms=True)

    """Optimize the structure and save the optimized POSCAR file"""
    def optimize_structure(self, save_path):
        opt = LBFGSLineSearch(self.atoms) 
        opt.run(fmax=0.0001) 
        write(save_path, self.atoms, format="vasp")
        return self.atoms.get_cell()

    """Calculate the second-order force constants and save them as an `fc2.hdf5` file"""
    def generate_fc2(self, output_path):
        phonon = Phonopy(
            PhonopyAtoms(
                symbols=self.atoms.get_chemical_symbols(),
                positions=self.atoms.get_positions(),
                cell=self.atoms.get_cell(),
            ),
            supercell_matrix=self.supercell_matrix  
        )
        phonon.generate_displacements(distance=0.03)
        set_of_forces = [self._calculate_forces(scell) for scell in phonon.supercells_with_displacements]
        phonon.forces = set_of_forces
        phonon.produce_force_constants()
        write_force_constants_to_hdf5(phonon.force_constants, filename=output_path)
        click.echo(
            f"-------------------The second-order force constants have been generated-------------------")

    """Calculate the third-order force constants and save them as an `fc3.hdf5` file"""
    def generate_fc3(self, output_path):
        unitcell = PhonopyAtoms(
            symbols=self.atoms.get_chemical_symbols(),
            positions=self.atoms.get_positions(),
            cell=self.atoms.get_cell(),
        )
        ph3 = Phono3py(unitcell, supercell_matrix=self.supercell_matrix, primitive_matrix='auto')
        ph3.generate_displacements()
        ph3.save('phono3py_disp.yaml')
        set_of_forces = [self._calculate_forces(scell) for scell in ph3.supercells_with_displacements]
        ph3.forces = set_of_forces
        ph3.produce_fc3()
        with h5py.File(output_path, 'w') as hdf:
            hdf.create_dataset('fc3', data=ph3.fc3)
            hdf.create_dataset('supercell_matrix', data=self.supercell_matrix)
            hdf.create_dataset('primitive_matrix', data=ph3.primitive_matrix)
        click.echo(
            f"-------------------The third-order force constants have been generated-------------------")

    """Generate the phonon spectrum using the 'sumo' tool"""
    def generate_phonon_bandplot(self, output_fc2, supercell):
        command = [
            "sumo-phonon-bandplot",
            "-f", output_fc2,
            "--dim", str(supercell[0]), str(supercell[1]), str(supercell[2])
        ]
        click.echo(
            f"-------------------Running sumo to generate the phonon spectra-------------------")
        subprocess.run(command, check=True)
        click.echo(f"-------------------The phonon spectra has been generated.-------------------")

    """Run phono3py to calculate thermal properties"""
    def run_phono3py(self, supercell, mesh, output_dir):
        # Copy fc2.hdf5 and fc3.hdf5 from output directory to current directory
        fc2_source = os.path.join(output_dir, 'fc2.hdf5')
        fc3_source = os.path.join(output_dir, 'fc3.hdf5')
        fc2_dest = os.path.join(os.getcwd(), 'fc2.hdf5')
        fc3_dest = os.path.join(os.getcwd(), 'fc3.hdf5')
        
        shutil.copy(fc2_source, fc2_dest)
        shutil.copy(fc3_source, fc3_dest)
        
        # Run phono3py command
        command = [
            "phono3py",
            "--fc3",
            "--fc2",
            "--dim", f"{supercell[0]} {supercell[1]} {supercell[2]}",
            "--mesh", f"{mesh[0]} {mesh[1]} {mesh[2]}",
            "-c", "POSCAR",
            "--br",
        ]
        click.echo(f"-------------------Running phono3py to calculate thermal properties-------------------")
        subprocess.run(command, check=True)
        click.echo(f"-------------------phono3py calculation completed-------------------")

    """Calculate the atomic forces of the supercell"""
    def _calculate_forces(self, scell):
        scell_atoms = Atoms(symbols=scell.symbols, positions=scell.positions, cell=scell.cell, pbc=True)
        scell_atoms.calc = self.calc
        return scell_atoms.get_forces()


def compute_phase_diagram(poscar_file, mpr_api_key, calculator_type, mace_model_path):
    """
    Compute the phase diagram and e_above_hull for a structure given a POSCAR file
    and a Materials Project API key.
    """
    structure = Structure.from_file(poscar_file)
    mpr = MPRester(mpr_api_key)
  
    elements = [el.symbol for el in structure.composition.elements]
    competing_entries = mpr.get_entries_in_chemsys(elements)
    
    # Prepare calculator and ASE structure
    if calculator_type == 'mattersim':
        calc = MatterSimCalculator()
    elif calculator_type == 'mace':
        if not mace_model_path:
            raise ValueError("The path to the MACE model was not provided")
        calc = mace_mp(model=mace_model_path, default_dtype='float64')
    else:
        raise ValueError(f"Unknown calculator")

    structure_ase = AseAtomsAdaptor.get_atoms(structure)
    cons = UnitCellFilter(structure_ase)
    structure_ase.set_calculator(calc)
    opt = LBFGSLineSearch(cons)
    opt.run(fmax=0.05)

    # Create target entry for the structure
    target_entry = ComputedStructureEntry(AseAtomsAdaptor.get_structure(structure_ase), structure_ase.get_potential_energy())
    compatibility = MaterialsProject2020Compatibility(check_potcar=False)
    target_entry.parameters['software'] = 'vasp'
    target_entry.parameters['run_type'] = 'GGA+U'
    compatibility.process_entries([target_entry])

    pd = PhaseDiagram([target_entry] + competing_entries)
    e_above_hull = pd.get_e_above_hull(target_entry)
    plotter = PDPlotter(pd, show_unstable=True, ternary_style='2d')
    #plotter.fig.savefig('phase_diagram.png')
    return e_above_hull, plotter


@click.command()
@click.option('--supercell', '-s', type=(int, int, int), default=(3, 3, 3), help='Supercell size')
@click.option('--strain', '-e', type=float, default=1.0, help='Volumetric strain factor')
@click.option('--calculator', '-c', type=click.Choice(['mattersim', 'mace']), default='mattersim', help='Calculator type (mattersim or mace)')
@click.option('--mace_model', '-m', type=str, default=None, help='MACE model path (only used when the calculator type is mace)')
@click.option('--mesh', type=(int, int, int), default=(19, 19, 19), help='Mesh size for phono3py calculation')
@click.option('--do_compute_phase_diagram', '-p', is_flag=True, help='Compute phase diagram and e_above_hull')
@click.option('--mpr_api_key', '-k',  type=str, default=None, help='Materials Project API key (required if --do_compute_phase_diagram is set)')
def main(supercell, strain, calculator, mace_model, mesh, do_compute_phase_diagram, mpr_api_key):
    """Main function: Read the POSCAR file, optimize the structure, 
    calculate the force constants, generate the phonon spectrum, and run phono3py."""
    base_dir = os.getcwd()
    poscar_path = os.path.join(base_dir, 'POSCAR')
    if not os.path.exists(poscar_path):
        click.echo("The POSCAR file does not exist")
        return

    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    phonon_calculator = PhononCalculator(poscar_path, strain, supercell, calculator, mace_model)
    phonon_calculator.apply_strain()
    optimized_poscar = os.path.join(output_dir, 'POSCAR_optimized.vasp')
    phonon_calculator.optimize_structure(optimized_poscar)
    output_fc2 = os.path.join(output_dir, 'fc2.hdf5')
    phonon_calculator.generate_fc2(output_fc2)
    phonon_calculator.generate_fc3(os.path.join(output_dir, 'fc3.hdf5'))

    # Generate the phonon spectra
    phonon_calculator.generate_phonon_bandplot(output_fc2, supercell)

    # Run phono3py to calculate thermal properties
    phonon_calculator.run_phono3py(supercell, mesh, output_dir)


  
    if do_compute_phase_diagram:
        if not mpr_api_key:
            click.echo("Materials Project API key is required when --do_compute_phase_diagram is set.")
            return
        e_above_hull, plotter = compute_phase_diagram(optimized_poscar, mpr_api_key, calculator, mace_model)
        click.echo(f"E_above_hull: {e_above_hull}")
        plotter.get_plot().write_html('phase_diagram.html')

        
    # Move all files in the current directory except the POSCAR file to the "output" folder.
    for filename in os.listdir(base_dir):
        file_path = os.path.join(base_dir, filename)
        if os.path.isfile(file_path) and filename != 'POSCAR':
            shutil.move(file_path, os.path.join(output_dir, filename))
    click.echo(f"-------------------All calculations have been completed-------------------")


if __name__ == "__main__":
    main()
