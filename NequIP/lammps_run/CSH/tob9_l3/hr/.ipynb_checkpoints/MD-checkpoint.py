from ase import Atoms, io
import torch
from nequip.dynamics.nequip_calculator import NequIPCalculator
from ase.io import read, write
import os
from ase.calculators.calculator import Calculator, all_changes
import nequip.scripts.deploy

device = torch.device("cuda")

nequip.scripts.deploy.load_deployed_model(
            model_path=model_path, device=device
        )
print('done')

calculator = NequIPCalculator.from_deployed_model(
    model_path="tob9_l3-deployed.pth",
    species_to_type_name = {
        "Ca": "NequIPTypeNameForCalcium",
        "Si": "NequIPTypeNameForSilicon",
        "O": "NequIPTypeNameForOxygen",
        "H": "NequIPTypeNameForHydrogen",
    },
    device=device
)
print('done')

from ase.optimize import QuasiNewton

ase_dir = './'
name = "optimization"
molecule_path = 'tob9.xyz'
molecule = read(molecule_path)

molecule.set_calculator(calculator)

optimize_file = os.path.join(ase_dir, name)
optimizer = QuasiNewton(
    molecule,
    trajectory="{:s}.traj".format(optimize_file),
    restart="{:s}.pkl".format(optimize_file),
)
optimizer.run(fmax=1e-2, steps=1000)

def save_molecule(self, name: str, file_format: str = "xyz", append: bool = False):
        
        molecule_path = os.path.join(
            ase_dir, "{:s}.{:s}".format(name, file_format)
        )
        write(molecule_path, molecule, format=file_format, append=append)


save_molecule(name, file_format="extxyz")