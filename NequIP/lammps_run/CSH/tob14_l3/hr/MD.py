from typing import Union
import torch

from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from nequip.ase import NequIPCalculator
from ase import Atoms, io
import torch
#from nequip.dynamics.nequip_calculator import NequIPCalculator
from nequip.ase import NequIPCalculator
from ase.io import read, write
from ase import Atoms, io
from ase.md import VelocityVerlet, Langevin, MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.io.trajectory import Trajectory
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations
from ase import units
from typing import Optional, List, Union, Dict
import os
from ase.calculators.calculator import Calculator, all_changes
import nequip.scripts.deploy
device = torch.device("cuda")


device = torch.device("cuda")


calculator = NequIPCalculator.from_deployed_model(
    model_path="./tob14_l3-deployed.pth",
    device = device,
)

def save_molecule(working_dir, molecule,name: str, file_format: str = "xyz", append: bool = False):
    molecule_path = os.path.join(working_dir, "{:s}.{:s}".format(name, file_format))
    write(molecule_path, molecule, format=file_format, append=append)

    

ase_dir = './'
name = "optimization"
molecule_path = './tob14.xyz'
molecule = read(molecule_path)

molecule.calc = calculator

optimize_file = os.path.join(ase_dir, name)
optimizer = QuasiNewton(
    molecule,
    trajectory="{:s}.traj".format(optimize_file),
    restart="{:s}.pkl".format(optimize_file),
)
if os.path.exists("optimization.extxyz"):
    print('Optimization has been done!')
else:
    optimizer.run(fmax=1e-2, steps=1000)
    save_molecule(working_dir = ase_dir, molecule = molecule, name = name, file_format="extxyz")

def init_velocities(
        molecule,
        temp_init: float = 300,
        remove_translation: bool = True,
        remove_rotation: bool = True,
    ):
        MaxwellBoltzmannDistribution(molecule, temp_init * units.kB)
        if remove_translation:
            Stationary(molecule)
        if remove_rotation:
            ZeroRotation(molecule)


ase_dir = './'
dynamics = None
name = 'simulation_300K'
time_step=0.5 #fs
temp_init=300
temp_bath=300 #NVT None:NVE
reset=True
interval=100
molecule_path = 'optimization.extxyz'
molecule = read(molecule_path)
molecule.calc = calculator


# init md
if dynamics is None or reset:
    init_velocities(molecule = molecule, temp_init=temp_init)

if temp_bath is None:
    dynamics = VelocityVerlet(molecule, time_step * units.fs)
else:
    dynamics = Langevin(
        molecule,
        time_step * units.fs,
        temp_bath * units.kB,
        1.0 / (100.0 * units.fs),
    )

logfile = os.path.join(ase_dir, "{:s}.log".format(name))
trajfile = os.path.join(ase_dir, "{:s}.traj".format(name))
logger = MDLogger(
    dynamics,
    molecule,
    logfile,
    stress=False,
    peratom=False,
    header=True,
    mode="a",
)
trajectory = Trajectory(trajfile, "w", molecule)

dynamics.attach(logger, interval=interval)
dynamics.attach(trajectory.write, interval=interval)

dynamics.run(1000000)
