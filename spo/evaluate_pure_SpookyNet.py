#!/usr/bin/env python3
import argparse
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import *
import torch
from spookynet import SpookyNet

"""
Spookynet: Learning force fields with electronic degrees of freedom and nonlocal effects.
OT Unke, S Chmiela, M Gastegger, KT Schütt, HE Sauceda, KR Müller
Nat. Commun. 12(1), 2021, 1-14, (2021).
"""

"""Initialises FFSpookyNet
Args:

   spookyNet_model: filename containing the parameters
   ini_xyz: xyz file of the structure (to initialize atomic charges etc.)
   nbeads: how many beads are used
   charge: total molecular charge
   magmom: number of unpaired electrons (singlet = 0.0)
   use_gpu: use the GPU if it is available (falls back to cpu automatically if there is no GPU)
"""

#parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
required.add_argument("-nn", "--neural_network",  type=str,   help=".pth file from which to load the nn model",  required=True)
required.add_argument("-i", "--input",  type=str,   help="input xyz",  required=True)
optional = parser.add_argument_group("optional arguments")
optional.add_argument("--charge",     type=int,   help="total charge",  default=0)
optional.add_argument("--magmom",     type=int,   help="total spin (singlet = 0, doublet=1, etc.)",  default=0)

args = parser.parse_args()
print("input ", args.input)
'best_model.pth.tar'
#read input file, atoms is an ASE object
atoms = read(args.input)

spookyNet_model = args.neural_network
charge = args.charge
magmom = args.magmom
nbeads = 1
use_gpu = True

# load SpookyNet model
try:
    model = SpookyNet(load_from=spookyNet_model)
    model.to(torch.float32)
    model.eval()
    print(
        " @ForceField: SpookyNet model " + spookyNet_model + " loaded"
    )
except ValueError:
    raise ValueError(
        "ERROR: Reading SpookyNet model " + spookyNet_model + " file failed."
    )

if use_gpu and not torch.cuda.is_available():
    print(
        " @ForceField: No GPU available: Evaluation on GPU was requested"
        + " but no GPU device is available. Falling back to CPU."
    )

if use_gpu and torch.cuda.is_available():
    model.cuda()
    device = 'cuda'
else:
    device = 'cpu'
print('magmom', magmom)
print(torch.cuda.is_available())
# generate idx lists for finding neighboring atoms
idx = torch.arange(len(atoms), dtype=torch.int64, device=device)
idx_i = idx.view(-1, 1).expand(-1, len(atoms)).reshape(-1)
idx_j = idx.view(1, -1).expand(len(atoms), -1).reshape(-1)
# exclude self-interactions
idx_i, idx_j = idx_i[idx_i != idx_j], idx_j[idx_i != idx_j]
# create input dictionary
inputs = {
    "Z": torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int64, device=device).repeat(nbeads),
    "Q": torch.full((nbeads,), charge, dtype=torch.float32, device=device),
    "S": torch.full((nbeads,), magmom, dtype=torch.float32, device=device),
    "idx_i": torch.cat([idx_i + i * len(atoms) for i in range(nbeads)]),
    "idx_j": torch.cat([idx_j + i * len(atoms) for i in range(nbeads)]),
    "batch_seg": torch.arange(nbeads, dtype=torch.int64, device=device).repeat_interleave(len(atoms)),
    "num_batch": nbeads
}

print(
    " @ForceField: IMPORTANT: It is always assumed that the units in"
    + " the provided model file are in Angstroms and eV."
)

"""Evaluate the energy and forces."""
#gather coordinates
R = torch.tensor([atoms.positions],
                 dtype=torch.float32,
                 device=device).view(-1, 3)
R.requires_grad = True
inputs["R"] = R

outputs = model.energy_and_forces(**inputs)
energy = outputs[0].detach().cpu().numpy()
forces = outputs[1].detach().cpu().numpy().reshape((inputs['num_batch'], -1, 3))

print("energy [eV]", energy[0])
print(energy,type( energy[0]))