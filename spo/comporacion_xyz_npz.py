#checaremos como se ven los archivos para poder despues cambiar el au.xyz a npz
import numpy as np
from ase import Atoms
from ase.io import read, write
import torch

ethanol_data = np.load('ethanol_ccsd_t-train.npz')
nxyz_data = np.dstack((np.array([ethanol_data.f.z]*1000).reshape(1000, -1, 1), np.array(ethanol_data.f.R)))
nxyz_data.shape

force_data = ethanol_data.f.F
force_data.shape

energy_data = ethanol_data.f.E.squeeze() - ethanol_data.f.E.mean()
energy_data.shape

ethanol_data.f.z
ethanol_data.shape


lst = ethanol_data.files


for item in lst:
    print(item)
    print(ethanol_data[item].shape)


nxyz_data = np.dstack((np.array([ethanol_data.f.z]*49863).reshape(49863, -1, 1), np.array(ethanol_data.f.R)))



neural_network = 'Au_model.pth'
input_v = 'Au_particle.xyz'
charge = 1
magmom = 0



#read input file, atoms is an ASE object
atoms = read(input_v)
atoms.set_initial_charges()

type(atoms)
spookyNet_model = neural_network
charge = charge
magmom = magmom
nbeads = 1


if torch.cuda.is_available():

    device = 'cuda'
else:
    device = 'cpu'
atoms.positions
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
