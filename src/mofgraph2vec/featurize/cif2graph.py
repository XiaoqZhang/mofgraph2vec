# Build MOF graph from CIF file
import os
import numpy as np
import yaml
from typing import Tuple

from pymatgen.core.structure import Structure, Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CutOffDictNN

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_THIS_DIR, "tuned_vesta.yml"), "r", encoding="utf8") as handle:
    _VESTA_CUTOFFS = yaml.load(handle, Loader=yaml.UnsafeLoader) 

class MOFDataset:
    def __init__(
        self,
        strategy: str="vesta",
    ) -> None:
        if strategy == "vesta":
            self.strategy = CutOffDictNN(cut_off_dict=_VESTA_CUTOFFS)

    def from_cif(self, path: str):
        meta_path = path.replace("cifs", "meta")
        meta_path = meta_path.replace(".cif", ".pt")

        if not os.path.exists(meta_path):
            structure = Structure.from_file(path)
            sg = StructureGraph.with_local_env_strategy(structure, self.strategy)

            x = self._get_node_features(structure)
            edge_idx, edge_attr = self._get_edge_index_and_lengths(sg)
            data = Data(x=x, edge_index=torch.Tensor(edge_idx), edge_attr=edge_attr)
            torch.save(data, meta_path)
        else:
            data = torch.load(meta_path)

        return data

    def _get_node_features(self, structure: Structure):
        x = [site.specie.Z for site in structure]
        return np.vstack(x)

    def _get_edge_index_and_lengths(self, sg: StructureGraph):
        edge_idx = []
        distances = []

        lattice = sg.structure.lattice
        structure = sg.structure
        for edge in sg.graph.edges(keys=True, data=True):
            fc_0 = structure.frac_coords[edge[0]]
            fc_1 = structure.frac_coords[edge[1]]
            d = _get_distance(lattice, fc_0, fc_1, edge[-1]["to_jimage"])
            distances.append(d)
            edge_idx.append([edge[0], edge[1]])

        return (
            np.array(edge_idx).T,
            np.array(distances)
        )
    
    def to_WL_machine(self, path):
        data = self.from_cif(path)
        graph = to_networkx(data)
        features_to_WL = {}
        for i, item in enumerate(data.x.flatten()):
            features_to_WL.update({i: item})
        return graph, features_to_WL


def _get_distance(
    lattice: Lattice, frac_coords_0: np.array, frac_coords_1: np.array, jimage: Tuple[int, int, int]
) -> float:
    """Get the distance between two fractional coordinates taking into account periodic boundary conditions.
    Parameters
    ----------
    lattice : Lattice
        pymatgen Lattice object
    frac_coords_0 : np.array
        fractional coordinates of the first atom
    frac_coords_1 : np.array
        fractional coordinates of the second atom
    jimage : Tuple[int, int, int]
        image of the second atom
    Returns
    -------
    float
        Distance between the two atoms
    """
    jimage = np.array(jimage)
    mapped_vec = lattice.get_cartesian_coords(jimage + frac_coords_1 - frac_coords_0)
    return np.linalg.norm(mapped_vec)

