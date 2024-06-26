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
        """parse the CIF file to a graph

        Args:
            strategy (str, optional): how to define bonds in crystals. Defaults to "vesta".
        """
        if strategy == "vesta":
            self.strategy = CutOffDictNN(cut_off_dict=_VESTA_CUTOFFS)

    def from_cif(self, path: str):
        meta_folder = os.path.join(os.path.dirname(path), "../meta")
        if not os.path.exists(meta_folder):
            os.makedirs(meta_folder)

        metaname = os.path.basename(path).replace(".cif", ".pt")
        meta_path = os.path.join(meta_folder, metaname)

        if not os.path.exists(meta_path):
            structure = Structure.from_file(path)
            sg = StructureGraph.with_local_env_strategy(structure, self.strategy)

            x = self._get_node_features(structure)
            edge_idx, edge_attr = self._get_edge_index_and_lengths(sg)
            data = Data(
                x=x, 
                edge_index=torch.Tensor(edge_idx), 
                edge_attr=edge_attr, 
                        )
            torch.save(data, meta_path)
        else:
            data = torch.load(meta_path)

        return data

    def _get_node_features(self, structure: Structure):
        x = [site.species_string for site in structure]
        return np.vstack(x)

    def _get_edge_index_and_lengths(self, sg: StructureGraph):
        edge_idx = []
        distances = []

        lattice = sg.structure.lattice
        structure = sg.structure
        for node_idx in range(len(structure.sites)):
            nbr = sg.get_connected_sites(node_idx)
            fc_0 = structure.frac_coords[node_idx]
            d = [_get_distance(lattice, fc_0, structure.frac_coords[nn.index], nn.jimage) for nn in nbr]
            distances += d
            edge_idx += [[node_idx, nn.index] for nn in nbr]

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

    Args:
        lattice (Lattice): pymatgen Lattice object
        frac_coords_0 (np.array): the fractional coordinates of the first atom
        frac_coords_1 (np.array): the fractional coordinates of the second atom
        jimage (Tuple[int, int, int]): the periodic boundary conditions

    Returns:
        float: the distance between the two atoms
    """
    jimage = np.array(jimage)
    mapped_vec = lattice.get_cartesian_coords(jimage + frac_coords_1 - frac_coords_0)
    return np.linalg.norm(mapped_vec)

