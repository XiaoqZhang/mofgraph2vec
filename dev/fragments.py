import os
from moffragmentor import MOF
import json
from tqdm import tqdm

cifs = os.listdir("../data/cifs/rsm/")

sb_info = []
for cif in tqdm(cifs):
    mof = MOF.from_cif(os.path.join("../data/cifs/rsm", cif))
    fragments = mof.fragment()
    if fragments is not None:
        sb_info.append(
            {
                "name": cif.rsplit(".cif")[0],
                "nodes": [node.smiles for node in fragments.nodes],
                "linkers": [linker.smiles for linker in fragments.linkers]
            }
        )

#json_object = json.dumps(sb_info, indent = 4) 
with open("./rsm_bb.json", "w") as file:
    json.dump(sb_info, file, indent=4)