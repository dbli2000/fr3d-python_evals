"""
Amino Acid Hydrogen Inference for FR3D.
"""
import numpy as np
from fr3d.data.atoms import Atom

# For data-driven approach
AA_HYDROGEN_RULES = {
    # "ALA": [
    #    {"type": "pyramidal", "atoms_for_plane": ["C", "CA", "CB"], "hydrogen_name": "HA", "invert": True/False?}, # Simplified
    #    ...
    # ],
    # ... more amino acids
}

def unit_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v # Or raise error
    return v / norm

def angle_between_vectors(vec1, vec2):
    if len(vec1) == 3 and len(vec2) == 3:
        cosang = np.dot(vec1, vec2)
        sinang = np.linalg.norm(np.cross(vec1, vec2))
        angle = np.arctan2(sinang, cosang)
        return 180*angle/np.pi
    else:
        return None

def angle_between_three_points(P1,P2,P3):
    if len(P1) == 3 and len(P2) == 3 and len(P3) == 3:
        return angle_between_vectors(P1-P2,P3-P2)
    else:
        return None

def pyramidal_hydrogens(P1,C,P2,bondLength=1):
    V1_orig = np.array(P1)
    C_orig = np.array(C)
    V2_orig = np.array(P2)

    # infer positions one way
    V1 = V1_orig
    V2 = V2_orig
    u = unit_vector(C_orig-V2)
    W = np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
    R_matrix = np.identity(3) + (np.sqrt(3)/2)*W + 1.5 * np.dot(W,W) # Corrected: W is already a matrix
    V3 = C_orig + bondLength * unit_vector(np.dot(R_matrix,V1-C_orig))
    V4 = C_orig + bondLength * unit_vector(np.dot(np.transpose(R_matrix),V1-C_orig))

    # infer positions the other way
    V1 = V2_orig
    V2 = V1_orig
    u = unit_vector(C_orig-V2)
    W = np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
    R_matrix = np.identity(3) + (np.sqrt(3)/2)*W + 1.5 * np.dot(W,W) # Corrected: W is already a matrix
    VV4 = C_orig + bondLength * unit_vector(np.dot(R_matrix,V1-C_orig))
    VV3 = C_orig + bondLength * unit_vector(np.dot(np.transpose(R_matrix),V1-C_orig))

    P3 = (V3+VV3)/2
    P4 = (V4+VV4)/2
    return P3, P4

def planar_hydrogens(P1,P2,P3,bondLength=1):
    P1_arr, P2_arr, P3_arr = np.array(P1), np.array(P2), np.array(P3)
    A = unit_vector(P2_arr - P1_arr)
    A1 = P3_arr + A*bondLength
    B = unit_vector(P3_arr - P2_arr)
    A2 = P3_arr + unit_vector(B - A)*bondLength # Fixed: unit_vector around B-A
    return A1, A2

def planar_ring_hydrogen(P1,P2,P3,bondlength=1):
    P1_arr, P2_arr, P3_arr = np.array(P1), np.array(P2), np.array(P3)
    u=unit_vector(P2_arr-P1_arr)
    v=unit_vector(P2_arr-P3_arr)
    w = unit_vector(u + v)
    A1= P2_arr + bondlength * w
    return A1


def infer_aa_hydrogens(component):
    """
    Infers hydrogen atom positions for an amino acid component.
    Modifies component._atoms in place.
    """
    # Original logic from Component.infer_amino_acid_hydrogens will be moved here
    # and then refactored to use AA_HYDROGEN_RULES.
    """
    Infers hydrogen atom positions for an amino acid component.
    Modifies component._atoms in place.
    """
    # This function needs access to:
    # component.sequence, component.centers, component._atoms
    # component.unit_id() (for logging, if needed)
    # NHBondLength (can be defined locally or passed)
    NHBondLength = 1 # Assuming this is a constant, define locally.

    try:
        if component.sequence == "ALA":
            A1,A2 = pyramidal_hydrogens(component.centers["C"],component.centers["CA"],component.centers["CB"])
            component._atoms.append(Atom(name="HA",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = planar_hydrogens(component.centers["C"],component.centers["CA"],component.centers["CB"],NHBondLength)
            component._atoms.append(Atom(name="HB1",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CA"],component.centers["CB"],component.centers["HB1"])
            component._atoms.append(Atom(name="HB3",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HB2",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))

        elif component.sequence == "ARG":
            A1,A2 = planar_hydrogens(component.centers["NE"],component.centers["CZ"],component.centers["NH1"],NHBondLength)
            component._atoms.append(Atom(name="HH11",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HH12",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = planar_hydrogens(component.centers["NE"],component.centers["CZ"],component.centers["NH2"])
            component._atoms.append(Atom(name="HH22",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HH21",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = planar_hydrogens(component.centers["NH1"],component.centers["CZ"],component.centers["NE"])
            component._atoms.append(Atom(name="HE",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CG"],component.centers["CD"],component.centers["NE"])
            component._atoms.append(Atom(name="HD3",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HD2",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CB"],component.centers["CG"],component.centers["CD"])
            component._atoms.append(Atom(name="HG2",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HG3",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CA"],component.centers["CB"],component.centers["CG"])
            component._atoms.append(Atom(name="HB2",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HB3",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["C"],component.centers["CA"],component.centers["CB"])
            component._atoms.append(Atom(name="HA",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))

        elif component.sequence == "ASN":
            A1,A2 = pyramidal_hydrogens(component.centers["C"],component.centers["CA"],component.centers["CB"])
            component._atoms.append(Atom(name="HA",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CA"],component.centers["CB"],component.centers["CG"])
            component._atoms.append(Atom(name="HB3",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HB2",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = planar_hydrogens(component.centers["CB"],component.centers["CG"],component.centers["ND2"],NHBondLength)
            component._atoms.append(Atom(name="HD22",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = planar_hydrogens(component.centers["OD1"],component.centers["CG"],component.centers["ND2"],NHBondLength)
            component._atoms.append(Atom(name="HD21",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))

        elif component.sequence == "ASP":
            A1,A2 = pyramidal_hydrogens(component.centers["C"],component.centers["CA"],component.centers["CB"])
            component._atoms.append(Atom(name="HA",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CA"],component.centers["CB"],component.centers["CG"])
            component._atoms.append(Atom(name="HB3",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HB2",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = planar_hydrogens(component.centers["CB"],component.centers["CG"],component.centers["OD2"],NHBondLength)
            component._atoms.append(Atom(name="HD2",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))

        elif component.sequence == "CYS":
            A1,A2 = pyramidal_hydrogens(component.centers["C"],component.centers["CA"],component.centers["CB"])
            component._atoms.append(Atom(name="HA",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CA"],component.centers["CB"],component.centers["SG"])
            component._atoms.append(Atom(name="HB3",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HB2",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = planar_hydrogens(component.centers["CA"],component.centers["CB"],component.centers["SG"],NHBondLength)
            component._atoms.append(Atom(name="HG",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))

        elif component.sequence == "GLU":
            A1,A2 = pyramidal_hydrogens(component.centers["C"],component.centers["CA"],component.centers["CB"])
            component._atoms.append(Atom(name="HA",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CA"],component.centers["CB"],component.centers["CG"])
            component._atoms.append(Atom(name="HB3",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HB2",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CB"],component.centers["CG"],component.centers["CD"])
            component._atoms.append(Atom(name="HG3",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HG2",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))

        elif component.sequence == "GLY":
            A1,A2 = pyramidal_hydrogens(component.centers["N"],component.centers["CA"],component.centers["C"])
            component._atoms.append(Atom(name="HA3",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HA2",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))

        elif component.sequence == "HIS":
            A1,A2 = pyramidal_hydrogens(component.centers["C"],component.centers["CA"],component.centers["CB"])
            component._atoms.append(Atom(name="HA",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CA"],component.centers["CB"],component.centers["CG"])
            component._atoms.append(Atom(name="HB3",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HB2",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1 = planar_ring_hydrogen(component.centers["CG"],component.centers["ND1"],component.centers["CE1"],NHBondLength)
            component._atoms.append(Atom(name="HD1",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1 = planar_ring_hydrogen(component.centers["NE2"],component.centers["CE1"],component.centers["ND1"],NHBondLength)
            component._atoms.append(Atom(name="HE1",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1 = planar_ring_hydrogen(component.centers["CE1"],component.centers["NE2"],component.centers["CD2"],NHBondLength)
            component._atoms.append(Atom(name="HE2",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1 = planar_ring_hydrogen(component.centers["NE2"],component.centers["CD2"],component.centers["CG"],NHBondLength)
            component._atoms.append(Atom(name="HD2",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))

        elif component.sequence == "ILE":
            A1,A2 = pyramidal_hydrogens(component.centers["C"],component.centers["CA"],component.centers["CB"])
            component._atoms.append(Atom(name="HA",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CB"],component.centers["CG1"],component.centers["CD1"])
            component._atoms.append(Atom(name="HG12",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HG13",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = planar_hydrogens(component.centers["CG1"],component.centers["CB"],component.centers["CG2"],NHBondLength)
            component._atoms.append(Atom(name="HG23",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CB"],component.centers["CG2"],component.centers["HG23"])
            component._atoms.append(Atom(name="HG22",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HG21",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = planar_hydrogens(component.centers["CB"],component.centers["CG1"],component.centers["CD1"],NHBondLength)
            component._atoms.append(Atom(name="HD11",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CG1"],component.centers["CD1"],component.centers["HD11"])
            component._atoms.append(Atom(name="HD12",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HD13",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))

        elif component.sequence == "LEU":
            A1,A2 = pyramidal_hydrogens(component.centers["C"],component.centers["CA"],component.centers["CB"])
            component._atoms.append(Atom(name="HA",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CA"],component.centers["CB"],component.centers["CG"])
            component._atoms.append(Atom(name="HB2",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HB3",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = planar_hydrogens(component.centers["CA"],component.centers["N"],component.centers["CG"]) # Note: Planar_hydrogens might not be ideal for HG on LEU. Re-evaluate geometry if needed.
            component._atoms.append(Atom(name="HG",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = planar_hydrogens(component.centers["CB"],component.centers["HB3"],component.centers["CD1"]) # Using HB3 to help define plane for CD1 hydrogens
            component._atoms.append(Atom(name="HD12",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CG"],component.centers["CD1"],component.centers["HD12"]) # Using HD12 to complete tetrahedron
            component._atoms.append(Atom(name="HD11",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HD13",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = planar_hydrogens(component.centers["CB"],component.centers["HB2"],component.centers["CD2"]) # Using HB2 for CD2 hydrogens
            component._atoms.append(Atom(name="HD21",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CG"],component.centers["CD2"],component.centers["HD21"]) # Using HD21
            component._atoms.append(Atom(name="HD22",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HD23",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))

        elif component.sequence == "LYS":
            A1,A2 = pyramidal_hydrogens(component.centers["CG"],component.centers["CD"],component.centers["CE"])
            component._atoms.append(Atom(name="HD3",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HD2",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CB"],component.centers["CG"],component.centers["CD"])
            component._atoms.append(Atom(name="HG3",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HG2",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CD"],component.centers["CE"],component.centers["NZ"])
            component._atoms.append(Atom(name="HE3",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HE2",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CA"],component.centers["CB"],component.centers["CG"])
            component._atoms.append(Atom(name="HB3",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HB2",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["C"],component.centers["CA"],component.centers["CB"])
            component._atoms.append(Atom(name="HA",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = planar_hydrogens(component.centers["CD"],component.centers["CE"],component.centers["NZ"],NHBondLength) # HZ3 is one of these
            component._atoms.append(Atom(name="HZ3",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry)) # Assuming A1 is HZ3 for now
            A1_other,A2_other = pyramidal_hydrogens(component.centers["CE"],component.centers["NZ"],component.centers["HZ3"]) # Use the just added HZ3
            component._atoms.append(Atom(name="HZ2",x=A1_other[0],y=A1_other[1],z=A1_other[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HZ1",x=A2_other[0],y=A2_other[1],z=A2_other[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))

        elif component.sequence == "MET":
            A1,A2 = pyramidal_hydrogens(component.centers["C"],component.centers["CA"],component.centers["CB"])
            component._atoms.append(Atom(name="HA",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CA"],component.centers["CB"],component.centers["CG"])
            component._atoms.append(Atom(name="HB3",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HB2",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CB"],component.centers["CG"],component.centers["SD"])
            component._atoms.append(Atom(name="HG3",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HG2",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = planar_hydrogens(component.centers["CG"],component.centers["SD"],component.centers["CE"],NHBondLength)
            component._atoms.append(Atom(name="HE1",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry)) # HE1 is one of these
            A1_other,A2_other = pyramidal_hydrogens(component.centers["SD"],component.centers["CE"],component.centers["HE1"]) # Use the just added HE1
            component._atoms.append(Atom(name="HE3",x=A1_other[0],y=A1_other[1],z=A1_other[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HE2",x=A2_other[0],y=A2_other[1],z=A2_other[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))

        elif component.sequence == "PHE":
            A1,A2 = pyramidal_hydrogens(component.centers["C"],component.centers["CA"],component.centers["CB"])
            component._atoms.append(Atom(name="HA",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CA"],component.centers["CB"],component.centers["CG"])
            component._atoms.append(Atom(name="HB3",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HB2",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1 = planar_ring_hydrogen(component.centers["CG"],component.centers["CD1"],component.centers["CE1"],NHBondLength)
            component._atoms.append(Atom(name="HD1",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1 = planar_ring_hydrogen(component.centers["CD1"],component.centers["CE1"],component.centers["CZ"],NHBondLength)
            component._atoms.append(Atom(name="HE1",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1 = planar_ring_hydrogen(component.centers["CE1"],component.centers["CZ"],component.centers["CE2"],NHBondLength)
            component._atoms.append(Atom(name="HZ",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1 = planar_ring_hydrogen(component.centers["CZ"],component.centers["CE2"],component.centers["CD2"],NHBondLength)
            component._atoms.append(Atom(name="HE2",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1 = planar_ring_hydrogen(component.centers["CG"],component.centers["CD2"],component.centers["CE2"],NHBondLength)
            component._atoms.append(Atom(name="HD2",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))

        elif component.sequence == "PRO":
            A1,A2 = pyramidal_hydrogens(component.centers["C"],component.centers["CA"],component.centers["CB"])
            component._atoms.append(Atom(name="HA",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            # H on Nitrogen in PRO is often absent or needs special handling due to ring.
            # The original code used pyramidal_hydrogens(self.centers["CA"],self.centers["N"],self.centers["CD"]) for "H"
            # This might need re-evaluation based on standard Proline geometry if issues arise.
            if "N" in component.centers and "CA" in component.centers and "CD" in component.centers:
                 A1_N_H,A2_N_H = pyramidal_hydrogens(component.centers["CA"],component.centers["N"],component.centers["CD"])
                 component._atoms.append(Atom(name="H",x=A1_N_H[0],y=A1_N_H[1],z=A1_N_H[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))

            A1,A2 = pyramidal_hydrogens(component.centers["N"],component.centers["CD"],component.centers["CG"])
            component._atoms.append(Atom(name="HD2",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HD3",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CD"],component.centers["CG"],component.centers["CB"])
            component._atoms.append(Atom(name="HG2",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HG3",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CA"],component.centers["CB"],component.centers["CG"])
            component._atoms.append(Atom(name="HB3",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HB2",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))

        elif component.sequence == "SER":
            A1,A2 = pyramidal_hydrogens(component.centers["C"],component.centers["CA"],component.centers["CB"])
            component._atoms.append(Atom(name="HA",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CA"],component.centers["CB"],component.centers["OG"])
            component._atoms.append(Atom(name="HB3",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HB2",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = planar_hydrogens(component.centers["CA"],component.centers["CB"],component.centers["OG"],NHBondLength)
            component._atoms.append(Atom(name="HG",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))

        elif component.sequence == "THR":
            A1,A2 = pyramidal_hydrogens(component.centers["C"],component.centers["CA"],component.centers["CB"])
            component._atoms.append(Atom(name="HA",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CA"],component.centers["CB"],component.centers["CG2"]) # HB uses CG2
            component._atoms.append(Atom(name="HB",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry)) # Assuming A1 is HB
            A1,A2 = planar_hydrogens(component.centers["CA"],component.centers["CB"],component.centers["CG2"],NHBondLength)
            component._atoms.append(Atom(name="HG21",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1_other,A2_other = pyramidal_hydrogens(component.centers["CB"],component.centers["CG2"],component.centers["HG21"])
            component._atoms.append(Atom(name="HG23",x=A1_other[0],y=A1_other[1],z=A1_other[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HG22",x=A2_other[0],y=A2_other[1],z=A2_other[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = planar_hydrogens(component.centers["CG2"],component.centers["HG23"],component.centers["OG1"]) # Using HG23 to define plane for OG1 hydrogen
            component._atoms.append(Atom(name="HG1",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))

        elif component.sequence == "TRP":
            A1,A2 = pyramidal_hydrogens(component.centers["C"],component.centers["CA"],component.centers["CB"])
            component._atoms.append(Atom(name="HA",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CA"],component.centers["CB"],component.centers["CG"])
            component._atoms.append(Atom(name="HB3",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HB2",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1 = planar_ring_hydrogen(component.centers["CG"],component.centers["CD1"],component.centers["NE1"],NHBondLength)
            component._atoms.append(Atom(name="HD1",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1 = planar_ring_hydrogen(component.centers["CD1"],component.centers["NE1"],component.centers["CE2"],NHBondLength)
            component._atoms.append(Atom(name="HE1",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1 = planar_ring_hydrogen(component.centers["CE2"],component.centers["CZ2"],component.centers["CH2"],NHBondLength)
            component._atoms.append(Atom(name="HZ2",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1 = planar_ring_hydrogen(component.centers["CZ2"],component.centers["CH2"],component.centers["CZ3"],NHBondLength)
            component._atoms.append(Atom(name="HH2",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1 = planar_ring_hydrogen(component.centers["CH2"],component.centers["CZ3"],component.centers["CE3"],NHBondLength)
            component._atoms.append(Atom(name="HZ3",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1 = planar_ring_hydrogen(component.centers["CZ3"],component.centers["CE3"],component.centers["CD2"],NHBondLength)
            component._atoms.append(Atom(name="HE3",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))

        elif component.sequence == "TYR":
            A1,A2 = pyramidal_hydrogens(component.centers["C"],component.centers["CA"],component.centers["CB"])
            component._atoms.append(Atom(name="HA",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CA"],component.centers["CB"],component.centers["CG"])
            component._atoms.append(Atom(name="HB3",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HB2",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1 = planar_ring_hydrogen(component.centers["CG"],component.centers["CD2"],component.centers["CE2"],NHBondLength)
            component._atoms.append(Atom(name="HD2",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1 = planar_ring_hydrogen(component.centers["CD2"],component.centers["CE2"],component.centers["CZ"],NHBondLength)
            component._atoms.append(Atom(name="HE2",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1 = planar_ring_hydrogen(component.centers["CZ"],component.centers["CE1"],component.centers["CD1"],NHBondLength)
            component._atoms.append(Atom(name="HE1",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1 = planar_ring_hydrogen(component.centers["CG"],component.centers["CD1"],component.centers["CE1"],NHBondLength)
            component._atoms.append(Atom(name="HD1",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            # HG for TYR (on OH)
            if "OH" in component.centers and "CZ" in component.centers and "CE1" in component.centers : # Ensure OH and other atoms exist
                A1_HG, _ = planar_hydrogens(component.centers["CE1"], component.centers["CZ"], component.centers["OH"], NHBondLength) # Simplified, might need adjustment
                component._atoms.append(Atom(name="HH", x=A1_HG[0], y=A1_HG[1], z=A1_HG[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))


        elif component.sequence == "VAL":
            A1,A2 = pyramidal_hydrogens(component.centers["C"],component.centers["CA"],component.centers["CB"])
            component._atoms.append(Atom(name="HA",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = pyramidal_hydrogens(component.centers["CA"],component.centers["CB"],component.centers["CG1"]) # HB uses CG1
            component._atoms.append(Atom(name="HB",x=A2[0],y=A2[1],z=A2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry)) # Assuming A2 is HB
            A1,A2 = planar_hydrogens(component.centers["CA"],component.centers["CB"],component.centers["CG1"],NHBondLength)
            component._atoms.append(Atom(name="HG11",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1_other,A2_other = pyramidal_hydrogens(component.centers["CB"],component.centers["CG1"],component.centers["HG11"])
            component._atoms.append(Atom(name="HG12",x=A1_other[0],y=A1_other[1],z=A1_other[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HG13",x=A2_other[0],y=A2_other[1],z=A2_other[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            A1,A2 = planar_hydrogens(component.centers["CA"],component.centers["CB"],component.centers["CG2"],NHBondLength)
            component._atoms.append(Atom(name="HG23",x=A1[0],y=A1[1],z=A1[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry)) # HG23 is one of these
            A1_other_2,A2_other_2 = pyramidal_hydrogens(component.centers["CB"],component.centers["CG2"],component.centers["HG23"])
            component._atoms.append(Atom(name="HG21",x=A1_other_2[0],y=A1_other_2[1],z=A1_other_2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))
            component._atoms.append(Atom(name="HG22",x=A2_other_2[0],y=A2_other_2[1],z=A2_other_2[2], pdb=component.pdb, model=component.model, chain=component.chain, component_id=component.sequence, component_number=component.number, alt_id=component.alt_id, insertion_code=component.insertion_code, symmetry=component.symmetry))

    except (KeyError, AttributeError, TypeError, IndexError, ValueError) as e:
        # It's good practice to log which component failed, if possible.
        unit_id_str = component.unit_id() if hasattr(component, 'unit_id') else 'UnknownComponent'
        print(f"{unit_id_str} Adding amino acid hydrogens failed: {type(e).__name__} - {e}")
    except Exception as e: # Catch any other unexpected errors
        unit_id_str = component.unit_id() if hasattr(component, 'unit_id') else 'UnknownComponent'
        print(f"{unit_id_str} Adding amino acid hydrogens failed with an unexpected error: {type(e).__name__} - {e}")
