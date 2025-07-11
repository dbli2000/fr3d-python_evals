"""
Nucleic Acid Hydrogen Inference for FR3D.
"""
import numpy as np
from fr3d import definitions as defs
from fr3d.data.atoms import Atom
from fr3d.data.mapping import modified_base_to_parent, parent_atom_to_modified, \
                               modified_base_atom_list, modified_base_to_hydrogens, \
                               modified_base_to_hydrogens_coordinates

def infer_na_hydrogens(component):
    """
    Infers and/or corrects hydrogen atom positions for a nucleic acid component.
    Modifies component._atoms in place.
    """
    """
    Infers and/or corrects hydrogen atom positions for a nucleic acid component.
    Modifies component._atoms in place.
    """
    # This function needs access to:
    # component.sequence, component._atoms, component.base_center, component.rotation_matrix
    # component.unit_id() (for logging, if needed)

    def get_amino_hydrogen_coords(target_component, heavy_atom_name, amino_h1_name, amino_h2_name):
        """
        Helper function to retrieve the coordinates of amino hydrogens and specified heavy atom.
        Separate processing for modified nucleotides.
        If modified nucleotide has a mapping, checks to see if the atom maps to the name of the passed in parent.
        Returns 3 triples of atom coordinates of a (heavy_atom_coords, amino_h1_coords, amino_h2_coords)
        """
        heavy_coords_found = None
        amino_h1_coords_found = None
        amino_h2_coords_found = None

        # Standard base
        if target_component.sequence in defs.NAbasehydrogens:
            for atom_obj in target_component._atoms:
                if atom_obj.name == heavy_atom_name:
                    heavy_coords_found = atom_obj.coordinates()
                elif atom_obj.name == amino_h1_name:
                    amino_h1_coords_found = atom_obj.coordinates()
                elif atom_obj.name == amino_h2_name:
                    amino_h2_coords_found = atom_obj.coordinates()
        # Mapped modified base
        elif target_component.sequence in modified_base_to_parent:
            # Ensure mappings are loaded and accessible for the current modified nucleotide
            if target_component.sequence not in parent_atom_to_modified or \
               target_component.sequence not in modified_base_atom_list:
                # This case should ideally not happen if mappings are correctly loaded
                # print(f"Warning: Mapping data missing for {target_component.sequence}")
                return None, None, None

            for atom_obj in target_component._atoms:
                # Check if atom_obj.name is a key in the specific modified nucleotide's mapping dict
                if atom_obj.name in parent_atom_to_modified.get(target_component.sequence, {}):
                    parent_equivalent_atom_name = parent_atom_to_modified[target_component.sequence][atom_obj.name]
                    if parent_equivalent_atom_name == heavy_atom_name:
                        heavy_coords_found = atom_obj.coordinates()
                    elif parent_equivalent_atom_name == amino_h1_name:
                        amino_h1_coords_found = atom_obj.coordinates()
                    elif parent_equivalent_atom_name == amino_h2_name:
                        amino_h2_coords_found = atom_obj.coordinates()
                # Fallback for atoms that might not be in parent_atom_to_modified but are part of the base
                # This part of the original logic was a bit unclear; ensuring direct name match for safety if unmapped.
                elif atom_obj.name in modified_base_atom_list.get(target_component.sequence, []):
                    if atom_obj.name == heavy_atom_name: # Direct match if parent name is same
                         heavy_coords_found = atom_obj.coordinates()
                    elif atom_obj.name == amino_h1_name:
                         amino_h1_coords_found = atom_obj.coordinates()
                    elif atom_obj.name == amino_h2_name:
                         amino_h2_coords_found = atom_obj.coordinates()


        return heavy_coords_found, amino_h1_coords_found, amino_h2_coords_found

    try:
        # Common logic for adding/fixing hydrogens
        parent_sequence_for_hydrogens = None
        hydrogens_to_add_rules = None # Standard coordinates for hydrogens

        is_standard_base = component.sequence in defs.NAbasehydrogens
        is_mapped_modified_base = component.sequence in modified_base_to_parent

        if is_standard_base:
            parent_sequence_for_hydrogens = component.sequence
            hydrogens_to_add_rules = defs.NAbasehydrogens.get(component.sequence, [])
            standard_coords_for_hydrogens = defs.NAbasecoordinates.get(component.sequence, {})
        elif is_mapped_modified_base:
            parent_sequence_for_hydrogens = modified_base_to_parent.get(component.sequence)
            # For modified, we use specific hydrogen lists and their mapped parent coordinates
            hydrogens_to_add_rules = modified_base_to_hydrogens.get(component.sequence, [])
            standard_coords_for_hydrogens = modified_base_to_hydrogens_coordinates.get(component.sequence, {})
        else:
            return # Not a standard or mapped modified NA base

        # Correct existing amino hydrogens
        # This logic needs careful adaptation for standard vs modified
        heavy_ref_atom = None
        h1_name, h2_name = None, None

        current_parent_seq = parent_sequence_for_hydrogens # Could be original or parent of modified
        if current_parent_seq in ["A", "DA"]:
            heavy_ref_atom, h1_name, h2_name = "N7", "H61", "H62"
        elif current_parent_seq in ["C", "DC"]:
            heavy_ref_atom, h1_name, h2_name = "C5", "H41", "H42"
        elif current_parent_seq in ["G", "DG"]:
            heavy_ref_atom, h1_name, h2_name = "N1", "H21", "H22"

        if heavy_ref_atom:
            # Get coordinates based on whether it's standard or mapped modified
            h_heavy_coords, h_amino1_coords, h_amino2_coords = get_amino_hydrogen_coords(component, heavy_ref_atom, h1_name, h2_name)

            if h_heavy_coords is not None and h_amino1_coords is not None and h_amino2_coords is not None:
                dist1_sq = np.sum((h_heavy_coords - h_amino1_coords)**2)
                dist2_sq = np.sum((h_heavy_coords - h_amino2_coords)**2)

                # H62 should be closer to N7 than H61 (A), H42 closer to C5 than H41 (C), H22 closer to N1 than H21 (G)
                # The names h1_name and h2_name are parent names. We need to find the actual atoms in component.

                # Find the actual atom objects in the component to swap their coordinates
                atom_to_swap1 = None
                atom_to_swap2 = None

                if is_standard_base:
                    for atom_obj in component._atoms:
                        if atom_obj.name == h1_name: atom_to_swap1 = atom_obj
                        if atom_obj.name == h2_name: atom_to_swap2 = atom_obj
                elif is_mapped_modified_base:
                     # For modified, find atoms whose parent mapping matches h1_name, h2_name
                    map_to_parent = parent_atom_to_modified.get(component.sequence, {})
                    for atom_obj in component._atoms:
                        if map_to_parent.get(atom_obj.name) == h1_name: atom_to_swap1 = atom_obj
                        if map_to_parent.get(atom_obj.name) == h2_name: atom_to_swap2 = atom_obj

                if dist1_sq < dist2_sq: # Incorrect labeling, swap needed
                    if atom_to_swap1 and atom_to_swap2:
                        # print(f"Swapping {atom_to_swap1.name} and {atom_to_swap2.name} for {component.unit_id()}")
                        temp_coords = atom_to_swap1.coordinates().copy()
                        atom_to_swap1.x, atom_to_swap1.y, atom_to_swap1.z = atom_to_swap2.x, atom_to_swap2.y, atom_to_swap2.z
                        atom_to_swap2.x, atom_to_swap2.y, atom_to_swap2.z = temp_coords[0], temp_coords[1], temp_coords[2]


        # Add missing hydrogens
        if component.rotation_matrix is None or component.base_center is None:
            # print(f"Warning: Cannot add hydrogens for {component.unit_id()} due to missing rotation matrix or base center.")
            return

        existing_atom_names = {atom.name for atom in component._atoms}

        hydrogens_to_actually_add = []
        if is_standard_base:
            for h_name in hydrogens_to_add_rules: # these are parent hydrogen names
                 if h_name not in existing_atom_names:
                    hydrogens_to_actually_add.append(h_name)
        elif is_mapped_modified_base:
            # hydrogens_to_add_rules contains actual modified hydrogen names
            # standard_coords_for_hydrogens maps these modified H names to parent standard H coords
            for mod_h_name in hydrogens_to_add_rules:
                if mod_h_name not in existing_atom_names:
                     hydrogens_to_actually_add.append(mod_h_name)


        for h_name_to_add in hydrogens_to_actually_add:
            # h_name_to_add is the actual name for the new hydrogen atom
            # standard_h_coords are the coordinates of its PARENT equivalent in standard orientation
            standard_h_coords = standard_coords_for_hydrogens.get(h_name_to_add)

            if standard_h_coords is None:
                # print(f"Warning: No standard coordinates for hydrogen {h_name_to_add} in {component.sequence} ({parent_sequence_for_hydrogens}). Skipping.")
                continue

            # Need to ensure standard_h_coords is a 1D array for dot product
            standard_h_coords_arr = np.array(standard_h_coords).flatten()
            if standard_h_coords_arr.shape != (3,):
                # print(f"Warning: Incorrect shape for standard_h_coords for {h_name_to_add}. Shape: {standard_h_coords_arr.shape}")
                continue

            # Transform parent standard H coords to current component's frame
            # component.base_center is a 1D array (3,)
            # component.rotation_matrix is (3,3)
            # np.dot(self.rotation_matrix, standard_h_coords_arr) is (3,)
            # newcoordinates should be (3,)
            newcoordinates_arr = component.base_center + np.dot(component.rotation_matrix, standard_h_coords_arr)

            component._atoms.append(Atom(name=h_name_to_add,
                                         x=newcoordinates_arr[0],
                                         y=newcoordinates_arr[1],
                                         z=newcoordinates_arr[2],
                                         pdb=component.pdb, model=component.model, chain=component.chain,
                                         component_id=component.sequence, component_number=component.number,
                                         alt_id=component.alt_id, insertion_code=component.insertion_code,
                                         symmetry=component.symmetry
                                         ))

    except (KeyError, AttributeError, TypeError, IndexError, ValueError) as e:
        print(f"{component.unit_id()} Adding NA hydrogens failed: {type(e).__name__} - {e}")
    except Exception as e: # Catch any other unexpected errors
        print(f"{component.unit_id()} Adding NA hydrogens failed with an unexpected error: {type(e).__name__} - {e}")
