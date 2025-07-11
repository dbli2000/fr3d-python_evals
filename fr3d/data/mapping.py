# Read file called atom_mappings.txt, create dictionary from this where keys are modified
# nucleotides and values are a triple of parent sequence, parent atom, then corresponding
# atom from modified nucleotide.

# NOTES:
    # How the following dictionaries work

    # modified_atom_to_parent['4EN']['N8'] = 'C8': 4EN is modified, N8 is from the modified 4EN, corresponds to parent A's C8
    # parent_atom_to_modified['4EN']['C8'] = 'N8': 4EN is modified, C8 is from parent A, corresponds to modified 4EN's N8

    # modified_base_to_parent['PSU'] = 'U': PSU key yields parent U
    # modified_base_to_parent['4EN'] = 'A': 4EN key yields parent A
    # modified_base_to_hydrogens['PSU']: list of hydrogens on the base of PSU
    # modified_base_to_hydrogens_coordinates['PSU']['HN1']: triple of coordinates of H5, the hydrogen of U that PSU HN1 is mapped to
    # modified_base_atom_list['PSU']: list of names of all atoms in PSU

from fr3d import definitions as defs
import os
import sys
import pkg_resources # Added for robust data file access

# Python 3 specific read mode
read_mode = 'rt'

def create_modified_nucleotide_to_parent_mappings():
    # Use pkg_resources to find the data file
    try:
        # Assuming 'fr3d' is the package name and 'data/atom_mappings.txt' is the path within the package
        filename = pkg_resources.resource_filename('fr3d', 'data/atom_mappings.txt')
    except ImportError:
        # Fallback for environments where pkg_resources might not work as expected
        # (e.g., running script directly without full package installation)
        # This fallback maintains previous behavior but is less robust.
        current_path, _ = os.path.split(os.path.abspath(__file__))
        filename = os.path.join(current_path, "atom_mappings.txt")
        # Consider logging a warning here if relying on fallback.

    #print(f'mapping.py is trying to open {filename}')

    try:
        with open(filename, read_mode) as fid:
            lines = fid.readlines()
    except FileNotFoundError:
        print(f"Error: mapping.py could not find atom_mappings.txt at {filename}")
        print("This can happen after installing with 'python setup.py install' with no known fix.")
        print("Instead, from the directory where setup.py is, use 'python -m pip install .'")
        # Re-raise or handle as appropriate for the application
        # For now, to maintain existing behavior of failing somewhat silently if file not found by create_map..
        raise # Or return empty dicts to prevent downstream errors if that's preferred

    modified_atom_map = {}

    for line in lines:
        fields = line.split()
        if len(fields) == 4:
            if not fields[2] in modified_atom_map:
                modified_atom_map[fields[2]] = []
            modified_atom_map[fields[2]].append((fields[0], fields[1], fields[3]))

    """
    for line in mapping_text.split("\n"):
        fields = line.split()
        if len(fields) == 4:
            if not fields[2] in modified_atom_map:
                modified_atom_map[fields[2]] = []
            modified_atom_map[fields[2]].append((fields[0], fields[1], fields[3]))
    """

    modified_base_to_hydrogens = {}
    modified_atom_to_parent = {}
    parent_atom_to_modified = {}
    modified_base_to_parent = {}
    modified_base_atom_list = {}
    modified_base_to_hydrogens_coordinates = {}

    for modified_nucleotide in modified_atom_map:
        modified_base_to_hydrogens[modified_nucleotide] = []
        modified_base_to_parent[modified_nucleotide] = {}
        modified_base_to_parent[modified_nucleotide] = modified_atom_map[modified_nucleotide][0][0]
        modified_base_to_hydrogens_coordinates[modified_nucleotide] = {}
        modified_base_atom_list[modified_nucleotide] = []
        modified_atom_to_parent[modified_nucleotide] = {}
        parent_atom_to_modified[modified_nucleotide] = {}

        for atom in modified_atom_map[modified_nucleotide]:
            if len(atom) == 3:
                modified_atom_to_parent[modified_nucleotide][atom[2]] = atom[1]
                parent_atom_to_modified[modified_nucleotide][atom[1]] = atom[2]
                if atom[1] in defs.NAbaseheavyatoms[atom[0]] or atom[1] in defs.NAbasehydrogens[atom[0]]: # The parent mapping is in the base
                    modified_base_atom_list[modified_nucleotide].append(atom[2])
                    modified_base_to_hydrogens[modified_nucleotide].append(atom[2])
                    modified_base_to_hydrogens_coordinates[modified_nucleotide][atom[2]] = (defs.NAbasecoordinates[atom[0]][atom[1]])

    return modified_base_to_hydrogens, modified_atom_to_parent, parent_atom_to_modified, modified_base_to_parent, modified_base_atom_list,  modified_base_to_hydrogens_coordinates

try:
    modified_base_to_hydrogens, modified_atom_to_parent, parent_atom_to_modified, modified_base_to_parent, modified_base_atom_list,  modified_base_to_hydrogens_coordinates = create_modified_nucleotide_to_parent_mappings()
    # print("Modified nucleotide mappings read successfully.")
except Exception as e:
    print("mapping.py is unable to load mappings for modified nucleotides.")
    print("This can happen after installing with 'python setup.py install' with no known fix.")
    print("Instead, from the directory where setup.py is, use 'python -m pip install .'")
    print('Error message: %s' % str(e))
