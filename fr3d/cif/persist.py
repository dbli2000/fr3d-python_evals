"""This package deals with writing and restoring cif file objects.
"""

import json
import warnings
from fr3d.cif.reader import Cif
from pdbx.reader.PdbxContainers import DataContainer, DataCategory


def _category_to_dict(category):
    """Converts a DataCategory object to a dictionary."""
    cat_dict = {'name': category.name, 'attributes': category.attribute_list, 'rows': []}
    for i in range(category.row_count):
        cat_dict['rows'].append(category.row_list[i])
    return cat_dict

def _container_to_dict(container):
    """Converts a DataContainer object to a dictionary."""
    cont_dict = {'name': container.name, 'categories': []}
    for cat_name in container.get_object_name_list():
        cont_dict['categories'].append(_category_to_dict(container.get_object(cat_name)))
    return cont_dict

def _dict_to_category(cat_dict):
    """Converts a dictionary back to a DataCategory object."""
    category = DataCategory(cat_dict['name'])
    for attr in cat_dict['attributes']:
        category.append_attribute(attr)
    for row_data in cat_dict['rows']:
        category.append(row_data)
    return category

def _dict_to_container(cont_dict):
    """Converts a dictionary back to a DataContainer object."""
    container = DataContainer(cont_dict['name'])
    for cat_data in cont_dict['categories']:
        container.append(_dict_to_category(cat_data))
    return container


def serialize(handle, cif_obj):
    """Serialize a Cif object to a JSON formatted file.

    :handle: The filehandle to write to.
    :cif_obj: The Cif object to persist.
    """
    if not isinstance(cif_obj, Cif):
        raise TypeError("Expected a Cif object for serialization.")

    # Convert DataContainer to a serializable dictionary
    # This assumes cif_obj.data is a DataContainer
    # and that DataContainer has methods to access its contents
    # in a way that can be converted to basic Python types.

    # If cif_obj.data is already a dictionary or simple structure, this might be simpler.
    # Based on the original pickle code, cif_obj.data seems to be a complex object.
    # We need to ensure all relevant parts of the DataContainer are captured.

    # A more robust approach might involve defining __json__ methods
    # in the PDBx classes if we control them, or writing more specific converters here.

    data_to_serialize = _container_to_dict(cif_obj.data)
    json.dump(data_to_serialize, handle, indent=4)


def deserialize(handle):
    """Load a Cif object from a JSON formatted file.

    :handle: The filehandle to read from.
    :returns: A new Cif object from the persisted data.
    """
    try:
        data_from_json = json.load(handle)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON: {e}")

    # Convert the dictionary back to a DataContainer object
    reconstructed_data_container = _dict_to_container(data_from_json)

    # Reconstruct the Cif object
    # This assumes Cif can be reconstructed with a DataContainer instance
    return Cif(data=reconstructed_data_container)
