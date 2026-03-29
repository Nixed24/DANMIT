# -*- coding: utf-8 -*-
"""
Started on Sun Dec 22 2024

Displacement And Normal Map Interface Tool

This program allows Source 1 Engine texture bump (normal) maps to be generated
via a displacement map located in a VMF file.

It requires a VMF populated by brushes whose top faces are displacements parallel to the Top view,
with a uniform displacement power across all of them.

REQUIRED LIBRARIES:
    os (built-in)
    numpy
    Python Image Library
    VMF Deserialiser (included)
    
@author: Nicks24 (GitHub)
"""

from os import getcwd, system
import numpy as np
from PIL import Image
import vmf_deserialiser

#######------USER DEFINED CONSTANTS------#######

DISTANCE_DIVISOR = 1
# Factor to multiply the displacement distances by before the normal map is computed.
# Convenient if you do not want to paint normal maps exactly to scale so that they're easier to work with in Hammer.

VTEX_PATH = 'C:\\PROGRA~2\\Steam\\steamapps\\common\\half-life 2\\bin\\vtex.exe'
# Change to vtex executable for the game you're working with.

ENABLE_VTF_EXPORT = True
# Enables/Disables auto-export to VTF. 
# Will only export the normal map as an image file (with chosen type) if set to False.

TEXTURE_FILENAME_NOEXT = "new_danmit_test"
# Filename of the material whose normal map is to be generated (no extension).

NMAP_IMAGE_EXTENSION = "tga"
# File type of the exported normal map image if ENABLE_VTF_EXPORT = False.

VMF_FILENAME = "vmf_deserialiser_test.vmf"
# Filename of the VMF + file extension with the target displacement map.

RES_X = 32
# Horizontal resolution of normal map in pixels. Must be an integer power of 2.

RES_Y = 32
# Vertical resolution of normal map in pixels. Must be an integer power of 2.

TILING_MODE = 0
# The tiling mode for the normal map. 0 = untiled, 1 = tiled
# The pixels on the leftmost column/bottom row will be replaced by those on the rightmost column/top row.

################################################

NMAP_FILENAME = TEXTURE_FILENAME_NOEXT + "_normal.tga"

def dispinfo_substructure_to_ndarray(substructure):
    returned_array = np.array([], dtype = float)
    for i in range (10000):
        try:
            current_row = substructure[("row" + str(i))]
        except:
            if len(returned_array) < 2:
                raise IndexError("Cannot have dispinfo substructure with less than 3 rows.")
            returned_array = returned_array[1:]
            break
        else:
            if i == 0:
                current_array = np.array(current_row.split(" "))
                returned_array = np.zeros(len(current_array), dtype=float)
            current_array = np.array(current_row.split(" "))
            returned_array = np.vstack((returned_array,current_array))
    returned_array = np.array(returned_array, dtype=float)
    return returned_array
def get_centre(vertexes_array):
    """Gets the average of all points specified when said points are a vertically-stacked ndarray of positions.
    Parameters
    ----------
    vertexes_array : np.ndarray
        Vertically stacked ndarray containing vertex positions.

    Returns
    -------
    centre_coord : np.ndarray
        Coordinates of the average of the positions inputted."""
    vertexes_array = np.array(vertexes_array, dtype=float)
    centre_coord = np.zeros(3)
    for axis_i in range (3):
        centre_coord[axis_i] = np.average(vertexes_array[:,axis_i])
    #Take the average position of all vertices to get the center. This almost definetly wouldn't work for planes.
    return centre_coord
def get_fourth_vertex(plane_points):
    """
    Given a set of three vertices that correspond to a side that is perpendicular to 1 spatial (x/y/z) axis,
    find the fourth vertex of that side.

    Parameters
    ----------
    plane_points : np.ndarray
        Vertically stacked ndarray of points corresponding to the three vertices given in the plane field of the side

    Raises
    ------
    ValueError
        When the 'perpendicular to 1 axis' condition is not met.

    Returns
    -------
    fourth_point : np.ndarray
        A numpy array corresponding (x/y/z wise) to the 4th vertex of the side.

    """
    #Assuming plane is a rectangle w.r.t x-y-z
    #And the plane points are 3 vertices of that rectangle (they usually are in Hammer)
    #We could generalise this to work in any axis but we'd have to map our plane to this situation,
    #by choosing a different basis
    plane_points = np.array(plane_points, dtype=float)
    fourth_point = np.zeros(3)
    x_s, y_s, z_s = plane_points[:,0], plane_points[:,1], plane_points[:,2]
    point_array = np.array([x_s, y_s, z_s])
    for c_i, c in enumerate(point_array):
        if (np.average(c) - c[0]) == 0:
            const_axis_value = c[0]
            ax_nt = np.delete(np.arange(3), c_i) #axes_nontrivial
            points_of_interest = point_array
            fourth_point[c_i] = const_axis_value
    if not('ax_nt' in locals() and 'points_of_interest' in locals()):
        raise ValueError('get_fourth_vertex could not find constant axis. Are all your brushes cuboidal?')
    #We know the plane that the points form will have normal vector in the direction of axis specified by const_axis_value   
    dupe_table = np.zeros(points_of_interest.shape)
    for y in range(points_of_interest.shape[0]):
        for x in range(points_of_interest.shape[1]):
            # the value on a given axis with no dupes (i.e that value does not appear anywhere else on that axis)
            # will be the value on that axis of our missing point, but the constant axis will of course be constant.
            dupe_table[y][x] = np.sum(np.count_nonzero(points_of_interest[y] == points_of_interest[y][x]))
    for a in ax_nt:
        fourth_point[a] = point_array[a][dupe_table[a] == 1][0]
    return fourth_point
def get_startpos_configuration(start_pos, centre):
    """
    Obtains the start position configuration of the specified dispinfo field and centre
    of the brush the dispinfo field belongs to.

    Parameters
    ----------
    startpos : np.ndarray
        Cpprds of the startpos of the displacement on the solid
    centre : np.ndarray
        Coords of the centre of the solid 

    Returns
    -------
    startpos_config : Integer
        The orientation number of the displacement field
    """
    start_pos = np.array(start_pos, dtype=float)
    centre = np.array(centre, dtype=float)
    startpos_config = None
    if start_pos[1] < centre[1]: # Bottom
        if start_pos[0] < centre[0]: # Left
            startpos_config = 2
        else: # Right
            startpos_config = 3
    if start_pos[1] > centre[1]: # Top
        if start_pos[0] < centre[0]: # Left
            startpos_config = 0
        else: # Right
            startpos_config = 1
    return startpos_config

############################## Reading VMF ####################################

VMF_FILENAME = "vmf_deserialiser_test.vmf"

vmf_deserialiser.settings["tagFirstInstance"] = True

vmf_dict = vmf_deserialiser.deserialise_vmf(VMF_FILENAME)

world_dict = vmf_dict["world&0"]

distance_fields = []
normals_fields = []
solid_centres = []
startpos_list = []
startpos_orientations = []
displacement_fields = []

on_dispinfo = False

for i in range (100000):
    try:
        current_solid = world_dict[("solid&" + str(i))]
        current_solid_vertices = np.zeros((1,3))
    except: # no more solids
        #print(f"{i} solids found")
        break
    for j in range (6):
        current_side = current_solid[("side&" + str(j))]
        try:
            current_dispinfo = current_side["dispinfo&0"]
            on_dispinfo = True
            current_start_pos = current_dispinfo["startposition"].replace("]", "").replace("[", "").split(" ")
            startpos_list.append(current_start_pos)
        except KeyError:
            pass
        finally:
            current_plane_points = np.array(current_side["plane"].replace(")", "").replace("(", "").split(" "))
            current_plane_points = current_plane_points.reshape((3,3))
            current_fourth_vertex = get_fourth_vertex(current_plane_points)
            vertices = np.vstack((current_plane_points, current_fourth_vertex))            
            current_solid_vertices = np.vstack((current_solid_vertices, vertices))
            if j == 5:
                current_solid_vertices = current_solid_vertices[1:]
                current_solid_centre = get_centre(current_solid_vertices)
                solid_centres.append(current_solid_centre)
                startpos_orientations.append(get_startpos_configuration(current_start_pos, current_solid_centre))
    if on_dispinfo:
        distance_fields.append(dispinfo_substructure_to_ndarray(current_dispinfo["distances&0"]))
        normals_fields.append(dispinfo_substructure_to_ndarray(current_dispinfo["normals&0"])[:,2::3])
        on_dispinfo = False

for i in range (len(distance_fields)):
    displacement_fields.append(normals_fields[i] * distance_fields[i])

nodes_per_solid = len(distance_fields[0][0])

centres = np.array(solid_centres)

x_min = np.min( centres[:,0] )
y_min = np.min( centres[:,1] )

x_max = np.max( centres[:,0] )
y_max = np.max( centres[:,1] )

x_length_solids = np.count_nonzero(centres[:,1] == y_min)   
x_length_solid_units = (x_max - x_min) / (x_length_solids-1) #Max x and y is of centre, so we go over 1 less solid in the difference.

y_length_solids = np.count_nonzero(centres[:,0] == x_min)
y_length_solid_units = (y_max - y_min) / (y_length_solids-1)

total_nodes_x = nodes_per_solid * x_length_solids

total_nodes_y = nodes_per_solid * y_length_solids

###############################################################################

def correct_displacement_field_orientation(displacement_field, config_num):
    """
    Given a distance/displacement field and orientation, rotate it such that it begins from the top-left
    and scans like a document.

    Parameters
    ----------
    displacement_field : np.ndarray
    The displacement field.
    
    config_num : Integer
    The orientation number of the displacement field, found by get_startpos_configuration()
    
    Returns
    -------
    transformed displacement_field
        Reorientated displacement field
    """
    if config_num == 3:
        return np.flipud(np.fliplr(displacement_field))
    if config_num == 2:
        return np.flipud(displacement_field)
    if config_num == 1:
        return np.fliplr(displacement_field)
    return displacement_field
def gen_normal_map(displacement_field):
    """
    Generates a corresponding normal map from the distance field,
    interpolating between distance points to find the normal vector.
    
    Parameters
    ----------
    displacement_field : np.ndarray
        The displacement field.
    Returns
    -------
    normal_map : np.ndarray
        The corresponding normal map.
    """
    #First we will go through left-right for each row to get the R
    #Top-Bottom for G
    #Then we will find the node with the highest and lowest distances, and
    #normalise it to 128->256 for the blue
    
    node_distance_x = x_length_solid_units / nodes_per_solid
    node_distance_y = y_length_solid_units / nodes_per_solid
    
    shape = displacement_field.shape
    shape = shape + (3,) # This 2D array of single floats will be a 2D array of 3 values (RGB values)
    normal_map = np.zeros(shape, dtype=int)
    
    for row in range(displacement_field.shape[0]):
        for index_x in range (len(displacement_field[0,:])-1):
            d = displacement_field[row][index_x] - displacement_field[row][index_x+1]
            theta_x = np.arcsin(d/np.sqrt(d**2+(node_distance_x)**2))
            normal_map[row][index_x][0] = np.floor(255 * ((theta_x / np.pi) + 0.5))
            
    for column in range(displacement_field.shape[1]):
        for index_y in range (len(displacement_field[:,0])-1):
            d = displacement_field[index_y][column] - displacement_field[index_y+1][column]
            theta_y = np.arcsin(d/np.sqrt(d**2+(node_distance_y)**2))
            normal_map[index_y][column][1] = np.floor(255 * ((theta_y / np.pi) + 0.5))
            
    normal_map[:,:,2] = displacement_field
    return normal_map
def normalise_distance_field(distance_field):
    #May need to rework this function
    """
    Normalises the inputted distance field such that its values are on the bounds
    [128,256) and follow a proper order.
     
    Parameters
    ----------
    distance_field : np.ndarray
       A distance field.
    Returns
    -------
    normalised_distance_field : np.ndarray
        The inputted distance field after normalisation.
    """
    max_dist = np.max(distance_field)
    min_dist = np.min(distance_field)
    if max_dist == min_dist:
        print("Cannot normalise distance field since max_dist = min_dist\n\
              Aborted normalisation, blue channel will be uniform")
        return distance_field
    normalised_distance_field = np.floor(distance_field * 128/(max_dist - min_dist))
    normalised_distance_field = normalised_distance_field + 128
    normalised_distance_field = np.where((normalised_distance_field == 128),\
                                         192,\
                                         normalised_distance_field) #set zeroes
    normalised_distance_field = np.where((normalised_distance_field > 255),\
                                         255,\
                                         normalised_distance_field) #Restrict upper
    normalised_distance_field = np.where((normalised_distance_field < 0),\
                                         0,\
                                         normalised_distance_field) #Restrict lower
    return normalised_distance_field
def gen_normal_map_image(normal_map):
    """
    Generates the correspnding image for an inputted normal map, which can
    then be imported by a VTF editor into a VTF file.
    
    Parameters
    ----------
    normal_map : np.ndarray
        The array representing the normal map.
    Returns
    -------
    img : PIL.Image.Image
        Corresponding image of the normal map exportable directly.
    """
    img = Image.new("RGB", (total_nodes_y, total_nodes_x), (128, 128, 255))
    pixels = img.load()
    for c in range(img.size[1]):    # For every column
        for r in range(img.size[0]):    # For every row
            pixels[r,c] = (normal_map[c][r][0],\
                           normal_map[c][r][1],\
                           normal_map[c][r][2])
    img = img.resize((RES_X, RES_Y))
    #img.save(NMAP_FILENAME_NOEXT + (".png") + "")
    return img

def main():
    grand_displacement_field = np.zeros((nodes_per_solid * y_length_solids, nodes_per_solid * x_length_solids))
    
    for x_index in range (x_length_solids):
        for y_index in range(y_length_solids):
            grand_displacement_field[y_index * nodes_per_solid : (y_index + 1) * nodes_per_solid,\
                                     x_index * nodes_per_solid : (x_index + 1) * nodes_per_solid]\
                                    = displacement_fields[y_index + (x_length_solids * x_index)] * DISTANCE_DIVISOR
                                    
    output_normal_map = gen_normal_map(grand_displacement_field)
    output_normal_map[:,:,2] = normalise_distance_field(output_normal_map[:,:,2])
    
    # TILING #
    if TILING_MODE == 1:
        output_normal_map[-1,:] = output_normal_map[0,:]
        output_normal_map[:,-1] = output_normal_map[:,0]
    elif TILING_MODE == 0:
        output_normal_map[-1,:] = np.array([127,127,192])
        output_normal_map[:,-1] = np.array([127,127,192])
    #  #  #  #  #
    
    normal_map_image = gen_normal_map_image(output_normal_map) #pil.image.image
    
    if ENABLE_VTF_EXPORT:
        normal_map_param_path = getcwd() + "\\" + TEXTURE_FILENAME_NOEXT + "_normal.txt"
        with open(normal_map_param_path, "w", encoding="UTF-8") as param_file:
            normal_map_image.save(NMAP_FILENAME)
            normal_map_path = getcwd() + "\\" + NMAP_FILENAME
            param_file.write('// This is a normal map parameter file auto-generated by DANMIT.\n')
            param_file.write('"nocompress" "1"\n')
            param_file.write('// "yourparam" "1"\n')
            param_file.close()
            vtex_instance = system(f'"{VTEX_PATH}" -nopause -dontusegamedir {normal_map_path}')
    else:
        normal_map_image.save(TEXTURE_FILENAME_NOEXT + "." + NMAP_IMAGE_EXTENSION)
        return
    
    main()
    