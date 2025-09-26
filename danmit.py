# -*- coding: utf-8 -*-
"""
Started on Sun Dec 22 2024

Displacement And Normal Map Interface Tool

This program allows Source 1 Engine texture bump (normal) maps to be generated
via a displacement map located in a VMF file.

It requires a VMF populated by brushes whose top faces are displacements parallel to the Top view,
with a uniform power (POWER, equal to 3 or 4) across all of them.

REQUIRED LIBRARIES:
    os (built-in)
    numpy
    Python Image Library

OPTIONAL:
    Making your VMF using Hammer++, or converting it by copying it from Hammer/Hammer++ into a new file in Hammer++,
    can speed up normal map creation for large maps.
    
@author: Nicks24 (GitHub)
"""
from os import getcwd, system
import numpy as np
from PIL import Image

#######------USER DEFINED CONSTANTS------#######
DISTANCE_DIVISOR = 1
# Factor to multiply the displacement distances by before the normal map is computed.
# Convenient if you do not want to paint normal maps exactly to scale so that they're easier to work with in Hammer.

VTEX_PATH = 'C:\\PROGRA~2\\Steam\\steamapps\\common\\half-life 2\\bin\\vtex.exe'
# Change to vtex executable for the game you're working with.

ENABLE_VTF_EXPORT = False
# Enables/Disables auto-export to VTF. 
# Will only export the normal map as an image file (with chosen type) if set to False.

TEXTURE_FILENAME_NOEXT = "material_name_here"
# Filename of the material whose normal map is to be generated (no extension).

NMAP_IMAGE_EXTENSION = "tga"
# File type of the exported normal map image if ENABLE_VTF_EXPORT = False.

VMF_FILENAME = "vmf_name_here.vmf"
# Filename of the VMF + file extension with the target displacement map.

POWER = 4
# Power of all displacement maps in the VMF under VMF_FILENAME (MUST BE CONSTANT)

RES_X = 32
# Horizontal resolution of normal map in pixels. Must be an integer power of 2.

RES_Y = 32
# Vertical resolution of normal map in pixels. Must be an integer power of 2.

TILING_MODE = 0
# The tiling mode for the normal map. 0 = untiled, 1 = tiled
# The pixels on the leftmost column/bottom row will be replaced by those on the rightmost column/top row.

HAMMER_PLUS_PLUS = False
# Is the VMF in Hammer++ format? Set to "False" if you're not sure/getting errors
################################################

def read_vmf(vmf_filename):
    """
    Opens the VMF with the inputted filename and reads its text data.
    Parameters
    ----------
    vmf_filename : String
        Filename of the VMF to be opened, including file extension.
    Returns
    -------
    vmf_data : String
        All the information stored as text in the VMF.
    """
    with open(vmf_filename, "r", encoding="UTF-8") as vmf:
        vmf_text = vmf.read()
        vmf.close()
    return vmf_text
def get_bracket_sandwich(vmf_data, string, indent_level=0):
    """
    Gets an isolated 'bracket sandwich' of a field with a given name and given identation level in the VMF.
    Parameters
    ----------
    vmf_data : String
        The VMF as a string
    string : String
        The name of the field, such as dispinfo, to isolate.
    indent_level : Integer
        The indentation level of the field, or how many identations between the name header and opening curly bracket.
    Returns
    -------
    vmf_data : String
        All the information stored as text in the VMF."""
    output = vmf_data[ vmf_data.find( (string) + ("\n") + ("\t" * indent_level) + ("{") ):]
    completion_correction = indent_level + 2
    #Adding this fixed number onto the end of the slice, so the sandwich has the bottom bread (closing curly bracket)
    output = output[: (output.find( "\n" + ("\t" * indent_level) + ("}") )) + (completion_correction)]
    return output
def get_field_by_id(vmf_data, field, _id, indent_level=0):
    """
    Gets an isolated field in specified VMF with specified id and identation level (of field).
    Parameters

    Parameters
    ----------
    vmf_data : String
        VMF as a string to find the field in.
    field : String
        The field name.
    _id : Integer
        id number of the field.
    indent_level : Integer, optional
        Indentation level of the field. The default is 0.

    Returns
    -------
    output : String
        The field with specified parameters.
    """
    _index = vmf_data.find( (field) + ("\n") + ("\t" * indent_level) + ("{") + ("\n") + ("\t" * (indent_level + 1))\
                          + '"id" "' + (_id) + '"')
    output = vmf_data[_index:]
    completion_correction = indent_level + 2
    #Adding this fixed number onto the end of the slice, so the sandwich has the bottom bread (closing curly bracket)
    output = output[: (output.find( "\n" + ("\t" * indent_level) + ("}") )) + (completion_correction)]
    return output
def get_vertexes(vmf_data):
    """
    Obtains all vertex positions from specified VMF or VMF part as a vertically stacked ndarray.

    Parameters
    ----------
    vmf_data : String
        VMF data or part thereof

    Returns
    -------
    vertexes_array : np.ndarray
        Vertically stacked ndarray containing vertex positions.
    """
    vertexes_array = np.zeros(3)
    while vmf_data.find('"v" "') != -1:
        vmf_data = vmf_data[ vmf_data.find('"v" "') + 5 : ]
        #We want the index pointing to the major digit of the 1st number so we add 5 to the index to get there.
        _string = vmf_data[:vmf_data.find('"')]
        _string = np.array(_string.split(" "), dtype="float")
        vertexes_array = np.vstack((vertexes_array, _string))
    vertexes_array = vertexes_array[1:]
    return vertexes_array
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
    #by choosing a different basis instead of x-y-z such that we have the ideal situation.
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
def get_plane_points(vmf_data):
    """
    Obtains the vertex positions from every plane field from specified VMF or VMF part as a vertically stacked ndarray.

    Parameters
    ----------
    vmf_data : String
        VMF data or part thereof

    Returns
    -------
    vertexes_array : np.ndarray
        Vertically stacked ndarray containing vertex positions as given by plane field.
    """
    planes_array = np.zeros((3,3))
    while vmf_data.find('"plane" "') != -1:
        vmf_data = vmf_data[ vmf_data.find('"plane" "') + 9 : ] 
        #We want the index pointing to the major digit of the 1st number so we add 9 to the index to get there.
        _string = vmf_data[:vmf_data.find('"')]
        _string = _string.replace("(", "")
        _string = _string.replace(")", "")
        plane_array = _string.split(" ")
        plane_array = np.array(plane_array, dtype="float")
        plane_array = plane_array.reshape((3,3))
        planes_array = np.vstack((planes_array, plane_array))
    planes_array = planes_array[3:]
    return planes_array
def get_vertices_from_plane(plane_points):
    """
    Obtains the vertices of a side from its plane field.

    Parameters
    ----------
    plane_points : np.ndarray
        A vertically stacked ndarray of points given in the plane field of a side

    Returns
    -------
    vertices : np.ndarray
        A vertically stacked ndarray of points corresponding to the 4 vertices of that side.

    """
    fourth_point = get_fourth_vertex(plane_points)
    vertices = np.vstack((plane_points, fourth_point))
    return vertices

vmf_raw = read_vmf(VMF_FILENAME)

NMAP_FILENAME = TEXTURE_FILENAME_NOEXT + "_normal.tga"

NUM_OF_BRUSHES = vmf_raw.count("solid\n\t{")

#### Generating dictionary of { solid id : solid centre coord } ####
solid_dict = {}
counter = 0
solid_marker = vmf_raw.find("solid\n\t{")
trunc_vmf_data = vmf_raw
if HAMMER_PLUS_PLUS:  
    while solid_marker != -1:
        
        trunc_vmf_data = trunc_vmf_data[solid_marker:]
        
        end_marker = trunc_vmf_data[17:].find('"')
        
        solid_id = trunc_vmf_data[17:end_marker+17]
        current_solid = get_field_by_id(vmf_raw, "solid", solid_id, 1)
         
        vertexes_stack = get_vertexes(current_solid)

        solid_centre = get_centre(vertexes_stack)
        
        solid_dict[solid_id] = solid_centre
        
        counter += 1
        trunc_vmf_data = trunc_vmf_data[end_marker+17:]
        solid_marker = trunc_vmf_data.find("solid\n\t{")
else:
    while solid_marker != -1:
        
        trunc_vmf_data = trunc_vmf_data[solid_marker:]
        
        end_marker = trunc_vmf_data[17:].find('"')
        
        solid_id = trunc_vmf_data[17:end_marker+17]
        current_solid = get_field_by_id(vmf_raw, "solid", solid_id, 1)
        
        vertexes_stack = np.zeros(3)
        planes = get_plane_points(current_solid)
        for i in range (6): #we expect our solid to be 6-sided
            side_vertices = get_vertices_from_plane(planes[3*i : 3*(i+1)])
            vertexes_stack = np.vstack((vertexes_stack, side_vertices))
        vertexes_stack = vertexes_stack[1:]
        
        solid_centre = get_centre(vertexes_stack)
        
        solid_dict[solid_id] = solid_centre
        
        counter += 1
        trunc_vmf_data = trunc_vmf_data[end_marker+17:]
        solid_marker = trunc_vmf_data.find("solid\n\t{")
####################################################################

#### Ordering the dictionary from left->right top->bottom from topleft w.r.t hammer top view ####

#so least x, greatest y -> greatest x, least y; running through like a document
#Going from least x to greatest x, decreasing y, then so on

solid_ids = np.array( list(solid_dict.keys()) )
centres = np.array( list(solid_dict.values()) )

centres = np.round(centres) # Accounting for loss of floating-point precision by rounding

Z = centres[0,2] # all Z coords should be the same so we'll just use the top-left one

X_MIN = np.min( centres[:,0] )
Y_MIN = np.min( centres[:,1] )

X_MAX = np.max( centres[:,0] )
Y_MAX = np.max( centres[:,1] )

x_length_solids = np.count_nonzero(centres[:,1] == Y_MIN)   
x_length_solid_units = (X_MAX - X_MIN) / (x_length_solids-1) #Max x and y is of centre, so we go over 1 less solid in the difference.

y_length_solids = np.count_nonzero(centres[:,0] == X_MIN)
y_length_solid_units = (Y_MAX - Y_MIN) / (y_length_solids-1)

x_centres = (x_length_solid_units * np.arange(0, (x_length_solids))) + X_MIN
y_centres = (y_length_solid_units * (-1) * np.arange(0, (y_length_solids))) + Y_MAX


NODES_PER_SOLID = 2**POWER + 1 # Number of nodes along one direction for a displacement of given power.

RESOLUTION_X = NODES_PER_SOLID * x_length_solids
RESOLUTION_Y = NODES_PER_SOLID * y_length_solids
        
####---Initialising arrays---####
ordered_solid_id_list = np.zeros(NUM_OF_BRUSHES) # ordered array of solid ids (order detailed further above)
ordered_solid_centre_list = np.zeros(NUM_OF_BRUSHES, dtype=np.ndarray)
ordered_solid_field_list = np.zeros(NUM_OF_BRUSHES, dtype=np.ndarray) # ordered array of "solid" fields from brush
ordered_solid_dispinfo_list = np.zeros(NUM_OF_BRUSHES, dtype=np.ndarray) # ordered array of "dispinfo" fields from brush
ordered_solid_distance_list = np.zeros(NUM_OF_BRUSHES, dtype=np.ndarray)
ordered_solid_normals_list = np.zeros(NUM_OF_BRUSHES, dtype=np.ndarray)
ordered_solid_direction_list = np.zeros(NUM_OF_BRUSHES, dtype=np.ndarray)
ordered_solid_distance_field = np.zeros(NUM_OF_BRUSHES, dtype=np.ndarray)
ordered_startpos_config_list = np.zeros(NUM_OF_BRUSHES, dtype=np.ndarray)
ordered_solid_displacement_field = np.zeros(NUM_OF_BRUSHES, dtype=np.ndarray)
displacement_field_array = np.zeros((y_length_solids, x_length_solids, NODES_PER_SOLID, NODES_PER_SOLID))
grand_displacement_field = np.zeros((NODES_PER_SOLID * y_length_solids, NODES_PER_SOLID * x_length_solids))
####--------------------------####


####--- Declaring some functions that require variables that we declared earlier before initialising our arrays ---####
def get_normals_sign(data):
    """
    Finds the sign field that is applied to the distance field to produce the displacement field.
    The "normals" field of the dispinfo field is used.

    Parameters
    ----------
    data : String
        Normals or outer non-ambiguous field.

    Returns
    -------
    binary_normals_array : np.ndarray
        The sign field.

    """
    normals_array = np.zeros(3 * (NODES_PER_SOLID))
    while data.find('"row') != -1:
        
        data = data[ data.find('"row') + 2 : ]
        data = data[ data.find('"') + 3 : ]
    
        _string = data[:data.find('"')]
        _string = np.array(_string.split(" "), dtype="float")
        normals_array = np.vstack((normals_array, _string))
    
    normals_array = normals_array[1:]
    normals_array = normals_array[:,2::3]
    binary_normals_array = np.where(normals_array > 0, 1, -1)
    return binary_normals_array
def get_distances(data):
    """
    Given the distance field or outer non-ambiguous field, obtains an np.ndarray
    of the distance field.

    Parameters
    ----------
    data : String
        Distance field or non-ambiguous outer field.

    Returns
    -------
    distances_array : np.ndarray
        Distance field as an ndarray.
    """
    distances_array = np.zeros(NODES_PER_SOLID)
    while data.find('"row') != -1:
        data = data[ data.find('"row') + 2 : ]
        data = data[ data.find('"') + 3 : ]
        _string = data[:data.find('"')]
        _string = np.array(_string.split(" "), dtype="float")
        distances_array = np.vstack((distances_array, _string))
    distances_array = distances_array[1:]
    return distances_array
def get_dispinfo(vmf_data, _id, mode="solid"):
    """
    Obtains the dispinfo field from side or solid with specified ID in given VMF part.
    Assumes that the solid has only 1 dispinfo-having side.
    
    Parameters
    ----------
    vmf_data : String
        VMF data or part thereof.
    _id : Integer
        The ID of the side or solid to find the dispinfo of.
    mode : String, optional
        Whether to find dispinfo from solid with specified _id or side with specified _id. 
        The default is "solid".

    Returns
    -------
    dispinfo_data : String
        The dispinfo field matching the parameters.
    """
    if mode == "side":
        _indent_level = 2
    else:
        _indent_level = 1 # mode == solid
        
    object_data = get_field_by_id(vmf_data, mode, _id, _indent_level)
    
    dispinfo_data = get_bracket_sandwich(object_data, "dispinfo", 3)
    
    #For "normals" part of dispinfo, only z-coord (and particularly its sign) matters; this and "distances" should
    #give us all the necessary info.
    return dispinfo_data
def get_dispinfo_start_position(dispinfo):
    """
    Finds the "startposition" parameter of a dispinfo field.

    Parameters
    ----------
    dispinfo : String
        Dispinfo field or outer non-ambiguous field.

    Returns
    -------
    startpos : np.ndarray
        Coordinates specified by the startposition parameter.
    """
    data = dispinfo
    data = data[data.find('"startposition" "[') + 18 : ]
    data = data[:data.find(']')]
    startpos = np.array(data.split(" "), dtype="float")
    return startpos
# 0: do nothing
# 1: flip horizontal
# 2: flip vertical
# 3: flip horizontal and vertical
def get_startpos_configuration(dispinfo, centre):
    """
    Obtains the start position configuration of the specified dispinfo field and centre
    of the brush the dispinfo field belongs to.

    Parameters
    ----------
    dispinfo : String
        The dispinfo field of the solid's top face.
    centre : np.ndarray
        Coords of the centre of the solid 

    Returns
    -------
    startpos_config : Integer
        The orientation number of the displacement field
    """
    start_pos = get_dispinfo_start_position(dispinfo)
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
    
    node_distance_x = x_length_solid_units / NODES_PER_SOLID
    node_distance_y = y_length_solid_units / NODES_PER_SOLID
    
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
    img = Image.new("RGB", (RESOLUTION_Y, RESOLUTION_X), (128, 128, 255))
    pixels = img.load()
    for c in range(img.size[1]):    # For every column
        for r in range(img.size[0]):    # For every row
            pixels[r,c] = (normal_map[c][r][0],\
                           normal_map[c][r][1],\
                           normal_map[c][r][2])
    img = img.resize((RES_X, RES_Y))
    #img.save(NMAP_FILENAME_NOEXT + (".png") + "")
    return img
###############################################################################
# #- Getting distance fields, combining with normals field to produce displacement field, combining displacement fields
# into one grand displacement field -#
counter = 0
for y_index, current_y in enumerate(y_centres):
    
    for x_index, current_x in enumerate(x_centres):
        
        target_id_index_condition = np.logical_and(centres[:,0] == current_x, centres[:,1] == current_y) # gen cond
        target_id_index = np.argwhere(target_id_index_condition) #do cond
        
        target_id = np.squeeze(solid_ids[target_id_index])
        target_centre = np.squeeze(centres[target_id_index])
        ordered_solid_id_list[counter] = target_id
        ordered_solid_centre_list[counter] = target_centre
        ordered_solid_field_list[counter] = get_field_by_id(vmf_raw, "solid", target_id, 1)
        
        ##-- Generating ordered dispinfo dict, hence generating displacement field pair --##

        ordered_solid_dispinfo_list[counter] = get_bracket_sandwich(ordered_solid_field_list[counter], "dispinfo", 3)
        ordered_solid_distance_list[counter] = get_bracket_sandwich(ordered_solid_dispinfo_list[counter], "distances", 4)
        ordered_solid_normals_list[counter] = get_bracket_sandwich(ordered_solid_dispinfo_list[counter], "normals", 4)
        ordered_startpos_config_list[counter] = get_startpos_configuration(ordered_solid_dispinfo_list[counter], ordered_solid_centre_list[counter])
        ordered_solid_direction_list[counter] = get_normals_sign(ordered_solid_normals_list[counter])
        ordered_solid_distance_field[counter] = get_distances(ordered_solid_distance_list[counter])
        ordered_solid_displacement_field[counter] = ordered_solid_direction_list[counter] * ordered_solid_distance_field[counter]
        ordered_solid_displacement_field[counter] = correct_displacement_field_orientation(ordered_solid_displacement_field[counter], ordered_startpos_config_list[counter])
        
        displacement_field_array[y_index][x_index] = ordered_solid_displacement_field[counter]
        
        grand_displacement_field[y_index * NODES_PER_SOLID : (y_index + 1) * NODES_PER_SOLID,\
                                 x_index * NODES_PER_SOLID : (x_index + 1) * NODES_PER_SOLID]\
                                = displacement_field_array[y_index][x_index] * DISTANCE_DIVISOR
        counter += 1
###############################################################################

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