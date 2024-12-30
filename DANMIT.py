# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 01:13:11 2024

REQUIRED LIBRARIES:
    numpy
    Python Image Library (for generating the normal map image)

Source External Displacement-based Normal Map Tool

This program allows Source 1 Engine texture bump (a.k.a normal) maps to be generated
via a displacement map located in a VMF file.

It requires a VMF populated exclusively by square brushes whose top faces (located at Z_CEILING)
are displacements, with a uniform power (POWER) across all of them (usually 3 or 4)

@author: NS
"""

import numpy as np
from PIL import Image

#######------USER DEFINED CONSTANTS------#######
MAT_FILENAME_NOEXT = "normalmap_test001a"
#Filename of the material whose normal map is to be generated. Currently unused.

VMF_FILENAME = "normalmaptest6.vmf"
#Filename of the VMF with the target displacement map.

POWER = 4
#Power of all displacement maps in the VMF under VMF_FILENAME (MUST BE CONSTANT)

IN_NODES = 2**POWER
#Unused

Z_CEILING = 0
#The top of all of the brushes in the VMF should be located at this Z coord.

Z_FLOOR = -64
#The bottom of all of the brushes in the VMF should be located at this Z coord.
################################################











################################################
def find_all(a_str, sub):
    """
    Finds all occurrences of a substring inductively within a string.
    Parameters
    ----------
    a_str : string
        String on which to perform the function.
    sub : string
        Target substring.
    Yields
    ------
    start : GENERATOR
        Generator of target indices.
    """
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)
####Thank you Karl Knetchel for this function####
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
    vmf = open(vmf_filename, "r")
    data = vmf.read()
    vmf.close()
    return data
def get_brush_data(vmf_data):
    """
    Gets (noisy) data for all brushes in a VMF, is currently only
    programmed to find brush vertexes as necessary.
    Parameters
    ----------
    vmf_data : string
        The text body of a VMF.
    Returns
    -------
    list
        List with index 0 the number of brushes in the VMF as an integer
        Index 1 is a noisy array with all vertexes of the level inside.
    """
    brushdata = vmf_data[vmf_data.find("world"):].split("solid")
    numbrushes = len(brushdata) - 1
    vertexdata = vmf_data[vmf_data.find("world"):].split("vertices_plus")
    vertexdata = vertexdata[1:]
    for index, element in enumerate(vertexdata):
        vertexdata[index] = element[element.find("{")+1:element.find("}")]
    vertexdata = " ".join(str(m) for m in vertexdata)
    vertexdata = vertexdata.split('"')
    _vertexdata = []
    for n in range(3,(len(vertexdata) - 1),4):
        _vertexdata.append(vertexdata[n])
    _vertexdata = " ".join(str(m) for m in _vertexdata)
    _vertexdata = _vertexdata.split(' ')
    _vertexdata = np.array(_vertexdata, dtype="float")
    _vertexdata.shape = (24*numbrushes,3)
    return [numbrushes, _vertexdata]
def get_disp_data(data):
    """
    Finds all data under "dispinfo {...}" for each brush side.
    Parameters
    ----------
    vmf_data : string
        The or part of the text body of a VMF.
    Returns
    -------
    disp_data : list
        Noisy array of displacement info for all sides of brushes in the VMF.
    """
    disp_data = data.split("dispinfo")
    disp_data = disp_data[1:]
    return disp_data
def get_disp_normal_vector_field(disp_data):
    """
    Reads the map of normal vectors from the given displacement info field
    (what is found under the dispinfo statement in a VMF.)
    This is used to allow the program to properly recognise displacement nodes
    located below Z_CEILING.
    Parameters
    ----------
    disp_data : string
        String containing the displacement data, usually the return string of a
        get_disp_data call.
    Returns
    -------
    _matrix : np.ndarray
        A 2D map of the normal vectors at each point in the displacement.
    """
    matrix = []
    _matrix = []
    normfield = disp_data[disp_data.find("normals") + 6:]
    normfield = normfield[normfield.find("{") + 1:normfield.find("}")]
    normfield = normfield.split('"')
    start_point = normfield.index("row0") + 2 #start point + 2
    counter = 0
    for x in range (start_point,(len(normfield) - 1),4):
        counter += 1
        matrix.append(normfield[x])
    matrix = " ".join(str(y) for y in matrix)
    matrix = matrix.split(" ")
    matrix = np.array(matrix)
    for i in range(2,len(matrix),3):
        _matrix.append(matrix[i])
    _matrix = np.array(_matrix)
    _matrix.shape = (2**POWER+1,2**POWER+1) #5, 9, 17 = 2**power + 1
    return _matrix
def get_dist_field(truncated_brush_side):
    """
    Finds the distance field of a displacement on a brush side.
    Parameters
    ----------
    truncated_brush_side : string
        The data of an individual brush side in the VMF.
    Returns
    -------
    matrix : np.ndarray
        The distance field of that side as a 2-D numpy array.
    """
    matrix = []
    distfield = truncated_brush_side[truncated_brush_side.find("distances") + 9:]
    distfield = distfield[distfield.find("{") + 1:distfield.find("}")]
    distfield = distfield.split('"')
    start_point = distfield.index("row0") + 2 #start point + 2
    counter = 0
    for x in range (start_point,(len(distfield) - 1),4):
        counter += 1
        matrix.append(distfield[x])
    matrix = " ".join(str(y) for y in matrix)
    matrix = matrix.split(" ")
    matrix = np.array(matrix, dtype="float")
    matrix.shape = (2**POWER+1,2**POWER+1)#5, 9, 17 = 2**power + 1
    sign_field = get_disp_normal_vector_field(truncated_brush_side).astype("float")
    matrix = np.where(sign_field < 0, -1 * matrix, matrix)
    return matrix
def gen_normal_map(distance_field):
    """
    Generates a corresponding normal map from the distance field,
    interpolating between distance points to find the normal vector.
    Parameters
    ----------
    distance_field : np.ndarray
        The distance field.
    Returns
    -------
    normalMap : np.ndarray
        The corresponding normal map.
    """
    #First we will go through left-right for each row to get the R
    #Top-Bottom for G
    #Then we will find the node with the highest and lowest distances, and
    #normalise it to 128->256 for the blue
    distance_field = distance_field.astype(float)
    shape = distance_field.shape
    shape = shape + (3,)
    normal_map = np.zeros(shape, dtype=int)
    theta_map = np.zeros(shape)
    for row in range(distance_field.shape[0]):
        for index_x in range (len(distance_field[0,:])-1):
            d = distance_field[row][index_x] - distance_field[row][index_x+1]
            theta_x = np.arcsin(d/np.sqrt(d**2+(node_distance)**2))
            theta_map[row][index_x][0] = theta_x
            normal_map[row][index_x][0] = np.floor((theta_x + (np.pi/2))*(255/np.pi))
    for column in range(distance_field.shape[1]):
        for index_y in range (len(distance_field[:,0])-1):
            d = distance_field[index_y][column] - distance_field[index_y+1][column]
            theta_y = np.arcsin(d/np.sqrt(d**2+(node_distance)**2))
            theta_map[index_y][column][1] = theta_y
            #normal_map[index_y][column][1] = np.floor((theta_y + (np.pi/2))*(255/np.pi))
            normal_map[index_y][column][1] = 255 - np.floor((theta_y + (np.pi/2))*(255/np.pi))
    normal_map[:,:,2] = distance_field
    normal_map = normal_map[:-1,:-1,:]
    return normal_map
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
    img = Image.new("RGB", (RESOLUTION, RESOLUTION), (128, 128, 255))
    pixels = img.load()
    for i in range(img.size[0]):    # for every col:
        for j in range(img.size[1]):    # For every row
            pixels[i,j] = (normal_map[i][j][0],\
                           normal_map[i][j][1],\
                           normal_map[i][j][2]) # set the colour accordingly
    img.save(VMF_FILENAME[0:-4] + (".png") + "")
    return img
def get_vertexes_from_top_left_vertex(vertexpos, _dims):
    """
    Finds all vertices of a brush given the brush dimensions and position of
    the top-left vertex as seen in Hammer's top view. This function assumes
    the brush is a cuboid.
    Parameters
    ----------
    vertexpos : array_like
        The position coordinates of the top-left vertex
    _dims : array_like
        Dimensions of the brush (length, width and height)
    Returns
    -------
    verts : np.ndarray
        Array of all the vertices of the corresponding brush.
    """
    verts = np.zeros((2,2,3))
    counterx = 0
    countery = -1
    for y in range (2):
        countery += 1
        counterx = 0
        for x in range (2):
            verts[x][y] = [(vertexpos[0] + _dims[0] * counterx),\
                           (vertexpos[1] - _dims[1] * countery),\
                           (vertexpos[2])]
            counterx += 1
    verts.shape = (4,3)
    return verts
def get_vertex_top_lefts(tltlvertex, _dims, num_brushes):
    """
    Gets all top-left vertices for every brush in the VMF given the top-left
    vertex of the top-left brush as a reference point.
    Parameters
    ----------
    tltlvertex : array_like
        Top-left vertex of the top-left brush.
    _dims : array_like
        Dimensions of each brush, these must be constant for all brushes.
    num_brushes : int
        Number of brushes present in the VMF. (Square number)
    Returns
    -------
    verts : np.ndarray 
        Array of all top-left vertices of every brush in the VMF.
    """
    sidelength = int(np.sqrt(num_brushes))
    verts = np.zeros((sidelength, sidelength, 3))
    counterx = 0
    countery = -1
    for y in range (sidelength):
        countery += 1
        counterx = 0
        for x in range (sidelength):
            verts[x][y] = [(tltlvertex[0] + counterx * _dims[0]),\
                           (tltlvertex[1] - countery * _dims[1]),\
                               tltlvertex[2]] #Only interested in sides at z = Z_CEILING
            counterx += 1
    _verts = verts
    _verts.shape = (sidelength**2, 3)
    return verts
def find_disp_from_vert_data(vmf_data, vertex):
    """
    Finds the displacement data of a brush side given its vertices.
    Parameters
    ----------
    vmf_data : string
        The text body of a VMF.
    vertex : np.ndarray
        The vertices of the target brush side.
    Returns
    -------
    disp_out : list
        The displacement data for that brush side.
    """
    occurrences = []
    vertex = np.array(vertex, dtype="int")
    _string = ""
    for A in vertex:
        for B in range(3):
            _string =  ' '.join(str(C) for C in A)
        _string = '"v" "' + (_string) + '"\n'
        _list = list(find_all(vmf_data, _string))
        occurrences += _list
    occurrences.sort()
    min_distance = float('inf')
    best_subset = []
    for D in range(len(occurrences)-3):
        subset = occurrences[D:D+4]
        distance = subset[-1] - subset[0]
        if distance < min_distance:
            min_distance = distance
            best_subset = subset
    vmf_data_chunk = vmf_data[best_subset[0]:]
    disp_out = get_disp_data(vmf_data_chunk)
    return disp_out
def get_vertex_set(toplefts):
    """
    Finds the complete set of all brush side vertices given the coordinates
    of the top-left vertex for each brush.
    Parameters
    ----------
    toplefts : np.ndarray
        The array containing the coordinates of the top-left vertices for
        every brush in the VMF.
    Returns
    -------
    vertexSet : np.ndarray
        The complete set of all brush side vertices for each brush in the VMF.
    """
    counter = 0
    _vertex_set = np.zeros((len(toplefts), 4, 3))
    for x in toplefts:
        _vertex_set[counter] = get_vertexes_from_top_left_vertex(x, dims)
        counter += 1
    return _vertex_set
def concatenate_normal_map_set(normalmap_set):
    """
    Concatenates a set of normal maps into one normal map, following the order
    in which vertex_set refers to individual brushes.
    The current order is top-left brush downwards, before looping back to the
    bottom brush of the next column and walking upwards.
    Parameters
    ----------
    normalmap_set : np.ndarray
        The set containing each normal map.
    Returns
    -------
    _currentrow : np.ndarray
        The concatenated, single normal map.
    """
    sidelength = int(np.sqrt(len(normalmap_set)))
    _currentcolumn = np.zeros((2**POWER,2**POWER,3))
    _currentrow = np.zeros((2**POWER*(sidelength),2**POWER,3))
    for a in range (sidelength):
        _currentcolumn = np.zeros((2**POWER,2**POWER,3))
        for b in range (sidelength):
            if a % 2 == 0:
                normalmap_set[b+sidelength*a] = np.flip(normalmap_set[b+sidelength*a],\
                                                        axis=0)
                _currentcolumn = np.vstack((_currentcolumn,\
                                            normalmap_set[b+sidelength*a]))
            else:
                _currentcolumn = np.vstack((normalmap_set[b+sidelength*a],\
                                            _currentcolumn))
        if a % 2 == 0:
            _currentcolumn = _currentcolumn[2**POWER:,:,:]
        else:
            _currentcolumn = _currentcolumn[:-2**POWER,:,:] #get rid of zeroes
        _currentrow = np.hstack((_currentrow, _currentcolumn))
    _currentrow = _currentrow[:,2**POWER:,:]
    for C in range(sidelength):
        if C % 2 == 0:
            _currentrow[:,2**POWER*C:2**POWER*(C+1),:] \
                = np.flipud(_currentrow[:,2**POWER*C:2**POWER*(C+1),:])
    return _currentrow
def normalise_distance_field(distance_field):
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
              Aborted normalisation.")
        return distance_field
    normalised_distance_field = np.floor(distance_field * 128/(max_dist - min_dist))
    normalised_distance_field = normalised_distance_field + 128
    normalised_distance_field = np.where((normalised_distance_field > 255),\
                                         255,\
                                         normalised_distance_field) #Restrict upper
    normalised_distance_field = np.where((normalised_distance_field < 192),\
                                         192,\
                                         normalised_distance_field) #Restrict lower
    return normalised_distance_field
def _normalise_distance_field(distance_field):
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
    distance_field = np.where(distance_field < 1/255,\
                              1/255, 1/distance_field)
    if max_dist == min_dist:
        print("Cannot normalise distance field since max_dist and min_dist are the same.\n\
              Aborted normalisation.")
        return distance_field
    normalised_distance_field = np.floor(distance_field * 128/(max_dist - min_dist))
    normalised_distance_field = normalised_distance_field + 128
    #normalised_distance_field = normalised_distance_field[:-1,:-1]
    normalised_distance_field = np.where((normalised_distance_field > 255),\
                                         255,\
                                         normalised_distance_field) #Restrict upper
    normalised_distance_field = np.where((normalised_distance_field < 192),\
                                         192,\
                                         normalised_distance_field) #Restrict lower
    return normalised_distance_field




#####-----PROGRAM-DEFINED CONSTANTS-----#####





vmf_data = read_vmf(VMF_FILENAME)
dist_data = get_dist_field(vmf_data)
num_of_brushes, vertexes = get_brush_data(vmf_data)
RESOLUTION = int(2**POWER * np.sqrt(num_of_brushes))
xLen = np.abs(np.max(vertexes[:,0]) - np.min(vertexes[:,0]))
yLen = np.abs(np.max(vertexes[:,1]) - np.min(vertexes[:,1]))
topleft_x = np.min(vertexes[:,0])
topleft_y = np.max(vertexes[:,1])
topleft_z = np.max(vertexes[:,2]) # = Z_CEILING
brush_xLen = xLen / np.sqrt(num_of_brushes)
brush_yLen = yLen / np.sqrt(num_of_brushes)
dims = [brush_xLen, brush_yLen,(Z_CEILING - Z_FLOOR)]
topleft_coords = [topleft_x, topleft_y, Z_CEILING]
node_distance = (brush_xLen) / (2**POWER)
num_of_points = num_of_brushes * (2**POWER + 1)**2
#Min X and Max Y is the top-left in Hammer's top view, so we'll use that
#Since we know brush dims, num of brushes and a reference point, we can easily
#find the vertices of all brushes
top_lefts = get_vertex_top_lefts(topleft_coords, dims, num_of_brushes)
top_left_brush_vertexes = get_vertexes_from_top_left_vertex(topleft_coords, dims)
vertex_set = get_vertex_set(top_lefts)
normal_map_set = np.zeros((len(vertex_set),2**POWER,2**POWER,3), dtype="float")





##############################################
#vertex_set vertexes goes from ToLe->BoLe->BoRi->ToRi
#brushes for normal_map_set go vertically downwards then loop to the next column (makes sense)
for i, e in enumerate(vertex_set):
#    normal_map_set[P] = gen_normal_map\
#        (get_dist_field\
#         (find_disp_from_vert_data\
#          (vmf_data, vertex_set[P])[0]))
    element = find_disp_from_vert_data(vmf_data, e)
    if len(element) <= 0:
        print("WARNING: Null displacement map detected."\
              "\nIs the top of your brush at z=" + (Z_FLOOR) + "?")
        normal_map_set[i] = gen_normal_map(np.zeros((2**POWER+1, 2**POWER+1)))
    else:
        element = element[0]
        element = get_dist_field(element)
        normal_map_set[i] = gen_normal_map(element)
combined_normal_map = concatenate_normal_map_set(normal_map_set)
combined_normal_map[:,:,2] = normalise_distance_field(combined_normal_map[:,:,2])
combined_normal_map = combined_normal_map.astype(int)
combined_normal_map = np.rot90(combined_normal_map, k=3, axes=(0,1))
output = gen_normal_map_image(combined_normal_map)
output.show()
output.close()
#We assume and hope that the brushes are nice, equally sized squares that could
#tile infinetely
#A power N displacement has (2^N + 1)^2 points
#Don't care about Z-len since displacement map is only for a 2D face
