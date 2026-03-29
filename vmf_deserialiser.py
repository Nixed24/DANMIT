# -*- coding: utf-8 -*-
"""
Started on Feb 22 2026

VMF Deserialiser

This module allows Source VMF files to be turned into Python dictionaries.

The function to be used is deserialise_vmf(), which should be called with a path to a VMF file, which it will then open and deserialise, returning a dict that represents the VMF.

REQUIRED LIBRARIES:
    None, VMF Deserialiser only uses builtins.
    
@author: Nicks24 (GitHub)
"""

settings = {"tagFirstInstance": True}

def read_in_vmf(vmf_filename):
    with open(vmf_filename, "r") as vmf_file:
        vmf_text = vmf_file.read()
    return vmf_text
def read_keyval(text, index):
    i = index
    marks_counter = 0
    key_buffer = ""
    value_buffer = ""
    while i < len(text):
        character = text[i]
        
        if character == "{":
            print("Warning: keyvalue encountered {")
        if character == "}":
            print("Warning: keyvalue encountered }")
            
        if character == '"': # First iteration should trigger this!
            marks_counter += 1
            i += 1
            continue
        
        if marks_counter == 1:
            key_buffer += character
        if marks_counter == 3:
            value_buffer += character
        if marks_counter == 4:
            return (key_buffer, value_buffer, (i-index))
            break
        i += 1
    return
def read_block(text, index):
    duped_names_dict = {}
    name_buffer = ""
    reading_name = False
    for i_char in range(index,-1,-1):
        if text[i_char] == "\n":
            reading_name = True
            continue
        if text[i_char] == "\t" or text[i_char] == "}": # For some reason linebreaks are only read half the time
            if reading_name == True:
                break
            reading_name = False
        if reading_name == True:
            name_buffer += text[i_char]
            
    name = name_buffer[::-1]
    
    block_dict = {}
    
    i = index + 1
    
    while i < len(text):
        character = text[i]
        if character == "{":
            block_buffer = read_block(text, i)
            
            try:
                duped_num = duped_names_dict[block_buffer[0]]
                block_name = block_buffer[0] + "&" + str(duped_num)
                duped_names_dict[block_buffer[0]] += 1
            except KeyError:
                duped_names_dict[block_buffer[0]] = 1
                duped_num = 0
                if settings["tagFirstInstance"] == True:
                    block_name = block_buffer[0] + "&" + str(duped_num)
                else:
                    block_name = block_buffer[0]
            else:
                block_name = block_buffer[0] + "&" + str(duped_num)
            # Why does this work?
            
            block_dict[block_name] = block_buffer[1]
            i += block_buffer[-1]
            i += 1 #take it off the last quotation mark
            continue
        if character == "}":
            return (name, block_dict, (i-index))
        if character == '"':
            keyval_buffer = read_keyval(text, i)
            block_dict[keyval_buffer[0]] = keyval_buffer[1]
            i += keyval_buffer[-1]
            
            i += 1 #take it off the last quotation mark
            continue #
            
        i += 1
    return
def read_loop(vmf_text):
    duped_names_dict = {}
    vmf_dict = {}
    i = 0
    while i < len(vmf_text):
        character = vmf_text[i]
        if character == "{":
            block_buffer = read_block(vmf_text, i)

            try:
                duped_num = duped_names_dict[block_buffer[0]]
                block_name = block_buffer[0] + "&" + str(duped_num)
                duped_names_dict[block_buffer[0]] += 1
            except KeyError:
                duped_names_dict[block_buffer[0]] = 1
                duped_num = 0
                if settings["tagFirstInstance"] == True:
                    block_name = block_buffer[0] + "&" + str(duped_num)
                else:
                    block_name = block_buffer[0]
            else:
                block_name = block_buffer[0] + "&" + str(duped_num)


            vmf_dict[block_name] = block_buffer[1]
            i += block_buffer[-1]
            i += 1 #take it off the last quotation mark
            continue #
        if character == "}":
            print("Warning: encountered vacuum }")
        if character == '"':
            print('Warning: encountered vacuum "')
        i += 1
    return vmf_dict
def deserialise_vmf(filename):
    """Deserialises the VMF file located at the specified path, returning it as a dict."""
    text = read_in_vmf(filename)
    vmf_dict = read_loop(text)
    return vmf_dict
