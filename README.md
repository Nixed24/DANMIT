# DANMIT
Displacement and Normal Map interface tool for the Source Engine.

Converts in-Level-editor displacement maps into bump maps for use in textures.

This is an alternative to using complex and typically quite expensive (both financially and computationally) modelling software to do the same job,
using programs that are all free/open-source.

PREREQUISITES:
os (built-in)
subprocess (built-in)
numpy
PIL (Python Image Library)
Hammer or Hammer++

Both of these libraries can be installed with PIP.

INSTRUCTIONS:
Drag DANMIT.py to a folder with your displacement VMF inside, edit the VMF_FILENAME variable to point to your VMF file, and NMAP_FILENAME_NOEXT to point to your output normal map, specify the POWER variable as the power of your displacements and customise other settings to your liking. Save DANMIT.py and execute it, and your normal map VTF should shortly be created in the same folder as DANMIT with the specified filename. You may want to have DANMIT in the same folder you are exporting your normal-mapped material to for this reason.
TIPS:
For painting normal maps, it is best to use the Hard-Edge setting when painting geometry.
You may want to take advantage of the "Raise To" option if you want to paint sudden bumps in your texture
