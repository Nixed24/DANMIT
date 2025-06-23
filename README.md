# DANMIT
Displacement and Normal Map interface tool for the Source Engine.

Converts in-Level-editor displacement maps into bump maps for use in textures.

This is an alternative to using complex and typically quite expensive (both financially and computationally) modelling software to do the same job,
using programs that are all free and open-source.

TIPS:
For painting normal maps, it is best to use the Hard-Edge setting when painting geometry.
For the moment, if you are trying to get a discrete set of height values (blue channel),
it is probably best to use the "Raise To" effect until this is added.

TODO:
-BIG: Allow two-way generation, i.e displacement maps FROM normal maps

-Allow discrete values for the blue channel as an option -- i.e can only have 3 distinct values for the entire normal map.

-Add export directly to VTF for normal map image
