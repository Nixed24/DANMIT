# DANMIT
Displacement and Normal Map interface tool for the Source Engine.


TIPS:
For painting normal maps, it is best to use the Hard-Edge setting when painting geometry.
For the moment, if you are trying to get a discrete set of height values (blue channel),
it is probably best to use the "Raise To" effect until this is added.

TODO:
-BIG: Allow two-way generation, i.e displacement maps FROM normal maps
-Allow discrete values for the blue channel as an option -- i.e can only have 3 distinct values for the entire normal map.
-Generalise program for Z if possible, or make it failsafe if the user specifies it incorrectly.
-Add export directly to VTF for normal map image
