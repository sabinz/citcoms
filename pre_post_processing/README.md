#=====================================================================
#                 Geodynamic Framework Python Scripts for 
#         Preprocessing, Data Assimilation, and Postprocessing
#
#                 AUTHORS: Dan J. Bower, Mark Turner
#
#                  ---------------------------------
#             (c) California Institute of Technology 2013
#                        ALL RIGHTS RESERVED
#=====================================================================

This framework is a set of Python scripts, modules, and configuration control files, 
for the processing geodyanmic and plate tectonic data.

The scripts use an input configuration file containing parameter and value pairs.
These paramters determine workflow directives, basic input data,
paths to model output data files, and, links to other resources.

The scripts and modules are self-documenting by running with no arguments.

The scrtipts will generate an example configuration with the optional -e argument.

The modules are self-testing, and different tests may be enabled or disabled.
See each module for details.

For resumable, oldest-first parallel preparation of dynamic-topography restart
inputs, see [restart_citcoms_parallel.md](restart_citcoms_parallel.md) and the
[example configuration](sample_data/restart_citcoms_parallel.cfg). Set the
concurrent age limit with `workers` in the configuration file.


