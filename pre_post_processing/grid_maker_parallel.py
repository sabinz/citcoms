#!/usr/bin/env python
#=====================================================================
#                Geodynamic Framework Scripts for 
#         Preprocessing, Data Assimilation, and Postprocessing
#
#                 AUTHORS: Mark Turner, Dan J. Bower
#                  ---------------------------------
#             (c) California Institute of Technology 2014
#                  ---------------------------------
#                        ALL RIGHTS RESERVED
#=====================================================================
#=====================================================================
# grid_maker.py
#=====================================================================
# This script is a general purpose tool to process Citcoms output data
# into one ore more GMT style .grd or .nc format data files.  
# Please see the usage() function below, and the sample configuration 
# file: /sample_data/grid_maker.cfg for more info.
#=====================================================================
#=====================================================================
# Cascade implementation and deployment notes: grid_maker_parallel.md.
import sys, string, os, shutil
import subprocess
from pathlib import Path

_CASCADE_CONTEXT = None
import numpy as np
#=====================================================================
import Core_Citcom
import Core_isoGMT
import Core_Util
from Core_Util import now

#=====================================================================
#=====================================================================
def usage():
    '''print usage message, and exit'''

    print('''usage: grid_maker_parallel.py [-e] configuration_file.cfg [procs]

Options and arguments:
  
-e : if the optional -e argument is given this script will print to standard out an example configuration control file.
   The parameter values in the example config.cfg file may need to be edited or commented out depending on intended use.

'configuration_file.cfg' : is a geodynamic framework formatted control file, with at least these entries: 

    pid_file = /path/to/a/citcoms_pid_file # the path of a citcoms pid0000.cfg file 
    time_spec = multi-value specification (single value, comma delimted list, start/stop/step trio),
    level_spec = multi-value specification (single value, comma delimted list, start/stop/step trio),

and at least one sub-section:

    [Subsection], where 'Subsection' may be any string, followed by:
    field = standard Citcom field name ('temp', 'visc', 'comp', etc. - see Core_Citcom.py for more info)
      
and where each sub-section may have one or more of the following optional entries:

     dimensional = True to also generate a dimensionalized grid 
     blockmedian_I = value to pass to GMT blockmedian -I option
     surface_I = value to pass to GMT surface -I option

See the example config.cfg file for more info.
''')
    sys.exit()
#=====================================================================
#=====================================================================
def initialise_variables(configFile=None,verbose=False):
    if _CASCADE_CONTEXT is not None:
        return _CASCADE_CONTEXT['initialized']
    if configFile is None:
        configFile = sys.argv[1]
    # get the .cfg file as a dictionary
    visible_directory(Path.cwd())
    visible_file(configFile)
    control_d = Core_Util.parse_configuration_file( configFile, False, False )
    if control_d.get("coord_dir"):
        visible_directory(control_d["coord_dir"])
    
    print(f"{now()} Config file dictionary:")
    if verbose: Core_Util.tree_print( control_d )

    # set the pid file 
    pid_file = control_d['pid_file']
    visible_pid(pid_file)
    
    # get the master dictionary and define aliases
    master_d = Core_Citcom.get_all_pid_data( pid_file, verbose=verbose )
    coor_d = master_d['coor_d']
    pid_d = master_d['pid_d']
    
    # set up working variables
    # get basic info about the model run
    datadir       = pid_d['datadir']
    datafile      = pid_d['datafile']
    start_age     = pid_d['start_age']
    output_format = pid_d['output_format']

    depth_list = coor_d['depth_km']
    nodez      = pid_d['nodez']
    nproc_surf = pid_d['nproc_surf']

    return control_d, pid_file, master_d, coor_d, pid_d, datadir, datafile, start_age, output_format, depth_list, nodez, nproc_surf

def main():
    return run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)


def times_parallel(task):
    """Process a resolved timestep; depth work stays inside this age worker."""
    control_d, pid_file, master_d, coor_d, pid_d, datadir, datafile, start_age, output_format, depth_list, nodez, nproc_surf = initialise_variables()
    timestep = task['timestep']
    age_Ma_storing = task['age']
    age_Ma = '%03d' % task['age']
    runtime_Myr = task['runtime']
    verbose = control_d.get('verbose', False)
    lon, lat = _CASCADE_CONTEXT['lon'], _CASCADE_CONTEXT['lat']
    if nproc_surf == 12:
        grid_R = 'd' if control_d.get('shift_lon', False) else 'g'
    else:
        grid_R = '/'.join(str(pid_d[key]) for key in ('lon_min', 'lon_max', 'lat_min', 'lat_max'))
    file_format_cache, file_data = None, None
    for ss, section in enumerate(control_d['_SECTIONS_']):
        requested = control_d[section]['field']
        field_name = {'Vx': 'vx', 'Vy': 'vy', 'Vz': 'vz'}.get(requested, requested)
        read_field = 'vx' if field_name == 'horiz_vmag' else field_name
        mapping = Core_Citcom.field_to_file_map[read_field]
        pattern = task['patterns'][section]
        if pattern != file_format_cache:
            nested = Core_Citcom.read_proc_files_to_cap_list(pid_d, pattern, read_field)
            file_data = Core_Util.flatten_nested_structure(nested)
            file_format_cache = pattern
        field_data = np.asarray([row[mapping['column']] for row in file_data], dtype=float)
        if field_name == 'horiz_vmag':
            north = np.asarray([row[Core_Citcom.field_to_file_map['vy']['column']] for row in file_data], dtype=float)
            field_data = np.hypot(field_data, north)
        if not np.isfinite(field_data).all():
            raise ValueError('Non-finite source values for %s' % section)
        for ll, level in enumerate(_CASCADE_CONTEXT['levels']):
            levels_parallel((ll, level, task['timestep'], ss, section, timestep, age_Ma,
                             runtime_Myr, field_name, field_data, age_Ma_storing,
                             lon, lat, grid_R, [], verbose))


def levels_parallel(varsList):
    ll = varsList[0]
    level = varsList[1]
    tt = varsList[2]
    ss = varsList[3]
    s = varsList[4]
    timestep = varsList[5]
    age_Ma = varsList[6]
    runtime_Myr = varsList[7]
    field_name = varsList[8]
    field_data = varsList[9]
    age_Ma_storing = varsList[10]
    lon = varsList[11]
    lat = varsList[12]
    grid_R = varsList[13]
    grid_list = varsList[14]
    verbose = varsList[15]
    print( now(), 'grid_maker.py: Processing level = ', level) 

    control_d, pid_file, master_d, coor_d, pid_d, datadir, datafile, start_age, output_format, depth_list, nodez, nproc_surf = initialise_variables()

    # ensure level is an int value 
    level = int(level)
    depth = int(depth_list[level])
    # pad the depth value 
    #depth = '%04d' % depth
    depth=str(depth)
    print( now(), '------------------------------------------------------------------------------')
    print( now(), 'grid_maker.py: tt,ss,ll = ', tt, ',', ss, ',', ll, ';')
    print( now(), 'grid_maker.py: summary for', s, ': timestep =', timestep, '; age =', age_Ma, '; runtime_Myr =', runtime_Myr, '; level =', level, '; depth =', depth, ' km; field_name =', field_name)
    print( now(), '------------------------------------------------------------------------------')


    if field_name.startswith('vertical_'):
        # perform a z slice for citcom data 
        field_slice = field_data[level::nodez] # FIXME : how to get a v slice 
       #xyz_filename = datafile + '-' + field_name + '-' + str(age_Ma_storing) + 'Ma-' + str(depth) + 'km.xyz'
        xyz_filename = datafile + '_' + field_name + '_t' + str(age_Ma_storing)+ '_' + str(depth) + '.xyz'


    else:
        # perform a z slice for citcom data 
        field_slice = field_data[level::nodez]
        #xyz_filename = datafile + '-' + field_name + '-' + str(timestep) + '-' + str(depth) + '.xyz'
        #xyz_filename = datafile + '-' + field_name + '-' + str(age_Ma_storing) + 'Ma-' + str(depth) + 'km.xyz'
        xyz_filename = datafile + '_' + field_name + '_t' + str(age_Ma_storing)+ '_' + str(depth) + '.xyz'

    print( now(), 'grid_maker.py: xyz_filename =', xyz_filename)
    
    if field_name == 'visc':
        if np.any(field_slice <= 0):
            raise ValueError('Viscosity must be positive before log10')
        field_slice = np.log10(field_slice)

    print( now(), 'grid_maker.py: type(field_slice) = ', type(field_slice) )
    print( now(), 'grid_maker.py:  len(field_slice) = ', len(field_slice) )
    print( now() )


    # create the xyz data
    xyz_data = np.column_stack( (lon, lat, field_slice) )
    np.savetxt( xyz_filename, xyz_data, fmt='%f %f %f' )

    #print( now(), 'grid_maker.py: type(xyz_data) = ', type(xyz_data) )
    #print( now(), 'grid_maker.py:  len(xyz_data) = ', len(xyz_data) )
    #print( now() )

    # recast the slice 
    #fs = np.array( field_slice )  
    #fs.shape = ( len(lat), len(lon) )
    #print( now(), 'grid_maker.py: type(fs) = ', type(field_slice) )
    #print( now(), 'grid_maker.py:  len(fs) = ', len(field_slice) )
    #print( now() )

    # check for a grid_R 
    if 'R' in control_d[s] :
        grid_R = control_d[s]['R']

    # create the median file 
    median_xyz_filename = xyz_filename.rstrip('xyz') + 'median.xyz'

    blockmedian_I = control_d[s].get('blockmedian_I', '0.5')
    cmd = xyz_filename + ' -I' + str(blockmedian_I) + ' -R' + grid_R

    Core_isoGMT.callgmt( 'blockmedian', cmd, '', '>', median_xyz_filename )

    # get a T value for median file 
    if not 'Ll' in control_d[s] or not 'Lu' in control_d[s]:
        T = Core_isoGMT.get_T_from_minmax( median_xyz_filename )
    else:
        dt = (control_d[s]['Lu']-control_d[s]['Ll'])/10
        T = '-T' + str(control_d[s]['Ll']) + '/'
        T += str(control_d[s]['Lu']) + '/' + str(dt)

    print( now(), 'grid_maker.py: T =', T)

    
    # create the grid
    grid_filename = xyz_filename.rstrip('xyz') + 'nc'

    interpolate(median_xyz_filename, grid_filename, control_d[s], grid_R)

    # Produce a grid showing deviation from average
    if control_d[s].get('deviation'):
        ave = np.median(field_slice)
        field_slice_dev = field_slice - ave

        xyz_filename_dev = 'deviation_' + datafile + '_' + field_name + '_t' + str(age_Ma_storing)+ '_' + str(depth) + '.xyz'
        # create the xyz data
        xyz_data_dev = np.column_stack( (lon, lat, field_slice_dev) )
        np.savetxt( xyz_filename_dev, xyz_data_dev, fmt='%f %f %f' )

        # create the median file 
        median_xyz_filename_dev = xyz_filename_dev.rstrip('xyz') + 'median.xyz'

        cmd = xyz_filename_dev + ' -I' + str(blockmedian_I) + ' -R' + grid_R
        Core_isoGMT.callgmt( 'blockmedian', cmd, '', '>', median_xyz_filename_dev )
        
        # create the grid
        grid_filename_dev = xyz_filename_dev.rstrip('xyz') + 'nc'

        interpolate(median_xyz_filename_dev, grid_filename_dev, control_d[s], grid_R)

        dev_dir_name = f'{field_name}_deviation'
        dev_grid_dir=f"{control_d['_grid_output_root']}/{dev_dir_name}/{age_Ma_storing}.inprogress"

        os.makedirs(f'{dev_grid_dir}', exist_ok=True)
        if os.path.isfile(f'{dev_grid_dir}/{grid_filename_dev}'):
            os.remove(f'{dev_grid_dir}/{grid_filename_dev}')
        shutil.move(grid_filename_dev, f'{dev_grid_dir}')

        if not control_d.get('debug', False):
            os.remove(xyz_filename_dev)
            os.remove(median_xyz_filename_dev
                )

    ### Jono- uncomment below to produce plots
    if control_d.get('debug', False):
        # label the variables
        
        # −Dxname/yname/zname/scale/offset/title/remark
        cmd = grid_filename + ' -D/=/=/' + str(field_name) + '/=/=/' + str(field_name) + '/' + str(field_name)
        Core_isoGMT.callgmt( 'grdedit', cmd, '', '', '')
    
    # Dimensionalize grid   
    if control_d[s].get('dimensional'):
        print( now(), 'grid_maker.py: dimensional = ', control_d[s]['dimensional'])
        dim_grid_name = grid_filename.replace('.nc', '_dimensional.nc')
        dim = _CASCADE_CONTEXT['dimension_map'][field_name]
        cmd = '%s %f MUL %s ADD' % (grid_filename, dim['coef'], dim['const'])
        Core_isoGMT.callgmt('grdmath', cmd, '', '=', dim_grid_name)

        dim_dir_name = f'{field_name}_dimensional'

    # save this grid and its age in a list
    if control_d[s].get('dimensional'):
        grid_list.append( (dim_grid_name, age_Ma) )
    else: 
        grid_list.append( (grid_filename, age_Ma) )

    J = control_d[s].get('J', 'X5/3')
    C = control_d[s].get('C', 'polar')

    # Optional step to transform grid to plate frame
    if control_d.get('make_plate_frame_grid', False):
        cmd = 'frame_change_pygplates.py %(age_Ma)s %(grid_filename)s %(grid_R)s' % vars()
        print(now(), 'grid_maker.py: cmd =', cmd)
        subprocess.check_call([sys.executable, str(Path(__file__).with_name('frame_change_pygplates.py')),
                               str(age_Ma), grid_filename, grid_R])


    # Assoicate this grid with GPlates exported line data in .xy format:
    # compute age value 
    age_float = 0.0

    # time_list values for citcom data uses timesteps; get age 
    time_triple = Core_Citcom.get_time_triple_from_timestep(master_d['time_d']['triples'], timestep, verbose=verbose)
    age_float = time_triple[1]

    if control_d.get('debug', False):
        # truncate to nearest int and make a string for the gplates .xy file name 
        if age_float < 0: age_float = 0.0
        xy_path = master_d['geoframe_d']['gplates_line_dir']
        #xy_filename = xy_path + '/' + 'topology_platepolygons_' + str(int(age_float)) + '.00Ma.xy'
        xy_filename = xy_path + '/' + 'topology_platepolygons_' + str(int(age_Ma)) + '.00Ma.xy'
        print( now(), 'grid_maker.py: xy_filename = ', xy_filename)


        # Make a plot of the grids
        J = 'X5/3' #'R0/6'
        #J = 'M5/3'
        if 'J' in control_d[s] :
            J = control_d[s]['J']

        C = 'polar'
        if 'C' in control_d[s] :
            C = control_d[s]['C']
    
        # citcoms 
        # plot non-dimensional grid
        Core_isoGMT.plot_grid( grid_filename, xy_filename, grid_R, T, J, C)

        # also plot dimensional grid 
        if control_d[s].get('dimensional') :
            print( now(), 'grid_maker.py: plotting dimensional = ', control_d[s]['dimensional'])
            dim_grid_name = grid_filename.replace('.nc', '_dimensional.nc')
            T = Core_isoGMT.get_T_from_grdinfo( dim_grid_name )
            Core_isoGMT.plot_grid( dim_grid_name, xy_filename, grid_R, T, J)

    # plot plate frame grid 
    if control_d.get('make_plate_frame_grid', False):
        plateframe_grid_name = grid_filename.replace('.nc', '-plateframe.nc')
        xy_filename = ''
        xy_path = master_d['geoframe_d']['gplates_line_dir']
        # present day plate outlines : use '0' 
        xy_filename = xy_path + '/' + 'topology_platepolygons_0.00Ma.xy' 
        print( now(), 'grid_maker.py: xy_filename = ', xy_filename)

        T = Core_isoGMT.get_T_from_grdinfo( plateframe_grid_name )
        print( now(), 'grid_maker.py: T =', T)
        Core_isoGMT.plot_grid( plateframe_grid_name, xy_filename, grid_R, T, J)
    # end of plotting 

    # For normal (non-debug) mode, the produced grids go into neat folders
    # JONO - create field and age directories if needed. Done here
    # os.makedirs(field_name, exist_ok=True)
    grid_dir=f"{control_d['_grid_output_root']}/{field_name}/{age_Ma_storing}.inprogress"
    os.makedirs(grid_dir, exist_ok=True)
    
    if os.path.isfile(f'{grid_dir}/{grid_filename}'):
        os.remove(f'{grid_dir}/{grid_filename}')
    shutil.move(grid_filename, f'{grid_dir}')
    if control_d.get('make_plate_frame_grid', False):
        for artifact in (plateframe_grid_name,
                         str(Path(plateframe_grid_name).with_suffix('.ps')),
                         str(Path(plateframe_grid_name).with_suffix('.png')),
                         str(Path(plateframe_grid_name).with_suffix('.cpt'))):
            shutil.move(artifact, grid_dir)

    # Add dimensionalised grid to its own folder
    if control_d[s].get('dimensional'):
        dim_grid_dir=f"{control_d['_grid_output_root']}/{dim_dir_name}/{age_Ma_storing}.inprogress"
        os.makedirs(f'{dim_grid_dir}', exist_ok=True)

        if os.path.isfile(f'{dim_grid_dir}/{dim_grid_name}'):
            os.remove(f'{dim_grid_dir}/{dim_grid_name}')
        shutil.move(dim_grid_name, f'{dim_grid_dir}')

    if control_d.get('debug', False):
        if os.path.isfile(f'{grid_dir}/{xyz_filename}'):
            os.remove(f'{grid_dir}/{xyz_filename}')
        shutil.move(xyz_filename, f'{grid_dir}')

        if os.path.isfile(f'{grid_dir}/{median_xyz_filename}'):
            os.remove(f'{grid_dir}/{median_xyz_filename}')
        shutil.move(median_xyz_filename, f'{grid_dir}')

        ps = grid_filename.rstrip('.nc') + '.ps'
        if os.path.isfile(f'{grid_dir}/{ps}'):
            os.remove(f'{grid_dir}/{ps}')
        shutil.move(ps, f'{grid_dir}')  

        png = grid_filename.rstrip('.nc') + '.png'
        if os.path.isfile(f'{grid_dir}/{png}'):
            os.remove(f'{grid_dir}/{png}')
        shutil.move(png, f'{grid_dir}')                                            

        cpt = grid_filename.rstrip('.nc') + '.cpt'
        if os.path.isfile(f'{grid_dir}/{cpt}'):
            os.remove(f'{grid_dir}/{cpt}')
        shutil.move(cpt, f'{grid_dir}')  

        if control_d[s].get('dimensional'):
            ps = dim_grid_name.rstrip('.nc') + '.ps'
            if os.path.isfile(f'{dim_grid_dir}/{ps}'):
                os.remove(f'{dim_grid_dir}/{ps}')
            shutil.move(ps, f'{dim_grid_dir}')  

            png = dim_grid_name.rstrip('.nc') + '.png'
            if os.path.isfile(f'{dim_grid_dir}/{png}'):
                os.remove(f'{dim_grid_dir}/{png}')
            shutil.move(png, f'{dim_grid_dir}')                                            

            cpt = dim_grid_name.rstrip('.nc') + '.cpt'
            if os.path.isfile(f'{dim_grid_dir}/{cpt}'):
                os.remove(f'{dim_grid_dir}/{cpt}')
            shutil.move(cpt, f'{dim_grid_dir}') 


    # remove some of the unneeded files
    if not control_d.get('debug', False):
        os.remove(xyz_filename)
        os.remove(median_xyz_filename)

#=====================================================================
#=====================================================================
# SAVE This code for reference:
#                    # optionally adjust the lon bounds of the grid to -180/180
#                    #if 'shift_lon' in control_d : 
#                    #    print( now(), 'grid_maker.py: shifting values to -180/+180')
#                    #    arg = grid_filename
#                    #    opts = {'R' : 'd', 'S' : '' }
#                    #    Core_isoGMT.callgmt('grdedit', arg, opts)
#=====================================================================
#=====================================================================
import contextlib
import copy
import datetime
import fcntl
import glob
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

import Core_Citcom
import Core_isoGMT
import Core_Util


MANIFEST = "grid_complete.json"
CONTEXT = None


def report(message):
    print(f"{Core_Util.now()} {message}", flush=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def visible_directory(path):
    """Reject hidden model directories, including targets of directory symlinks."""
    path = Path(path).absolute()
    for candidate in (path, path.resolve()):
        if any(part.startswith(".") and part not in (".", "..") for part in candidate.parts):
            raise ValueError(f"Hidden directories are not used by this workflow: {path}")
    return path


def visible_file(path):
    path = Path(path).absolute()
    visible_directory(path.parent)
    visible_directory(path.resolve().parent)
    return path


def visible_pid(path):
    visible_file(path)
    settings = Core_Util.parse_configuration_file(str(path))
    for key in ("datadir", "coor_file"):
        if key in settings:
            value = str(settings[key]).strip("\"'")
            if key == "datadir":
                visible_directory(value.replace("%RANK", "0").replace("#", "0"))
            else:
                visible_file(value)
    if "datafile" in settings and "datadir" in settings:
        datafile = str(settings["datafile"])
        datadir = str(settings["datadir"]).strip("\"'").replace("%RANK", "0")
        visible_file(f"{datafile}.time")
        visible_file(Path(datadir) / f"{datafile}.time")
        visible_file(Path(datadir) / f"{datafile}.coord.0")


def identity(path):
    path = visible_file(path).resolve()
    info = path.stat()
    if not path.is_file() or info.st_size <= 0:
        raise ValueError(f"Missing or empty input: {path}")
    return {"path": str(path), "size": info.st_size, "mtime_ns": info.st_mtime_ns}


def nonempty(path):
    path = visible_file(path)
    return path.is_file() and path.stat().st_size > 0


def parse_times(spec):
    def value(text):
        text = str(text).strip()
        unit = "Ma" if text.endswith("Ma") else "Myr" if text.endswith("Myr") else "step"
        number = float(text[:-len(unit)].strip() if unit != "step" else text)
        if not math.isfinite(number):
            raise ValueError("Time requests must be finite")
        return number, unit
    if isinstance(spec, (list, tuple)):
        if not spec:
            raise ValueError("Time/level requests cannot be empty")
        return [value(item) for item in spec]
    text = str(spec).strip().strip("[]")
    if text.endswith(".dat"):
        requests = []
        for line in visible_file(text).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                requests.extend(parse_times(line))
        if not requests:
            raise ValueError(f"Empty time/level request file: {text}")
        return requests
    if "/" not in text:
        return [value(item) for item in text.split(",")]
    parts = text.split("/")
    if len(parts) != 3:
        raise ValueError("time_spec range requires start/end/step")
    (start, unit), (end, end_unit), (step, step_unit) = map(value, parts)
    if unit != end_unit or step_unit not in (unit, "step") or step <= 0:
        raise ValueError("time_spec requires matching endpoint units and a positive step")
    count = int(math.floor(abs(end - start) / step + 1e-10)) + 1
    if count > 100000:
        raise ValueError("Too many time requests")
    direction = 1 if end >= start else -1
    return [(start + index * direction * step, unit) for index in range(count)]


def parse_levels(spec, nodez):
    # Same list/range syntax, including descending ranges and a single endpoint.
    values = parse_times(spec)
    if any(unit != "step" or int(value) != value or not 0 <= value < nodez for value, unit in values):
        raise ValueError(f"level_spec must contain integer levels from 0 to {nodez - 1}")
    return list(dict.fromkeys(int(value) for value, _ in values))


def output_field(field):
    return {"Vx": "vx", "Vy": "vy", "Vz": "vz"}.get(field, field)


def file_field(field):
    field = output_field(field)
    return "vx" if field == "horiz_vmag" else field


def inventory(pid, field):
    component = Core_Citcom.field_to_file_map[file_field(field)]["file"]
    reference_rank = pid["nprocz"] - 1 if component == "surf" else 0
    datadir = str(pid["datadir"]).strip('"\'')
    directories = [datadir.replace("%RANK", "#")] if "%RANK" in datadir else [datadir + "/#", datadir]
    directories += ["data/#", "Data/#"]
    directories += [str(Path(age) / "#") for age in sorted(glob.glob("Age*Ma"))]
    steps = {}
    for directory in dict.fromkeys(directories):
        visible_directory(directory.replace("#", str(reference_rank)))
        zero = str(Path(directory.replace("#", str(reference_rank))) /
                   f"{pid['datafile']}.{component}.{reference_rank}.")
        found = {}
        for name in glob.glob(glob.escape(zero) + "*"):
            suffix = name.rsplit(".", 1)[-1]
            if suffix.isdigit():
                found[int(suffix)] = str(Path(directory).absolute() / f"{pid['datafile']}.{component}.#.{suffix}")
        if found:
            if not directory.startswith("Age"):
                return found
            for step, pattern in found.items():
                if step in steps and steps[step] != pattern:
                    raise ValueError(f"Ambiguous source timestep {step} for {field}: multiple Age folders")
                steps[step] = pattern
    if not steps:
        raise ValueError(f"No processor-{reference_rank} {component} files found for {field}")
    return steps


def expected_outputs(control, pid, depths, levels, age):
    outputs = {}
    for section in control["_SECTIONS_"]:
        options = control[section]
        field = output_field(options["field"])
        for level in levels:
            stem = f"{pid['datafile']}_{field}_t{age}_{int(depths[level])}"
            products = {field: [stem + ".nc"]}
            if options.get("dimensional"):
                products[field + "_dimensional"] = [stem + "_dimensional.nc"]
            if options.get("deviation"):
                products[field + "_deviation"] = ["deviation_" + stem + ".nc"]
            if control.get("debug", False):
                products[field] += [stem + ".xyz", stem + ".median.xyz", stem + ".ps", stem + ".png", stem + ".cpt"]
                if options.get("dimensional"):
                    products[field + "_dimensional"] += [stem + "_dimensional" + suffix for suffix in (".ps", ".png", ".cpt")]
            if control.get("make_plate_frame_grid", False):
                products[field] += [stem + "-plateframe" + suffix for suffix in (".nc", ".ps", ".png", ".cpt")]
            for category, names in products.items():
                previous = outputs.setdefault(category, [])
                if set(previous) & set(names):
                    raise ValueError(f"Output filename collision for {category}; sections or rounded depths overlap")
                previous.extend(names)
    return outputs


def make_tasks(control, master, levels):
    pid = master["pid_d"]
    inventories = {section: inventory(pid, control[section]["field"]) for section in control["_SECTIONS_"]}
    common = set.intersection(*(set(steps) for steps in inventories.values()))
    triples = [tuple(triple) for triple in master["time_d"]["triples"] if int(triple[0]) in common]
    if not triples:
        raise ValueError("No timesteps are available for all requested fields")
    tasks, folders = {}, {}
    for request, unit in parse_times(control["time_spec"]):
        index = {"step": 0, "Ma": 1, "Myr": 2}[unit]
        step, actual_age, runtime = min(triples, key=lambda item: (abs(float(item[index]) - request), float(item[1])))
        step, actual_age = int(step), float(actual_age)
        age = int(np.around(actual_age))
        if age in folders and folders[age] != step:
            raise ValueError(f"Different timesteps round to the same output age {age} Ma")
        folders[age] = step
        if step in tasks:
            tasks[step]["requests"].append([request, unit])
            continue
        patterns = {section: steps[step] for section, steps in inventories.items()}
        sources = {}
        for section, pattern in patterns.items():
            names, _ = Core_Citcom.define_cap_or_proc_names(pid, pattern, "proc")
            component = Core_Citcom.field_to_file_map[file_field(control[section]["field"])]["file"]
            for rank, name in enumerate(names):
                # Only top/bottom radial processors have surface/bottom files.
                kk = rank % pid["nprocz"]
                required = component not in ("surf", "botm") or kk == (pid["nprocz"] - 1 if component == "surf" else 0)
                if required or Path(name).exists():
                    sources[name] = identity(name)
        tasks[step] = {"timestep": step, "actual_age": actual_age, "runtime": float(runtime),
                       "age": age, "patterns": patterns, "sources": list(sources.values()),
                       "requests": [[request, unit]], "outputs": expected_outputs(control, pid, master["coor_d"]["depth_km"], levels, age)}
    return sorted(tasks.values(), key=lambda item: (-item["actual_age"], item["timestep"]))


def signature(context, task):
    return digest({"context": context["signature"], "step": task["timestep"],
                   "age": task["actual_age"], "sources": task["sources"], "outputs": task["outputs"]})


def workspace(context, task):
    return visible_directory(Path(context["output_root"]) / "gridmaker-work" / f"{task['age']}Ma.inprogress")


def folder(context, task, category, temporary=False):
    return visible_directory(Path(context["output_root"]) / category / (str(task["age"]) + (".inprogress" if temporary else "")))


def checked_files(directory, names):
    directory = visible_directory(directory)
    return (all(nonempty(directory / name) for name in names)
            and all(path.stat().st_size > 0 for path in directory.iterdir() if not path.name.startswith(".") and path.is_file()))


def matching_folder(directory, names, wanted):
    if not checked_files(directory, names):
        return False
    try:
        record = json.loads((Path(directory) / MANIFEST).read_text())
        return (record["signature"] == wanted and set(record["files"]) == set(names)
                and all((Path(directory) / name).stat().st_size == size for name, size in record["files"].items()))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def check_sources(task):
    for source in task["sources"]:
        if identity(source["path"]) != source:
            raise ValueError(f"Source changed during gridding: {source['path']}")


def archive(path):
    if not path.exists():
        return
    suffix = datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    target = path.with_name(path.name + ".previous-" + suffix)
    path.rename(target)
    report(f"Preserved incomplete work: {target}")


def inspect_age(context, task):
    wanted = signature(context, task)
    finals = {category: folder(context, task, category) for category in task["outputs"]}
    for category, final in finals.items():
        if (final / MANIFEST).exists():
            record = json.loads((final / MANIFEST).read_text())
            if record.get("signature") != wanted:
                raise ValueError(f"Existing published output is from different settings: {final}")
    if all((matching_folder(finals[category], names, wanted) if (finals[category] / MANIFEST).exists()
            else checked_files(finals[category], names)) for category, names in task["outputs"].items()):
        if any(not (final / MANIFEST).exists() for final in finals.values()):
            report(f"Existing {task['age']} Ma passes file existence/size checks (legacy output without manifests)")
        return "published"
    # A parent killed during the multi-directory rename may have published only
    # some fields. Matching manifests allow the remaining renames to resume.
    if all(matching_folder(finals[category], names, wanted)
           or matching_folder(folder(context, task, category, True), names, wanted)
           for category, names in task["outputs"].items()):
        return "ready"
    work = workspace(context, task)
    work.parent.mkdir(parents=True, exist_ok=True)
    work.mkdir(exist_ok=True)
    with (work / ".worker.lock").open("a+") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"An earlier worker is still preparing {work}; wait before resuming") from error
        for category, final in finals.items():
            archive(final)
            archive(folder(context, task, category, True))
        # Do not archive a directory created just to check the lock.
        if len(list(work.iterdir())) > 1:
            archive(work)
    return None


def publish(context, task):
    wanted = signature(context, task)
    check_sources(task)
    # Validate ALL fields, levels and optional products before the first rename.
    for category, names in task["outputs"].items():
        final = folder(context, task, category)
        stage = folder(context, task, category, True)
        if not matching_folder(final, names, wanted) and not matching_folder(stage, names, wanted):
            raise ValueError(f"Missing, empty or unverified output for {task['age']} Ma: {category}")
    for category in task["outputs"]:
        final = folder(context, task, category)
        stage = folder(context, task, category, True)
        if final.exists():
            if not matching_folder(final, task["outputs"][category], wanted):
                raise ValueError(f"Refusing to replace unexpected final folder: {final}")
            continue
        stage.rename(final)
    work = workspace(context, task)
    work.mkdir(parents=True, exist_ok=True)
    write_json(work / "age_published.json", {"signature": wanted, "age": task["age"], "timestep": task["timestep"]})
    # Logs have their own completed name, matching the published age.
    destination = work.with_name(f"{task['age']}Ma.complete")
    if destination.exists():
        archive(destination)
    work.rename(destination)
    report(f"Published all fields for {task['age']} Ma")


def checked_callgmt(command, argument, opts="", redirect="", out=""):
    # Stock isogmt wrappers can discard GMT's exit status. Age workspaces already
    # isolate GMT files, so invoke gmt directly and preserve its real exit status.
    parts = ["gmt " + command]
    if argument:
        parts.append(argument)
    if opts:
        parts.extend("-" + str(key) + str(value) for key, value in opts.items())
    if redirect:
        parts.append(redirect)
    if out:
        parts.append(out)
    text = " ".join(parts)
    if Core_isoGMT.verbose:
        report(text)
    return subprocess.check_output(text, shell=True, universal_newlines=True).rstrip()


def interpolate(median_file, grid_file, options, region):
    base = f"{median_file} -I{options.get('surface_I', '0.25')} -R{region}"
    if any(key in options for key in ("Ll", "Lu", "T")):
        # These are surface-specific limits/tension, not sphinterpolate options.
        command = base + "".join(f" -{key}{options[key]}" for key in ("Ll", "Lu", "T") if key in options)
        report("Using surface to honor configured limits/tension")
        Core_isoGMT.callgmt("surface", command, "", "", " -G" + grid_file)
    else:
        try:
            Core_isoGMT.callgmt("sphinterpolate", base, "", "", " -G" + grid_file)
            if not nonempty(grid_file):
                raise ValueError("sphinterpolate did not create a nonempty grid")
        except (subprocess.CalledProcessError, ValueError) as error:
            report(f"Spherical interpolation failed ({error}); using surface")
            if Path(grid_file).exists():
                Path(grid_file).unlink()
            Core_isoGMT.callgmt("surface", base, "", "", " -G" + grid_file)
    if not nonempty(grid_file):
        raise ValueError(f"Interpolation did not produce a nonempty grid: {grid_file}")


def initialize(context):
    global CONTEXT
    CONTEXT = context
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def prepare(task):
    global _CASCADE_CONTEXT
    context = CONTEXT
    work = workspace(context, task)
    work.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    previous = Path.cwd()
    with (work / ".worker.lock").open("a+") as lease, (work / "grid.log").open("w", buffering=1) as log:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                os.chdir(work)
                check_sources(task)
                _CASCADE_CONTEXT = context
                Core_isoGMT.callgmt = checked_callgmt
                verbose = context["initialized"][0].get("verbose", False)
                Core_Util.verbose = Core_Citcom.verbose = Core_isoGMT.verbose = verbose
                times_parallel(task)
                check_sources(task)
                for category, names in task["outputs"].items():
                    stage = folder(context, task, category, True)
                    missing = [name for name in names if not nonempty(stage / name)]
                    if missing:
                        raise ValueError(f"Missing/empty outputs in {stage}: {missing}")
                # Manifests are written only after every output for the age passes.
                for category, names in task["outputs"].items():
                    stage = folder(context, task, category, True)
                    write_json(stage / MANIFEST, {"signature": signature(context, task),
                        "age": task["actual_age"], "timestep": task["timestep"],
                        "files": {name: (stage / name).stat().st_size for name in names}})
                return {"ok": True, "seconds": time.monotonic() - started}
            except (Exception, SystemExit) as error:
                traceback.print_exc()
                write_json(work / "failure.json", {"error": str(error), "timestep": task["timestep"]})
                return {"ok": False, "error": str(error)}
            finally:
                os.chdir(previous)


def cascade(context, tasks, workers, executor_factory=None):
    states = [inspect_age(context, task) for task in tasks]
    executor = (executor_factory() if executor_factory else ProcessPoolExecutor(max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"), initializer=initialize, initargs=(context,)))
    futures, frontier, failed = {}, 0, False
    try:
        while frontier < len(tasks):
            while frontier < len(tasks) and states[frontier] in ("published", "ready"):
                if states[frontier] == "ready":
                    publish(context, tasks[frontier])
                    states[frontier] = "published"
                else:
                    report(f"Skipping completed {tasks[frontier]['age']} Ma")
                frontier += 1
            if frontier == len(tasks):
                break
            for index in range(frontier, min(frontier + workers, len(tasks))):
                if states[index] is None:
                    report(f"Starting {tasks[index]['age']} Ma, timestep {tasks[index]['timestep']}")
                    futures[executor.submit(prepare, tasks[index])] = index
                    states[index] = "running"
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                index = futures.pop(future)
                try:
                    result = future.result()
                except (Exception, SystemExit) as error:
                    result = {"ok": False, "error": str(error)}
                states[index] = "ready" if result["ok"] else "failed"
                if result["ok"]:
                    report(f"Validated {tasks[index]['age']} Ma in {result['seconds']:.1f}s"
                           + ("; awaiting older ages before publication" if index > frontier else ""))
                else:
                    failed = True
                    report(f"FAILED {tasks[index]['age']} Ma: {result['error']}; log: {workspace(context, tasks[index]) / 'grid.log'}")
            if failed:
                report("Stopping new work; running ages will finish into temporary folders")
                break
    except KeyboardInterrupt:
        report("Interrupted; running ages will finish into temporary folders. Rerun to resume.")
        failed = True
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True)
    report(f"{states.count('published')}/{len(tasks)} ages published")
    return 1 if failed else 0


def load(config, command_workers=None):
    global _CASCADE_CONTEXT
    _CASCADE_CONTEXT = None
    Core_Util.verbose = Core_Citcom.verbose = False
    initialized = list(initialise_variables(str(visible_file(config))))
    control, pid_file, master, coor, pid = initialized[:5]
    workers = command_workers if command_workers is not None else control.get("procs", control.get("workers", control.get("cores", 1)))
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("procs must be a positive integer")
    if not master.get("time_d") or not master.get("coor_d"):
        raise ValueError("Master time and coordinate files are required")
    if pid.get("output_format") != "ascii":
        raise ValueError("Only ASCII processor data is supported")
    for nodes, processors in (("nodex", "nprocx"), ("nodey", "nprocy"), ("nodez", "nprocz")):
        if pid[processors] < 1 or pid[nodes] < 2 or (pid[nodes] - 1) % pid[processors]:
            raise ValueError(f"Invalid mesh decomposition: {nodes}/{processors}")
    if pid["nproc_surf"] != 12 and any(key not in pid for key in ("lon_min", "lon_max", "lat_min", "lat_max")):
        raise ValueError("Regional models require lon_min, lon_max, lat_min and lat_max in the PID")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(pid["datafile"])) or str(pid["datafile"]) in (".", ".."):
        raise ValueError("datafile must be a simple model filename prefix")
    sections = control.get("_SECTIONS_", [])
    if not sections:
        raise ValueError("At least one field section is required")
    levels = parse_levels(control["level_spec"], pid["nodez"])
    if len(coor["depth_km"]) != pid["nodez"] or not np.isfinite(coor["depth_km"]).all():
        raise ValueError("Invalid depth coordinate array")
    for section in sections:
        options = control[section]
        if "field" not in options or file_field(options["field"]) not in Core_Citcom.field_to_file_map:
            raise ValueError(f"Missing or unknown field in {section}")
        component = Core_Citcom.field_to_file_map[file_field(options["field"])]["file"]
        if component in ("surf", "botm") and levels != [pid["nodez"] - 1 if component == "surf" else 0]:
            raise ValueError(f"{section} requires only its surface/bottom level")
        if options.get("multiprocess_depths"):
            report(f"{section}: multiprocess_depths is ignored; concurrency is bounded across ages")
    # Read coordinates from this model. Never trust the old cwd-wide text caches.
    if control.get("coord_dir"):
        names = str(Path(control["coord_dir"]).resolve() / f"{pid['datafile']}.coord.#")
        pid["coord_type"], pid["coord_file_in_use"] = Core_Citcom.read_citcom_coor_type(pid, names)
    if pid.get("coord_type") not in ("proc", "cap"):
        raise ValueError("Surface longitude/latitude extraction requires processor or cap coordinate files")
    coordinate_template = pid["coord_file_in_use"]
    coordinate_names, _ = Core_Citcom.define_cap_or_proc_names(pid, coordinate_template, pid["coord_type"])
    for name in coordinate_names:
        identity(name)
    if pid["coord_type"] == "cap":
        cap_data = Core_Citcom.read_cap_files_to_cap_list(pid, coordinate_template)
        coords = [(np.degrees(row[1]), 90 - np.degrees(row[0]))
                  for cap in cap_data for row in cap[::pid["nodez"]]]
    else:
        coords = Core_Util.flatten_nested_structure(Core_Citcom.read_citcom_surface_coor(pid))
    lon, lat = np.asarray([row[0] for row in coords]), np.asarray([row[1] for row in coords])
    if len(lon) != pid["nproc_surf"] * pid["nodex"] * pid["nodey"] or not np.isfinite(lon).all() or not np.isfinite(lat).all():
        raise ValueError("Invalid surface coordinates")
    root = Path.cwd()
    output_root = visible_directory(root / str(pid["datafile"]))
    output_root.mkdir(exist_ok=True)
    control["_grid_output_root"] = str(output_root)
    initialized[1] = str(Path(pid_file).resolve())
    for key in ("gplates_line_dir",):
        if key in master["geoframe_d"]:
            master["geoframe_d"][key] = str(visible_directory(master["geoframe_d"][key]).resolve())
    dimension_map = {}
    if any(control[section].get("dimensional") for section in sections):
        dimension_map = copy.deepcopy(Core_Citcom.populate_field_to_dimensional_map_from_pid(initialized[1], verbose=False))
        for section in sections:
            field = output_field(control[section]["field"])
            if control[section].get("dimensional") and field not in dimension_map:
                raise ValueError(f"No dimensionalization rule for {field}")
    semantic_control = {key: value for key, value in control.items() if key not in ("cores", "workers", "procs", "time_spec", "verbose")}
    code_hash = hashlib.sha256((Path(__file__).read_bytes()
                               + Path(Core_Citcom.__file__).read_bytes() + Path(Core_isoGMT.__file__).read_bytes())).hexdigest()
    context = {"initialized": initialized, "lon": lon, "lat": lat, "levels": levels,
               "dimension_map": dimension_map, "output_root": str(output_root),
               "signature": digest({"control": semantic_control, "pid": pid,
                 "depths": coor["depth_km"], "coordinate_hash": hashlib.sha256(lon.tobytes() + lat.tobytes()).hexdigest(),
                 "dimension_map": dimension_map, "code": code_hash})}
    tasks = make_tasks(control, master, levels)
    return context, tasks, workers


def run(config, command_workers=None):
    try:
        # Cwd-level lock also protects separate configurations for the same model.
        visible_directory(Path.cwd())
        with open(".grid_maker_parallel.lock", "a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("Another grid maker is using this model directory") from error
            context, tasks, workers = load(config, command_workers)
            report(f"Oldest-first cascade: {len(tasks)} ages, up to {workers} workers")
            for task in tasks:
                report(f"Requested {task['requests']} -> {task['actual_age']:.6g} Ma, timestep {task['timestep']}")
            return cascade(context, tasks, workers)
    except (Exception, SystemExit) as error:
        report(f"ERROR: {error}")
        return 1
    except KeyboardInterrupt:
        report("Interrupted before processing started")
        return 130


def make_example_config_file( ):
    '''print to standard out an example configuration file for this script'''

    text = '''#=====================================================================
# config.cfg file for the grid_maker.py script
# This example has information on creating grids for Citcom data.
# ==============================================================================
# Set the basic model coordinate information common to both source types:

# set path to the model pid file 
pid_file = pid18637.cfg 

# For RS set path
pid = model_restart_age.cfg

# CitcomS coordinate files by processor (i.e. [datafile].coord.[proc])
# first, look in this user-specified directory for all files
coord_dir = coord

# second, look in data/%RANK

# NOTE: grid_maker.py will fail if coord files cannot be located

# Optional global settings
# Concurrent ages, processed and published oldest first:
procs = 2
# Incomplete outputs use <age>.inprogress folders. Rerun to resume.

# If 'shift_lon' is set to True, then the grids will have data in the -180/+180 longitude range
# The default is for data in the 0/360 longitude range.
# shift_lon = True

# If 'make_plate_frame_grid' is set to True, then this script with produce additional data on the plate frame
#make_plate_frame_grid = True

# ==============================================================================
# Set the times to grid 

# Citcoms : use model time steps or reconstruction age, in one of these forms:

# single value:
#time_spec = 4500
#time_spec = 7Ma

# comma separated list:
#time_spec = 5400, 6200, 6961
#time_spec = 0Ma, 50Ma, 100Ma

# range of values: start/stop/step
# time_spec = 2000/2700/100
time_spec = 0Ma/10Ma/2Ma

# ==============================================================================
# Set the levels to grid 

# Citcoms : use int values from 0 to nodez-1, in one of these forms:

# single value:
#level_spec = 63

# comma separated list:
#level_spec = 64/0/10

# range of values: start/stop/step
#level_spec = 64/0/10

# NOTE: The level_spec settings must match the Citcoms field data types:

# Volume data fields : the level_spec may be values from 0 to nodez-1
level_spec = 63

# Surface (surf) data fields : the level_spec must be set to only nodez-1 
#level_spec = 64 ; for surf data 

# Botttom (botm) data fields : lthe evel_spec must be set to only 0 
#level_spec = 0 ; for bot data 

# ==============================================================================
# Set the fields to grid 
#
# Each field will be a separate section, delimited by brackets [Section_1], 
# each field requires the field name, e.g. 
# field = temp
# Each field may set optional arguments to set GMT parameters.
# Each field may set the optional parameter 'dimensional' to 'True',
# to produce an additional dimensionalized grid with the '.dimensional' filename component.
#
# See Core_Citcoms.field_to_file_map data struct for a list of field names.
#

# Citcoms :

[Grid_1]
field = temp
dimensional = True
#blockmedian_I = 0.5
#surface_I = 0.25
#Ll = 
#Lu = 
#T = 

#[Grid_2]
#field = surf_topography
#dimensional = True
#blockmedian_I = 0.5
#surface_I = 0.25
#Ll = 
#Lu =
#T = 
#=====================================================================
'''
    print( text )
#=====================================================================
#=====================================================================
if __name__ == "__main__":

    # print ( str(sys.version_info) ) 

    # check for script called wih no arguments
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        usage()
        sys.exit(-1)

    # create example config file 
    if sys.argv[1] == '-e':
        make_example_config_file()
        sys.exit(0)


    # run the main script workflow
    sys.exit(main())
#=====================================================================
#=====================================================================
