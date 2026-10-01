#!/usr/bin/env python3
#
#=====================================================================
#
#               Python Scripts for CitcomS Data Assimilation
#                  ---------------------------------
#
#                              Authors:
#                 Dan Bower, Mike Gurnis, Rakib Hassan
#             (c) California Institute of Technology 2014
#                        ALL RIGHTS RESERVED
#
#
#=====================================================================
#
#  Copyright January 2014, by the California Institute of Technology.
#
#  Modified: 14th July 2014 by DJB
#=====================================================================
'''Generate history files for slab assimilation in CitcomS using one
 processor OR the parallel infrastructure of CITerra.'''
#=====================================================================
#=====================================================================
import glob, os, shutil, subprocess, sys, multiprocessing, traceback
import Core_Util
from Core_Util import now
from subprocess import PIPE, Popen
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from make_history_for_age import basic_setup, isSerial
#=====================================================================
verbose = True

# name of the file (written into the job's top-level working directory)
# that accumulates every critical failure encountered while building
# per-age history files
CRITICAL_LOG_FILENAME = 'critical_errors.log'
#=====================================================================
#=====================================================================
#=====================================================================
def usage():
    """print usage message and exit"""

    print( now(),'''usage: Create_History.py [-d] [-e] configuration_file.cfg

options and arguments:

-d: if the optional -d argument is geiven this script will generate 
the (required) geodynamic_framework_defaults.conf file in the current 
working directory.  This file can then be modified by the user.

-e: if the optional -e argument is given this script will print to
standard out an example configuration control file.  The parameter 
values in the example configuration_file.cfg file may need to be 
edited or commented out depending on intended use.

citation:
    Bower, D.J., M. Gurnis, and N. Flament (2015)
    Assimilating lithosphere and slab history in 4-D Earth models,
    Physics of the Earth and Planetary Interiors,
    238, 8--22, doi:10.1016/j.pepi.2014.10.013
''' )

    sys.exit(0)

#====================================================================
#====================================================================
#====================================================================
def expected_outputs_present( control_d, age, is_ic ):
    '''Return True if every output file this age is configured to
    produce already exists (and is non-empty); False otherwise.

    Used both to decide whether an age can be skipped (resume mode,
    OVERWRITE_EXISTING = False) and, after a run, to verify that
    make_history_for_age.py actually wrote what it claims to have
    written - a clean (0) exit code alone is not sufficient evidence,
    since e.g. a problematic subduction boundary at a given age can
    leave an output file missing without the subprocess call itself
    reporting anything to Create_History.py.'''

    model_name = control_d['model_name']

    def have_file( dir_key, prefix ):
        base = os.path.join( control_d[dir_key], prefix )
        if os.path.isfile( base ) and os.path.getsize( base ) > 0:
            return True
        # full-sphere models write one file per cap: '<prefix>.<cap>'
        for name in glob.glob( base + '.*' ):
            if os.path.getsize( name ) > 0:
                return True
        return False

    checks = []

    if control_d.get('OUTPUT_LITH_AGE'):
        checks.append( have_file( 'lith_age_dir', '%s.lith.dat%s' % (model_name, age) ) )

    if control_d.get('OUTPUT_TEMP'):
        checks.append( have_file( 'hist_dir', '%s.hist.dat%s' % (model_name, age) ) )

    if control_d.get('OUTPUT_IVEL'):
        checks.append( have_file( 'ivel_dir', 'ivel.dat%s' % age ) )

    if control_d.get('SYNTHETIC') and control_d.get('OUTPUT_BVEL'):
        checks.append( have_file( 'bvel_dir', 'bvel.dat%s' % age ) )

    if is_ic:
        if control_d.get('OUTPUT_TEMP_IC'):
            checks.append( bool( glob.glob( os.path.join(
                control_d['ic_dir'], '%s.velo.*.%s' % (model_name, age) ) ) ) )
        if control_d.get('OUTPUT_TRAC_IC'):
            checks.append( have_file( 'trac_dir', '%s.tracer.%sMa' % (model_name, age) ) )

    # nothing configured to be output -> nothing to verify; treat this
    # (deliberately unusual) case as "not present" so it can never be
    # silently skipped or silently marked successful
    if not checks:
        return False

    return all( checks )
#end function

#====================================================================
#====================================================================
#====================================================================
def log_critical( critical_log_path, lock, message ):
    '''Thread-safe append of a timestamped message to the critical log.'''
    with lock:
        with open( critical_log_path, 'a' ) as f:
            f.write( '%s %s\n' % (now(), message) )
#end function

#====================================================================
#====================================================================
#====================================================================
def process_age( config_path, cwd, age, is_ic, control_d, overwrite,
                  critical_log_path, log_lock ):
    '''Build the history files for a single age.  Returns a dict
    describing the outcome; never raises - any problem is captured in
    the returned dict and in the critical log instead.'''

    age_dir = os.path.join( cwd, str(age) )
    result = { 'age': age, 'status': 'ok', 'detail': '' }

    try:
        if (not overwrite) and expected_outputs_present( control_d, age, is_ic ):
            result['status'] = 'skipped'
            if verbose: print( now(), 'age %s: outputs already present, skipping' % age )
            return result

        os.makedirs( age_dir, exist_ok=True )
        shutil.copy( os.path.join( cwd, 'geodynamic_framework_defaults.conf' ), age_dir )

        log_path = os.path.join( age_dir, 'make_history.log' )
        cmd = [ 'make_history_for_age.py', config_path, str(age), str(int(is_ic)) ]

        if verbose: print( now(), 'age %s: starting (log: %s)' % (age, log_path) )

        with open( log_path, 'w' ) as log_file:
            proc = subprocess.run( cmd, cwd=age_dir, stdout=log_file,
                                    stderr=subprocess.STDOUT )

        if proc.returncode != 0:
            result['status'] = 'failed'
            result['detail'] = 'make_history_for_age.py exited with code %d (see %s)' % \
                                (proc.returncode, log_path)
        elif not expected_outputs_present( control_d, age, is_ic ):
            result['status'] = 'failed'
            result['detail'] = 'exited cleanly but expected output file(s) are missing ' \
                                '(see %s)' % log_path
        else:
            if verbose: print( now(), 'age %s: done' % age )

    except Exception:
        result['status'] = 'failed'
        result['detail'] = 'unhandled exception while processing age %s:\n%s' % \
                            (age, traceback.format_exc())

    if result['status'] == 'failed':
        print( now(), 'age %s: CRITICAL FAILURE - %s' % (age, result['detail']) )
        log_critical( critical_log_path, log_lock, 'age %s FAILED: %s' % (age, result['detail']) )

    return result
#end function

#====================================================================
#====================================================================
#====================================================================
def report_summary( results, critical_log_path ):
    '''Print, and return the process exit status for, a final summary
    of every requested age once all processing has finished.'''

    ok = [ r for r in results if r['status'] == 'ok' ]
    skipped = [ r for r in results if r['status'] == 'skipped' ]
    failed = [ r for r in results if r['status'] == 'failed' ]

    print( now(), '=' * 70 )
    print( now(), 'Create_History.py: run summary' )
    print( now(), '  requested : %d' % len(results) )
    print( now(), '  succeeded : %d' % len(ok) )
    print( now(), '  skipped   : %d (already had valid output)' % len(skipped) )
    print( now(), '  FAILED    : %d' % len(failed) )

    if failed:
        failed_ages = sorted( (r['age'] for r in failed), reverse=True )
        print( now(), '  failed ages:', failed_ages )
        print( now(), '  see', critical_log_path, 'for details of each failure' )
    print( now(), '=' * 70 )

    return 1 if failed else 0
#end function

#====================================================================
#====================================================================
#====================================================================
def main():
    '''Main sequence of script actions.'''

    print( now(), 'Create_History.py:')
    print( now(), 'main:')

    # read settings from control file
    config_filename = sys.argv[1]
    control_d = Core_Util.parse_configuration_file( config_filename )
    
    # read job settings
    control_d['serial'] = isSerial(control_d)

    age_start = max( control_d['age_start'], control_d['age_end'] )
    age_end = min( control_d['age_end'], control_d['age_start'] )
    age_loop = list( range( age_end, age_start+1 ) )
    age_loop.reverse()

    # the oldest requested age is the one and only age that requires an
    # initial condition; this mirrors the original 'IC' flag which was
    # only ever True on the first (oldest) age processed
    oldest_age = age_loop[0] if age_loop else None

    # if False, ages that already have valid output (per the OUTPUT_*
    # switches) are left untouched instead of being recomputed - lets a
    # run be resumed after fixing whatever caused earlier failures
    # without reprocessing everything from scratch
    overwrite_existing = control_d.get('OVERWRITE_EXISTING', True)

    config_path = os.path.abspath( config_filename )

    job = control_d['job']

    # smp and serial branch
    if (job=='smp'):
        serial = control_d['serial']
        cwd = os.getcwd()
        critical_log_path = os.path.join( cwd, CRITICAL_LOG_FILENAME )
        log_lock = Lock()
        results = []

        if(serial):
            for age in age_loop:
                is_ic = (age == oldest_age)
                results.append( process_age( config_path, cwd, age, is_ic,
                                 control_d, overwrite_existing,
                                 critical_log_path, log_lock ) )
            #end for
        else:
            cpuCount = control_d['nproc'];
            if(cpuCount==-1): cpuCount = int(multiprocessing.cpu_count());
            else: cpuCount = min(cpuCount, int(multiprocessing.cpu_count()));

            # submit ages oldest-to-youngest; the pool always pulls the
            # next-oldest unstarted age as soon as a worker frees up, so
            # at most 'cpuCount' ages are ever in flight at once and the
            # completed set stays close to a clean, contiguous prefix
            # from the oldest requested age toward the present - rather
            # than scattered holes across the whole range that are hard
            # to find and resume from
            with ThreadPoolExecutor( max_workers=cpuCount ) as executor:
                futures = [
                    executor.submit( process_age, config_path, cwd, age,
                                      age == oldest_age, control_d,
                                      overwrite_existing, critical_log_path,
                                      log_lock )
                    for age in age_loop
                ]
                for future in as_completed( futures ):
                    results.append( future.result() )
            #end with
        #end if

        sys.exit( report_summary( results, critical_log_path ) )
    # parallel branch
    else:

        # total number of ages to create (inclusive)
        # and therefore number of processors to use
        # this is req for PBS submission script
        control_d['nodes'] = len(age_loop)

        batchfile = 'commands.batch'
        file = open( batchfile, 'w')

        cwd = os.getcwd()

        IC = 1

        # make directories for files of each age
        for age in age_loop:
            cmd = 'mkdir %(age)s' % vars()
            if verbose: print( now(), cmd)
            subprocess.call( cmd, shell=True )

            if (control_d['job']=='cluster'):
                # commands for batch file
                line = 'cp geodynamic_framework_defaults.conf %(cwd)s/%(age)s; ' % vars()
                line+= 'cd %(cwd)s/%(age)s; ' % vars()
                line+= 'source ~/.bash_profile; '
                line+= 'module load python; '
                line+= 'which python; '
                line+= 'module load gmt; '
                line+= 'make_history_for_age.py ' % vars()
                line+= '%(cwd)s/%(config_filename)s %(age)s ' % vars()
                line+= '%(IC)s\n' % vars()
                file.write( line )
                IC = 0

            if (control_d['job']=='raijin'):
                # commands for batch file
                line = 'cp geodynamic_framework_defaults.conf %(cwd)s/%(age)s; ' % vars()
                line+= 'cd %(cwd)s/%(age)s; ' % vars()
                line+= 'source ~/.profile; '
                line+= 'module load python3/3.3.0; '
                line+= 'module load python3/3.3.0-matplotlib; '
                line+= 'module load gmt/4.5.11; '
                line+= 'make_history_for_age.py ' % vars()
                line+= '%(cwd)s/%(config_filename)s %(age)s ' % vars()
                line+= '%(IC)s\n' % vars()
                file.write( line )
                IC = 0

            if (control_d['job']=='baloo'):
                # commands for batch file
                line = 'cp geodynamic_framework_defaults.conf %(cwd)s/%(age)s; ' % vars()
                line+= 'cd %(cwd)s/%(age)s; ' % vars()
                line+= 'source ~/.cshrc; '
                line+= 'make_history_for_age.py ' % vars()
                line+= '%(cwd)s/%(config_filename)s %(age)s ' % vars()
                line+= '%(IC)s\n' % vars()
                file.write( line )
                IC = 0

        file.close()

        # if cluster job:
        if (control_d['job']=='cluster'):
            make_pbs_sub_script( control_d )
        # if raijin cluster job:
        if (control_d['job']=='raijin'):
            make_raijin_pbs_sub_script( control_d )
        # if baloo cluster job:
        if (control_d['job']=='baloo'):
            make_baloo_pbs_sub_script( control_d )

        # for abigail
        #make_sbatch_sub_script( control_d )

        # submit to qsub
        qsub = control_d['qsub']
        cmd = 'qsub %(qsub)s' % vars()
        if verbose: print( now(), cmd)
        subprocess.call( cmd, shell=True )

#====================================================================
#====================================================================
#====================================================================
def make_raijin_pbs_sub_script( control_d ):

    '''Write PBS submission script to a file.'''

    if verbose: print( now(), 'make_raijin_pbs_submission_script:' )

    # get variables
    jobname = control_d.get('jobname','Create_History_Parallel.py')
    nodes = control_d['nodes']
    # 12 hour default walltime if not specified
    walltime = control_d.get('walltime','12:00:00')
    # by default send job information to Nico
    email = control_d.get('email','nicolas.flament@sydney.edu.au')
    # by default 8GB memory
    mem = control_d.get('mem','8')

    text='''#!/bin/bash
#PBS -N %(jobname)s
#PBS -l ncpus=%(nodes)s
#PBS -l mem=%(mem)dGB
#PBS -P q97
#PBS -l walltime=%(walltime)s
#PBS -r y
#PBS -m bae
#PBS -M %(email)s
#PBS -l wd
#PBS -j oe

# Set up job environment:
module load python3/3.3.0
module load python3/3.3.0-matplotlib
module load gmt/4.5.11
module load parallel/20150322

#change the working directory (default is home directory)
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

# Write out some information on the job
echo Running on host `hostname`
echo Time is `date`

### Define number of processors
NPROCS=`wc -l < $PBS_NODEFILE`
echo This job has allocated $NPROCS cpus

# Tell me which nodes it is run on
echo " "
echo This jobs runs on the following processors:
echo `cat $PBS_NODEFILE`
echo " "

# 
# Run the parallel job
#

parallel -a commands.batch''' % vars()

    filename = '%(jobname)s.pbs' % vars()
    control_d['qsub'] = filename
    file = open( filename, 'w' )
    file.write( '%(text)s' % vars() )
    file.close()


def make_baloo_pbs_sub_script( control_d ):

    '''Write PBS submission script to a file.'''

    if verbose: print( now(), 'make_baloo_pbs_submission_script:' )

    # get variables
    jobname = control_d.get('jobname','Create_History_Parallel.py')
    ppn = control_d.get('ppn','16')
    nodes = int(control_d['nodes']/ppn)
    # 12 hour default walltime if not specified
    walltime = control_d.get('walltime','12:00:00')
    # by default send job information to Nico
    email = control_d.get('email','rezg@statoil.com')

    text='''#!/bin/csh -f 
#PBS -N %(jobname)s
#PBS -l nodes=%(nodes)s:ppn=%(ppn)s
#PBS -l walltime=%(walltime)s
#PBS -m bae
#PBS -M %(email)s

#change the working directory (default is home directory)
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

# Write out some information on the job
echo Running on host `hostname`
echo Time is `date`

### Define number of processors
#NPROCS=`wc -l < $PBS_NODEFILE`
#echo This job has allocated $NPROCS cpus

# Tell me which nodes it is run on
echo " "
echo This jobs runs on the following processors:
echo `cat $PBS_NODEFILE`
echo " "

# 
# Run the parallel job
#

parallel -a commands.batch''' % vars()

    filename = '%(jobname)s.pbs' % vars()
    control_d['qsub'] = filename
    file = open( filename, 'w' )
    file.write( '%(text)s' % vars() )
    file.close()

#====================================================================
#====================================================================
#====================================================================
def make_pbs_sub_script( control_d ):

    '''Write PBS submission script to a file.'''

    if verbose: print( now(), 'make_pbs_submission_script:' )

    # get variables
    jobname = control_d.get('jobname','Create_History_Parallel.py')
    nodes = control_d['nodes']
    # 12 hour default walltime if not specified
    walltime = control_d.get('walltime','12:00:00')

    text='''#PBS -N %(jobname)s
#PBS -l nodes=%(nodes)s
#PBS -S /bin/bash
#PBS -V
#PBS -l walltime=%(walltime)s
#PBS -q default
#PBS -m ae
#PBS -o out.$PBS_JOBID.$PBS_JOBNAME
#PBS -e err.$PBS_JOBID.$PBS_JOBNAME

#change the working directory (default is home directory)
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

# Write out some information on the job
echo Running on host `hostname`
echo Time is `date`

### Define number of processors
NPROCS=`wc -l < $PBS_NODEFILE`
echo This job has allocated $NPROCS cpus

# Tell me which nodes it is run on
echo " "
echo This jobs runs on the following processors:
echo `cat $PBS_NODEFILE`
echo " "

# 
# Run the parallel job
#

parallel --sshloginfile $PBS_NODEFILE  -a commands.batch''' % vars()

    filename = 'qsub.Create_History_Parallel'
    control_d['qsub'] = filename
    file = open( filename, 'w' )
    file.write( '%(text)s' % vars() )
    file.close()

#====================================================================
#====================================================================
#====================================================================
def make_sbatch_sub_script( control_d ):

    '''Write SBATCH submission script to a file.'''

    if verbose: print( now(), 'make_sbatch_sub_script:' )

    # get variables
    jobname = control_d.get('jobname','Create_History_Parallel.py')
    nodes = control_d['nodes']
    # 2 hour default walltime if not specified
    #walltime = settings.get('walltime','12:00:00')

    text = '''#!/bin/bash

# Batch file for running CitcomS on Titan at UiO
# ALB Oct/Nov 2011

# Job Details
#SBATCH --job-name=%(jobname)s
#SBATCH --account=pgp
#SBATCH --constraint=intel
# Job time and memory limits
#SBATCH --time=96:00:00 ## YOU MUST CHANGE THIS FOR LONG JOBS
#SBATCH --mem-per-cpu=2GB
#
#Parallel and mpi settings
#SBATCH --ntasks=12 ## MPI KNOWS HOW MANY NODES IT CAN USE, DON'T SPECIFY THEM
#
# Set up job environment:
source /site/bin/jobsetup
module load python/2.6.2
module load openmpi/1.4.3.intel

## Copy the CASE1 dir to the scratch, just in case
srun --ntasks=$SLURM_JOB_NUM_NODES cp -r OUTPUTFILES/ $SCRATCH

#Run program
bin/citcoms cookbook1.cfg --solver.datadir=/usit/titan/u1/abigailb/CITCOM_S/CitcomS_CIG/OUTPUTFILES/Cookbook1''' % vars()

    filename = 'qsub.Create_History_Parallel'
    settings['qsub'] = filename
    file = open(filename,'w')
    file.write('%(text)s' % vars())
    file.close()

#====================================================================
#====================================================================
#====================================================================
def make_example_config_file():

    # get current working directory
    cwd = os.getcwd()
    text='''#============
# job details
#============
# N.B., keeping with python convention you must capitalize
# True and False for Booleans!

job = smp ; options are 'smp' or 'cluster' or 'raijin' or 'baloo'.
nproc = 1 ; -1 to use all available procs.
# For 'smp', 'nproc'=1 is the serial case.
# For 'cluster' or 'raijin' or 'baloo', 'nproc' is ignored.

jobname = mkhist ; for cluster or raijin or baloo  job only
walltime = 3:00:00 ; for cluster or raijin  or baloo job only
mem = 8 ; (in GB) for raijin job only 
ppn = 16 ; for baloo job only
email = rezg@statoil.com ; for raijin  or baloo job only
age_start = 1
age_end = 0

DEBUG = False ; generic switch for debugging
VERBOSE = True ; show terminal output

# for 'smp' jobs only: if False, ages that already have valid output
# files (per the OUTPUT_* switches below) are skipped rather than
# recomputed - useful for resuming a run after fixing whatever caused
# earlier ages to fail. if True (default), every requested age is
# always (re)computed, matching the original script behaviour.
OVERWRITE_EXISTING = True

# do not remove processed age and final temperature grids
KEEP_GRIDS = True
PLOT_SUMMARY_POSTSCRIPT = True; make a summary ps file for each depth
KEEP_PS = False

model_name = out ; history model name for output

#============
# data output
#============

# [INITIAL CONDITION]
# temperature initial condition
OUTPUT_TEMP_IC = False

# tracer initial condition
OUTPUT_TRAC_IC = False

# [HISTORY]
# slab temperature history
OUTPUT_TEMP = False

# internal velocity (slab descent) 
OUTPUT_IVEL = False

# thermal age of lithosphere
OUTPUT_LITH_AGE = True

#========================
# data output directories
#========================
# N.B. use an absolute path for job = cluster

grid_dir = %(cwd)s/grid ; intermediate grids
hist_dir = %(cwd)s/hist ; history
ic_dir = %(cwd)s/ic ; initial condition
ivel_dir = %(cwd)s/ivel ; ivel
lith_age_dir = %(cwd)s/age ; age
log_dir = %(cwd)s ; parameters
ps_dir = %(cwd)s/ps ; postscripts
trac_dir = %(cwd)s/trac ; tracers

#===========
# data input
#===========
# N.B. use an absolute path for job = cluster

pid_file = %(cwd)s/pid00000.cfg ; CitcomS pid file

# default coordinate file path is:
#     [datadir]/[proc]/[datafile].coord.[proc]
# or define a user-specified directory to all of the
# [datafile].coord.[proc] files:
coord_dir = %(cwd)s/coord ; CitcomS *.coord.* files

# spatial resolution of the prescribed ivel bcs
# levels = 1 ; prescribe at finest mesh
# levels = 2 ; coarsen by 2 in each dimension
# levels = 3 ; coarsen by 4 in each dimension
# for global models you'll likely need levels >= 2
# for regional models try levels = 1 and then increase
# if convergence is poor or solve time is unreasonable
levels = 2


#===================
# thermal parameters
#===================
# mantle temperature at the surface
temperature_mantle = 1.0

BUILD_LITHOSPHERE = True ; include an upper thermal boundary layer
UTBL_AGE_GRID = True ; True will use age grids
utbl_age = 300 ; if UTBL_AGE_GRID is False
lith_age_min = 0.01 ; minimum lithosphere thermal age
# note that the below only truncates oceanic regions
oceanic_lith_age_max = 300.0 ; maximum oceanic thermal age
# thermal age for non-oceanic regions if CONTINENTAL_TYPES = False
NaN_age = 200.0


BUILD_SLAB = True ; build slabs
radius_of_curvature = 200.0 ; km
# default values for slab dip and depth if GPML_HEADER = False
default_slab_dip = 45.0 ; degrees
default_slab_depth = 500.0 ; km
UM_advection = 1.0 ; non-dim factor
LM_advection = 3.0 ; non-dim factor
vertical_slab_depth = 660.0 ; depth at which to make slabs vertical

# GPML_HEADER must be True for subduction initiation
GPML_HEADER = True ; override defaults with GPML header data
slab_UM_descent_rate = 3.0 ; cm/yr
# from van der Meer et al. (2010)
slab_LM_descent_rate = 1.2 ; cm/yr

FLAT_SLAB = False ; include flat slabs

# lower thermal boundary layer
# with LTBL, the temperature of the CMB is always 1
BUILD_LTBL  = False ; lower thermal boundary layer
ltbl_age = 300.0 ; age (Ma) of tbl

# thermal blobs
BUILD_BLOB = False ; thermal blobs
blob_center_lon = 50, 130 ; degrees
blob_center_lat = 45, 45 ; degrees colat
blob_center_depth = 2867, 2867 ; km
blob_radius = 200, 400 ; km
blob_birth_age = 230, 220 ; Ma
blob_dT = 0.1, 0.1 ; non-dimensional temperature anomaly
blob_profile = constant, constant ; valid profiles (constant, exponential, gaussian1, gaussian2)

# thermal silos
BUILD_SILO = False ; thermal silos
silo_base_center_lon = 50, 130 ; degrees
silo_base_center_lat = 90, 90 ; degrees colat
silo_base_center_depth = 2867, 2867 ; km
silo_radius = 200, 400 ; km
silo_cylinder_height= 500, 500 ; km
silo_birth_age = 230, 220 ; Ma
silo_dT = 0.1, 0.1 ; non-dimensional temperature anomaly
silo_profile = constant, constant ; valid profiles (constant, exponential, gaussian1, gaussian2)

# ADIABAT only for extended-Boussinesq or compressible models
BUILD_ADIABAT = False ; linear temp increase across mantle
# non-dimensional adiabatic temperature drop scaled with respect
# to the total temperature drop across the model
adiabat_temp_drop = 0.3


#===========
# Continents
#===========
# Build upper thermal boundary layer with continents
# Also continental tracers (if OUTPUT_TRAC_IC is also True)
CONTINENTAL_TYPES = True

# For continental types, list stencil values with no spaces delimited 
# by a comma. Use negative integers.
# Then, for each stencil give the age for reassignment
# In the example below, -1 is Archean, -2 Proterozoic, -3 Phanerozoic,
# and -4 COB.
# ensure stencil_values are small negative integers

stencil_values = -4,-3,-2,-1
stencil_ages = 104,103,153,369

# No assimilation in areas that have been deformed
NO_ASSIM = True
no_ass_age = -1000
no_ass_padding = 100

# Exclude tracers in areas that have been deformed
# to a depth defined by 'tracer_no_ass_depth'
# also uses no_ass_age and no_ass_padding
# and CONTINENTAL_TYPES must be True
# also, no_ass_age must be more negative than the
# most negative stencil value
TRACER_NO_ASSIM = False
tracer_no_ass_depth = 350 ; km

# Build tracer field with continents using 'stencil_values'
# tracer flavors and depths

# for positive thermal ages (i.e., oceanic)
# note: must be '0' suffix
flavor_stencil_value_0 = 0
depth_stencil_value_0 = 410

# for stencil value -1
flavor_stencil_value_1 = 1,2
depth_stencil_value_1 = 40,250

# for stencil value -2
flavor_stencil_value_2 = 1,3
depth_stencil_value_2 = 40,160

# for stencil value -3
flavor_stencil_value_3 = 1,4
depth_stencil_value_3 = 40,130

# for stencil value -4
flavor_stencil_value_4 = 1,4
depth_stencil_value_4 = 40,130

# etc. for more stencil values, e.g.,
# flavor_stencil_value_5 = 0 
# depth_stencil_value_5 = 410

# set region around slabs to ambient flavor (0)
SLAB_STENCIL = True
# stencil width: 300 km is consistent with the default width of the thermal stencil
# wide stencils limit crustal thickening along convergent margins
# narrow stencils avoid a gap along convergent margins but may result in significant
# crustal thickening and unrealistic elevations along convergent margins
slab_stencil_width = 300 ; km - suggested range: 100-300 km

# uniform dense layer at base of mantle
DEEP_LAYER_TRACERS = False
deep_layer_thickness = 300 ; km - 113 km gives 2 per cent of Earth's volume.
# flavor should not be 0 (0 is always ambient flavor)
deep_layer_flavor = 5

# eliminate tracers between these bounds
# this saves memory when using the hybrid method to compute composition
NO_TRACER_REGION = True
no_tracer_min_depth = 410 ; km
no_tracer_max_depth = 2604 ; km

#==============================
# synthetic regional model only
#==============================
SYNTHETIC = False ; master switch for synthetic regional models
OUTPUT_BVEL = True ; output velocity boundary conditions
bvel_dir = %(cwd)s/bvel ; bvel
fi_trench = 0.8 ; radians
TRENCH_CURVING = False
curving_trench_lat = 0.0 ; degrees
subduction_zone_age = 100 ; Ma
plate_velocity = 5 ; cm/yr
plate_velocity_theta = -1 ; direction of velocity (non-dim)
plate_velocity_phi = 1 ; direction of velocity (non-dim)
velocity_smooth = 200 ; bvel smoothing (Gaussian filter)
no_of_edge_nodes_to_zero = 3 ; no of edge nodes to smooth across (x and y)
overriding_age = 50 ; Ma
rollback_start_age = 100 ; Ma
rollback_cm_yr = 0 ; cm/yr (direction is always -phi)
''' % vars()

    print( text )

#====================================================================
#====================================================================
#====================================================================

if __name__ == "__main__":

    # check for script called wih no arguments
    if len(sys.argv) < 2:
        usage()
        sys.exit(-1)

    # create example config file 
    if '-e' in sys.argv:
        make_example_config_file()
        sys.exit(0)

    # create example geodynamic_framework_defaults.conf
    if '-d' in sys.argv:
        Core_Util.parse_geodynamic_framework_defaults()
        sys.exit(0)

    # run the main script workflow
    main()
    sys.exit(0)

#====================================================================
#====================================================================
#====================================================================
