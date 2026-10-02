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
import datetime, glob, os, shutil, subprocess, sys, time, multiprocessing, traceback
import Core_Util
from Core_Util import now
from subprocess import PIPE, Popen
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Thread, Event
from make_history_for_age import basic_setup, isSerial
#=====================================================================
verbose = True

# name of the file (written into the job's top-level working directory)
# that accumulates every critical failure encountered while building
# per-age history files
CRITICAL_LOG_FILENAME = 'critical_errors.log'

# name of the file that tallies non-fatal WARNING/ERROR-ish text found
# in per-age logs (almost always raw GMT stderr chatter) - informational
# only, kept separate from CRITICAL_LOG_FILENAME so real failures are
# never buried among routine GMT noise
WARNINGS_LOG_FILENAME = 'warnings_summary.log'
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
def format_duration( seconds ):
    '''Human-readable H:MM:SS (or D days, H:MM:SS) rendering of a
    duration in seconds, used for both per-age and whole-run timing.'''
    return str( datetime.timedelta( seconds=int( seconds ) ) )
#end function

#====================================================================
#====================================================================
#====================================================================
def tail_log_line( log_path ):
    '''Best-effort read of the last non-empty line of a per-age log -
    used to surface live progress (e.g. during the initial-condition
    heartbeat) instead of a bare "still running" ping. Returns '' if
    the file doesn't exist yet, is empty, or can't be read (e.g. the
    subprocess hasn't created it yet).'''
    try:
        with open( log_path ) as f:
            lines = [ line.rstrip() for line in f if line.strip() ]
        return lines[-1] if lines else ''
    except (IOError, OSError):
        return ''
#end function

#====================================================================
#====================================================================
#====================================================================
def print_status_block( in_flight, progress_lock, results, total, start_time ):
    '''Print one snapshot of overall progress plus, for every age
    currently in flight, its elapsed time and latest log line (e.g.
    'znode 32/65', 'track_grids_to_cap_list: grid 23/63') - so a long
    run shows not just how many ages are done, but what each of the
    currently-running ones is actually doing right now.'''

    done = len( results )
    failed_so_far = sum( 1 for r in results if r['status'] == 'failed' )
    elapsed = time.time() - start_time

    eta = ''
    if 0 < done < total:
        remaining = elapsed / done * (total-done)
        eta = ' | ETA %s' % format_duration( remaining )

    pct = int( 100*done/total ) if total else 100

    with progress_lock:
        snapshot = sorted( in_flight.items() )

    print( now(), '-' * 70 )
    print( now(), 'progress: %d/%d done (%d%%) | %d failed%s' %
           (done, total, pct, failed_so_far, eta) )
    if snapshot:
        print( now(), 'currently processing %d age(s):' % len(snapshot) )
        for age, (age_start, log_path) in snapshot:
            last_line = tail_log_line( log_path )
            print( now(), '  age %s (%s elapsed): %s' % (
                age, format_duration( time.time() - age_start ),
                last_line if last_line else 'starting...' ) )
    print( now(), '-' * 70 )
#end function

#====================================================================
#====================================================================
#====================================================================
def status_reporter_loop( in_flight, progress_lock, results, total, start_time,
                           stop_event, interval ):
    '''Runs in a background thread for the lifetime of the smp run
    (covering both the solo initial-condition call and the main pool),
    printing a status snapshot every 'interval' seconds until told to
    stop. A single mechanism for both phases - whatever age(s) are
    in 'in_flight' at the time, IC or not, get reported.'''
    while not stop_event.wait( interval ):
        print_status_block( in_flight, progress_lock, results, total, start_time )
    #end while
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
def extract_critical_lines( log_path ):
    '''Pull out every logging.critical(...) line (tagged 'CRITICAL' by
    the standard logging format) from a per-age log.  make_history_for_age.py
    now logs a CRITICAL line immediately before every fatal exit, so this
    is the actual root cause of a failure - not just a pointer to a file
    the user has to go open.'''
    lines = []
    try:
        with open( log_path ) as f:
            for line in f:
                if 'CRITICAL' in line:
                    lines.append( line.rstrip() )
    except (IOError, OSError):
        pass
    return lines
#end function

#====================================================================
#====================================================================
#====================================================================
def extract_traceback_blocks( log_path ):
    '''Pull out raw Python traceback blocks (unhandled exceptions,
    e.g. a KeyError from a missing config key) from a per-age log.
    These never go through logging.critical() - they're Python's own
    default crash output - so extract_critical_lines() alone misses
    them entirely, silently hiding the actual error behind a generic
    "exited with code 1, no CRITICAL message found" line.'''
    blocks = []
    try:
        with open( log_path ) as f:
            lines = [ line.rstrip('\n') for line in f ]
    except (IOError, OSError):
        return blocks

    ii, n = 0, len(lines)
    while ii < n:
        if lines[ii] == 'Traceback (most recent call last):':
            block = [ lines[ii] ]
            jj = ii + 1
            while jj < n:
                line = lines[jj]
                block.append( line )
                jj += 1
                # a traceback block ends at the first line that is not
                # indented - that line is the 'ExceptionType: message'
                # summary, and is itself part of the block
                if line and not line[0].isspace():
                    break
            blocks.append( block )
            ii = jj
        else:
            ii += 1
    return blocks
#end function

#====================================================================
#====================================================================
#====================================================================
def tally_log_noise( log_path ):
    '''Count non-fatal WARNING/ERROR-ish lines in a per-age log - in
    practice almost always raw GMT stderr text, which (per repeated
    observation) is frequently emitted for conditions that are not
    fatal to the run.  This is explicitly NOT the failure signal (that
    is the exit code + expected-output check in process_age()); it is
    purely informational, to make it easy to see which ages produced
    GMT chatter without having to open every single log file.  Lines
    already counted as CRITICAL are excluded so the two tallies never
    double up on the same root cause.'''
    warnings = errors = 0
    try:
        with open( log_path ) as f:
            for line in f:
                if 'CRITICAL' in line:
                    continue
                low = line.lower()
                if 'warning' in low:
                    warnings += 1
                elif 'error' in low:
                    errors += 1
    except (IOError, OSError):
        pass
    return warnings, errors
#end function

#====================================================================
#====================================================================
#====================================================================
def summarize_failure_from_log( log_path ):
    '''Build a human-readable summary of why an age failed, pulling in
    both logging.critical() lines and raw Python tracebacks - the two
    kinds of fatal signal a per-age log can contain. This is what
    actually ends up inline in critical_errors.log, so the reason is
    visible there directly instead of requiring a trip into the age's
    own log folder.'''
    critical_lines = extract_critical_lines( log_path )
    traceback_blocks = extract_traceback_blocks( log_path )

    parts = []
    if critical_lines:
        parts.append( '; '.join( critical_lines ) )
    for block in traceback_blocks:
        parts.append( '\n'.join( block ) )

    if not parts:
        return 'no CRITICAL message or traceback found in log - inspect it directly'
    return ' | '.join( parts )
#end function

#====================================================================
#====================================================================
#====================================================================
def process_age( config_path, cwd, age, is_ic, control_d, overwrite,
                  critical_log_path, log_lock, in_flight, progress_lock ):
    '''Build the history files for a single age.  Returns a dict
    describing the outcome; never raises - any problem is captured in
    the returned dict and in the critical log instead.

    While the actual subprocess is running, this age is registered in
    the shared 'in_flight' dict (age -> (start_time, log_path)) so the
    background status_reporter_loop can report its live progress -
    removed again once the subprocess finishes, success or not.'''

    age_dir = os.path.join( cwd, str(age) )
    log_path = os.path.join( age_dir, 'make_history.log' )
    result = { 'age': age, 'status': 'ok', 'detail': '' }
    t0 = time.time()

    try:
        if (not overwrite) and expected_outputs_present( control_d, age, is_ic ):
            result['status'] = 'skipped'
            result['duration'] = time.time() - t0
            if verbose: print( now(), 'age %s: outputs already present, skipping' % age )
            return result

        os.makedirs( age_dir, exist_ok=True )
        shutil.copy( os.path.join( cwd, 'geodynamic_framework_defaults.conf' ), age_dir )

        cmd = [ 'make_history_for_age.py', config_path, str(age), str(int(is_ic)) ]

        if verbose: print( now(), 'age %s: starting (log: %s)' % (age, log_path) )

        with progress_lock:
            in_flight[age] = (t0, log_path)
        try:
            with open( log_path, 'w' ) as log_file:
                proc = subprocess.run( cmd, cwd=age_dir, stdout=log_file,
                                        stderr=subprocess.STDOUT )
        finally:
            with progress_lock:
                in_flight.pop( age, None )

        warnings, errors = tally_log_noise( log_path )
        result['warnings'] = warnings
        result['errors'] = errors

        if proc.returncode != 0:
            reason = summarize_failure_from_log( log_path )
            result['status'] = 'failed'
            result['detail'] = 'make_history_for_age.py exited with code %d: %s (full log: %s)' % \
                                (proc.returncode, reason, log_path)
        elif not expected_outputs_present( control_d, age, is_ic ):
            reason = summarize_failure_from_log( log_path )
            result['status'] = 'failed'
            result['detail'] = 'exited cleanly but expected output file(s) are missing: %s ' \
                                '(full log: %s)' % (reason, log_path)
        else:
            if verbose: print( now(), 'age %s: done (%s)' %
                               (age, format_duration( time.time() - t0 )) )

    except Exception:
        result['status'] = 'failed'
        result['detail'] = 'unhandled exception while processing age %s:\n%s' % \
                            (age, traceback.format_exc())

    result['duration'] = time.time() - t0

    if result['status'] == 'failed':
        print( now(), 'age %s: CRITICAL FAILURE after %s - %s' %
               (age, format_duration( result['duration'] ), result['detail']) )
        log_critical( critical_log_path, log_lock, 'age %s FAILED after %s: %s' %
                      (age, format_duration( result['duration'] ), result['detail']) )

    return result
#end function

#====================================================================
#====================================================================
#====================================================================
def write_warnings_summary( results, warnings_log_path ):
    '''Write the non-fatal GMT WARNING/ERROR tally to its own file.
    Returns the number of ages that logged any such text.  This is
    purely informational - it is not evidence of a real failure, which
    is tracked separately in critical_errors.log.'''

    noisy = [ r for r in results if r.get('warnings') or r.get('errors') ]
    if not noisy:
        return 0

    with open( warnings_log_path, 'w' ) as f:
        f.write( '# informational only - NOT failures. GMT routinely prints\n' )
        f.write( '# WARNING/ERROR text for conditions that are not fatal. Real\n' )
        f.write( '# failures are reported separately in %s\n' % CRITICAL_LOG_FILENAME )
        f.write( '#\n' )
        f.write( '# age  warning_lines  error_lines\n' )
        for r in sorted( noisy, key=lambda r: r['age'], reverse=True ):
            f.write( '%s  %d  %d\n' % (r['age'], r.get('warnings',0), r.get('errors',0)) )

    return len(noisy)
#end function

#====================================================================
#====================================================================
#====================================================================
def report_summary( results, critical_log_path, warnings_log_path,
                     run_start_dt, run_end_dt ):
    '''Print, and return the process exit status for, a final summary
    of every requested age once all processing has finished.'''

    ok = [ r for r in results if r['status'] == 'ok' ]
    skipped = [ r for r in results if r['status'] == 'skipped' ]
    failed = [ r for r in results if r['status'] == 'failed' ]

    noisy_count = write_warnings_summary( results, warnings_log_path )

    print( now(), '=' * 70 )
    print( now(), 'Create_History.py: run summary' )
    print( now(), '  run started : %s' % run_start_dt.strftime( '%Y-%m-%d %H:%M:%S' ) )
    print( now(), '  run ended   : %s' % run_end_dt.strftime( '%Y-%m-%d %H:%M:%S' ) )
    print( now(), '  run duration: %s' % format_duration( (run_end_dt - run_start_dt).total_seconds() ) )
    print( now(), '  requested : %d' % len(results) )
    print( now(), '  succeeded : %d' % len(ok) )
    print( now(), '  skipped   : %d (already had valid output)' % len(skipped) )
    print( now(), '  FAILED    : %d' % len(failed) )

    # ages actually processed (not skipped) have a real 'duration' -
    # surface the slowest ones, since an unusually slow age can itself
    # be a useful signal even when it didn't outright fail
    timed = [ r for r in results if r['status'] != 'skipped' and r.get('duration') is not None ]
    if timed:
        slowest = sorted( timed, key=lambda r: r['duration'], reverse=True )[:5]
        avg = sum( r['duration'] for r in timed ) / len(timed)
        print( now(), '  average time per age: %s' % format_duration( avg ) )
        print( now(), '  slowest ages:' )
        for r in slowest:
            print( now(), '    age %s: %s%s' % (
                r['age'], format_duration( r['duration'] ),
                ' (FAILED)' if r['status'] == 'failed' else '' ) )

    if failed:
        failed_ages = sorted( (r['age'] for r in failed), reverse=True )
        print( now(), '  failed ages:', failed_ages )
        print( now(), '  see', critical_log_path, 'for the reason each one failed' )

    if noisy_count:
        print( now(), '  note: %d age(s) logged non-fatal GMT WARNING/ERROR text '
                       '(informational only) - see %s' % (noisy_count, warnings_log_path) )
    print( now(), '=' * 70 )

    return 1 if failed else 0
#end function

#====================================================================
#====================================================================
#====================================================================
def preflight_check( control_d, cwd ):
    '''Validate the handful of things that, if wrong, would otherwise
    make every single worker fail identically (e.g. a bad pid_file
    path) - catch that once, up front, with one clear message, instead
    of burning a parallel run on N copies of the same root cause.'''

    problems = []

    defaults_conf = os.path.join( cwd, 'geodynamic_framework_defaults.conf' )
    if not os.path.isfile( defaults_conf ):
        problems.append( "missing %s (generate it with: Create_History.py -d)" % defaults_conf )

    pid_file = control_d.get('pid_file')
    if not pid_file:
        problems.append( "'pid_file' is not set in the configuration file" )
    elif not os.path.isfile( pid_file ):
        problems.append( "pid_file not found: %s" % pid_file )

    coord_dir = control_d.get('coord_dir')
    if coord_dir and not os.path.isdir( coord_dir ):
        problems.append( "coord_dir not found: %s" % coord_dir )

    if not control_d.get('model_name'):
        problems.append( "'model_name' is not set in the configuration file" )

    return problems
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

    # Andres - This update allows processing an arbitrary list of ages in the configuration file.
    if 'age_list' in control_d:
        raw_age_list = control_d['age_list']

        if isinstance(raw_age_list, (list, tuple)):
            age_loop = [int(a) for a in raw_age_list]
        else:
            raw_age_list = str(raw_age_list)
            raw_age_list = raw_age_list.replace('[', '').replace(']', '')
            age_loop = [int(a.strip()) for a in raw_age_list.split(',') if a.strip()]

        if not age_loop:
            raise ValueError('age_list must contain at least one integer age')
    else:
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

    # fail fast with one clear message rather than letting every
    # worker discover (and separately report) the same broken config
    problems = preflight_check( control_d, os.getcwd() )
    if problems:
        print( now(), 'CRITICAL: pre-flight check failed - fix the following before running:' )
        for p in problems:
            print( now(), ' -', p )
        sys.exit(1)

    # smp and serial branch
    if (job=='smp'):
        serial = control_d['serial']
        cwd = os.getcwd()
        critical_log_path = os.path.join( cwd, CRITICAL_LOG_FILENAME )
        warnings_log_path = os.path.join( cwd, WARNINGS_LOG_FILENAME )
        log_lock = Lock()
        progress_lock = Lock()
        in_flight = {} # age -> (start_time, log_path), only while actually running
        results = []

        start_time = time.time()
        run_start_dt = datetime.datetime.now()
        print( now(), 'run started: %s' % run_start_dt.strftime( '%Y-%m-%d %H:%M:%S' ) )

        # one background thread reports progress for the whole smp run -
        # both the solo initial-condition call below and the main pool
        # after it, since both register themselves in 'in_flight' the
        # same way. Prints a full snapshot (overall % done, ETA, and
        # every currently-running age's latest log line) periodically,
        # so a long run shows what's actually happening instead of
        # going silent or showing a bare, uninformative bar.
        total_ages = len( age_loop )
        status_interval = 20 # seconds
        stop_status = Event()
        status_thread = Thread( target=status_reporter_loop,
                                 args=(in_flight, progress_lock, results, total_ages,
                                       start_time, stop_status, status_interval) )
        status_thread.daemon = True
        status_thread.start()

        def stop_status_reporter():
            stop_status.set()
            status_thread.join( timeout=5 )
        #end function

        # process the initial-condition age (always the oldest requested
        # age) on its own, before any other age starts. Tracer IC
        # generation in particular can be very memory-hungry, and that
        # peak would otherwise land at exactly the moment every other
        # worker is also starting up (its own heaviest point too) - the
        # worst possible time for it. The IC age is also the one every
        # other age's CitcomS run ultimately depends on, so a failure
        # here is treated as critical and aborts the rest of this run
        # immediately rather than burning compute on ages built on top
        # of a broken initial condition.
        remaining_ages = age_loop
        if oldest_age is not None:
            print( now(), 'processing initial-condition age %s on its own first...' % oldest_age )
            print( now(), 'this is typically the slowest single age (tracer generation can be '
                           'memory- and time-intensive)' )

            ic_result = process_age( config_path, cwd, oldest_age, True,
                                      control_d, overwrite_existing,
                                      critical_log_path, log_lock, in_flight, progress_lock )
            results.append( ic_result )

            if ic_result['status'] == 'failed':
                stop_status_reporter()
                print( now(), '=' * 70 )
                print( now(), 'CRITICAL: initial-condition age %s FAILED - aborting the '
                               'rest of this run.' % oldest_age )
                print( now(), 'CRITICAL:', ic_result['detail'] )
                print( now(), '=' * 70 )
                sys.exit( report_summary( results, critical_log_path, warnings_log_path,
                                           run_start_dt, datetime.datetime.now() ) )
            #end if

            remaining_ages = age_loop[1:]
        #end if

        if(serial):
            for age in remaining_ages:
                results.append( process_age( config_path, cwd, age, False,
                                 control_d, overwrite_existing,
                                 critical_log_path, log_lock, in_flight, progress_lock ) )
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
                    executor.submit( process_age, config_path, cwd, age, False,
                                      control_d, overwrite_existing,
                                      critical_log_path, log_lock, in_flight, progress_lock )
                    for age in remaining_ages
                ]
                for future in as_completed( futures ):
                    results.append( future.result() )
            #end with
        #end if

        stop_status_reporter()
        sys.exit( report_summary( results, critical_log_path, warnings_log_path,
                                   run_start_dt, datetime.datetime.now() ) )
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

# GMT's own internal (OpenMP) multithreading per GMT call. Ages are
# normally already processed with many ages in parallel (see 'nproc'
# above), so left unconstrained, each worker's GMT calls would also
# multithread internally and oversubscribe the available CPUs.
# default (1) pins each GMT call to a single thread, which is usually
# faster under heavy outer parallelism (nproc > 1). Set to -1 to leave
# GMT's own default threading unconstrained (e.g. for serial / small
# nproc runs where individual GMT calls dominate runtime), or to a
# specific integer to tune manually. If this slows preprocessing down
# for your job, try -1 first.
GMT_NUM_THREADS = 1

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

# scales the slab's thermal anomaly (colder slab reduces slab
# break-off). Despite the name, this is NOT gated by
# BUILD_WEAK_INTERFACE below - it is read unconditionally in the core
# slab temperature construction (Core_Util.make_slab_temperature_xyz)
# whenever BUILD_SLAB is True, so it is required regardless of the
# weak-interface toggle. 1.0 reproduces the original (pre-this-factor)
# behaviour; lower values make the slab colder.
SLAB_STRENGTH_FACTOR = 1.0

# GPML_HEADER must be True for subduction initiation
GPML_HEADER = True ; override defaults with GPML header data
slab_UM_descent_rate = 3.0 ; cm/yr
# from van der Meer et al. (2010)
slab_LM_descent_rate = 1.2 ; cm/yr

# optional additional depth cutoff for slab temperature assimilation,
# beyond whatever other depth limits already apply (e.g. slab_depth_gen).
# 0 (default) disables this extra cutoff - no behaviour change.
MAX_ASSIMILATION_DEPTH = 0 ; km - 0 disables this cutoff

FLAT_SLAB = False ; include flat slabs

# lower thermal boundary layer
# with LTBL, the temperature of the CMB is always 1
BUILD_LTBL  = False ; lower thermal boundary layer
ltbl_age = 300.0 ; age (Ma) of tbl

# weak interface along the top of the subducting slab - a newer,
# less-documented feature. Defaults to off; the two parameters below
# are only read at all when this is True, and are conservative
# placeholders (not verified physically-motivated defaults) - treat
# them as "must be set deliberately before turning this on", not as
# tuned values.
BUILD_WEAK_INTERFACE = False ; weak interface along top of slab
MAX_DEPTH_WEAK_INTERFACE = 300.0 ; km - placeholder, verify before enabling
WEAK_INTERFACE_TEMP_CAP = 1.0 ; mantle temperature is divided by this factor - placeholder, verify before enabling

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
# only used by make_synthetic_age_grid (SYNTHETIC models only) -
# distinct from oceanic_lith_age_max above
lith_age_max = 300.0 ; Ma - synthetic age grid clip, SYNTHETIC only
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
