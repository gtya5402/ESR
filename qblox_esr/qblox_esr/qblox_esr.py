"""
Module to run a QRM-RF from qblox as an esr spectrometer.

author: jbv
"""

from textwrap import dedent
from qblox_instruments import Cluster, PlugAndPlay
from .simple_convert import simple_convert_q1asm
from .advanced_convert import advanced_convert_q1asm
import matplotlib.pyplot as plt
import numpy as np
import copy
import ast


def connect2cluster(device_name):
    """
    Connects to the qblox cluster.
    """
    # Scan for available devices and display
    with PlugAndPlay() as p:
        # get info of all devices
        device_list = p.list_devices()
        device_keys = list(device_list.keys())

    connect_options = [(device_list[key]["description"]["name"]) for key in device_list.keys()]
    
    Cluster.close_all()  # close all previous connections to the cluster

    # Retrieve IP address
    device_number = connect_options.index(device_name)
    ip_address = device_list[device_keys[device_number]]["identity"]["ip"]

    # connect to the cluster and reset
    cluster = Cluster(device_name, ip_address)

    cluster.reset()
    print(f"{device_name} connected at {ip_address}")
    
    return cluster


def connect2module(cluster, slot_nb):
    """
    Connect to a module from a qblox cluster.
    """
    # Find all QRM/QCM modules
    available_slots = {}
    for module in cluster.modules:
        # if module is currently present in stack
        if cluster._get_modules_present(module.slot_idx):
            # check if QxM is RF or baseband
            if module.is_rf_type:
                available_slots[f"module{module.slot_idx}"] = ["QCM-RF", "QRM-RF"][
                    module.is_qrm_type
                ]
            else:
                available_slots[f"module{module.slot_idx}"] = ["QCM", "QRM"][
                    module.is_qrm_type
                ]

    # Connect to the cluster on input slot_nb
    slot_module = getattr(cluster, 'module'+str(slot_nb))

    print(f"{available_slots['module'+str(slot_nb)]} connected")
    print(cluster.get_system_status())
    
    return slot_module


def create_sequence(waveforms={}, weights={}, acqs={}, seq_prog={}):
    """
    Creates dictionary with the sequence information.
    """
    # TODO - not very useful function, combine with another/delete?
    return {"waveforms": waveforms,
            "weights": weights,
            "acquisitions": acqs,
            "program": seq_prog}


def qrm_rf_awg(qrm_rf, lo_freq=3e9, nco_freq=100e6, in0_att=30, out0_att=60):
    """
    Necessary commands to run sequence with QRM-RF (or at least, some).

    Notes
    -----
    Run time: a few 10s of ms
    """
    qrm_rf.out0_in0_lo_freq(lo_freq)  # main carrier frequency essentially
    qrm_rf.sequencer0.mod_en_awg(True)
    qrm_rf.sequencer0.nco_freq(nco_freq)  # additional modulation
    # NB: carrier frequency = lo_freq + nco_freq

    qrm_rf.in0_att(in0_att)  # input attenuation
    qrm_rf.out0_att(out0_att)  # output attenuation


def config_acq(qrm_rf, integ_len=None):
    """
    Configure I/Q acquisition.

    Parameters
    ----------
    qrm_rf: instance of qblox_instruments.qcodes_drivers.qcm_qrm.QcmQrm
        QRM-RF module connector
    integ_len: int, default None
        integration length (ns)

    Acquisition is configured on sequencer 0, averages are enabled. When 
    integration is desired, integ_len should be specified.
    Run time: a few 10s of ms (using integration +50%)
    """
    # enable hardware averaging
    qrm_rf.scope_acq_avg_mode_en_path0(True)
    qrm_rf.scope_acq_avg_mode_en_path1(True)

    # integration configuration (does not affect scope acq.)
    if integ_len is not None:
        qrm_rf.sequencer0.demod_en_acq(True)
        qrm_rf.sequencer0.integration_length_acq(integ_len)

    # configure the sequencer 0 to trigger the scope acquisition
    qrm_rf.scope_acq_sequencer_select(0)
    qrm_rf.scope_acq_trigger_mode_path0("sequencer")


def get_scope_acq_0(qrm_rf, acq_name, timeout=60):
    """
    Get acquisition list from on sequencer 0
    
    Parameters
    ----------
    acq_name: string
    name of the acquisition used for the sequencer    
    """
    # Print status of sequencer.
    print("Acquisition status:")
    print(qrm_rf.get_sequencer_status(0))

    # Wait for the acquisition to finish with a timeout period of one minute.
    qrm_rf.get_acquisition_status(0, timeout=timeout)

    # Move acquisition data from temporary memory to acquisition list.
    qrm_rf.store_scope_acquisition(0, acq_name)

    # Get acquisition list from instrument.
    return qrm_rf.get_acquisitions(0)


def seq2list(waveforms, seq_prog):
    
    # inner function for basic delay extension
    def delay(td, t, m1, m2, m3, m4, acq,
              m1_val, m2_val, m3_val, m4_val, acq_val):

        # time extended and incremented according to delay value
        if t == []:
            t.extend([0])
            t.extend(list(range(t[-1],t[-1]+td-1)))
        else:
            t.extend(list(range(t[-1],t[-1]+td)))

        # markers and acquisition extended by their original values
        m1.extend([m1_val]*td)
        m2.extend([m2_val]*td)
        m3.extend([m3_val]*td)
        m4.extend([m4_val]*td)

        acq.extend([acq_val]*td)
    
    t, wfm1, wfm2 = [], [], []
    m1, m2, m3, m4, acq = [], [], [], [], []
    m1_val, m2_val, m3_val, m4_val, acq_val = 0, 0, 0, 0, 0

    for line in seq_prog.splitlines():

        # delete comments
        if '#' in line:
            line=line[:line.find('#')]

        # replace comma delimiters between numbers by space
        if ',' in line:
            line=line.replace(',', ' ')

        # get numbers present on the line
        nb = [int(s) for s in line.split() if s.isdigit()]

        if 'move' in line:
            # constant - could extract its value and use it later on
            # future TODO, a bit too advanced for now
            pass

        elif q1asm_line_issimplewait(line) and nb[0]>0:
            delay(nb[0], t, m1, m2, m3, m4, acq, m1_val, m2_val, m3_val, m4_val, acq_val)
            # waveforms at 0 unless not fully played
            # .extend([0]*int(s))
            wfm1.extend([0]*nb[0])
            wfm2.extend([0]*nb[0])

        elif 'play' in line:
            delay(nb[2], t, m1, m2, m3, m4, acq, m1_val, m2_val, m3_val, m4_val, acq_val)

            # TODO need to take into account case where the waveform isn't fully played...
            # create boolean fully_played =
            for key in waveforms:
                wfm=waveforms[key]

                if int(wfm["index"]) == nb[0]:
                    wfm1.extend(wfm["data"])

                if wfm["index"] == nb[1]:
                    wfm2.extend(wfm["data"])

        elif 'set_mrk' in line:  # the markers are updated the value from the command
            # extract marker values from binary number (0 and 1)
            m1_val, m2_val, m3_val, m4_val = f'{nb[0]:04b}'
            m1_val, m2_val, m3_val, m4_val = int(m1_val), int(m2_val), int(m3_val), int(m4_val)

        elif 'upd_param' in line:
            delay(nb[0], t, m1, m2, m3, m4, acq, m1_val, m2_val, m3_val, m4_val, acq_val)
            wfm1.extend([0]*nb[0])
            wfm2.extend([0]*nb[0])

        elif 'acquire' in line:
            # acquistion get turned on for the rest of the sequence
            # TODO unless it is > 16ms - ?
            # TODO acquire delay itself - what does it do?
            acq_val = 1
            
            # acquisition command faster (no 146ns delay)
            # if early in the sequence, starts earlier than what has been created so far
            if len(acq) < 146:
                t_ext = 146 - len(acq)
                
                # time extended and incremented according to delay value
                if t == []:  # case where acquire is the first command of sequence program
                    delay(t_ext, t, m1, m2, m3, m4, acq,
                          m1_val, m2_val, m3_val, m4_val, acq_val)
                    wfm1.extend([0]*t_ext)
                    wfm2.extend([0]*t_ext)
                else:
                    t.extend(list(range(t[-1],t[-1]+t_ext)))
                    m1.insert(0, [m1[1]]*t_ext)
                    m2.insert(0, [m2[1]]*t_ext)
                    m3.insert(0, [m3[1]]*t_ext)
                    m4.insert(0, [m4[1]]*t_ext)
                    acq.insert(0, [acq_val]*t_ext)
                    wfm1.insert(0, [0]*t_ext)
                    wfm2.insert(0, [0]*t_ext)
            else:
                # TODO acquisition own delay?
                pass

        elif 'loop' in line:
            # TODO: could add brackets with a loop index
            # not super important and a bit hard
            # need to be careful about things like @loop
            pass

        elif 'stop' in line:
            break
        
    return t, m1, m2, m3, m4, acq, wfm1, wfm2


def plot_seq(waveforms, seq_prog):
    
    t, m1, m2, m3, m4, acq, wfm1, wfm2 = seq2list(waveforms, seq_prog)
    
    fig, axs = plt.subplots(7, 1, figsize=(15, 7.5))

    plt.subplot(7,1,1)
    plt.plot(t, wfm1)
    plt.ylim(-1.1, 1.1)

    plt.subplot(7,1,2)
    plt.plot(t, wfm2)
    plt.ylim(-1.1, 1.1)

    plt.subplot(7,1,3)
    plt.plot(t, m1)
    plt.ylim(-0.1, 1.1)

    plt.subplot(7,1,4)
    plt.plot(t, m2)
    plt.ylim(-0.1, 1.1)

    plt.subplot(7,1,5)
    plt.plot(t, m3)
    plt.ylim(-0.1, 1.1)

    plt.subplot(7,1,6)
    plt.plot(t, m4)
    plt.ylim(-0.1, 1.1)

    plt.subplot(7,1,7)
    plt.plot(t, acq)
    plt.ylim(-0.1, 1.1)
    
    plt.xlabel('time (ns)')
    
    ylabels = ['wfm1', 'wfm2', 'M1', 'M2', 'M3', 'M4', 'acq']

    for i, ax in enumerate(axs):
        # get rid of right and top boundary
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_ylabel(ylabels[i])
        ax.set_xlim(0, t[-1])
        if i < len(axs):
            ax.set_xticks([])
    
    return fig, axs


def q1asm_count_play(prog):
    """
    Count the number of waveforms played in a q1asm program.

    Parameters
    ----------
    prog : str
        Q1ASM or simplified Q1ASM program

    Returns
    -------
    wfm_nb : TYPE
        Number of occurences of the instruction play
    """
    # TODO CPMG loop?
    play_nb = 0
    for line in prog.splitlines():
        
        line, nb = q1asm_get_line_info(line)

        if 'play' in line:
            play_nb += 1

    return play_nb


def q1asm_line_issimplewait(line):
    """
    Check that line is wait instruction with numerical value
    """
    
    line, nb = q1asm_get_line_info(line)

    return 'wait' in line and nb != [] and 'wait_sync' not in line


def q1asm_delay(delay, loop_index=1):
    """
    Creates Q1ASM instructions for delay, including delays >65535ns.
    
    Parameters
    ----------
    delay: int
        duration of delay (ns)
    loop_index: int, default 1
        loop index used for long delay loop address loop_delay{loop_index}
    
    Returns
    -------
    delay_prog: string
        the Q1ASM instructions for the delay
    loop_used: boolen
        indicates if a loop was created in order to create the delay
    
    Notes
    -----
    Q1ASM maximum input for the delay instruction "wait" is 65535, this
    function creares several wait instructions or loops around it.
    """
    delay_prog = ''
    loop_used = False
    
    # delay short enough for single Q1ASM line
    if delay < 65535:
        delay_prog += f'wait      {delay}\n'
    
    # long delay requires several wait (possibly in loop)
    else:
        loop_nb = delay // 65535
        remainder = delay % 65535
        
        if 0 < remainder < 4:  # wait instruction must be more than 4ns
            loop_nb -= 1
            low_remainder = True
        else:
            low_remainder = False
        
        if loop_nb < 5:
            for i in range(loop_nb):
                delay_prog += 'wait      65535\n'
        else:
            delay_prog += f'move      {loop_nb},R1\n'
            delay_prog += f'loop_delay{loop_index}:\n'
            delay_prog += 'wait      65535\n'
            delay_prog += f'loop      R1,@loop_delay{loop_index}\n'
            loop_used = True

        if remainder > 0:
            if low_remainder is True:
                delay_prog += f'wait      {65535-4}\n'
                remainder +=4
            delay_prog += f'wait      {remainder}\n'
    
    delay_prog = delay_prog.rstrip()
    
    return delay_prog, loop_used


def q1asm_check_amp_overtrigger(prog, max_time=5000):
    """
    Returns the longuest duration the amplifier is on.

    Parameters
    ----------
    prog : str
        Q1ASM program
    max_time : int
        maximum time the amplifier should be left on, default 5us

    Returns
    -------
    amp_time : int
        Longuest time for which the amplifier is triggered (ns)
    """
    # TODO loops?
    amp = False
    amp_time = 0
    amp_time_max = 0
    for line in prog.splitlines():

        line, nb = q1asm_get_line_info(line)
        
        if 'set_mrk' in line:
            if nb[0] in [15, 1111]:
                amp = True
            else:
                amp = False
                if amp_time > amp_time_max:    
                    amp_time_max = amp_time
                amp_time=0
        
        if amp is True:

            if q1asm_line_issimplewait(line) or 'upd_param' in line:
                amp_time += nb[0]
            
            elif 'play' in line:
                amp_time += nb[-1]

    assert amp is False, 'amplifier trigger should be stopped'
    assert amp_time_max < max_time, \
        f'amplifier trigger longer than {max_time}ns'

    return amp_time_max


def q1asm_get_line_info(line):
    """
    Remove line comments and extract numerical values.

    Parameters
    ----------
    line : str
        line to be treated

    Returns
    -------
    cleanline : str
        line without comments
    numbers : list of int
        line numbers
    """
    cleanline = line
    if '#' in line:
        cleanline=cleanline[:cleanline.find('#')] # delete comments

    # replace comma delimiters between numbers by space
    cleanline_nospace=cleanline.replace(',', ' ')
    
    # get numbers present on the line
    numbers = [int(s) for s in cleanline_nospace.split() if s.lstrip('-').isdigit()]
    
    return cleanline, numbers


def q1asm_convert_markers(simple_prog):
    """
    Transforms binary commands for marker to single int as required by Q1ASM.
    
    Parameters
    ----------
    prog : str
        Q1ASM or simplified Q1ASM program

    Returns
    -------
    prog : str
        Q1ASM or simplified Q1ASM program

    Notes
    -----
    Makes it easier to think of the markers for user. Example:
    'set_mrk   0100' becomes 'set_mrk   3'
    
    The argument of `set_mrk` is the decimal representation of the 4 bit number
    m4m3m2m1 where mi is the state (either 0 or 1) of the corresponding marker
    channel. The number goes from 0 to 16 (corresponding to 0000 to 1111).
    For QRM-RF module:
    - bit indices 0 & 1 correspond to input 1 and output 1 switches
      respectively,
    - indices 2 & 3 correspond to marker outputs 1 and 2 respectively
      (M1 and M2).
    """
    prog = str()
    for line in simple_prog.splitlines():
        
        # check if binary command for set_mrk
        if line[0:7] == 'set_mrk' and len(line.split()[1]) > 2:
                
                assert len(line.split()[1]) == 4, \
                    'marker binary command should have a length of 4.'

                # int cmd: convert binary to int
                int_cmd = int(line.split()[1], 2)
            
                assert 0 <= int_cmd <= 15, \
                    'binary command not correctly converted to 0 <= int <= 15.'

                prog += f'set_mrk   {int_cmd}\n'

        else:
            prog += (line+'\n')
        
    return prog.rstrip()
            

def q1asm_insert_twt_switch_markers(simple_prog,
                                    switch_open_postdelay=0,
                                    amp_on_postdelay=250,
                                    amp_off_predelay=50,
                                    amp_off_postdelay=250,
                                    switch_closed_postdelay=150):
    """
    Inserts markers to trigger TWT during pulse sequence and open protection
    switch.
    """

    for delay in [switch_open_postdelay, amp_on_postdelay, amp_off_predelay,
                  amp_off_postdelay, switch_closed_postdelay]:
        assert delay == 0 or delay > 4, "optional delays should be 0 or >4"

    # count number of waveforms
    wfm_nb = q1asm_count_play(simple_prog)

    prog = str()

    amp = False  # amplifer is off at the start
    switch_open = False  # switched is closed at the start
    wfm_counter = 0
    delay_reduction = 0

    for line in simple_prog.splitlines():

        line, nb = q1asm_get_line_info(line)

        if q1asm_line_issimplewait(line):
            # turn off amplifier if interpulse delay is long
            # avoid overtriggering TWT amplifier
            if 1 <= wfm_counter < wfm_nb and nb[0] > 1000 and amp == True:
                if amp_off_predelay > 4:
                    prog += f'wait      {amp_off_predelay}\n'
                    delay_reduction += amp_off_predelay
                prog += 'set_mrk   11\n'
                prog += 'upd_param 4\n'

                delay_reduction += switch_open_postdelay + amp_on_postdelay + 4
                amp = False

            if delay_reduction > 0:
                assert nb[0]-delay_reduction > 0, \
                    f'wait delay of {nb[0]}ns is too short'
                nb[0] -= delay_reduction
                delay_reduction = 0

            # write delay instructions
            prog += f'wait      {nb[0]}\n'

        elif 'play' in line:
            wfm_counter += 1

            # open switch before if required
            if switch_open is False and switch_open_postdelay > 0:
                prog += 'set_mrk   7\n'
                prog += f'upd_param {switch_open_postdelay}\n'
                switch_open = True

            # turn on amplifier before if required (switch opened here too)
            if amp is False:
                prog += 'set_mrk   15\n'
                prog += f'upd_param {amp_on_postdelay}\n'
                amp = True
                switch_open = True

            # play the waveform
            prog += (line+'\n')

            # turn off amplifier and close switch if last pulse
            if wfm_counter == wfm_nb:
                assert delay_reduction == 0, \
                    "previous delay reduction not applied"
                if amp_off_predelay > 4:
                    prog += f'wait      {amp_off_predelay}\n'
                    delay_reduction += amp_off_predelay
                if amp_off_predelay > 4 or amp_off_postdelay > 4:
                    prog += 'set_mrk   11\n'  # turn off amplifier
                if amp_off_postdelay > 4:
                    prog += f'upd_param {amp_off_postdelay}\n'
                    delay_reduction += amp_off_postdelay

                # TODO only close switch if acquire instruction?
                prog += 'set_mrk   3\n'  # turn off amplifier and switch

                if switch_closed_postdelay > 4:
                    prog += f'upd_param {switch_closed_postdelay}\n'
                    delay_reduction += switch_closed_postdelay

        else:
            prog += (line+'\n')
            
            # special case with for loop
            # check that enough delay in the loop
            if line.strip() == 'end':
                assert delay_reduction == 0, \
                    'wait time expected before end instruction'
        
    assert delay_reduction == 0, \
        'error with delay for markers (ex: missing delay after last pulse)'

    return prog.rstrip()


def q1asm_write_loops(simple_prog):

    prog = ''
    for_counter = 0
    loop_info = []
    for line in simple_prog.splitlines():

        if 'for' in line[0:3]:
            
            if line[-1] == ':':
                raise SyntaxError(f'Do not use : at the end of that line: {line}')
            # retrieve and save loop information        
            for_counter += 1
            start, step, stop = q1asm_get_line_info(line)[1]
            register = line.split()[1]
            loop_info.append([register, start, step, stop, for_counter])

            # initialise
            if start < 0:
                prog += f'move      0,{register}\n'
                prog += 'nop\n'
                prog += f'sub       {register},{start},{register}\n'
            elif stop < 0:
                raise ValueError ('negative stop not supported yet')
                # TODO think about stop < 0
                # need to store stop value in register
                # will make loop more prone to underflow
                # prog += f'move      0,{register}\n'
                # prog += 'nop\n'
                # prog += f'sub       {register},{stop},{register}\n'
            else:
                prog += f'move      {start},{register}\n'

            # write loop start
            prog += f'loop_for{for_counter}:\n'

        elif line.strip() == 'end':
            # more advanced condition to avoid conflits with labels starting
            # by 'end'

            # get loop info and delete it from information list
            # (avoid conflicts between nested and non-nested loops)
            register, start, step, stop, loop_index = loop_info.pop()

            if start < stop:

                prog += f'add       {register},{step},{register}\n'
                prog += 'nop\n'
                prog += f'jlt       {register},{stop+1},@loop_for{loop_index}\n'

            elif start > stop:

                prog += f'sub       {register},{step},{register}\n'
                prog += 'nop\n'
                prog += f'jge       {register},{stop},@loop_for{loop_index}\n'

        else:
            prog += (line+'\n')
            
    assert loop_info == [], 'each for should be matched with an end'
    
    return prog.rstrip()


def q1asm_get_prog_rec(steps, ctp):
    """
    Creates the Q1ASM intructions to generate the receiver phase adapted to an
    input phase cycling

    Parameters
    ----------
    steps: list
        phase steps of the phase cycling list (length = number of pulses)
    ctp: list
        coherence transfer pathway list (length = number of pulses)

    Returns
    -------
    prog_rec: string
        Q1ASM instructions to generate the receiver phase
    
    Notes
    -----
    This creates instructions which are equivalent to:
    receiver_phase = 0
    for i in range(pulse_number):
        receiver_phase += step[i]*ctp[i]
    
    example: 90  -  180 - 180 - detection
    ctp = [-1, +2, -2]
    +1                 ******
    0 **********      *      *
    -1          ******        *************
    
    steps = [180, 180, 90]
    
    ph1 =    0 0 0 0 0 0 0 0 2 2 2 2 2 2 2 2
    ph2 =    0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2
    ph3 =    0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3
    ph_rec = 0 0 0 0 2 2 2 2 2 2 2 2 0 0 0 0
    
    where 0=90deg, 1=90deg, 2=180deg, 3=270deg
    """
    prog_rec = str()

    # initialize receiver phase to high value to avoid negative numbers
    prog_rec = f'move      {abs(min(ctp))*360*2777777},R40\nnop\n'


    # ph1*step1+ph2*step2+...
    j = 0  # phase cycle step counter
    for step, mult in zip(steps, ctp):

        if step != 0:
            if mult > 0:
                while mult > 0:
                    prog_rec += f'add       R40,R4{j+1},R40\nnop\n'
                    mult -= 1
            elif mult < 0:
                while mult < 0:
                    prog_rec += f'sub       R40,R4{j+1},R40\nnop\n'
                    mult += 1
            j += 1

    # modulo 360 (subtract 360 while >360)
    prog_rec += dedent(f"""
                       loop_mod360:
                       sub       R40,{360*2777777},R40
                       nop
                       jge       R40,{360*2777777},@loop_mod360
                       """).lstrip('\n')

    return prog_rec.rstrip()


def q1asm_compensate_delay(prog, line_idx, delay, position='after'):

    if position == 'after':
        pos = 1
    elif position == 'before':
        pos = -1
    else: raise ValueError('position can only be \'before\' or \'after\'.')
    
    prog_comp = prog.splitlines()

    try:
        i = 1
        uncompensated = True
        while uncompensated:
            
            idx = line_idx + pos*i
            line, nb = q1asm_get_line_info(prog_comp[idx])
            
            if q1asm_line_issimplewait(line):
                if nb[0] > delay+4:
                    prog_comp[idx] = f'wait      {nb[0]-delay}'
                    uncompensated = False
    
            elif 'upd_param' in line:
                if nb[0] > delay+4:
                    prog_comp[idx] = f'upd_param {nb[0]-delay}'
                    uncompensated = False
            i += 1
    except IndexError:

        raise IndexError('list index out of range, most likely because could'
                         f'not find a delay long enough {position} line index'
                         f'{line_idx} for compensation.\n')
    
    return '\n'.join(prog_comp)


def q1asm_insert_phase_cycling(prog, steps, ctp):

    # TODO several receivers?
    # TODO negative steps?
    # TODO phase offset (e.g. y-pulse in x-pulses SIFTER) - add init_phase list? expect user to modify its waveform?
    # TODO use q1asm_phase()
    
    # TODO steps should be called phase_values instead

    # initialise phase
    # create loop
    # repeat for each pulse
    # calculate receiver phase
    # play sequence
    # set_ph pulses
    # set_ph receiver
    # reloop

    # count number of waveforms
    wfm_nb = q1asm_count_play(prog)

    assert wfm_nb == len(steps), 'steps dimensions should match number of pulses'
    assert wfm_nb == len(ctp), 'ctp dimensions should match number of pulses'
    for line in prog.splitlines():
        if 'loop_ph' in line and ':' in line:
            print(line)
            raise ValueError('labels should not contain \'loop_ph\' '
                             'when using phase cycling.')

    step_count = 0
    tot_step_count = np.sum(np.array(steps)!=np.zeros(wfm_nb))
    for step in steps:

        if step != 0:
            j = tot_step_count-step_count

            prog = f'loop_ph{j}:\n' + prog  # create phase loop 
            prog = f'move      0,R4{j}\n' + prog  # initialize phase
    
            prog += dedent(f"""
                           add       R4{j},{step*2777777},R4{j}
                           nop
                           jlt       R4{j},{360*2777777},@loop_ph{j}
                           """).rstrip()
            step_count += 1

    wfm_count = 0
    step_count = 0
    new_prog = prog.splitlines()
    idx2=0
    for idx, line in enumerate(prog.splitlines()):

        line, nb = q1asm_get_line_info(line)

        if 'play' in line:
            if steps[wfm_count] > 0:
                new_prog.insert(idx2, f'set_ph    R4{step_count+1}')
                idx2 += 1
                step_count += 1

            elif steps[wfm_count] == 0:
                new_prog.insert(idx2, 'set_ph    0')
                idx2 += 1

            wfm_count += 1
        
        elif 'acquire' in line:
            # special case for receiver phase if all steps are 0
            if (steps == np.zeros(len(steps))).all():
                new_prog.insert(idx2, 'set_ph    0')
            else:
                new_prog.insert(idx2, 'set_ph    R40')
            idx2 += 1

        if 'acquire' in line or 'play' in line:
            new_prog.insert(idx2, 'upd_param 8')
            idx2 += 1

            new_prog = q1asm_compensate_delay('\n'.join(new_prog), idx2, 8, position='before')
            new_prog = new_prog.splitlines()
        
        idx2 += 1

    # add receiver calculation after last phase cycling loop start
    for idx, line in enumerate(new_prog[::-1]):
        
        # reverse search the last phase cycling loop
        if 'loop_ph' in line and ':' in line:
            prog_rec = q1asm_get_prog_rec(steps, ctp)
            new_prog.insert(len(new_prog)-idx, prog_rec)
            break

    return '\n'.join(new_prog)


def q1asm_transform_long_delay(simple_prog):

    loop_delay_idx = 0  # avoid having the same label twice
    prog = str()
    for line in simple_prog.splitlines():

        if q1asm_line_issimplewait(line):
            
            line, nb = q1asm_get_line_info(line)
                
            # write delay instruction
            wait_instruction, looped = q1asm_delay(nb[0], loop_index=loop_delay_idx+1)
            if looped: loop_delay_idx += 1
            
            prog += wait_instruction + '\n'

        else:
            prog += (line+'\n')
    
    return prog.rstrip()


def q1asm_transform_long_waveforms(simple_prog, waveforms, step=28):
    """
    Transforms long waveforms to shorter ones played in a Q1ASM loop.

    Parameters
    ----------
    simple_prog: str
        Q1ASM or simplified Q1ASM program
    waveforms: dict
        dictionary with waveforms to potentially modify
    step: int, optional
        step of waveform lenth to loop over, default 28

    Returns
    -------
    prog: str
        Q1ASM program with transformed play instructions which are looped over
    wfms: dict
        waveforms dictionary with transformed waveforms

    Notes
    -----
    The default step is 28ns and was tested up to 560ms for a single
    waveform. The minimum step size is 24ns to avoid underruns, which can be
    caused by looping over short instruction here. This is the recommended
    minimum value in the qblox-instruments documentation, but it seemed to 
    start failing when going to loop total length superior to several
    microseconds.
    """
    
    wfms = copy.deepcopy(waveforms)  # avoid conflicts for user

    # find waveforms to be shortened
    wfm_2loop = {}
    wfm_2loop_indexes = {}  # to retrieve waveform name from index later on
    nb_loop = 0
    for wfm_name, wfm_value in waveforms.items():
        if len(wfm_value['data']) > 1000:
            wfm_2loop[wfm_name] = wfm_value
            wfm_2loop_indexes[wfm_value['index']] = [wfm_name, nb_loop]
            assert step < len(wfm_value['data'])/3, \
                f'step value should be < 3*len({wfm_name}[\'data\'])'
                
            assert step >= 24, \
                f'step value should be >= 24ns, high risk of underflow'

    if len(wfm_2loop)>0:

        # reduce waveform size, add complementary waveform if not multiple of step
        nb_compl = 0  # number of complentary waveforms
        for name, wfm in wfm_2loop.items():

            wfm = wfms[name]['data']
            assert wfm.count(wfm[0]) == len(wfm), \
                'waveform to be looped should be constant'
            # count number of loops
            wfm_duration = len(wfm)
            nb_loop = wfm_duration // step  # number of iterations TODO needed for later

            # complementary waveform
            if wfm_duration % step > 0:
                               
                comp_name = name + '_compl'
                comp_idx = 1023 - nb_compl
                comp_duration = wfm_duration % step
                
                # ensuring the complementary waveform is > 4ns
                if comp_duration < 4:
                    comp_duration += step
                    nb_loop -= 1  # adjusting iteration number

                comp_data = wfms[name]['data'][0:comp_duration]

                assert comp_name not in wfms.keys(), dedent("""
                    waveform for waveform {name} needs waveform index 
                    {comp_idx}, free that index to proceed or make the 
                    waveform at multiple of your step value""")

                wfms[comp_name] = {'data': comp_data,
                                        'index': comp_idx}

                nb_compl +=1
            
            wfm_2loop_indexes[wfms[name]['index']][1] = nb_loop
            # reduce size of waveform to step ns to be looped over
            wfms[name]['data'] = wfms[name]['data'][0:step]
    
        # create loop over waveforms
        prog = ''
        loop_index = 1
        for line in simple_prog.splitlines():
            if 'play' in line[0:4]:
                # extract info
                line, nb = q1asm_get_line_info(line)

                if (nb[0] in wfm_2loop_indexes.keys() and
                    nb[1] in wfm_2loop_indexes.keys()):

                    nb_loop = wfm_2loop_indexes[nb[0]][1]
                    
                    line = f'move      {nb_loop},R2\n'
                    line += f'loop_play{loop_index}:\n'
                    line += f'play      {nb[0]},{nb[1]},{step}\n'
                    line += f'loop      R2,@loop_play{loop_index}'
                    
                    loop_index += 1

                    wfm_name1 = wfm_2loop_indexes[nb[0]][0]
                    wfm_name2 = wfm_2loop_indexes[nb[1]][0]

                    if wfm_name1+'_compl' in wfms.keys():

                        idx1 = wfms[wfm_name1+'_compl']['index']
                        idx2 = wfms[wfm_name2+'_compl']['index']
                        wait_time = len(wfms[wfm_name1+'_compl']['data'])

                        line += f'\nplay      {idx1},{idx2},{wait_time}'
                        
                elif (nb[0] in wfm_2loop_indexes.keys() or
                      nb[1] in wfm_2loop_indexes.keys()):
                    # only I or Q is set up to be compressed
                    raise ValueError(f'waveforms of index {nb[0]} and {nb[1]} '
                                    'should be of the same length for use of '
                                    'q1asm_transform_long_waveforms()')

            prog += (line+'\n')

        prog = prog.rstrip()

    else: prog = simple_prog

    return prog, wfms


def q1asm_transform_long_chirps(simple_prog, waveforms, gain=0.3):
    """
    Transform play_lg_chirp commands into the appropriate Q1ASM instructions.

    Parameters
    ----------
    simple_prog: str
        Q1ASM or simplified Q1ASM program
    waveforms: dict
        dictionary with waveforms of the sequence
    gain: float
        waveforms gain (>30% leads to stronger LO leakage and spurious freq.)

    Returns
    -------
    prog: str
        Q1ASM program with transformed play instructions which are looped over
    waveforms: dict
        input waveforms dictionary with added waveforms for chirp smoothing

    Notes
    -----
    To be used on chirp waveforms which are typically too long to be
    conveniently uploaded to qblox (i.e. >1us).

    Example of use for 12us chirp:
        f'play_lg_chirp({chirp_param}),12008'

    Other chirp parameters are transferred in the form of a dictionary, e.g.:
        chirp_param = {'bw': int(20e6), 'sm': 10, 'delta_f': nco_f}
    Mandatory chirp parameters:
        - bw, the chirp bandwidth (Hz)
        - sm, the smoothing percentage (%) which will be applied as a linear
        ramp on each side of the chirp
        - delta_f, the frequency position of the chirp, typically the NCO
        frequency for a centred chirp
    IMPORTANT: once the chirp is finished, the NCO stays fixed on delta_f. Do
    not forget to adjust the NCO back for non-centred chirps!
    Optional chirp parameters:
        - step, time step for the NCO sweep (in ns)

    The maximum bandwidth reachable is limited by the smallest time step
    available before underflow of the qblox processor. It did not fail down to
    50ns (20MHz chirp bandwidth at Nyquist condition) for chirp durations of up
    to 50us. Longer duration will require larger steps.

    Reverse sweep can be indicated with a negative bandwidth.

    The smoothing is a simple linear ramp. Compared to regular chirp smoothing
    (sinsmoothed, WURST, superGaussian...), it has better broadband capabilities
    but reduced B1-insensitivity (very small difference). The ramp is created
    with a gain offset loop, completed by a small ramp waveform which is added
    to the waveforms list (waveform index starts at 301). It has a duration 
    equal to the chirp time step.
    """
    gain = int(gain*32767)  # adjust gain to value used by Q1ASM
    prog = ''
    chirp_index = 1
    for line in simple_prog.splitlines():

        if 'play_lg_chirp' in line[0:13]:

            line, nb = q1asm_get_line_info(line)
            # get dictionary with chirp parameters
            chirp_param = ast.literal_eval(line[line.find('{'):line.find('}')+1])
            # extract parameters from dictionary
            try:
                keys =  ['bw', 'sm', 'delta_f']
                bw, sm, delta_f = [chirp_param[key] for key in keys]           
            except KeyError:
                print(f'chirp mandatory parameters must be in {keys}: '
                      f'\'{key}\' missing')
            if delta_f < 0:
                raise ValueError('negative centre frequency (delta_f<0) not '
                                 'supported')
                
            # time step default value is 50ns
            if 'step' not in chirp_param.keys():
               step = 50
            else: step = chirp_param['step']

            duration = q1asm_get_line_info(line)[1][-1] - 8

            if duration % step != 0:
                raise ValueError('duration must be a multiple of step with an '
                                 'added small delay at the end of the chirp of '
                                 '8ns.')
            if duration*sm % step != 0:
                raise ValueError('sm*(duration-8) must be a multiple of step.')

            ns = int(duration/step)  # number of sample/points of the chirp
        
            # NCO frequency is adjusted with 4e9 steps between -500 and 500 MHz
            # and expressed as an integer between -2e9 and 2e9 (e.g.1 MHz=4e6)
            bw_q1asm = int(bw*4)
            chirp_start = int((delta_f - bw/2)*4)
            chirp_end = int((delta_f + bw/2)*4)

            end_ramp_up = int(chirp_start + bw_q1asm*sm/100)
            end_cst = int(chirp_end - bw_q1asm*sm/100)
            end_ramp_down = chirp_end

            if bw > 0:
                sweep_cond, sweep_cmd = 'jlt', 'add'
                end_ramp_up += 1
                end_cst += 1
                end_ramp_down += 1
            else:  # reverse sweep case
                sweep_cond, sweep_cmd = 'jge', 'sub'

            # -1 for the way the sweep is created in Q1ASM loop
            freq_step = int(4*np.abs(bw)/(ns-1))

            # add ramp waveform to waveforms list
            ramp = [np.round((i/step)/(ns*sm/100), 8) for i in range(0, step)]
            waveforms[f'chirpsm{chirp_index}'] = {
                                        "data": ramp, "index": chirp_index+300
            }

            # The chirp is achieved thanks to 3 loops:
            # - ramping up: triangular waveform is played on top of ascending
            #   "staircase" (AWG gain offset increased) as the NCO sweep starts
            # - sweep: amplitude stays constant as NCO sweep goes on
            # - ramping down: triangular waveform is played backward (thanks to 
            #   opposite gain) on a descending staircase
            line = dedent(f"""
            set_awg_gain {gain}, {gain}
            move      0,R9
            move      {chirp_start},R8
            ramp_up{chirp_index}:
            add       R9,{int(gain/(ns*sm/100))},R9
            set_freq  R8
            play      {chirp_index+300},{chirp_index+300},{step}
            {sweep_cmd}       R8,{freq_step},R8
            set_awg_offs R9,R9
            {sweep_cond}       R8,{end_ramp_up},@ramp_up{chirp_index}
            sweep{chirp_index}:
            set_freq  R8
            {sweep_cmd}       R8,{freq_step},R8
            upd_param {step}
            {sweep_cond}       R8,{end_cst},@sweep{chirp_index}
            set_awg_gain {-gain}, {-gain}
            ramp_down{chirp_index}:
            sub       R9,{int(gain/(ns*sm/100))},R9
            set_freq  R8
            play      {chirp_index+300},{chirp_index+300},{step}
            {sweep_cmd}       R8,{freq_step},R8
            set_awg_offs R9,R9
            {sweep_cond}       R8,{end_ramp_down},@ramp_down{chirp_index}
            set_freq  {int(delta_f*4)}
            set_awg_gain {gain}, {gain}
            set_awg_offs 0,0
            upd_param 8""").lstrip('\n')  # lstrip gets rid of leading character

            chirp_index += 1

        prog += (line+'\n')

    return prog.rstrip(), waveforms


def q1asm_add_avg_loop(prog, avg_nb, loop_register='R0', loop_label='loop_avg'):

    # TODO see where reset_ph goes (depending on phase cycling too)
    
    # averaging loop start (lstrip gets rid of leading character)
    prog = dedent(f"""
                  move      {avg_nb},{loop_register}
                  {loop_label}:
                  """).lstrip('\n') + prog

    # averaging loop end
    prog += f'\nloop      {loop_register},@{loop_label}'

    return prog


def q1asm_get_prog_dummy(prog, dummy_nb):
    
    # TODO dummy register can be R0
    
    # replace acquisition instruction by their waiting time
    prog_dummy = str()
    for line in prog.splitlines():
        if 'acquire' in line:
            line, nb = q1asm_get_line_info(line)
            prog_dummy += f'wait      {nb[-1]}\n'
        else:
            prog_dummy += (line+'\n')
            
    prog_dummy = prog_dummy.rstrip()

    # loop if necessary
    if dummy_nb > 1:
        prog_dummy = q1asm_add_avg_loop(prog_dummy, dummy_nb,
                                        loop_register='R63',
                                        loop_label='loop_dummy')

    return prog_dummy


def get_grid_duration(prog):
    
    grid_duration = 0
    for line in prog.splitlines():
        line, nb = q1asm_get_line_info(line)
        if q1asm_line_issimplewait(line) or 'upd_param' in line:
            grid_duration += nb[0]
        if 'play' in line or 'acquire' in line:
            grid_duration += nb[-1]

    return grid_duration


def simple2real_Q1ASM(simple_prog, avg_nb=1, shot_nb=1, dummy_nb=0,
                      steps=None, ctp=None,
                      twt_triggers=True,
                      switch_open_postdelay=0, amp_on_postdelay=250,
                      amp_off_predelay=50, amp_off_postdelay=250,
                      switch_closed_postdelay=150,
                      waveforms=None):

    # order of execution:
    # - add TWT and protection swtich markers
    # - put on 4ns grid (if several scans required)
    # - create dummy shot program
    # - add phase cycling
    # - add shot loop
    # - add average loop
    # - add dummy shots
    # - rewrite constant delays > 65535us into several delays or a loop
    # - add stop
    # ?. sweeps?

    # TODO sweeps
    # TODO communicate on added delays before sequence
    simple_prog = simple_convert_q1asm(simple_prog)

    # TODO convert this into function +pick assigned markers
    for line in simple_prog.splitlines():

        if line[0:4] == 'move':
            line_space=line.replace(',', ' ')
            if line_space.split()[2] in ['R0', 'R1', 'R2']:
                # R0 averages
                # R1 long delays
                # R2 long waveforms
                # TODO R3 for loops negative last point (stop < 0)
                # TODO might be possible to use R1 for R2
                # TODO R40+ for phase cycling
                # TODO R63 dummy shots?
                raise NameError(line_space.split()[2] + ' is reserved.')

    # convert binary marker commands to Q1ASM int marker commands
    if 'set_mrk' in simple_prog:
        prog_mrk = q1asm_convert_markers(simple_prog)
    else: # other marker uses? - not for now
        prog_mrk = simple_prog

    # TWT and switch markers
    if q1asm_count_play(simple_prog) >= 1 and twt_triggers==True:
        # TODO write arguments as tuple?
        prog_mrk = q1asm_insert_twt_switch_markers(prog_mrk,
                            switch_open_postdelay=switch_open_postdelay,
                            amp_on_postdelay=amp_on_postdelay,
                            amp_off_predelay=amp_off_predelay,
                            amp_off_postdelay=amp_off_postdelay,
                            switch_closed_postdelay=switch_closed_postdelay)

    # taking care of NCO grid if necessary
    if avg_nb > 1 or shot_nb > 1 or dummy_nb > 0 or (steps is not None and ctp is not None):
        
        # additional delay to make sure the NCO grid on 4ns grid
        # and on a 10ns grid for scope acquisition (multiple of 20)
        # additional % 20 to avoid adding extra delay if remained is 0
        delay_grid = (20 - get_grid_duration(prog_mrk) % 20) % 20
        # make sure upd_param has a delay >8ns
        if delay_grid < 8: delay_grid += 20

        # phase reset to synchronize NCO between scans
        # upd_param implement the change (important if NCO phase change after)
        prog_mrk = f'upd_param {delay_grid}\n' + prog_mrk
        prog_mrk = 'reset_ph\n' + prog_mrk

    # dummy shots
    if dummy_nb > 0:
        prog_dummy = q1asm_get_prog_dummy(prog_mrk, dummy_nb)

    prog = prog_mrk

    # TODO should I loop shots around phase cyling or on each step?
    # phase cycling
    if steps is not None and ctp is not None:
        prog = q1asm_insert_phase_cycling(prog, steps, ctp)

    # shots
    if shot_nb > 1:
        prog = q1asm_add_avg_loop(prog, shot_nb, loop_register='R51',
                                  loop_label='loop_shot')

    # averages
    if avg_nb > 1:
        prog = q1asm_add_avg_loop(prog, avg_nb)

    # dummy shots are added
    if dummy_nb > 0:
        prog = prog_dummy + '\n' + prog
    
    # rewrite "for" loops into Q1ASM loops
    prog = q1asm_write_loops(prog)

    # delays - important to execute this after functions which creates delay
    # reduction (marker insertion, NCO phase changes from phase cycling...)
    # and on the final program (i.e., after addition of dummy shots)
    prog = q1asm_transform_long_delay(prog)

    # check if TWT is not getting overtriggered
    if twt_triggers==True:
        q1asm_check_amp_overtrigger(prog, max_time=5000)

    if waveforms is not None:
        prog, wfms = q1asm_transform_long_waveforms(prog, waveforms, step=32)
        prog, wfms = q1asm_transform_long_chirps(prog, waveforms, gain=0.3)
 
    # add stop instruction
    prog += '\nstop'
    
    if waveforms is not None:
        return prog, wfms
    else:
        return prog


def sequence_converter(sequence, avg_nb=1, shot_nb=1, dummy_nb=0,
                       steps=None, ctp=None,
                       twt_triggers=True,
                       switch_open_postdelay=0, amp_on_postdelay=250,
                       amp_off_predelay=50, amp_off_postdelay=250,
                       switch_closed_postdelay=150,
                       waveforms=None):
    """
    Convert a simple sequence to one which is compatible with qblox.
    
    Notes
    -----
    The bulk of the conversion happens on the sequence program to generate Q1ASM
    instruction.
    
    WORK IN PROGRESS
    """
    
    # apply basic operations
    # unit conversion with q1asm_ph(), q1asm_gain(), and q1asm_freq()
    # operators: =, +, -, *
    # function with integrated unit conversion:
    # - set_ph(), set_ph_delta()
    # - set_awg_gain(), set_awg_offs()
    # - set_freq()
    sequence['program'] = simple_convert_q1asm(sequence['program'])
    
    # TODO: advanced converter (essentially what used to be simple2real_Q1ASM())

    sequence = advanced_convert_q1asm(sequence)

    return sequence