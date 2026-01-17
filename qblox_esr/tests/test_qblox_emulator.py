from qblox_emulator import Q1ASMEmulator
import numpy as np
import scipy.signal
import pytest


def test_markers_on_off():
    
    waveform_length = 22
    Marker_waveforms = {
        "gaussian": {
            "data": scipy.signal.windows.gaussian(waveform_length, std=0.12 * waveform_length).tolist(),
            "index": 0,
        },
        "block": {"data": [1.0 for i in range(0, waveform_length)], "index": 1},
    }



    Marker_program = """
        move      1,R0        # Start at marker output channel 0 (move 1 into R0)
        nop                   # Wait a cycle for R0 to be available.

    loop: set_mrk   R0          # Set marker output channels to R0
        upd_param 1000        # Update marker output channels and wait 1μs.
        asl       R0,1,R0     # Move to next marker output channel (left-shift R0).
        nop                   # Wait a cycle for R0 to be available.
        jlt       R0,16,@loop # Loop until all 4 marker output channels have been set once.

        set_mrk   0           # Reset marker output channels.
        upd_param 400           # Update marker output channels.
        stop                  # Stop sequencer.
    """
    acquisitions = {
        "single": {"num_bins": 301, "index": 0},
        "binzer": {"num_bins": 301, "index": 1},
    }

    marker_sequence = {
        "waveforms": {},
        "weights": {},
        "acquisitions": acquisitions,
        "program": Marker_program,
    }


    Marker_cycle = Q1ASMEmulator(marker_sequence)
    assert Marker_cycle.markers == [(0, 0), (0, 1), (1000, 2), (2000, 4), (3000, 8), (4000, 0), (4400, 0)] 
    

def test_registeration():
    with pytest.raises(Exception):
        Register_prog = """
        move      100,R0   
        move      20,R1    
        move      5, R65   # should break 
        """
        register_sequence = {
        "waveforms": {},
        "weights": {},
        "acquisitions": {},
        " program": Register_prog,
        } 
        register_assignement_test = Q1ASMEmulator(register_sequence)

def test_basic_sequencing():
    waveform_length = 22 
    pulses = {
        "gaussian": {
            "data": scipy.signal.windows.gaussian(waveform_length, std=0.12 * waveform_length).tolist(),
            "index": 0,
        },
        "block": {"data": [1.0 for i in range(0, waveform_length)], "index": 1},
    }

    pulse_prog = """
            move      100,R0   #Loop iterator.
            move      20,R1    #Initial wait period in ns.
            wait_sync 4        #Wait for sequencers to synchronize and then wait another 4 ns.

    loop:   set_mrk   15       #Set all marker outputs to high. For RF-modules, this also enables the output switches.
            play      0,1,4    #Play a gaussian and a block on output path 0 and 1 respectively and wait 4 ns.
            set_mrk   3        #Reset marker outputs.
            upd_param 18       #Update parameters and wait the remaining 18 ns of the waveforms.

            wait      R1       #Wait period.

            play      1,0,22   #Play a block and a gaussian on output path 0 and 1 respectively and wait 22 ns.
            wait      10     #Wait a 1us in between iterations.
            add       R1,20,R1 #Increase wait period by 20 ns.
            loop      R0,@loop #Subtract one from loop iterator.

            stop               #Stop the sequence after the last iteration.
    """
    pulse_sequence = {
        "waveforms": pulses,
        "weights": {},
        "acquisitions": {},
        "program": pulse_prog,
    }

    sequencing_emulator = Q1ASMEmulator(pulse_sequence)
    correct_data = np.load('basic_sequencing.npy')
    assert np.allclose(sequencing_emulator.I, correct_data)


def test_awg_control():
    waveform_length = 20
    awg_checker_waveforms = {
        "block_1":{"data": [0.5 for i in range(0, waveform_length)], "index": 0},
        "block 2":{"data": [0.2 for i in range(0, waveform_length)], "index": 1},
    }



    awg_checker_prog = """
            move      4 ,R0   #Loop iterator.
            move      5 ,R1

    loop:   
            set_awg_gain  R0, R1   
            play      0,1,4    #Play a gaussian and a block on output path 0 and 1 respectively and wait 4 ns.           
            add       R1,1,R1 #Increase wait period by 20 ns.
            loop      R0,@loop #Subtract one from loop iterator.

            stop               #Stop the sequence after the last iteration.
    """


    awg_checker_sequence = {
        "waveforms": awg_checker_waveforms,
        "weights": {},
        "acquisitions": {},
        "program": awg_checker_prog,
    }


    awg_checker = Q1ASMEmulator(awg_checker_sequence)
    assert np.allclose(awg_checker.I[::4], np.array([0.7924465962305568, 0.7062687723113772, 0.6294627058970836, 0.5610092271509817]))
    assert np.allclose(awg_checker.Q[::4], np.array([0.3556558820077846, 0.39905246299377595, 0.4477442277136679, 0.502377286301916]))
    