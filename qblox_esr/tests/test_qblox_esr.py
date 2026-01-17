import pytest
from textwrap import dedent
import numpy as np
import qblox_esr as qbl


wfm_len = 21


def test_q1asm_line_issimplewait():

    assert qbl.q1asm_line_issimplewait('wait      4') == True
    assert qbl.q1asm_line_issimplewait('wait_sync 40') == False
    assert qbl.q1asm_line_issimplewait('wait      R3') == False


def test_q1asm_delay():
    
    q1asm_prog =  'wait      517'
    assert qbl.q1asm_delay(517) == (q1asm_prog, False)
    
    # dedent removes function indent from string
    q1asm_prog = dedent("""
    wait      65535
    wait      65535
    wait      65535
    wait      65535
    wait      65531
    wait      6""").lstrip('\n')  # lstrip gets rid of leading character
    assert qbl.q1asm_delay(65535*5+2) == (q1asm_prog, False)

    q1asm_prog = dedent("""
    move      16,R1
    loop_delay1:
    wait      65535
    loop      R1,@loop_delay1
    wait      65531
    wait      5""").lstrip('\n')
    assert qbl.q1asm_delay(65535*17+1) == (q1asm_prog, True)


def test_q1asm_check_amp_overtrigger():

    prog = dedent("""move      64,R0
    loop_avg:
    reset_ph
    upd_param 8
    acquire   0,0,4
    set_mrk   15
    upd_param 250
    play      4,2,8
    wait      500
    play      5,3,16
    wait      50
    set_mrk   11
    upd_param 250
    set_mrk   3
    upd_param 150
    wait      50
    move      15,R1
    loop_delay1:
    wait      65535
    loop      R1,@loop_delay1
    loop      R0,@loop_avg
    stop""").lstrip('\n')
    
    total_time = 250+500+50+8+16

    assert qbl.q1asm_check_amp_overtrigger(prog, max_time=5000) == total_time

    # when there is a long delay between pulse, only the longuest trigger
    # is calculated
    prog = dedent("""set_mrk   15
    upd_param 242
    set_ph    0
    upd_param 8
    play      4,2,350
    wait      50
    set_mrk   11
    upd_param 4
    wait      65535
    wait      65535
    wait      18626
    set_mrk   15
    upd_param 242
    set_ph    0
    upd_param 8
    play      5,3,700
    wait      50
    set_mrk   11
    upd_param 250
    set_mrk   3
    upd_param 150
    wait      65535
    wait      65535
    wait      10972
    set_ph    0
    upd_param 8
    acquire   0,0,4
    stop""").lstrip('\n')

    assert qbl.q1asm_check_amp_overtrigger(prog, max_time=5000) == 242+8+700+50
    
    # a way to overtrigger the TWT is to play multiple waveforms with short
    # interpulse delays
    prog = dedent("""
    set_mrk   15
    upd_param 250
    play      4,2,8
    wait      999
    play      4,2,8
    wait      999
    play      4,2,8
    wait      999
    play      4,2,8
    wait      999
    play      4,2,8
    wait      999
    stop""").lstrip('\n')
    
    with pytest.raises(AssertionError):
        qbl.q1asm_check_amp_overtrigger(prog, max_time=5000)


def test_q1asm_get_line_info():
    
    assert qbl.q1asm_get_line_info('play 0,1,17')[1] == [0,1,17]
    assert qbl.q1asm_get_line_info('set_freq  -4')[1] == [-4]


def test_q1asm_convert_markers():
    
    # check that binary commands are written correctly without destroying
    # regular int commands
    q1asm_prog = dedent("""
    set_mrk   0
    set_mrk   0
    set_mrk   1
    set_mrk   1
    set_mrk   2
    set_mrk   2
    set_mrk   3
    set_mrk   3
    set_mrk   4
    set_mrk   4
    set_mrk   5
    set_mrk   5
    set_mrk   6
    set_mrk   6
    set_mrk   7
    set_mrk   7
    set_mrk   8
    set_mrk   8
    set_mrk   9
    set_mrk   9
    set_mrk   10
    set_mrk   10
    set_mrk   11
    set_mrk   11
    set_mrk   12
    set_mrk   12
    set_mrk   13
    set_mrk   13
    set_mrk   14
    set_mrk   14
    set_mrk   15
    set_mrk   15""").lstrip('\n')  # lstrip gets rid of leading character
   
    simple_prog = dedent("""
    set_mrk   0
    set_mrk   0000
    set_mrk   1
    set_mrk   0001
    set_mrk   2
    set_mrk   0010
    set_mrk   3
    set_mrk   0011
    set_mrk   4
    set_mrk   0100
    set_mrk   5
    set_mrk   0101
    set_mrk   6
    set_mrk   0110
    set_mrk   7
    set_mrk   0111
    set_mrk   8
    set_mrk   1000
    set_mrk   9
    set_mrk   1001
    set_mrk   10
    set_mrk   1010
    set_mrk   11
    set_mrk   1011
    set_mrk   12
    set_mrk   1100
    set_mrk   13
    set_mrk   1101
    set_mrk   14
    set_mrk   1110
    set_mrk   15
    set_mrk   1111""").lstrip('\n')  # lstrip gets rid of leading character

    assert qbl.q1asm_convert_markers(simple_prog) == q1asm_prog


def test_q1asm_write_loops():

    start, step, stop = 100, 50, 500
    
    # simple loop
    q1asm_prog = dedent(f"""
    wait      500
    move      {start},R3
    loop_for1:
    wait      R3
    add       R3,{step},R3
    nop
    jlt       R3,{stop+1},@loop_for1
    acquire   0,0,4
    stop""").lstrip('\n')
    
    simple_prog = dedent(f"""
    wait      500
    for R3 in {start}, {step}, {stop}
    wait      R3
    end
    acquire   0,0,4
    stop""").lstrip('\n')
    
    assert q1asm_prog == qbl.q1asm_write_loops(simple_prog)
    
    # nested loops
    q1asm_prog = dedent(f"""
    wait      500
    move      {start},R3
    loop_for1:
    move      {start+50},R4
    loop_for2:
    move      {start},R5
    loop_for3:
    wait      R3
    wait      R4
    wait      R5
    add       R5,{step+50},R5
    nop
    jlt       R5,{stop+100+1},@loop_for3
    add       R4,{step-25},R4
    nop
    jlt       R4,{stop+1},@loop_for2
    add       R3,{step},R3
    nop
    jlt       R3,{stop+1},@loop_for1
    acquire   0,0,4
    stop""").lstrip('\n')
    
    simple_prog = dedent(f"""
    wait      500
    for  R3 in {start}, {step}, {stop}
    for   R4 in {start+50}, {step-25}, {stop}
    for    R5 in {start}, {step+50}, {stop+100}
    wait      R3
    wait      R4
    wait      R5
    end
    end
    end
    acquire   0,0,4
    stop""").lstrip('\n')
    
    assert q1asm_prog == qbl.q1asm_write_loops(simple_prog)
    
    # loops following each other and negative step
    q1asm_prog = dedent(f"""
    wait      500
    move      {start},R3
    loop_for1:
    wait      R3
    add       R3,{step},R3
    nop
    jlt       R3,{stop+1},@loop_for1
    wait      50
    move      {stop},R4
    loop_for2:
    wait      R4
    sub       R4,{step},R4
    nop
    jge       R4,{start},@loop_for2
    acquire   0,0,4
    stop""").lstrip('\n')
    
    simple_prog = dedent(f"""
    wait      500
    for R3 in {start}, {step}, {stop}
    wait      R3
    end
    wait      50
    for R4 in {stop}, {step}, {start}
    wait      R4
    end
    acquire   0,0,4
    stop""").lstrip('\n')
   
    assert q1asm_prog == qbl.q1asm_write_loops(simple_prog)
    
    # negative start
    start, step, stop = -100e6, 50e6, 500e6
    
    # simple loop
    q1asm_prog = dedent(f"""
    move     0,R3
    nop
    sub      R3,{start},R3
    loop_for1:
    set_freq  R3
    add       R3,{step},R3
    nop
    jlt       R3,{stop+1},@loop_for1
    """).lstrip('\n')
    
    simple_prog = dedent(f"""
    for R3 in {start}, {step}, {stop}
    set_freq  R3
    end
    """).lstrip('\n')


def test_q1asm_compensate_delay():
    
    prog1 = dedent("""
    wait      500
    play      6,7,23
    upd_param 300
    stop""").lstrip('\n')
    
    prog2 = dedent(f"""
    wait      500
    play      6,7,23
    upd_param {300-217}
    stop""").lstrip('\n')
    
    assert prog2 == qbl.q1asm_compensate_delay(prog1, 1, 217)

    prog2 = dedent(f"""
    wait      {500-15}
    play      6,7,23
    upd_param 300
    stop""").lstrip('\n')
    
    assert prog2 == qbl.q1asm_compensate_delay(prog1, 1, 15, position='before')

    with pytest.raises(IndexError):
        qbl.q1asm_compensate_delay(prog1, 1, 297)
    with pytest.raises(IndexError):
        qbl.q1asm_compensate_delay(prog1, 1, 550, position='before')


def test_q1asm_transform_long_waveforms():

    waveforms = {
        "block1": {"data": [0.0 for i in range(0, 28000)], "index": 1},
        "block2": {"data": [1.0 for i in range(0, 28000)], "index": 2},
        "block3": {"data": [0.0 for i in range(0, 9735)], "index": 3},
        "block4": {"data": [1.0 for i in range(0, 9735)], "index": 4},
    }

    original_waveforms = {
        "block1": {"data": [0.0 for i in range(0, 28)], "index": 1},
        "block2": {"data": [1.0 for i in range(0, 28)], "index": 2},
        "block3": {"data": [0.0 for i in range(0, 28)], "index": 3},
        "block4": {"data": [1.0 for i in range(0, 28)], "index": 4},
        "block3_compl": {"data": [0.0 for i in range(0, 19)], "index": 1023},
        "block4_compl": {"data": [1.0 for i in range(0, 19)], "index": 1023-1},
    }

    # if the waveforms are not concerned, does not change the program
    assert 'play 6,7,40' == \
        qbl.q1asm_transform_long_waveforms(
            'play 6,7,40', original_waveforms)[0]
    
    q1asm_prog = dedent("""
    move      1000,R2
    loop_play1:
    play      1,2,28
    loop      R2,@loop_play1
    move      347,R2
    loop_play2:
    play      3,4,28
    loop      R2,@loop_play2
    play      1023,1022,19""").lstrip('\n')
    
    simple_prog = dedent("""
    play      1,2,28000
    play      3,4,9735""").lstrip('\n')
   
    with pytest.raises(AssertionError):
         qbl.q1asm_transform_long_waveforms(
             simple_prog, waveforms, step=8001)

    prog, wfms = qbl.q1asm_transform_long_waveforms(simple_prog, waveforms)

    assert prog == q1asm_prog and wfms == original_waveforms
    
    # check that it does not affect other instructions
    q1asm_prog = dedent("""
    acquire   0,0,4
    move      1000,R2
    loop_play1:
    play      1,2,28
    loop      R2,@loop_play1
    wait      1005
    upd_param 50
    move      347,R2
    loop_play2:
    play      3,4,28
    loop      R2,@loop_play2
    play      1023,1022,19
    set_mrk   3""").lstrip('\n')
    
    simple_prog = dedent("""
    acquire   0,0,4
    play      1,2,28000
    wait      1005
    upd_param 50
    play      3,4,9735
    set_mrk   3""").lstrip('\n')

    prog, wfms = qbl.q1asm_transform_long_waveforms(simple_prog, waveforms)

    assert prog == q1asm_prog and wfms == original_waveforms
    
    # check complementary waveform not a multiple of 4ns
    waveforms = {
        "block1": {"data": [0.0 for i in range(0, 28003)], "index": 1},
    }

    original_waveforms = {
        "block1": {"data": [0.0 for i in range(0, 28)], "index": 1},
        "block1_compl": {"data": [0.0 for i in range(0, 31)], "index": 1023},
    }
    
    q1asm_prog = dedent("""
    move      999,R2
    loop_play1:
    play      1,1,28
    loop      R2,@loop_play1
    play      1023,1023,31""").lstrip('\n')

    simple_prog = dedent("""
    play      1,1,28003""").lstrip('\n')

    prog, wfms = qbl.q1asm_transform_long_waveforms(simple_prog, waveforms)

    assert prog == q1asm_prog and wfms == original_waveforms
    
    # check complementary waveform not a multiple of 4ns
    # check that does not interfere with other waveforms
    waveforms = {
        "block1": {"data": [0.0 for i in range(0, 28001)], "index": 0},
        "block2": {"data": [1.0 for i in range(0, 28001)], "index": 2},
        "block02": {"data": [0.0 for i in range(0, 17)], "index": 6},
        "block12": {"data": [1.0 for i in range(0, 17)], "index": 7},
    }

    original_waveforms = {
        "block1": {"data": [0.0 for i in range(0, 28)], "index": 0},
        "block2": {"data": [1.0 for i in range(0, 28)], "index": 2},
        "block1_compl": {"data": [0.0 for i in range(0, 29)], "index": 1023},
        "block2_compl": {"data": [1.0 for i in range(0, 29)], "index": 1023-1},
        "block02": {"data": [0.0 for i in range(0, 17)], "index": 6},
        "block12": {"data": [1.0 for i in range(0, 17)], "index": 7},
    }
    
    q1asm_prog = dedent("""
    move      999,R2
    loop_play1:
    play      0,2,28
    loop      R2,@loop_play1
    play      1023,1022,29""").lstrip('\n')

    simple_prog = dedent("""
    play      0,2,28001""").lstrip('\n')

    prog, wfms = qbl.q1asm_transform_long_waveforms(simple_prog, waveforms)

    assert prog == q1asm_prog and wfms == original_waveforms


def test_q1asm_transform_long_chirps():
    
    chirp_param = {'bw': int(-20e6), 'sm': 10, 'delta_f': int(100e6)}

    original_waveforms = {}
    
    sm = chirp_param['sm']
    step = 50
    ns = int(12000/step)
    chirp_index = 1
    ramp = [np.round((i/step)/(ns*sm/100), 8) for i in range(0, step)]
    original_waveforms = {f'chirpsm{chirp_index}':
                            {"data": ramp, "index": chirp_index+300}
    }

    q1asm_prog = dedent("""
    set_mrk   3
    upd_param 146
    acquire   0,0,4
    wait      500
    set_awg_gain 9830, 9830
    move      0,R9
    move      440000000,R8
    ramp_up1:
    add       R9,409,R9
    set_freq  R8
    play      301,301,50
    sub       R8,334728,R8
    set_awg_offs R9,R9
    jge       R8,432000000,@ramp_up1
    sweep1:
    set_freq  R8
    sub       R8,334728,R8
    upd_param 50
    jge       R8,368000000,@sweep1
    set_awg_gain -9830, -9830
    ramp_down1:
    sub       R9,409,R9
    set_freq  R8
    play      301,301,50
    sub       R8,334728,R8
    set_awg_offs R9,R9
    jge       R8,360000000,@ramp_down1
    set_freq  400000000
    set_awg_gain 9830, 9830
    set_awg_offs 0,0
    upd_param 8
    wait      16000""").lstrip('\n')

    simple_prog = dedent(f"""
    set_mrk   3
    upd_param 146
    acquire   0,0,4
    wait      500
    play_lg_chirp({chirp_param}),12008
    wait      16000""").lstrip('\n')

    prog, wfms = qbl.q1asm_transform_long_chirps(simple_prog, {})

    assert prog == q1asm_prog and wfms == original_waveforms


def test_simple2real_Q1ASM():
    
    q1asm_prog = dedent("""
    set_mrk   15
    upd_param 250
    play      5,3,40
    wait      50
    set_mrk   11
    upd_param 250
    set_mrk   3
    upd_param 150
    wait      300
    stop""").lstrip('\n')
    
    simple_prog = "play      5,3,40\nwait      750"
    
    assert qbl.simple2real_Q1ASM(simple_prog) == q1asm_prog 

    # markers correctly placed on simple Hahn-echo
    q1asm_prog = dedent(f"""
    acquire   0,0,4
    set_mrk   15
    upd_param 250
    play      4,2,{wfm_len}
    wait      500
    play      5,3,{2*wfm_len}
    wait      50
    set_mrk   11
    upd_param 250
    set_mrk   3
    upd_param 150
    wait      50
    stop""").lstrip('\n')
    
    simple_prog = dedent(f"""
    acquire   0,0,4
    play      4,2,{wfm_len}
    wait      500
    play      5,3,{2*wfm_len}
    wait      500""").lstrip('\n')
    
    assert qbl.simple2real_Q1ASM(simple_prog) == q1asm_prog
    
    # long delays converted appropriately
    q1asm_prog = dedent("""
    acquire   0,0,4
    move      6,R1
    loop_delay1:
    wait      65535
    loop      R1,@loop_delay1
    wait      4
    move      10,R1
    loop_delay2:
    wait      65535
    loop      R1,@loop_delay2
    stop""").lstrip('\n')

    simple_prog = dedent(f"""
    acquire   0,0,4
    wait      {65535*6+4}
    wait      {65535*10}""").lstrip('\n')

    assert qbl.simple2real_Q1ASM(simple_prog) == q1asm_prog

    # test long interpulse delay (should turn off amplifier)
    q1asm_prog = dedent(f"""
    acquire   0,0,4
    set_mrk   15
    upd_param 250
    play      4,2,{wfm_len}
    wait      50
    set_mrk   11
    upd_param 4
    wait      {2000-50-4-250}
    set_mrk   15
    upd_param 250
    play      5,3,{2*wfm_len}
    wait      50
    set_mrk   11
    upd_param 250
    set_mrk   3
    upd_param 150
    wait      550
    move      16,R1
    loop_delay1:
    wait      65535
    loop      R1,@loop_delay1
    stop""").lstrip('\n')

    simple_prog = dedent(f"""
    acquire   0,0,4
    play      4,2,{wfm_len}
    wait      2000
    play      5,3,{2*wfm_len}
    wait      1000
    wait      {16*65535}""").lstrip('\n')

    assert qbl.simple2real_Q1ASM(simple_prog) == q1asm_prog

    # test with acquire at the end of the sequence
    q1asm_prog = dedent(f"""
    set_mrk   15
    upd_param 250
    play      4,2,{wfm_len}
    wait      50
    set_mrk   11
    upd_param 4
    wait      {2000-50-4-250}
    set_mrk   15
    upd_param 250
    play      5,3,{2*wfm_len}
    wait      50
    set_mrk   11
    upd_param 250
    set_mrk   3
    upd_param 150
    wait      550
    move      16,R1
    loop_delay1:
    wait      65535
    loop      R1,@loop_delay1
    acquire   0,0,4
    stop""").lstrip('\n')

    simple_prog = dedent(f"""
    play      4,2,{wfm_len}
    wait      2000
    play      5,3,{2*wfm_len}
    wait      1000
    wait      {16*65535}
    acquire   0,0,4""").lstrip('\n')

    assert qbl.simple2real_Q1ASM(simple_prog) == q1asm_prog

    # test that reserved registers R# raise an error
    with pytest.raises(NameError):
        qbl.simple2real_Q1ASM('move      0,R0')
    with pytest.raises(NameError):
        qbl.simple2real_Q1ASM('move      0,R1')
    with pytest.raises(NameError):
        qbl.simple2real_Q1ASM('move      0,R2')

    # test loop average added
    q1asm_prog = dedent("""
    move      64,R0
    loop_avg:
    reset_ph
    upd_param 17
    acquire   0,0,4
    set_mrk   15
    upd_param 250
    play      4,2,8
    wait      500
    play      5,3,16
    wait      50
    set_mrk   11
    upd_param 250
    set_mrk   3
    upd_param 150
    wait      50
    move      15,R1
    loop_delay1:
    wait      65535
    loop      R1,@loop_delay1
    loop      R0,@loop_avg
    stop""").lstrip('\n')

    simple_prog = dedent(f"""
    acquire   0,0,4
    play      4,2,8
    wait      500
    play      5,3,16
    wait      500
    wait      {65535*15}""").lstrip('\n')

    assert qbl.simple2real_Q1ASM(simple_prog, avg_nb=64) == q1asm_prog

    # test shot loop added
    q1asm_prog = dedent("""
    move      17,R0
    loop_avg:
    move      8,R51
    loop_shot:
    reset_ph
    upd_param 10
    acquire   0,0,4
    set_mrk   15
    upd_param 250
    play      4,2,8
    wait      462
    play      5,3,16
    wait      50
    set_mrk   11
    upd_param 250
    set_mrk   3
    upd_param 150
    wait      50
    move      6,R1
    loop_delay1:
    wait      65535
    loop      R1,@loop_delay1
    loop      R51,@loop_shot
    loop      R0,@loop_avg
    stop""").lstrip('\n')

    simple_prog = dedent(f"""
    acquire   0,0,4
    play      4,2,8
    wait      462
    play      5,3,16
    wait      500
    wait      {65535*6}""").lstrip('\n')

    assert qbl.simple2real_Q1ASM(simple_prog, avg_nb=17, shot_nb=8) == q1asm_prog

    # test dummy shots added with single dummy shot number
    # Note:
    # - reset_ph added at the top and bottom of the program
    q1asm_prog = dedent(f"""
    reset_ph
    upd_param 23
    set_mrk   15
    upd_param 250
    play      4,2,{wfm_len}
    wait      500
    play      5,3,{2*wfm_len}
    wait      50
    set_mrk   11
    upd_param 250
    set_mrk   3
    upd_param 150
    wait      50
    wait      4
    reset_ph
    upd_param 23
    set_mrk   15
    upd_param 250
    play      4,2,{wfm_len}
    wait      500
    play      5,3,{2*wfm_len}
    wait      50
    set_mrk   11
    upd_param 250
    set_mrk   3
    upd_param 150
    wait      50
    acquire   0,0,4
    stop""").lstrip('\n')

    simple_prog = dedent(f"""
    play      4,2,{wfm_len}
    wait      500
    play      5,3,{2*wfm_len}
    wait      500
    acquire   0,0,4""").lstrip('\n')

    assert qbl.simple2real_Q1ASM(simple_prog, dummy_nb=1) == q1asm_prog

    # test several dummy shots added
    # Notes:
    # - wait instruction replaces acquire in dummy part of the seqeunce
    # - loop_delay index increasing in normal sequence
    q1asm_prog = dedent("""
    move      2,R63
    loop_dummy:
    reset_ph
    upd_param 17
    wait      4
    set_mrk   15
    upd_param 250
    play      4,2,8
    wait      500
    play      5,3,16
    wait      50
    set_mrk   11
    upd_param 250
    set_mrk   3
    upd_param 150
    wait      50
    move      15,R1
    loop_delay1:
    wait      65535
    loop      R1,@loop_delay1
    loop      R63,@loop_dummy
    move      64,R0
    loop_avg:
    reset_ph
    upd_param 17
    acquire   0,0,4
    set_mrk   15
    upd_param 250
    play      4,2,8
    wait      500
    play      5,3,16
    wait      50
    set_mrk   11
    upd_param 250
    set_mrk   3
    upd_param 150
    wait      50
    move      15,R1
    loop_delay2:
    wait      65535
    loop      R1,@loop_delay2
    loop      R0,@loop_avg
    stop""").lstrip('\n')

    simple_prog = dedent(f"""
    acquire   0,0,4
    play      4,2,8
    wait      500
    play      5,3,16
    wait      500
    wait      {65535*15}""").lstrip('\n')

    assert qbl.simple2real_Q1ASM(simple_prog, avg_nb=64, dummy_nb=2) == q1asm_prog

    steps = [90, 90]
    ctp = [-1, +2]
    q1asm_prog = dedent(f"""
    move      0,R41
    loop_ph1:
    move      0,R42
    loop_ph2:
    move      {(abs(min(ctp))*360)*2777777},R40
    nop
    sub       R40,R41,R40
    nop
    add       R40,R42,R40
    nop
    add       R40,R42,R40
    nop
    loop_mod360:
    sub       R40,{360*2777777},R40
    nop
    jge       R40,{360*2777777},@loop_mod360
    reset_ph
    upd_param 27
    set_mrk   15
    upd_param 242
    set_ph    R41
    upd_param 8
    play      0,1,18
    wait      492
    set_ph    R42
    upd_param 8
    play      2,3,36
    wait      50
    set_mrk   11
    upd_param 250
    set_mrk   3
    upd_param 150
    wait      42
    set_ph    R40
    upd_param 8
    acquire   0,0,4
    move      15,R1
    loop_delay1:
    wait      65535
    loop      R1,@loop_delay1
    add       R42,{steps[1]*2777777},R42
    nop
    jlt       R42,{360*2777777},@loop_ph2
    add       R41,{steps[0]*2777777},R41
    nop
    jlt       R41,{360*2777777},@loop_ph1
    stop""").lstrip('\n')

    simple_prog = dedent(f"""
    play      0,1,18
    wait      500
    play      2,3,36
    wait      500 # {50+250+138+8}
    acquire   0,0,4
    wait      {15*65535}""").lstrip('\n')

    assert qbl.simple2real_Q1ASM(simple_prog, steps=steps, ctp=ctp) == q1asm_prog

    # test with steps of 0
    q1asm_prog = dedent("""
    move      8,R0
    loop_avg:
    reset_ph
    upd_param 16
    set_mrk   15
    upd_param 242
    set_ph    0
    upd_param 8
    play      4,2,350
    wait      50
    set_mrk   11
    upd_param 4
    wait      65535
    wait      65535
    wait      18626
    set_mrk   15
    upd_param 242
    set_ph    0
    upd_param 8
    play      5,3,700
    wait      50
    set_mrk   11
    upd_param 250
    set_mrk   3
    upd_param 150
    wait      65535
    wait      65535
    wait      10972
    set_ph    0
    upd_param 8
    acquire   0,0,4
    move      76,R1
    loop_delay1:
    wait      65535
    loop      R1,@loop_delay1
    wait      26840
    loop      R0,@loop_avg
    stop""").lstrip('\n')

    prog = dedent(f"""
    play      4,2,{350}
    wait      {150000}
    play      5,3,{2*350}
    wait      {150000-7500}
    acquire   0,0,4
    wait      {int(5e6)+7500}""").lstrip('\n')

    full_q1asm = qbl.simple2real_Q1ASM(prog, avg_nb=8,
                                       steps=[0, 0], ctp=[-1, +2],
                                       switch_open_postdelay=0,
                                       amp_on_postdelay=250,
                                       amp_off_predelay=50,
                                       amp_off_postdelay=250,
                                       switch_closed_postdelay=150)

    assert full_q1asm == q1asm_prog

    # test with steps other than 90
    steps = [90, 180]
    ctp = [+1, -2]

    q1asm_prog = dedent(f"""
    move      2,R63
    loop_dummy:
    reset_ph
    upd_param 24
    set_mrk   15
    upd_param 250
    play      4,2,21
    wait      500
    play      5,3,21
    wait      50
    set_mrk   11
    upd_param 250
    set_mrk   3
    upd_param 150
    wait      50
    wait      4
    move      15,R1
    loop_delay1:
    wait      65535
    loop      R1,@loop_delay1
    wait      16975
    loop      R63,@loop_dummy
    move      64,R0
    loop_avg:
    move      0,R41
    loop_ph1:
    move      0,R42
    loop_ph2:
    move      {(abs(min(ctp))*360)*2777777},R40
    nop
    add       R40,R41,R40
    nop
    sub       R40,R42,R40
    nop
    sub       R40,R42,R40
    nop
    loop_mod360:
    sub       R40,999999720,R40
    nop
    jge       R40,999999720,@loop_mod360
    reset_ph
    upd_param 24
    set_mrk   15
    upd_param 242
    set_ph    R41
    upd_param 8
    play      4,2,21
    wait      492
    set_ph    R42
    upd_param 8
    play      5,3,21
    wait      50
    set_mrk   11
    upd_param 250
    set_mrk   3
    upd_param 150
    wait      42
    set_ph    R40
    upd_param 8
    acquire   0,0,4
    move      15,R1
    loop_delay2:
    wait      65535
    loop      R1,@loop_delay2
    wait      16975
    add       R42,249999930,R42
    nop
    jlt       R42,999999720,@loop_ph2
    add       R41,499999860,R41
    nop
    jlt       R41,999999720,@loop_ph1
    loop      R0,@loop_avg
    stop""").lstrip('\n')

    simple_prog = dedent(f"""
    play      4,2,21
    wait      500
    play      5,3,21
    wait      500
    acquire   0,0,4
    wait      {int(1e6)}""").lstrip('\n')

    full_q1asm = qbl.simple2real_Q1ASM(simple_prog, dummy_nb=2, avg_nb=64, steps=steps, ctp=ctp)

    assert q1asm_prog == full_q1asm


    # test with one step number at 0
    steps = [0, 180]
    ctp = [-1, +2]
    q1asm_prog = dedent(f"""
    move      0,R41
    loop_ph1:
    move      {(abs(min(ctp))*360)*2777777},R40
    nop
    add       R40,R41,R40
    nop
    add       R40,R41,R40
    nop
    loop_mod360:
    sub       R40,{360*2777777},R40
    nop
    jge       R40,{360*2777777},@loop_mod360
    reset_ph
    upd_param 27
    set_mrk   15
    upd_param 242
    set_ph    0
    upd_param 8
    play      0,1,18
    wait      492
    set_ph    R41
    upd_param 8
    play      2,3,36
    wait      50
    set_mrk   11
    upd_param 250
    set_mrk   3
    upd_param 150
    wait      42
    set_ph    R40
    upd_param 8
    acquire   0,0,4
    move      15,R1
    loop_delay1:
    wait      65535
    loop      R1,@loop_delay1
    add       R41,{steps[1]*2777777},R41
    nop
    jlt       R41,{360*2777777},@loop_ph1
    stop""").lstrip('\n')

    simple_prog = dedent(f"""
    play      0,1,18
    wait      500
    play      2,3,36
    wait      500
    acquire   0,0,4
    wait      {15*65535}""").lstrip('\n')

    full_q1asm = qbl.simple2real_Q1ASM(simple_prog, steps=[0, 180], ctp=[-1, +2])

    assert qbl.simple2real_Q1ASM(simple_prog, steps=steps, ctp=ctp) == q1asm_prog

    # mix order with step of 0
    q1asm_prog = dedent("""
    move      0,R41
    loop_ph1:
    move      999999720,R40
    nop
    sub       R40,R41,R40
    nop
    loop_mod360:
    sub       R40,999999720,R40
    nop
    jge       R40,999999720,@loop_mod360
    reset_ph
    upd_param 27
    set_mrk   15
    upd_param 242
    set_ph    R41
    upd_param 8
    play      0,1,18
    wait      492
    set_ph    0
    upd_param 8
    play      2,3,36
    wait      50
    set_mrk   11
    upd_param 250
    set_mrk   3
    upd_param 150
    wait      42
    set_ph    R40
    upd_param 8
    acquire   0,0,4
    move      15,R1
    loop_delay1:
    wait      65535
    loop      R1,@loop_delay1
    add       R41,499999860,R41
    nop
    jlt       R41,999999720,@loop_ph1
    stop""").lstrip('\n')

    assert qbl.simple2real_Q1ASM(simple_prog, steps=[180, 0], ctp=ctp) == q1asm_prog


def test_simple_convert_q1asm():
    
    simple_prog = dedent("""
    R11 = 2
    R21  =    3    # some comment =
    R20 = 40
    R11 = R21
    R20 = R20 + 2
    R20=2+R20
    R7 = R3 * 1
    R7 = R3 * 3
    R7 = R3 * 6
    
    set_mrk(t1t2t3t4=1010)   # synthax: m4m3m2m1
        set_ph(  90  )   # messing with spaces
    set_ph(90.0251)
    set_ph(2e-5)
    set_ph_delta(65)
    for R11 in q1asm_ph(0), q1asm_ph(1), q1asm_ph(180)
        set_ph    R11
    end
    # set_ph(R1)
    set_awg_gain(0.75, 0.75)
    set_freq(6.8e6)""").lstrip('\n')  # lstrip gets rid of leading character
    
    q1asm_prog = dedent("""
    move      R11,2
    move      R21,3
    move      R20,40
    move      R11,R21
    add       R20,2,R20
    add       R20,2,R20
    move      R7,R3
    move      R7,R3
    add       R7,R3,R7
    add       R7,R3,R7
    move      R7,R3
    move      R1,5
    loop_mult0:
    add       R7,R3,R7
    nop
    loop     R1,@loop_mult0
    
    set_mrk   5
    set_ph    250000000
    set_ph    250069722
    set_ph    56
    set_ph_delta 180555556
    move      0,R11
    loop_for1:
    set_ph    R11
    add       R11,2777778,R11
    nop
    jlt       R11,500000001,@loop_for1
    
    set_awg_gain 24575,24575
    set_freq  27200000
    stop""").lstrip('\n')

    assert qbl.simple2real_Q1ASM(simple_prog) == q1asm_prog
