import pytest
from textwrap import dedent
from qblox_esr.simple_convert import simple_convert_q1asm, q1asm_ph, q1asm_freq, q1asm_gain


# reserved registers
# -----------------------------------------------------------------------------
def test_R1_reserved():
    with pytest.raises(NameError):
        simple_convert_q1asm("R1 = 0")

# assignment and arithmetic operations
# -----------------------------------------------------------------------------
def test_simple_assignment():
    assert simple_convert_q1asm("R11 = R2") == "move      R11,R2"

def test_addition():
    assert simple_convert_q1asm("R11 = R2 + R3") == "add       R2,R3,R11"

def test_subtraction():
    assert simple_convert_q1asm("R11 = R2 - R3") == "sub       R2,R3,R11"

def test_addition_with_immediate():
    assert simple_convert_q1asm("R11 = 2 + R2") == "add       R2,2,R11"
    assert simple_convert_q1asm("R11 = R2 + 2") == "add       R2,2,R11"

def test_multiplication_small_constant():
    result = simple_convert_q1asm("R11 = R2 * 3")
    expected = "\n".join([
        "move      R11,R2",
        "add       R11,R2,R11",
        "add       R11,R2,R11"
    ])
    assert result == expected

def test_multiplication_large_constant():
    result = simple_convert_q1asm("R11 = R2 * 6")
    assert "loop_mult0:" in result
    assert "loop     R1,@loop_mult0" in result

# # test arithmetic operations error cases
def test_addition_constants_error():
    with pytest.raises(NotImplementedError):
        simple_convert_q1asm("R11 = 2 + 3")

def test_multiplication_constants_error():
    with pytest.raises(NotImplementedError):
        simple_convert_q1asm("R11 = 2 * 3")

def test_multiplication_two_registers_error():
    with pytest.raises(NotImplementedError):
        simple_convert_q1asm("R11 = R2 * R3")

def test_multiplication_same_register_error():
    with pytest.raises(NotImplementedError):
        simple_convert_q1asm("R11 = R11 * 3")

def test_division_error():
    with pytest.raises(NotImplementedError):
        simple_convert_q1asm("R11 = R2 / 2")

# set_ph() and set_ph_delta()
# -----------------------------------------------------------------------------
def test_set_ph_with_spaces_and_comment():
    inp = "  set_ph(  90  )   # virtual Z"
    exp = f"set_ph    {q1asm_ph(90)}"
    assert simple_convert_q1asm(inp) == exp

def test_set_ph_negative_wraps():
    inp = "set_ph(-90)"
    exp = f"set_ph    {q1asm_ph(-90)}"
    assert simple_convert_q1asm(inp) == exp

def test_set_ph_sci_notation():
    inp = "set_ph(2e-5)"
    # tiny angle: still converts deterministically
    exp = f"set_ph    {q1asm_ph(2e-5)}"
    assert simple_convert_q1asm(inp) == exp

def test_set_ph_delta():
    inp = "set_ph_delta(65)"
    exp = f"set_ph_delta {q1asm_ph(65)}"
    assert simple_convert_q1asm(inp) == exp
    
def test_register_error():
    with pytest.raises(ValueError):
        simple_convert_q1asm('set_ph(R11)')
        
    with pytest.raises(ValueError):
        simple_convert_q1asm('  set_ph( R18 )  # oklm')

# ---- in-line q1asm_ph(...) (can appear multiple times per line)
def test_inline_q1asm_ph_single():
    inp = "for q1asm_ph(180),1:"
    # simple_convert_q1asm replaces q1asm_ph(...) anywhere on the line, then passes through
    exp = f"for {q1asm_ph(180)},1:"
    assert simple_convert_q1asm(inp) == exp


def test_inline_q1asm_ph_multiple():
    inp = "for q1asm_ph(0), q1asm_ph(180), 1:"
    exp = f"for {q1asm_ph(0)}, {q1asm_ph(180)}, 1:"
    assert simple_convert_q1asm(inp) == exp

# set_awg_gain()
# -----------------------------------------------------------------------------
def test_set_awg_gain_wrong_arg_count_raises():
    with pytest.raises(ValueError):
        simple_convert_q1asm("set_awg_gain(0.5)")
    with pytest.raises(ValueError):
        simple_convert_q1asm("set_awg_gain(0.1, 0.2, 0.3)")

def test_set_awg_gain_with_spaces_and_comment():
    inp = "  set_awg_gain(  0.5  , 0.5)   # half scale"
    exp = f"set_awg_gain {q1asm_gain(0.5)},{q1asm_gain(0.5)}"
    assert simple_convert_q1asm(inp) == exp

def test_set_awg_gain_negative_fullscale():
    inp = "set_awg_gain(-1,-0.17)"
    exp = f"set_awg_gain {q1asm_gain(-1)},{q1asm_gain(-0.17)}"
    assert simple_convert_q1asm(inp) == exp

def test_set_awg_gain_sci_notation():
    inp = "set_awg_gain(5e-1,74e-2)"
    exp = f"set_awg_gain {q1asm_gain(0.5)},{q1asm_gain(0.74)}"
    assert simple_convert_q1asm(inp) == exp

def test_set_awg_gain_out_of_range_raises():
    with pytest.raises(ValueError):
        simple_convert_q1asm("q1asm_gain(1.0001,1)")
    with pytest.raises(ValueError):
        simple_convert_q1asm("q1asm_gain(0.5,-1.01)")

def test_set_awg_gain_register_error():
    with pytest.raises(ValueError):
        simple_convert_q1asm("q1asm_gain(R11, 1)")
    with pytest.raises(ValueError):
        simple_convert_q1asm("  q1asm_gain(3, R2 )  # not allowed")


# set_freq()
# -----------------------------------------------------------------------------
def test_set_freq_with_spaces_and_comment():
    inp = "  set_freq(  100e6  )   # 100 MHz"
    exp = f"set_freq  {q1asm_freq(100e6)}"
    assert simple_convert_q1asm(inp) == exp

def test_set_freq_negative():
    inp = "set_freq(-250e6)"
    exp = f"set_freq  {q1asm_freq(-250e6)}"
    assert simple_convert_q1asm(inp) == exp

def test_set_freq_sci_notation():
    inp = "set_freq(2.5e8)"  # 250 MHz
    exp = f"set_freq  {q1asm_freq(250e6)}"
    assert simple_convert_q1asm(inp) == exp

def test_set_freq_step_rounding():
    # 1 step = 0.25 Hz. Choose a non-integer-step value to test rounding.
    # 0.26 Hz -> 1.04 steps -> rounds to 1 step
    inp = "set_freq(0.26)"
    exp = f"set_freq  {q1asm_freq(0.26)}"
    assert simple_convert_q1asm(inp) == exp

def test_set_freq_exact_step():
    # Exactly one step
    inp = "set_freq(0.25)"
    exp = f"set_freq  {q1asm_freq(0.25)}"  # -> 1
    assert simple_convert_q1asm(inp) == exp

def test_set_freq_endpoints():
    # Check both ends of the allowed range
    assert simple_convert_q1asm("set_freq(500e6)") == f"set_freq  {q1asm_freq(500e6)}"
    assert simple_convert_q1asm("set_freq(-500e6)") == f"set_freq  {q1asm_freq(-500e6)}"

def test_set_freq_out_of_range_raises():
    with pytest.raises(ValueError):
        simple_convert_q1asm("set_freq(600e6)")
    with pytest.raises(ValueError):
        simple_convert_q1asm("set_freq(-501e6)")

def test_set_freq_register_error():
    with pytest.raises(ValueError):
        simple_convert_q1asm("set_freq(R11)")
    with pytest.raises(ValueError):
        simple_convert_q1asm("  set_freq( R18 )  # not allowed")

# set_mrk()
# -----------------------------------------------------------------------------
def test_set_mrk_t1t2t3t4_binary():
    # In simple_convert.py: the code reverses the 4-bit string before int(..., 2)
    # "1010" -> reversed "0101" -> 5
    inp = "set_mrk(t1t2t3t4=1010)"
    exp = "set_mrk   5"
    assert simple_convert_q1asm(inp) == exp

def test_set_mrk_all_ones():
    # "1111" -> reversed "1111" -> 15
    assert simple_convert_q1asm("set_mrk(t1t2t3t4=1111)") == "set_mrk   15"

def test_set_mrk_invalid_length_raises():
    # The implementation asserts length==4
    # short input should trigger AssertionError inside set_mrk()
    with pytest.raises(AssertionError):
        simple_convert_q1asm("set_mrk(t1t2t3t4=101)")

# passthrough of unknown commands (default branch)
# -----------------------------------------------------------------------------
def test_default():
    assert simple_convert_q1asm("passthrough()") == "passthrough()"
    
    
# general test with multiple lines
# -----------------------------------------------------------------------------
def test_simple_convert_q1asm():
    
    q1asm_prog = dedent("""
    move      R11,2
    move      R2,3
    move      R20,40
    move      R11,R2
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
    for R11 in 0, 500000000, 1:
    set_ph    R11
    end
    
    set_awg_gain 24575,24575
    set_freq  27200000
    passthrough()""").lstrip('\n')
    
    simple_prog = dedent("""
    R11 = 2
    R2  =    3    # some comment =
    R20 = 40
    R11 = R2
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
    for R11 in q1asm_ph(0), q1asm_ph(180), 1:
        set_ph    R11
    end
    # set_ph(R1)
    set_awg_gain(0.75, 0.75)
    set_freq(6.8e6)
    passthrough()""").lstrip('\n')  # lstrip gets rid of leading character

    assert simple_convert_q1asm(simple_prog) == q1asm_prog