"""
Module to convert simple Pythonic instructions to Q1ASM instructions with simple_convert_q1asm().

author: jbv
"""

import re
import numpy as np


# Utilities (internal)
# -----------------------------------------------------------------------------
def _arg_between_parens(s):
    """
    Extract the substring inside the first top-level parentheses.

    Parameters
    ----------
    s: str
        Input string that contains a '(' followed by a matching ')'.

    Returns
    -------
    str
        Substring between the first '(' and its matching ')'.

    Raises
    ------
    ValueError
        If '(' or ')' is not found in the expected order.
    """
    a = s.find('(')
    if a == -1:
        raise ValueError("No '(' found.")
    b = s.find(')', a + 1)
    if b == -1:
        raise ValueError("No ')' found.")
    return s[a + 1:b].strip()


def _replace_function_calls(line, func_name, converter):
    """
    Replace all occurrences of func_name(<number>) in a line with converter(<number>).

    Parameters
    ----------
    line : str
        Input line containing zero or more calls to func_name(...).
    func_name : str
        Name of the function to replace (e.g., 'q1asm_ph').
    converter : callable
        Function that takes a float and returns an int or str.

    Returns
    -------
    str
        Line with all calls replaced by their computed values.
    """
    out = []
    i = 0
    n = len(line)
    token = func_name + '('
    tlen = len(token)

    while i < n:
        j = line.find(token, i)
        if j == -1:
            out.append(line[i:])
            break

        # Emit text before the call
        out.append(line[i:j])

        # Find matching ')'
        k = j + tlen
        depth = 1
        p = k
        while p < n and depth:
            if line[p] == '(':
                depth += 1
            elif line[p] == ')':
                depth -= 1
            p += 1
        if depth:
            raise ValueError(f"Unmatched ')' in {func_name} call.")

        # Extract argument and replace
        arg_text = line[k:p-1].strip()
        val = float(arg_text)
        out.append(str(converter(val)))

        i = p  # continue after ')'

    return ''.join(out)


def _check_arg_not_register(line, arg):
    """
    Raises ValueError if argument is a register (e.g., 'R1', 'R2', ...).
    """

    if arg.startswith('R'):
        raise ValueError(
            'Do not use registers (variables) with set_ph(), use Q1ASM command '
            'set_ph Register and q1asm_ph() on the numbers used to set it up. '
            f'Spotted at line:\n{line}'
        )


# Unit conversion
# -----------------------------------------------------------------------------
def q1asm_ph(phase, deg=True):
    """
    Convert a phase to the integer argument expected by Q1ASM NCO phase instructions.

    Parameters
    ----------
    phase: float
        phase value (degrees or radians, see deg)
    deg: bool, optional
        phase in degrees when True (default), radians if False.

    Returns
    -------
    q1asm_phase: int
        NCO phase in the range [0, 1e9] suitable for Q1ASM commands.

    Notes
    -----
    360 degrees (2π rad) is 1e9 steps on Qblox sequencers. Values are wrapped
    (mod 360° or 2π) so that 360° and 2π both map to 0.

    Examples
    --------
    >>> q1asm_ph(90)  # 90° -> 1e9/4 steps
    250000000
    >>> q1asm_ph(-90)  # -90° ≡ 270°
    750000000
    """
    if not deg: phase = np.rad2deg(phase)
    phase_wrapped = phase % 360
    q1asm_phase = phase_wrapped * 2777777.777778  # 1e9/360 steps

    return int(np.round(q1asm_phase))


def q1asm_gain(gain: float) -> int:
    """
    Convert a normalized AWG gain in [-1, 1] to the signed 16-bit integer
    expected by Q1ASM AWG gain instructions.

    Parameters
    ----------
    gain : float
        normalized amplitude in the range [-1, 1].

    Returns
    -------
    int16_gain : int
        signed 16-bit value in [-32768, 32767].

    Notes
    -----
    Signed 16-bit domain asymmetric:
      - gain == -1 -> -32768
      - gain == +1 -> +32767
    this function uses a piecewise scale (32768 on the negative side,
    32767 on the non-negative side) before rounding.

    Example
    --------
    >>> set_awg_gain(0.5)  # 0.5 * 32767 -> 16383.5 -> rounds to 16384
    16384
    """
    if gain < -1.0 or gain > 1.0:
        raise ValueError("gain out of range [-1, 1]")

    # Piecewise scaling to achieve -1 -> -32768 and +1 -> +32767.
    scale = 32767 if gain >= 0.0 else 32768
    return int(np.round(gain * scale))


def q1asm_freq(freq: float) -> int:
    """
    Convert a frequency in Hz to the integer 'frequency steps' expected by
    Q1ASM ``set_freq``.

    Parameters
    ----------
    freq : float
        frequency in Hz.

    Returns
    -------
    steps : int
        integer frequency word in [-2e9, 2e9] for Q1ASM ``set_freq``.

    Examples
    --------
    >>> q1asm_freq(100e6)          # 100 MHz -> 100e6 * 4 = 400,000,000
    400000000
    >>> q1asm_freq(0.25)           # 0.25 Hz -> 1 step
    1
    """
    if freq < -500e6 or freq > 500e6:
        raise ValueError("frequency out of range [-500e6, 500e6] Hz")

    return int(np.round(freq * 4.0))


# Command handles
# -----------------------------------------------------------------------------
def set_mrk(line):
    """
    Convert a simplified set_mrk(t1t2t3t4=xxxx) command to Q1ASM format.
    
    Parameters
    ----------
    line : str
        Line containing the simplified marker command.
    
    Returns
    -------
    str
        Q1ASM-compatible marker command.
    
    Example
    -------
    set_mrk(t1t2t3t4=0000) is transformed to set_mrk 0
    The trigger numbers follow the more natural marker on the module
    (unlike Q1ASM set_mrk). For example, on QRM-RF, t1t2t3t4=O1I1M1M2.
    """

    match = re.match(
        r'^\s*set_mrk\(\s*t1t2t3t4\s*=\s*([01]{4})\s*\)\s*$', line)

    assert match is not None, "invalid set_mrk() syntax"

    binary = match.group(1)            # e.g., "1010"

    # Q1ASM maker command, 4-bits (t4t3t2t1!) as integer
    binary = binary[::-1]  # reverse
    int_cmd = int(binary, 2)       
    assert 0 <= int_cmd <= 15, 'marker value must be in [0..15]'
    
    return f'set_mrk   {int_cmd}'


def set_ph(line):
    """
    Convert a simplified set_ph() command to Q1ASM format using q1asm_ph().
    """
    
    arg = _arg_between_parens(line)
    _check_arg_not_register(line, arg)

    q1asm_phase = q1asm_ph(float(arg))

    return f'set_ph    {q1asm_phase}'


def set_ph_delta(line):
    """
    See set_ph() documentation.
    """
    return set_ph(line).replace("set_ph   ", "set_ph_delta", 1)


def set_awg_gain(line):
    """
    Convert a simplified set_awg_gain() command to Q1ASM format using q1asm_gain().
    """
    args_str = _arg_between_parens(line)
    args = [p for p in args_str.split(',')]
    if len(args) != 2:
        raise ValueError(f"Expected 2 arguments, got {len(args)}: {args}")
    _check_arg_not_register(line, args[0])
    _check_arg_not_register(line, args[1])
    I = int(q1asm_gain(float(args[0])))
    Q = int(q1asm_gain(float(args[1])))

    return f"set_awg_gain {I},{Q}"


def set_awg_offs(line):
    """
    See set_awg_gain() documentation.
    """
    return set_awg_gain(line).replace("set_awg_gain", "set_awg_offs", 1)


def set_freq(line):
    """
    Convert a simplified set_freq() command to Q1ASM format using q1asm_freq().
    """
    arg = _arg_between_parens(line)
    _check_arg_not_register(line, arg)

    q1asm_frequency = q1asm_freq(float(arg))

    return f'set_freq  {q1asm_frequency}'


# Operators
# -----------------------------------------------------------------------------
def equal2q1asm(line):
    """
    Convert a simple assignment (e.g., R1 = R2) to Q1ASM move instruction.
    
    Parameters
    ----------
    line : str
        Line containing the assignment.
    
    Returns
    -------
    str
        Q1ASM move instruction.
    """


    match = re.match(r'^\s*(\w+)\s*=\s*(\w+)\s*$', line)
    destination, source = match.groups()

    return f'move      {destination},{source}'


def operator2q1asm(line, mult_loop_index=None):
    """
    Convert arithmetic operations to Q1ASM instructions.
    
    Parameters
    ----------
    line : str
        Line containing the arithmetic operation.
    mult_loop_index : int or None
        Index for labeling multiplication loops.
    
    Returns
    -------
    q1asm_line : str
        Q1ASM instruction(s)
    mult_loop_index: int or None
         updated loop index.
    """

    match = re.match(r'^\s*(\w+)\s*=\s*(\w+)\s*([+\-*/])\s*(\w+)\s*$', line)
       
    dest, op1, operator, op2 = match.groups()

    if operator in ['+', '-']:
        # dest = op1 + op2 converted to add op1, op2, dest
        # (same for - and sub)
        if operator == '+':
            op_cmd = 'add'
        else: op_cmd = 'sub'

        if op1[0] != 'R' and op2[0] != 'R':
            raise NotImplementedError(
                f'Addition/Subtraction between constants (immediate values) '
                f'not supported, spotted at line:\n{line}'
                )
        elif op1[0] != 'R':
            # immediate value (integer) must be placed second
            q1asm_line = f'{op_cmd}       {op2},{op1},{dest}'
        else:
            q1asm_line = f'{op_cmd}       {op1},{op2},{dest}'
    
    elif operator == '*':
        # NB: might be possible to write the operation bit-wise
        # by looping over and, add, asl

        # Only support multiplication of other register by integer constant
        # dest = reg * count
        if op1[0] != 'R' and op2[0] != 'R':
            # possible to implement, but should be avoid by the user
            raise NotImplementedError(
                f'Multiplication between constants (immediate values) not '
                f'supported (and not recommended, do this off-board), spotted '
                f'at line:\n{line}')
        elif op1[0] == 'R' and op2[0] == 'R':
            # potentially more difficult to implement
            # need a loop strategy:
            # move dest, 0
            # move count, R1
            # loop:
            # add dest, reg1, dest
            # loop R1, @loop
            raise NotImplementedError(
                f'Multiplication between 2 variables (registers) not '
                f'supported, spotted at line:\n{line}')
        elif op1[0] == 'R':
            reg, count = op1, op2
        else:
            count, reg = op2, op1

        if reg == dest:
            # possible to implement, just need to reserve a register
            # and initialize it
            # need to reserve 2 if used twice (square)
            raise NotImplementedError(f'Multiplication on the same variable '
                                      f'(registers) not supported, spotted at '
                                      f'line:\n{line}')

        # initialize destination
        q1asm_line = f'move      {dest},{reg}\n'
        count = int(count) - 1

        if count < 0:
            # nb: this would not be recognized by the regex anyway
            raise NotImplementedError('Negative constants for multiplication '
                                      'not supported.'
                                      )
        if count < 5:
            while count > 0:
                q1asm_line += f'add       {dest},{reg},{dest}\n'
                count -= 1
        else:  # create a loop to avoid using too many lines of code
            q1asm_line += f'move      R1,{count}\n'  # register R1 reserved
            q1asm_line += f'loop_mult{mult_loop_index}:\n'
            q1asm_line += f'add       {dest},{reg},{dest}\n'
            q1asm_line += 'nop\n'
            q1asm_line += f'loop     R1,@loop_mult{mult_loop_index}\n'
            mult_loop_index += 1
        q1asm_line = q1asm_line.strip()
    elif operator == '/':
        raise NotImplementedError("Division is not supported.")

    return q1asm_line, mult_loop_index


# Main conversion function
# -----------------------------------------------------------------------------
def simple_convert_q1asm(simple_prog):
    """
    Convert simple Pythonic instructions to Q1ASM instructions.
    
    Parameters
    ----------
    simple_prog : str
        Q1ASM or simplified Q1ASM program.

    Returns
    -------
    prog: str
        program with simple commands converted to Q1ASM commands.

    Notes:
    ------   
    Allows to mix Q1ASM code with more Python-like code:
       - supported operators:
           - = (Q1ASM move)
           - + (Q1ASM add) and - (Q1ASM sub), not supported between constants
             as unefficient use of FPGA
           - *, only supported between variable (register) and constant
       - unit conversion (cannot be used on registers):
           - q1asm_ph()
           - q1asm_gain()
           - q1asm_freq()
       - set commands integrating unit conversion (cannot be used on registers):
           - set_ph() and set_ph_delta() in degrees 
           - set_awg_gain() and set_awg_offs() as a simple mulitplier from -1 to 1
           - set_freq() in Hz
       - simple trigger (marker) controls with markers:
           - set_mrk(t1t2t3t4=0000), where the trigger numbers follow
             the more natural marker number on the module, unlike Q1ASM set_mrk
             which goes as set_mrk m4m3m2m1 and is expressed as a single integer.
             on QRM-RF, t1t2t3t4=NoneO1M1M2 whereas set_mrk M2M1O1None.
     
    This is separated from advanced commands for efficiency as this allows to 
    only scan the program once and does not require anything else other than 
    the sequence program.
    
    Example:
    --------
    simple_prog = dedent(\"""
    R20 = 40
    R20 = R20 + 2
    R16 = 2
    R20 = R16 * 1
    set_mrk(t1t2t3t4=1010)   # synthax: m4m3m2m1
    set_ph(90.0251)
    set_ph(2e-5)
    set_ph_delta(65)
    for R20 in q1asm_ph(0), q1asm_ph(180), 1:
        set_ph    R20
    end
    set_awg_gain(0.75)
    set_freq(6.8e6)\""")

    simple_convert_q1asm(simple_prog)
    """
    prog = simple_prog.splitlines()  # list of strings

    # loop organized with elif when possible to avoid multiple passes of lines
    mult_loop_index = 0
    for i, line in enumerate(prog):

        # remove comments when looking for commands
        if '#' in line:
            line = line[:line.find('#')]

        if re.search(r'\bR1\b', line):
            raise NameError('R1 is reserved.')

        # remove leading/trailing spaces
        # regex anchors and function detection simpler
        line = line.lstrip().rstrip()

        # unit conversion, necessary to execute before the rest
        if 'q1asm_ph(' in line:
            line = _replace_function_calls(line, 'q1asm_ph', q1asm_ph)
        elif 'q1asm_gain(' in line:
            line = _replace_function_calls(line, 'q1asm_gain', q1asm_gain)
        elif 'q1asm_freq(' in line:
            line = _replace_function_calls(line, 'q1asm_freq', q1asm_freq)

        # variable assignment ('=' operator)
        # regex only match "=" with two "words" on each side
        # ^ and $ anchor the reg, "s" handles spaces
        if re.match(r'^\s*(\w+)\s*=\s*(\w+)\s*$', line):
            prog[i] = equal2q1asm(line)

        # arithmetic expressions ('+', '-', '*' operators)
        elif re.match(r'^\s*(\w+)\s*=\s*(\w+)\s*([+\-*/])\s*(\w+)\s*$', line):
            
            prog[i], mult_loop_i = operator2q1asm(line, mult_loop_index)
            
            if mult_loop_i is not None: mult_loop_index = mult_loop_i

        # 3) triggers/marker commands
        # t1t2t3t4 stands for trigger to avoid confustion with marker number
        # for RF modules. For example:
        # for QRM-RF, t1 is inactive, t2 activates the switch of output 1
        # t3 activates marker 1, t4 activates marker 2
        elif line.startswith('set_mrk('):
            prog[i] = set_mrk(line)

        # functions for Q1ASM commands
        # phase
        elif line.startswith('set_ph('):
            prog[i] = set_ph(line)
        elif line.startswith('set_ph_delta('):
            prog[i] = set_ph_delta(line)
        # AWG gain
        elif line.startswith('set_awg_gain('):
            prog[i] = set_awg_gain(line)
        elif line.startswith('set_awg_offs('):
            prog[i] = set_awg_offs(line)
        # NCO frequency
        elif line.startswith('set_freq('):
            prog[i] = set_freq(line)

        # 4) default passthrough
        else:
            prog[i] = line.strip()

        # TODO auto nop?

    return '\n'.join(prog)
