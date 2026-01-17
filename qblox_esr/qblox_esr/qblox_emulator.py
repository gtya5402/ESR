"""
Module that emulates the qlblox hardware
authors: JBV & GokulM
"""


import matplotlib.pyplot as plt
import numpy as np


def q1asm_get_line_info(line):
    """
    Remove line comments and extract numerical values.

    Parameters
    ----------
    line : str
        Line to be treated.

    Returns
    -------
    cleanline : str
        Line without comments.
    numbers : list of int
        Line numbers.
    """
    cleanline = line
    if '#' in line:
        cleanline = cleanline[:cleanline.find('#')]
    cleanline_nospace = cleanline.replace(',', ' ')
#     print(cleanline_nospace)
#     numbers = [int(s) for s in cleanline_nospace.split() if not s.startswith('R') and s.lstrip('-').isdigit()]
#     print(numbers)
    numbers = [int(s) for s in cleanline_nospace.split()
               if s.lstrip('-').isdigit()]
    

    return cleanline_nospace.strip(), numbers

def convert_gain(gain_in_db):
    gain = 10**(gain_in_db/20)
    return gain


class Q1ASMEmulator:
    def __init__(self, sequence, integration_time = None, ):
        registers_locations = [f'R{i}' for i in range(0,64)]
        self.program = sequence['program']
        self.weights = sequence['weights']
        
        self.registers = {location: 0 for location in registers_locations}
        self.awg_gain_I = 0
        self.awg_gain_Q = 0
        self.nco_phase = 0
        self.nco_phase_stream = []
        self.awg_offset_I = 0
        self.awg_offset_Q = 0
        self.timing = 0
        self.time = []
        self.acquire = False
        
        self.acquisitions = {key :{
                             **value,
                             'scope': {'path0': [], 'path1': []}, 
                              'bins': {'path0': [], 'path1': []} }for key, value in sequence['acquisitions'].items()  
                            }
        for channel in self.acquisitions.keys():
            number_of_bins = self.acquisitions[channel]['num_bins']
            self.acquisitions[channel]['bins']['path0'] = np.zeros((number_of_bins, 2), dtype=np.float64)
            self.acquisitions[channel]['bins']['path1'] = np.zeros((number_of_bins, 2), dtype = np.float64)

        self.integration_time = integration_time
        self.acquisition_channel = None
        self.index_to_aquisitions = {info["index"]: name for name, info in self.acquisitions.items()}
        self.acquisition_bin = None

        self.waveforms = sequence['waveforms'] 
        self.markers = []
        self.marker_state = 0
        self.labels = {}
        self.I = []
        self.Q = []
        self.current_i_waveform = None
        self.current_q_waveform = None
        self.current_i_end_time = 0
        self.current_q_end_time = 0
        self.current_i_start_time = 0
        self.current_q_start_time = 0
        self.index_to_name = {info["index"]: name for name, info in self.waveforms.items()}
        self.run(self.program)
        
    def advance_time(self, duration):
        """Advance I/Q waveforms for the given number of time steps."""

        G_I = convert_gain(self.awg_gain_I)
        G_Q = convert_gain(self.awg_gain_Q)

        for _ in range(duration):
            # Compute relative indices for I and Q
            i_idx = self.timing - self.current_i_start_time
            q_idx = self.timing - self.current_q_start_time

            # Helper function to get waveform values
            def get_waveform_value(waveform, idx):
                return waveform[idx] if waveform and 0 <= idx < len(waveform) else 0

            
            i_val = G_I * get_waveform_value(self.current_i_waveform, i_idx) + self.awg_offset_I
            q_val = G_Q* get_waveform_value(self.current_q_waveform, q_idx) + self.awg_offset_Q

            if self.acquire:
                # Append raw values
                self.I.append(i_val)
                self.Q.append(q_val)

                # Append scaled values to acquisition scope
                self.acquisitions[self.acquisition_channel]['scope']['path0'].append(i_val)
                self.acquisitions[self.acquisition_channel]['scope']['path1'].append(q_val)

                # (Optional) Uncomment these lines if bin accumulation is required
                # self.acquisitions[self.acquisition_channel]['bins']['path0'][self.acquisition_bin][0][0] = G_I * i_val
                # self.acquisitions[self.acquisition_channel]['bins']['path1'][self.acquisition_bin][0][0] = G_Q * q_val
            else:
                # Append only scaled waveform values when not acquiring
                self.I.append(i_val)
                self.Q.append(q_val)

            # Advance timing and record NCO phase
            self.nco_phase_stream.append(self.nco_phase)
            self.timing += 1
   

    def run(self, q1asm_code):
        lines = q1asm_code.split('\n')
        # auto detect label
        for i, line in enumerate(lines):
            line = line.strip()
            if line.endswith(':'):
                label = line[:-1]
                self.labels[label] = i

        # apply each line instructions
        pc=0
        self.markers.append((self.timing, self.marker_state))
        while pc < len(lines):
            line = lines[pc].strip()
            cleanline, numbers = q1asm_get_line_info(line)
            if not cleanline:
                pc += 1
                continue
            parts = cleanline.split()

            if parts[0].endswith(':'):
                label = parts[0][:-1]
                self.labels[label] = pc
                parts.pop(0)
            if not parts:
                pc +=1
                continue
            instruction = parts[0]
            
            if instruction == 'asl':
                self.registers[parts[3]] = self.registers[parts[1]] << numbers[0]
                pc +=1
            
            if instruction == 'asr':
                self.registers[parts[3]] = self.registers[parts[1]] >> numbers[0]
                pc +=1
                
            if instruction == 'move':
                if numbers[0] > 2**32 - 1:
                    raise Exception('Registars can only store 32 bit integers')
                self.registers[parts[2]] = numbers[0]
                if len(self.registers) > 64:
                    raise Exception('Registar locations only in range 0 to 63')
                pc += 1
                        
            elif instruction == 'add':
                a = self.registers.get(parts[1], 0)
                if not numbers:
                    b = self.registers.get(parts[2], 0)
                else: 
                    b = numbers[0]
                
                self.registers[parts[3]] = a + b
                                          
                pc += 1
                
            elif instruction == 'sub':
                
                a = self.registers.get(parts[1], 0)
                if not numbers:
                    b = self.registers.get(parts[2], 0)
                else: 
                    b = numbers[0]
                
                self.registers[parts[3]] = a - b
                
                pc += 1
            
            elif instruction == 'jmp':
                pc = self.labels[parts[1]]
            
            elif instruction == 'jge':
                reg = parts[1]
                if self.registers.get(reg, 0) >= 0:
                    pc = self.labels[parts[2]]
                else:
                    pc += 1
            elif instruction == 'jlt':
                reg = parts[1]
                immediate = numbers[0]
                jump_adress = parts[3]
                
                
                if self.registers.get(reg, 0) < immediate:
                    pc = self.labels[parts[3][1:]]
                else:
                    pc += 1
            
            elif instruction == 'loop':
                
                reg = parts[1]
                jump_adress = parts[2][1:]
                self.registers[reg] = self.registers.get(reg, 0) - 1
                if self.registers.get(reg, 0) > 0:
                    pc = self.labels[jump_adress]
                else:
                    pc += 1
            
            
            elif instruction == 'play':
                
                waveform_index_1 = numbers[0]
                waveform_index_2 = numbers[1]
                wait_time = numbers[2]
                waveform_1_name = self.index_to_name[waveform_index_1]
                waveform_2_name = self.index_to_name[waveform_index_2]
                
                self.markers.append((self.timing, self.marker_state))
                self.current_i_waveform = self.waveforms[waveform_1_name]["data"]
                self.current_q_waveform = self.waveforms[waveform_2_name]["data"]
                self.current_i_start_time = self.timing
                self.current_q_start_time = self.timing
                
                self.advance_time(wait_time)       
                pc += 1
                
            elif instruction == 'set_mrk':
                
                self.marker_state = numbers[0] if numbers else self.registers[parts[1]] 
                if self.marker_state > 15 and self.marker_state < 0:
                    raise  Exception('Marker state has to be a valid 4 bit binary number')
                
                pc += 1
                
            elif instruction == 'wait':
            
                wait_time = numbers[0] if numbers else self.registers[parts[1]]
                self.advance_time(wait_time)
                pc+=1
               
            elif instruction == 'upd_param':
                wait_time = numbers[0]
                self.markers.append((self.timing, self.marker_state))
                self.advance_time(wait_time)     
                pc += 1
            
            elif instruction =='acquire':
                
                self.markers.append((self.timing, self.marker_state))
                self.acquire = True
                acquisition_index = numbers[0]
                self.acquisition_channel = self.index_to_aquisitions[acquisition_index]
                
                if len(numbers) != 3:
                    self.acquisition_bin = self.registers[parts[2]]
                else:
                    self.acquisition_bin = numbers[1]

                duration = numbers[-1]
                
                print(duration)
                self.advance_time(duration)
           
                pc+=1 
            elif instruction =='reset_ph':
                self.nco_phase = 0
                pc+=1  

            elif instruction =='set_ph':
                Phase_number = numbers[0] if numbers else self.registers[parts[1]]
                self.nco_phase = np.deg2rad (Phase_number*360/1e9)
                pc+=1

            elif instruction =='set_awg_gain':
                if numbers:
                    self.awg_gain_I = numbers[0]
                    self.awg_gain_Q = numbers[1]
                else: 
                    self.awg_gain_I= self.registers[parts[1]]
                    self.awg_gain_Q= self.registers[parts[2]]
                pc+=1
                     
            elif instruction == 'set_awg_offs':
                if numbers:
                    self.awg_offset_I = numbers[0]
                    self.awg_offset_Q = numbers[1]
                else: 
                    self.awg_offset_I= self.registers[parts[1]]
                    self.awg_offset_Q= self.registers[parts[2]]
                pc+=1
                    
            #control instructions
            elif instruction == 'illegal':
                raise Exception('Some sort of flags are meant to be raised')
            
            elif instruction == 'nop':
                pc+=1
                
            elif instruction == 'stop':
                self.markers.append((self.timing, self.marker_state))
                break
            else:
                pc += 1
    

    def plot_channels_and_markers(self, window = [False,0,100], plot_channels = True, return_registers = False):
        if not self.I and not self.Q and not self.markers:
            print("No channels or markers to plot.")
            return
        
        if plot_channels:
            fig, ax = plt.subplots(5, 1, figsize=(10, 8), sharex= True, gridspec_kw={'height_ratios': [3,1,1,1,1]})

            ax[0].plot(self.I, label='I Channel')
            ax[0].plot(self.Q, label='Q Channel')
            ax[0].set_ylabel('Amplitude')
            ax[0].set_title('I and Q Channels')
            ax[0].legend()
            ax[0].grid(True, linestyle = '--')
            if self.markers:
                for marker_channel in range(4):
                    times = [m[0] for m in self.markers]
                    states = [(m[1] >> marker_channel) & 1 for m in self.markers]
                    ax[marker_channel+1].step(times, states, where='post',
                             label=f'Marker {marker_channel}', linestyle='-')
                    ax[marker_channel+1].set_ylabel('State')
                    ax[marker_channel+1].legend()
    #                 ax[marker_channel+1].set_title(f'Marker {marker_channel+1}')

            for a in ax:
                a.grid(True, which='both', axis='x', linestyle = '--')
                if window[0]:
                    a.set_xlim(window[1],window[2])


            plt.tight_layout()
            plt.show()

        if return_registers:
            plt.figure(figsize=(10,4))
            plt.bar(self.registers.keys(), self.registers.values())
            plt.title('Final Register Values')
            plt.xticks(rotation = 'vertical')
            plt.xlabel('Register')
            plt.ylabel('Final stored value')
            plt.grid(axis='y', linestyle = '--')
            plt.tight_layout()
            plt.show()