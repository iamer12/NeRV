import gc

import math
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from datetime import datetime

import os


###############################################
# snerv
# Adding some utility functions to evaluate various modes of quantization with the scalability feature
###############################################

def quantize_per_tensor_mdlns(t, bit=8, first_base=2, sec_base=3, sec_base_bits=3, start=0.1, end=5.0, step=0.1, auto_scale=1, axis=-1):

    #t_valid = t!=0
    #t_min, t_max =  t[t_valid].min(), t[t_valid].max()
    # removed this line because it would eventually trigger a run-time error when passed input tensor is all zeros. Keeping it like this likely leads to inputs traversing to outputs untouched

    qt = torch.empty_like(t)
    nt = torch.empty_like(t)

    c_qt = torch.empty_like(t)
    c_nt = torch.empty_like(t)
   
    first_base_bits = bit - sec_base_bits - 1

    bx_range = get_signed_range(first_base_bits)
    tx_range = get_signed_range(sec_base_bits)  # For easiness, I called it tx as "trenary exponent of x". It is known that it does not have to be "trenary" per se, and that the second (non binary) exponent can be anything

    # Compose conversion LUT
    size_of_lut = (bx_range.stop-bx_range.start) * (tx_range.stop-tx_range.start)
    conv_lut = [0] * size_of_lut    # array for the conversion value of (2 ** bx) * (sec_base ** tx)
    b_lut = [0] * size_of_lut       # arrary for the binary (first) exponent
    t_lut = [0] * size_of_lut       # array for the trenary (second) exponent
    
    first_base_selected = first_base
    sec_base_selected = sec_base
    if first_base != 1000 and sec_base != 1000: # both first and second bases are given
        index = 0
        for bx in bx_range:
            for tx in tx_range:
                conv_lut[index] = (first_base_selected ** bx) * (sec_base_selected ** tx)
                b_lut[index] = bx
                t_lut[index] = tx
                index = index + 1
        qt, nt = map_range(t, conv_lut, b_lut, t_lut, bit, sec_base_bits, auto_scale)
    elif first_base != 1000 and sec_base == 1000:   #first base is given. sweep second base only.
        candidate_sec_base = start
        max_qsnr = float('-inf')
        while candidate_sec_base <= end:
            index = 0
            for bx in bx_range:
                for tx in tx_range:
                    conv_lut[index] = (first_base_selected ** bx) * (candidate_sec_base ** tx)
                    b_lut[index] = bx
                    t_lut[index] = tx
                    index = index + 1
            c_qt, c_nt, sorted = map_range(t, conv_lut, b_lut, t_lut, bit, sec_base_bits, auto_scale)

            c_qsnr = calc_qsnr(t, c_nt)
            if c_qsnr > max_qsnr:
                max_qsnr = c_qsnr
                sec_base_selected = candidate_sec_base
                qt = c_qt
                nt = c_nt

            candidate_sec_base = candidate_sec_base + step
    elif first_base == 1000 and sec_base != 1000:   #second base is given. sweep first base only.
        candidate_first_base = start
        max_qsnr = float('-inf')
        while candidate_first_base <= end:
            index = 0
            for bx in bx_range:
                for tx in tx_range:
                    conv_lut[index] = (candidate_first_base ** bx) * (sec_base_selected ** tx)
                    b_lut[index] = bx
                    t_lut[index] = tx
                    index = index + 1
            c_qt, c_nt = map_range(t, conv_lut, b_lut, t_lut, bit, sec_base_bits, auto_scale)

            c_qsnr = calc_qsnr(t, c_nt)
            if c_qsnr > max_qsnr:
                max_qsnr = c_qsnr
                first_base_selected = candidate_first_base
                qt = c_qt
                nt = c_nt

            candidate_first_base = candidate_first_base + step
    else:   # both bases to be sweeped
        max_qsnr = float('-inf')
        candidate_first_base = start
        while candidate_first_base <= end:
            candidate_sec_base = start
            while candidate_sec_base <= end:
                index = 0
                for bx in bx_range:
                    for tx in tx_range:
                        conv_lut[index] = (candidate_first_base ** bx) * (candidate_sec_base ** tx)
                        b_lut[index] = bx
                        t_lut[index] = tx
                        index = index + 1
                c_qt, c_nt = map_range(t, conv_lut, b_lut, t_lut, bit, sec_base_bits, auto_scale)

                c_qsnr = calc_qsnr(t, c_nt)
                if c_qsnr > max_qsnr:
                    max_qsnr = c_qsnr
                    first_base_selected = candidate_first_base
                    sec_base_selected = candidate_sec_base
                    qt = c_qt
                    nt = c_nt

                candidate_sec_base = candidate_sec_base + step
            candidate_first_base = candidate_first_base + step

    
    del c_qt, c_nt, conv_lut, b_lut, t_lut

    gc.collect()

    return first_base_selected, sec_base_selected, qt, nt, sorted
###############################################
def calc_qsnr(original: torch.Tensor, noisy: torch.Tensor) -> float:
    
    # Calculates QSNR (Quantized Signal-to-Noise Ratio) in dB.
    # QSNR = 10 * log10 (signal_power / noise_power)

    # Parameters:
    #     original (torch.Tensor): Original clean signal.
    #     noisy (torch.Tensor): Noisy signal.

    # Returns:
    #     float: QSNR value in decibels.
    
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean((original - noisy) ** 2)

    if noise_power == 0:
        return float('inf')  # Perfect reconstruction, infinite QSNR
    
    qsnr = 10 * torch.log10(signal_power / noise_power)
    return qsnr.item()
###############################################
# def map_range(t, conv_lut, b_lut, nb_lut, bit=8, sec_base_bits=3, auto_scale=0):
#     # t --> tensor, b --> binary, nb --> non-binary, conv --> integer to MDLNS conversion LUT

#     conv_lut_signed = torch.tensor(conv_lut + [-x for x in conv_lut]) # conv_lut_signed  contains all the elements from conv_lut plus their negative counterparts
#     # Notice that the above line also converts from conv_lut array to conv_lut_signed tensor
    
#     b_lut_expanded = torch.tensor(b_lut + b_lut)
#     nb_lut_expanded = torch.tensor(nb_lut + nb_lut)

#     if auto_scale == 1:
#         t_min, t_max = t.min(), t.max()
#         lut_min, lut_max = conv_lut_signed.min(), conv_lut_signed.max()

#         # Scale t to fit within the range of conv_lut_signed
#         t_scaled = (t - t_min) / (t_max - t_min)
#         t_scaled = t_scaled * (lut_max - lut_min) + lut_min

#         # Compute differences and get indices of closest LUT entries
#         diff = torch.abs(t_scaled.unsqueeze(-1) - conv_lut_signed)
#         indices = torch.argmin(diff, dim=-1)  # shape: (B, L)

#         # Quantized values using those indices
#         t_dequantized = conv_lut_signed[indices]  # shape: (B, L)
        
#         # Reconstruct t in the original range
#         # Reverse scaling: map from [lut_min, lut_max] back to [t_min, t_max]
#         t_reconstructed = (t_dequantized - lut_min) / (lut_max - lut_min)
#         t_reconstructed = t_reconstructed * (t_max - t_min) + t_min
    
#         # Adding code
#         b_dequantized = b_lut_expanded[indices]
#         nb_dequantized = nb_lut_expanded[indices]
    
#     else:        
#         # Compute differences and get indices of closest LUT entries
#         diff = torch.abs(t.unsqueeze(-1) - conv_lut_signed)
#         indices = torch.argmin(diff, dim=-1)  # shape: (B, L)

#         # Quantized values using those indices - could have been one assignment straight to t_reconstructed
#         t_dequantized = conv_lut_signed[indices]  # shape: (B, L)
#         t_reconstructed = t_dequantized
        
#         # Adding code
#         b_dequantized = b_lut_expanded[indices]
#         nb_dequantized = nb_lut_expanded[indices]
        

#     bin_base_bits = bit - sec_base_bits - 1
#     qt = encode_qt_vector(t, b_dequantized, nb_dequantized, bin_base_bits, sec_base_bits)
#     #notice that I am passing "t" as the first argument to use it to extract the signs later in the function


#     del b_dequantized
#     del nb_dequantized
#     del t_dequantized
#     del diff
#     del indices
#     del conv_lut_signed
#     del b_lut_expanded
#     del nb_lut_expanded
    

#     gc.collect()

#     return qt, t_reconstructed


def map_range(t, conv_lut, b_lut, nb_lut, bit=8, sec_base_bits=3, auto_scale=0, chunk_size=512):
    # Convert LUTs to float32 or int32 tensors
    conv_lut_signed = torch.tensor(conv_lut + [-x for x in conv_lut], dtype=torch.float32)

    sorted_conv_lut_signed, _ = torch.sort(conv_lut_signed)

    b_lut_expanded = torch.tensor(b_lut + b_lut, dtype=torch.int32)
    nb_lut_expanded = torch.tensor(nb_lut + nb_lut, dtype=torch.int32)

    t = t.to(torch.float32)

    if auto_scale == 1:
        t_min, t_max = t.min(), t.max()
        lut_min, lut_max = conv_lut_signed.min(), conv_lut_signed.max()

        # Scale t to fit within the LUT range
        t_scaled = (t - t_min) / (t_max - t_min)
        t_scaled = t_scaled * (lut_max - lut_min) + lut_min
    else:
        t_scaled = t  # no scaling needed

    # Memory-safe chunked diff + argmin
    # indices_list = []
    # for i in range(0, t_scaled.size(0), chunk_size):
    #     t_chunk = t_scaled[i:i+chunk_size]
    #     diff_chunk = torch.abs(t_chunk.unsqueeze(-1) - conv_lut_signed)  # shape: (chunk_size, N)
    #     idx_chunk = torch.argmin(diff_chunk, dim=-1)
    #     indices_list.append(idx_chunk)

    # indices = torch.cat(indices_list, dim=0)


    #indices = safe_argmin_diff(t_scaled, conv_lut_signed, chunk_size_t=256, chunk_size_lut=1024)
    indices = safe_argmin_diff(t_scaled, conv_lut_signed, chunk_size_t=4096, chunk_size_lut=8192)



    # Quantized values using LUT
    t_dequantized = conv_lut_signed[indices]

    # Reconstruct original range if scaled
    if auto_scale == 1:
        t_reconstructed = (t_dequantized - lut_min) / (lut_max - lut_min)
        t_reconstructed = t_reconstructed * (t_max - t_min) + t_min
    else:
        t_reconstructed = t_dequantized

    # Gather bit representations
    b_dequantized = b_lut_expanded[indices]
    nb_dequantized = nb_lut_expanded[indices]

    bin_base_bits = bit - sec_base_bits - 1
    qt = encode_qt_vector(t, b_dequantized, nb_dequantized, bin_base_bits, sec_base_bits)

    # Clean up
    del b_dequantized, nb_dequantized, t_dequantized, indices, conv_lut_signed
    del b_lut_expanded, nb_lut_expanded, t_scaled
    gc.collect()

    return qt, t_reconstructed, sorted_conv_lut_signed



###############################################

def safe_argmin_diff(t_scaled, conv_lut_signed, chunk_size_t=256, chunk_size_lut=1024):
    original_shape = t_scaled.shape
    t_flat = t_scaled.view(-1)  # Flatten to 1D

    indices_list = []
    for i in range(0, t_flat.size(0), chunk_size_t):
        t_chunk = t_flat[i:i+chunk_size_t]  # (chunk_size,)
        best_diff = None
        best_idx = None

        for j in range(0, conv_lut_signed.size(0), chunk_size_lut):
            lut_chunk = conv_lut_signed[j:j+chunk_size_lut]  # shape: (lut_chunk_size,)
            diff_chunk = torch.abs(t_chunk.unsqueeze(-1) - lut_chunk)  # shape: (chunk_size, lut_chunk_size)
            idx_chunk = torch.argmin(diff_chunk, dim=-1)  # shape: (chunk_size,)
            val_chunk = diff_chunk[torch.arange(diff_chunk.size(0)), idx_chunk]

            if best_diff is None:
                best_diff = val_chunk
                best_idx = idx_chunk + j
            else:
                update_mask = val_chunk < best_diff
                best_diff[update_mask] = val_chunk[update_mask]
                best_idx[update_mask] = idx_chunk[update_mask] + j

            del diff_chunk, idx_chunk, val_chunk, lut_chunk
            gc.collect()

        indices_list.append(best_idx)

    indices_flat = torch.cat(indices_list, dim=0)
    return indices_flat.view(original_shape)  # Reshape back to match input

###############################################

def signed_to_unsigned_vector(val, bits):
    # Convert signed integers to unsigned using two's complement, vectorized
    mask = val < 0
    result = val.clone()
    result[mask] = (1 << bits) + val[mask]
    return result
###############################################
def encode_qt_vector(signt, bt_min, tt_min, bin_base_bits, sec_base_bits):
    sign_bit = (signt < 0).int()  # 1 if negative, else 0

    bt_unsigned = signed_to_unsigned_vector(bt_min, bin_base_bits)
    tt_unsigned = signed_to_unsigned_vector(tt_min, sec_base_bits)

    qt = (sign_bit << (bin_base_bits + sec_base_bits)) | \
         (bt_unsigned << sec_base_bits) | \
         tt_unsigned

    return qt
###############################################
# Compute ranges for signed integers
def get_signed_range(bits):
    min_val = -2**(bits - 1)
    max_val = 2**(bits - 1) - 1
    return range(min_val, max_val + 1)
###############################################
###############################################
###############################################
def quantize_per_tensor(t, bit=8, axis=-1):
    if axis == -1:
        t_valid = t!=0
        #t_min, t_max =  t[t_valid].min(), t[t_valid].max()
        # replaced the above line with this one below because it would eventually trigger a run-time error when passed input tensor is all zeros. Keeping it like this likely leads to inputs traversing to outputs untouched
        t_min, t_max =  t.min(), t.max() 
        
        scale = (t_max - t_min) / 2**bit
    elif axis == 0:
        min_max_list = []
        for i in range(t.size(0)):
            t_valid = t[i]!=0
            if t_valid.sum():
                min_max_list.append([t[i][t_valid].min(), t[i][t_valid].max()])
            else:
                min_max_list.append([0, 0])
        min_max_tf = torch.tensor(min_max_list).to(t.device)        
        scale = (min_max_tf[:,1] - min_max_tf[:,0]) / 2**bit
        if t.dim() == 4:
            scale = scale[:,None,None,None]
            t_min = min_max_tf[:,0,None,None,None]
        elif t.dim() == 2:
            scale = scale[:,None]
            t_min = min_max_tf[:,0,None]
    elif axis == 1:
        min_max_list = []
        for i in range(t.size(1)):
            t_valid = t[:,i]!=0
            if t_valid.sum():
                min_max_list.append([t[:,i][t_valid].min(), t[:,i][t_valid].max()])
            else:
                min_max_list.append([0, 0])
        min_max_tf = torch.tensor(min_max_list).to(t.device)             
        scale = (min_max_tf[:,1] - min_max_tf[:,0]) / 2**bit
        if t.dim() == 4:
            scale = scale[None,:,None,None]
            t_min = min_max_tf[None,:,0,None,None]
        elif t.dim() == 2:
            scale = scale[None,:]
            t_min = min_max_tf[None,:,0]            
    
    # import pdb; pdb.set_trace; from IPython import embed; embed()       
    quant_t = ((t - t_min) / (scale + 1e-19)).round()
    new_t = t_min + scale * quant_t
    return quant_t, new_t
    
def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


def all_reduce(tensors, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.
    """

    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors



################################################################


#def quantize_to_lns(tensor, bit, lns_base, exp_bits):
def quantize_per_tensor_lns(tensor, lns_base=2, exp_bits=3):
    
    tensor = tensor.float()
    sign_bit = (tensor < 0).to(torch.uint8)
    tensor_abs = tensor.abs()

    # def compute_qsnr(original, reconstructed):
    #     signal_power = torch.sum(original**2)
    #     error_power = torch.sum((original - reconstructed)**2) + 1e-12  # avoid division by zero
    #     return 10 * torch.log10(signal_power / error_power)

    def compute_qsnr(original, reconstructed, eps=1e-12):
        signal_power = torch.sum(original**2)
        error_power = torch.sum((original - reconstructed)**2) + eps  # safe epsilon for stability

        # If signal power is too small, define QSNR as 0 (or some fallback)
        if signal_power < eps:
            #return torch.tensor(0.0, device=original.device)
            return torch.tensor(-1000.0, device=original.device)

        ratio = signal_power / error_power

        # Clamp ratio to avoid taking log(0) or log(negative)
        ratio = torch.clamp(ratio, min=eps)

        return 10 * torch.log10(ratio)


    def quantize_lns(tensor_abs, base, exp_bits):
        log_tensor = torch.log(tensor_abs + 1e-12) / torch.log(torch.tensor(base))
        exp_max = 2**exp_bits - 1
        exp_min = 0

        log_min = log_tensor.min()
        log_max = log_tensor.max()
        scale = (log_max - log_min) if (log_max - log_min) != 0 else 1.0

        norm_log_tensor = (log_tensor - log_min) / scale * exp_max
        #quantized_exp = torch.clamp(norm_log_tensor.round(), exp_min, exp_max).to(torch.uint32)
        quantized_exp = torch.clamp(norm_log_tensor.round(), exp_min, exp_max).to(torch.int32)

        # Dequantize
        dequant_log = quantized_exp.float() / exp_max * scale + log_min
        tensor_dequant = base ** dequant_log
        
        return quantized_exp, tensor_dequant

    test = tensor_abs.max()/tensor_abs.min()
    if lns_base == 1000:
        
        best_qsnr = -float('inf')
        #base_candidates = torch.linspace(1.1, 10, 200)
        #base_candidates = torch.linspace(0.1, 10, 200)
        
        #Initialize
        # base_selected = 2.0
        # quantized_exp_init, tensor_dequant_init = quantize_lns(tensor_abs, 2.0, exp_bits)
        # best_quantized_exp = quantized_exp_init
        # best_tensor_dequant = tensor_dequant_init

        base_candidates = torch.exp(torch.linspace(torch.log(torch.tensor(1.2)), torch.log(torch.tensor(3.5)), 200))

        for base_candidate in base_candidates:
            quantized_exp, tensor_dequant_candidate = quantize_lns(tensor_abs, base_candidate.item(), exp_bits)
            qsnr = compute_qsnr(tensor_abs, tensor_dequant_candidate)
            if qsnr > best_qsnr:
                best_qsnr = qsnr
                base_selected = base_candidate.item()
                best_quantized_exp = quantized_exp
                best_tensor_dequant = tensor_dequant_candidate
    else:
        base_selected = lns_base
        best_quantized_exp, best_tensor_dequant = quantize_lns(tensor_abs, base_selected, exp_bits)

    # Build codewords
    #codewords = (sign_bit.to(torch.uint32) << exp_bits) | best_quantized_exp
    codewords = (sign_bit.to(torch.int32) << exp_bits) | best_quantized_exp

    # Restore sign
    tensor_dequant = best_tensor_dequant * (1 - 2 * sign_bit.float())

    return base_selected, codewords, tensor_dequant

################################################################

def quantize_per_tensor_minifloat(tensor, bit=8, exp_bits=4):
    assert bit > exp_bits + 1, "Total bit width must be greater than exponent bits + 1 for sign."

    mantissa_bits = bit - exp_bits - 1
    exp_bias = 2**(exp_bits - 1) - 1  # Bias for exponent

    codewords = []
    dequantized = []

    # Loop through each element in the tensor
    for val in tensor.view(-1):
        val = val.item()  # Convert tensor element to Python float

        # Handle special case zero
        if val == 0.0:
            codewords.append(0)  # 0 bitfield for zero
            dequantized.append(0.0)
            continue

        sign_bit = 0 if val >= 0 else 1
        val = abs(val)

        exp = int(torch.floor(torch.log2(torch.tensor(val))))  # Use PyTorch log2 function
        mant = val / (2 ** exp) - 1.0  # Normalized mantissa (between 0 and 1)

        exp_q = exp + exp_bias
        if exp_q <= 0:
            # Underflow to zero (no subnormals in this basic implementation)
            codewords.append(0)  # 0 bitfield for underflow
            dequantized.append(0.0)
            continue
        elif exp_q >= 2**exp_bits - 1:
            # Overflow to max representable value
            exp_q = 2**exp_bits - 1
            mant_q = (1 - 2 ** -mantissa_bits)
        else:
            mant_q = round(float(mant * (2 ** mantissa_bits))) / (2 ** mantissa_bits)

        # Pack codeword into integer bitfield
        exp_bits_int = int(exp_q)
        mant_bits_int = int(mant_q * (2 ** mantissa_bits))
        codeword = (sign_bit << (bit - 1)) | (exp_bits_int << mantissa_bits) | mant_bits_int
        codewords.append(codeword)

        # Dequantization
        quantized_val = ((-1)**sign_bit) * (1 + mant_q) * (2 ** (exp_q - exp_bias))
        dequantized.append(quantized_val)

    # Return a tensor of integer codewords and dequantized floating point values
    return torch.tensor(codewords, dtype=torch.int32).view(tensor.shape), torch.tensor(dequantized).view(tensor.shape)

################################################################
################################################################
################################################################
################################################################


# def maybe_plot_quantization(input_tensor, quantized_tensor, flag):
#     """
#     Plots quantized values and quantization errors if flag == 1.
    
#     Args:
#         input_tensor (torch.Tensor or np.ndarray): The original input tensor (full precision).
#         quantized_tensor (torch.Tensor or np.ndarray): The corresponding quantized tensor.
#         flag (int): 0 means do nothing, 1 means plot the curves.
#     """
#     if flag == 0:
#         return  # do nothing

#     # Convert to numpy if needed
#     if hasattr(input_tensor, 'detach'):
#         input_tensor = input_tensor.detach().cpu().numpy()
#     if hasattr(quantized_tensor, 'detach'):
#         quantized_tensor = quantized_tensor.detach().cpu().numpy()

#     # Flatten in case tensors are not 1D
#     input_tensor = input_tensor.flatten()
#     quantized_tensor = quantized_tensor.flatten()

#     # Sort the input values for cleaner plotting
#     sort_indices = np.argsort(input_tensor)
#     input_sorted = input_tensor[sort_indices]
#     quantized_sorted = quantized_tensor[sort_indices]

#     # Compute the error
#     error = input_sorted - quantized_sorted

#     # Plot
#     plt.figure(figsize=(10, 5))
#     plt.plot(input_sorted, quantized_sorted, label="Quantized Value", marker='.', linestyle='-', markersize=2)
#     plt.plot(input_sorted, error, label="Quantization Error", marker='.', linestyle='--', markersize=2)
#     plt.xlabel("Original Input Value")
#     plt.ylabel("Value")
#     plt.title("Quantization vs Error")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()



global_fig_counter = [1]  # Global figure counter

def maybe_plot_quantization(input_tensor, quantized_tensor, flag, qmode='integer', lns_base=2, mdlns_base=3, output_dir='output_plots', run_id=None):
    if flag == 1:


        if qmode == 'integer':
            st1 = 'Integer'
        elif qmode == 'minifloat':
            st1 = 'Minifloat'
        elif qmode == 'lns':
            if lns_base == 1000:
                st1 = 'LNS Sweep'
            else:
                st1 = 'LNS (2)'
        elif qmode == 'mdlns':
            if mdlns_base == 1000:
                st1 = 'MDLNS Sweep (2,X)'
            else:
                st1 = 'MDLNS (2,3)'

        
        if global_fig_counter[0] == 1:
            st2 = 'Base Layer'
        elif global_fig_counter[0] == 2:
            st2 = 'Enhancement Layer #1'
        elif global_fig_counter[0] == 3:
            st2 = 'Enhancement Layer #2'


        input_flat = input_tensor.flatten().cpu().numpy()
        quant_flat = quantized_tensor.flatten().cpu().numpy()

        # Create output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        fig, ax1 = plt.subplots()

        # Plot quantized values vs input values
        ax1.plot(input_flat, quant_flat, '.', markersize=2, label='Quantized vs Input')
        ax1.set_xlabel('Input Value')
        ax1.set_ylabel('Quantized Value')
        ax1.grid(True)

        # Plot error vs input values on secondary axis
        ax2 = ax1.twinx()
        error = abs(quant_flat - input_flat)
        ax2.plot(input_flat, error, '.', markersize=2, alpha=0.5, color='r', label='Abs Error')
        ax2.set_ylabel('Abs Error')
         
         # Set the y-axis range for the error plot
        max_abs_input = abs(input_flat).max()
        ax2.set_ylim(0, max_abs_input)

        #fig.suptitle(f'Quantization Plot #{global_fig_counter[0]}')
        fig.suptitle(f'{st1} Quantization Plot for {st2}')

        # Build filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_id is not None:
            #base_filename = f'figure_{global_fig_counter[0]}_run{run_id}_{timestamp}'
            base_filename = f'{st1}_{st2}_figure_{global_fig_counter[0]}_run{run_id}_{timestamp}'
        else:
            #base_filename = f'figure_{global_fig_counter[0]}_{timestamp}'
            base_filename = f'{st1}_{st2}_figure_{global_fig_counter[0]}_{timestamp}'

        png_path = os.path.join(output_dir, base_filename + '.png')
        pdf_path = os.path.join(output_dir, base_filename + '.pdf')

        # Save as PNG and PDF
        fig.savefig(png_path, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)  # Free memory

        print(f"Saved {png_path} and {pdf_path}")

        global_fig_counter[0] += 1

################################################################
################################################################


# def plot_tensor_histogram(tensor, title="Tensor Histogram", flag=1, save_path=None):
#     """
#     Plots a histogram of the values inside a tensor.

#     Args:
#         tensor (torch.Tensor): Input tensor (1D).
#         title (str): Title of the plot.
#         flag (int): If 0, do nothing. If 1, plot the histogram.
#         save_path (str or None): If provided, saves the figure to this path.
#     """
#     if flag == 0:
#         return

#     tensor_np = tensor.cpu().numpy()  # Convert to numpy array if needed
#     num_bins = min(50, max(10, tensor_np.shape[0] // 10))  # Adaptive number of bins

#     plt.figure(figsize=(8, 4))
#     plt.hist(tensor_np, bins=num_bins, edgecolor='black', alpha=0.75)
#     plt.title(title)
#     plt.xlabel('Value')
#     plt.ylabel('Frequency')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.tight_layout()

#     if save_path is not None:
#         plt.savefig(save_path, dpi=300)  # Save figure at high quality
#     plt.show()


import matplotlib.pyplot as plt
import torch
import numpy as np

# def plot_tensor_histogram(
#     tensor,
#     title="Tensor Histogram",
#     flag=1,
#     save_path=None,
#     new_tensor_int=None,
#     new_tensor_mdlns=None
# ):
    
# def plot_tensor_histogram(
#     tensor,
#     new_tensor_int=None,
#     new_tensor_mdlns=None,
#     title="Tensor Histogram",
#     flag=1,
#     save_path=None
# ):
#     """
#     Plots a histogram of the values inside a tensor, optionally overlaying
#     codeword distributions for INT and MDLNS quantizations as vertical lines.

#     Args:
#         tensor (torch.Tensor): Input tensor (1D).
#         title (str): Title of the plot.
#         flag (int): If 0, do nothing. If 1, plot the histogram.
#         save_path (str or None): If provided, saves the figure to this path.
#         new_tensor_int (torch.Tensor or None): Reconstructed tensor using INT quantization.
#         new_tensor_mdlns (torch.Tensor or None): Reconstructed tensor using MDLNS quantization.
#     """
#     if flag == 0:
#         return

#     tensor_np = tensor.cpu().numpy()
#     num_bins = min(50, max(10, tensor_np.shape[0] // 10))  # Adaptive number of bins

#     plt.figure(figsize=(10, 5))

#     # Plot histogram of the original tensor
#     plt.hist(tensor_np, bins=num_bins, edgecolor='black', alpha=0.6, color='gray', label='Original Tensor')


#     x_min, x_max = tensor_np.min(), tensor_np.max()
#     plt.xlim(x_min, x_max)

#     # Overlay INT codeword vertical lines
#     if new_tensor_int is not None:
#         new_int_np = new_tensor_int.cpu().numpy()
#         unique_int_values = np.unique(new_int_np)
#         for x in unique_int_values:
#             plt.vlines(x, ymin=0, ymax=0.5, colors='blue', linestyles='dashed', linewidth=1)

#     # Overlay MDLNS codeword vertical lines
#     if new_tensor_mdlns is not None:
#         new_mdlns_np = new_tensor_mdlns.cpu().numpy()
#         unique_mdlns_values = np.unique(new_mdlns_np)
#         for x in unique_mdlns_values:
#             plt.vlines(x, ymin=0, ymax=0.7, colors='red', linestyles='solid', linewidth=1)

#     plt.title(title)
#     plt.xlabel('Value')
#     plt.ylabel('Frequency')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend(["Original Tensor", "INT Codewords", "MDLNS Codewords"])
#     plt.tight_layout()

#     if save_path is not None:
#         plt.savefig(save_path, dpi=300)

#     plt.show()



# def plot_tensor_histogram(
#     tensor,
#     new_tensor_int=None,
#     new_tensor_mdlns=None,
#     title="Tensor Histogram",
#     flag=1,
#     save_path=None
# ):
#     """
#     Plots a histogram of the values inside a tensor, optionally overlaying
#     codeword distributions for INT and MDLNS quantizations as vertical lines.

#     Args:
#         tensor (torch.Tensor): Input tensor (1D).
#         new_tensor_int (torch.Tensor or None): Reconstructed tensor using INT quantization.
#         new_tensor_mdlns (torch.Tensor or None): Reconstructed tensor using MDLNS quantization.
#         title (str): Title of the plot.
#         flag (int): If 0, do nothing. If 1, plot the histogram.
#         save_path (str or None): If provided, saves the figure to this path.
#     """
#     if flag == 0:
#         return

#     tensor_np = tensor.cpu().numpy()
#     num_bins = min(50, max(10, tensor_np.shape[0] // 10))  # Adaptive number of bins

#     plt.figure(figsize=(10, 5))

#     # Plot histogram and capture histogram max height
#     n, bins, patches = plt.hist(tensor_np, bins=num_bins, edgecolor='black', alpha=0.6, color='gray')
#     hist_max_height = np.max(n)

#     # Fix x-axis range to match tensor
#     x_min, x_max = tensor_np.min(), tensor_np.max()
#     plt.xlim(x_min, x_max)

#     # Overlay INT codeword vertical lines
#     if new_tensor_int is not None:
#         new_int_np = new_tensor_int.cpu().numpy()
#         unique_int_values = np.unique(new_int_np)
#         for x in unique_int_values:
#             plt.vlines(x, ymin=0, ymax=4 * hist_max_height, colors='blue', linestyles='dashed', linewidth=1)

#     # Overlay MDLNS codeword vertical lines
#     if new_tensor_mdlns is not None:
#         new_mdlns_np = new_tensor_mdlns.cpu().numpy()
#         unique_mdlns_values = np.unique(new_mdlns_np)
#         for x in unique_mdlns_values:
#             plt.vlines(x, ymin=0, ymax=4 * hist_max_height, colors='red', linestyles='solid', linewidth=1)

#     plt.title(title)
#     plt.xlabel('Value')
#     plt.ylabel('Frequency')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend(["Original Tensor", "INT Codewords", "MDLNS Codewords"], loc='upper right')
#     plt.tight_layout()

#     if save_path is not None:
#         plt.savefig(save_path, dpi=300)

#     plt.show()



def plot_tensor_histogram(
    tensor,
    new_tensor_mdlns=None,
    quant_bits=None,
    title="Tensor Histogram",
    flag=1,
    save_path=None
):
    """
    Plots a histogram of the values inside a tensor, optionally overlaying
    codeword distributions for INT (uniform) and MDLNS (reconstructed) quantizations.

    Args:
        tensor (torch.Tensor): Input tensor (1D).
        new_tensor_mdlns (torch.Tensor or None): Reconstructed tensor using MDLNS quantization.
        quant_bits (int or None): Number of bits for uniform quantization (for INT).
        title (str): Title of the plot.
        flag (int): If 0, do nothing. If 1, plot the histogram and codeword distributions.
        save_path (str or None): If provided, saves the figure to this path.
    """
    if flag == 0:
        return

    tensor_np = tensor.cpu().numpy()
    num_bins = min(50, max(10, tensor_np.shape[0] // 10))  # Adaptive number of bins

    plt.figure(figsize=(10, 5))

    # Plot histogram of the original tensor
    plt.hist(tensor_np, bins=num_bins, edgecolor='black', alpha=0.6, color='gray', label='Original Tensor')

    x_min, x_max = tensor_np.min(), tensor_np.max()
    plt.xlim(x_min, x_max)

    hist_max = plt.gca().get_ylim()[1]

    # Plot MDLNS codeword sticks
    if new_tensor_mdlns is not None:
        new_mdlns_np = new_tensor_mdlns.cpu().numpy()
        unique_mdlns_values = np.unique(new_mdlns_np)
        for x in unique_mdlns_values:
            #plt.vlines(x, ymin=hist_max * 1.05, ymax=hist_max * 1.25, colors='red', linestyles='solid', linewidth=1)
            plt.vlines(x, ymin=hist_max * 1.05, ymax=hist_max * 1.15, colors='red', linestyles='solid', linewidth=1)

    # Plot INT uniform codeword sticks
    if quant_bits is not None:
        num_codewords = 2 ** quant_bits
        uniform_codewords = np.linspace(x_min, x_max, num_codewords)
        for x in uniform_codewords:
            #plt.vlines(x, ymin=hist_max * 1.30, ymax=hist_max * 1.50, colors='blue', linestyles='dashed', linewidth=1)
            plt.vlines(x, ymin=hist_max * 1.25, ymax=hist_max * 1.35, colors='blue', linestyles='dashed', linewidth=1)

    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)

    # plt.legend(["Binned Histogram of Sample First Enhacement Layer Input Tensor", "MDLNS Sweep (2,X) Codewords", "INT Codewords"], loc='center left')
    # plt.tight_layout()

    # plt.legend(
    # ["Binned Histogram of Sample First Enhancement Layer Input Tensor", 
    #     "MDLNS Sweep (2,X) Codewords", 
    #     "INT Codewords"],
    # loc='center left',
    # bbox_to_anchor=(0, 0.5, 0.33, 0.5),
    # frameon=True,
    # borderaxespad=0.5,
    # handletextpad=1.0,
    # fontsize='small'
    # )

    # plt.legend(
    # ["Binned Histogram of Sample\nFirst Enhancement Layer Input Tensor", 
    #     "MDLNS Sweep (2,X) Codewords", 
    #     "INT Codewords"],
    # loc='center left'
    # )


    # Create custom legend handles
    #hist_handle = mlines.Line2D([], [], color='gray', linewidth=10, label="Binned Histogram of Sample\nFirst Enhancement Layer Input Tensor")
    hist_handle = mlines.Line2D([], [], color='gray', linewidth=10, label="Binned Histogram of Sample Base Layer Input Tensor")
    mdlns_handle = mlines.Line2D([], [], color='red', linestyle='solid', linewidth=1, label="MDLNS Sweep (2,X) Codewords")
    int_handle = mlines.Line2D([], [], color='blue', linestyle='dashed', linewidth=1, label="INT Codewords")

    plt.legend(
        handles=[hist_handle, mdlns_handle, int_handle],
        loc='center left'
    )



    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()



# def plot_tensor_histogram(
#     tensor,
#     new_tensor_mdlns=None,
#     quant_bits=None,
#     title="Tensor Histogram",
#     flag=1,
#     save_path=None
# ):
#     """
#     Plots a histogram of the values inside a tensor, and optionally overlays
#     codeword distributions for INT (uniform) and MDLNS (actual) quantizations
#     as short sticks stacked above the histogram.

#     Args:
#         tensor (torch.Tensor): Input tensor (1D).
#         new_tensor_mdlns (torch.Tensor or None): Reconstructed tensor using MDLNS quantization.
#         quant_bits (int or None): Number of bits for INT codeword generation (2^quant_bits sticks).
#         title (str): Title of the plot.
#         flag (int): If 0, do nothing. If 1, plot the histogram and overlays.
#         save_path (str or None): If provided, saves the figure to this path.
#     """
#     if flag == 0:
#         return

#     tensor_np = tensor.cpu().numpy()
#     num_bins = min(50, max(10, tensor_np.shape[0] // 10))  # Adaptive number of bins

#     plt.figure(figsize=(10, 6))

#     # Plot histogram and capture histogram max height
#     n, bins, patches = plt.hist(tensor_np, bins=num_bins, edgecolor='black', alpha=0.6, color='gray')
#     hist_max_height = np.max(n)

#     # Fix x-axis range to match tensor
#     x_min, x_max = tensor_np.min(), tensor_np.max()
#     plt.xlim(x_min, x_max)

#     # Calculate small offsets
#     offset_mdlns = hist_max_height * 1.05
#     offset_int = hist_max_height * 1.15
#     stick_height = hist_max_height * 0.05  # Short stick

#     # Overlay MDLNS codeword vertical sticks
#     if new_tensor_mdlns is not None:
#         new_mdlns_np = new_tensor_mdlns.cpu().numpy()
#         unique_mdlns_values = np.unique(new_mdlns_np)
#         for x in unique_mdlns_values:
#             plt.vlines(x, ymin=offset_mdlns, ymax=offset_mdlns + stick_height, colors='red', linestyles='solid', linewidth=1)

#     # Overlay INT codeword vertical sticks (uniform spacing)
#     if quant_bits is not None:
#         num_codewords = 2 ** quant_bits
#         int_codewords = np.linspace(x_min, x_max, num_codewords)
#         for x in int_codewords:
#             plt.vlines(x, ymin=offset_int, ymax=offset_int + stick_height, colors='blue', linestyles='dashed', linewidth=1)

#     # Labels and grid
#     plt.title(title)
#     plt.xlabel('Value')
#     plt.ylabel('Frequency')
#     plt.grid(True, linestyle='--', alpha=0.7)

#     # Custom legend
#     custom_lines = [
#         plt.Line2D([0], [0], color='gray', lw=4, label='Original Tensor Histogram'),
#         plt.Line2D([0], [0], color='red', lw=2, linestyle='solid', label='MDLNS Codewords'),
#         plt.Line2D([0], [0], color='blue', lw=2, linestyle='dashed', label='INT Codewords (Uniform)')
#     ]
#     plt.legend(handles=custom_lines, loc='center left', bbox_to_anchor=(0.02, 0.5))  # Move legend to middle left

#     plt.tight_layout()

#     if save_path is not None:
#         plt.savefig(save_path, dpi=300)

#     plt.show()



# def plot_tensor_histogram(
#     tensor,
#     new_tensor_int=None,
#     new_tensor_mdlns=None,
#     title="Tensor Histogram",
#     flag=1,
#     save_path=None
# ):
#     """
#     Plots a histogram of the values inside a tensor, and optionally overlays
#     separate codeword distributions for INT and MDLNS quantizations as short
#     sticks stacked above the histogram.

#     Args:
#         tensor (torch.Tensor): Input tensor (1D).
#         new_tensor_int (torch.Tensor or None): Reconstructed tensor using INT quantization.
#         new_tensor_mdlns (torch.Tensor or None): Reconstructed tensor using MDLNS quantization.
#         title (str): Title of the plot.
#         flag (int): If 0, do nothing. If 1, plot the histogram and overlays.
#         save_path (str or None): If provided, saves the figure to this path.
#     """
#     if flag == 0:
#         return

#     tensor_np = tensor.cpu().numpy()
#     num_bins = min(50, max(10, tensor_np.shape[0] // 10))  # Adaptive number of bins

#     plt.figure(figsize=(10, 6))

#     # Plot histogram and capture histogram max height
#     n, bins, patches = plt.hist(tensor_np, bins=num_bins, edgecolor='black', alpha=0.6, color='gray')
#     hist_max_height = np.max(n)

#     # Fix x-axis range to match tensor
#     x_min, x_max = tensor_np.min(), tensor_np.max()
#     plt.xlim(x_min, x_max)

#     # Calculate small offsets
#     offset_mdlns = hist_max_height * 1.05
#     offset_int = hist_max_height * 1.15
#     stick_height = hist_max_height * 0.05  # Short stick

#     # Overlay MDLNS codeword vertical sticks
#     if new_tensor_mdlns is not None:
#         new_mdlns_np = new_tensor_mdlns.cpu().numpy()
#         unique_mdlns_values = np.unique(new_mdlns_np)
#         for x in unique_mdlns_values:
#             plt.vlines(x, ymin=offset_mdlns, ymax=offset_mdlns + stick_height, colors='red', linestyles='solid', linewidth=1)

#     # Overlay INT codeword vertical sticks
#     if new_tensor_int is not None:
#         new_int_np = new_tensor_int.cpu().numpy()
#         unique_int_values = np.unique(new_int_np)
#         for x in unique_int_values:
#             plt.vlines(x, ymin=offset_int, ymax=offset_int + stick_height, colors='blue', linestyles='dashed', linewidth=1)

#     # Labels and grid
#     plt.title(title)
#     plt.xlabel('Value')
#     plt.ylabel('Frequency')
#     plt.grid(True, linestyle='--', alpha=0.7)

#     # Custom legend
#     custom_lines = [
#         plt.Line2D([0], [0], color='gray', lw=4, label='Original Tensor Histogram'),
#         plt.Line2D([0], [0], color='red', lw=2, linestyle='solid', label='MDLNS Codewords'),
#         plt.Line2D([0], [0], color='blue', lw=2, linestyle='dashed', label='INT Codewords')
#     ]
#     plt.legend(handles=custom_lines, loc='center left', bbox_to_anchor=(0.02, 0.5))  # Move legend to middle left

#     plt.tight_layout()

#     if save_path is not None:
#         plt.savefig(save_path, dpi=300)

#     plt.show()



# import matplotlib.pyplot as plt
# import torch
# import numpy as np

# def plot_tensor_histogram(
#     tensor,
#     new_tensor_int=None,
#     new_tensor_mdlns=None,
#     title="Tensor Histogram",
#     flag=1,
#     save_path=None
# ):
#     """
#     Plots a histogram of the values inside a tensor, optionally overlaying
#     smoothed codeword density curves for INT and MDLNS reconstructions.

#     Args:
#         tensor (torch.Tensor): Input tensor (1D).
#         title (str): Title of the plot.
#         flag (int): If 0, do nothing. If 1, plot the histogram.
#         save_path (str or None): If provided, saves the figure to this path.
#         new_tensor_int (torch.Tensor or None): Reconstructed tensor using INT quantization.
#         new_tensor_mdlns (torch.Tensor or None): Reconstructed tensor using MDLNS quantization.
#     """
#     if flag == 0:
#         return

#     tensor_np = tensor.cpu().numpy()
#     num_bins = min(50, max(10, tensor_np.shape[0] // 10))  # Adaptive number of bins

#     plt.figure(figsize=(10, 5))

#     # Plot histogram of the original tensor
#     plt.hist(tensor_np, bins=num_bins, edgecolor='black', alpha=0.6, color='gray', label='Original Tensor')

#     # Overlay INT reconstructed values as smooth density curve
#     if new_tensor_int is not None:
#         new_int_np = new_tensor_int.cpu().numpy()
#         int_density, int_bins = np.histogram(new_int_np, bins=100, range=(tensor_np.min(), tensor_np.max()), density=True)
#         int_bin_centers = (int_bins[:-1] + int_bins[1:]) / 2
#         plt.plot(int_bin_centers, int_density, color='blue', linestyle='--', linewidth=2, label='INT Codeword Density')

#     # Overlay MDLNS reconstructed values as smooth density curve
#     if new_tensor_mdlns is not None:
#         new_mdlns_np = new_tensor_mdlns.cpu().numpy()
#         mdlns_density, mdlns_bins = np.histogram(new_mdlns_np, bins=100, range=(tensor_np.min(), tensor_np.max()), density=True)
#         mdlns_bin_centers = (mdlns_bins[:-1] + mdlns_bins[1:]) / 2
#         plt.plot(mdlns_bin_centers, mdlns_density, color='red', linestyle='-', linewidth=2, label='MDLNS Codeword Density')

#     plt.title(title)
#     plt.xlabel('Value')
#     plt.ylabel('Normalized Density')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend()
#     plt.tight_layout()

#     if save_path is not None:
#         plt.savefig(save_path, dpi=300)

#     plt.show()



################################################################
################################################################
################################################################
################################################################


class PositionalEncoding(nn.Module):
    def __init__(self, pe_embed):
        super(PositionalEncoding, self).__init__()
        self.pe_embed = pe_embed.lower()
        if self.pe_embed == 'none':
            self.embed_length = 1
        else:
            self.lbase, self.levels = [float(x) for x in pe_embed.split('_')]
            self.levels = int(self.levels)
            self.embed_length = 2 * self.levels

    def forward(self, pos):
        if self.pe_embed == 'none':
            return pos[:,None]
        else:
            pe_list = []
            for i in range(self.levels):
                temp_value = pos * self.lbase **(i) * math.pi
                pe_list += [torch.sin(temp_value), torch.cos(temp_value)]
            return torch.stack(pe_list, 1)


def psnr2(img1, img2):
    mse = (img1 - img2) ** 2
    PIXEL_MAX = 1
    psnr = -10 * torch.log10(mse)
    psnr = torch.clamp(psnr, min=0, max=50)
    return psnr

def loss_fn(pred, target, args):
    target = target.detach()

    if args.loss_type == 'L2':
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.mean()       
    elif args.loss_type == 'L1':
        loss = torch.mean(torch.abs(pred - target))
    elif args.loss_type == 'SSIM':
        loss = 1 - ssim(pred, target, data_range=1, size_average=True)
    elif args.loss_type == 'Fusion1':
        loss = 0.3 * F.mse_loss(pred, target) + 0.7 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion2':
        loss = 0.3 * torch.mean(torch.abs(pred - target)) + 0.7 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion3':
        loss = 0.5 * F.mse_loss(pred, target) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion4':
        loss = 0.5 * torch.mean(torch.abs(pred - target)) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion5':
        loss = 0.7 * F.mse_loss(pred, target) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion6':
        loss = 0.7 * torch.mean(torch.abs(pred - target)) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion7':
        loss = 0.7 * F.mse_loss(pred, target) + 0.3 * torch.mean(torch.abs(pred - target))
    elif args.loss_type == 'Fusion8':
        loss = 0.5 * F.mse_loss(pred, target) + 0.5 * torch.mean(torch.abs(pred - target))
    elif args.loss_type == 'Fusion9':
        loss = 0.9 * torch.mean(torch.abs(pred - target)) + 0.1 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion10':
        loss = 0.7 * torch.mean(torch.abs(pred - target)) + 0.3 * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion11':
        loss = 0.9 * torch.mean(torch.abs(pred - target)) + 0.1 * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion12':
        loss = 0.8 * torch.mean(torch.abs(pred - target)) + 0.2 * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    return loss

def psnr_fn(output_list, target_list):
    psnr_list = []
    for output, target in zip(output_list, target_list):
        l2_loss = F.mse_loss(output.detach(), target.detach(), reduction='mean')
        psnr = -10 * torch.log10(l2_loss)
        psnr = psnr.view(1, 1).expand(output.size(0), -1)
        psnr_list.append(psnr)
    psnr = torch.cat(psnr_list, dim=1) #(batchsize, num_stage)
    return psnr

def msssim_fn(output_list, target_list):
    msssim_list = []
    for output, target in zip(output_list, target_list):
        if output.size(-2) >= 160:
            msssim = ms_ssim(output.float().detach(), target.detach(), data_range=1, size_average=True)
        else:
            msssim = torch.tensor(0).to(output.device)
        msssim_list.append(msssim.view(1))
    msssim = torch.cat(msssim_list, dim=0) #(num_stage)
    msssim = msssim.view(1, -1).expand(output_list[-1].size(0), -1) #(batchsize, num_stage)
    return msssim

def RoundTensor(x, num=2, group_str=False):
    if group_str:
        str_list = []
        for i in range(x.size(0)):
            x_row =  [str(round(ele, num)) for ele in x[i].tolist()]
            str_list.append(','.join(x_row))
        out_str = '/'.join(str_list)
    else:
        str_list = [str(round(ele, num)) for ele in x.flatten().tolist()]
        out_str = ','.join(str_list)
    return out_str

def adjust_lr(optimizer, cur_epoch, cur_iter, data_size, args):
    cur_epoch = cur_epoch + (float(cur_iter) / data_size)
    if args.lr_type == 'cosine':
        lr_mult = 0.5 * (math.cos(math.pi * (cur_epoch - args.warmup)/ (args.epochs - args.warmup)) + 1.0)
    elif args.lr_type == 'step':
        lr_mult = 0.1 ** (sum(cur_epoch >= np.array(args.lr_steps)))
    elif args.lr_type == 'const':
        lr_mult = 1
    elif args.lr_type == 'plateau':
        lr_mult = 1
    else:
        raise NotImplementedError

    if cur_epoch < args.warmup:
        lr_mult = 0.1 + 0.9 * cur_epoch / args.warmup

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = args.lr * lr_mult

    return args.lr * lr_mult

def worker_init_fn(worker_id):
    """
    Re-seed each worker process to preserve reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return

class PositionalEncodingTrans(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, pos):
        index = torch.round(pos * self.max_len).long()
        p = self.pe[index]
        return p



