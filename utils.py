import math
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim

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
            c_qt, c_nt = map_range(t, conv_lut, b_lut, t_lut, bit, sec_base_bits, auto_scale)

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
            

        
    return first_base_selected, sec_base_selected, qt, nt
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
def map_range(t, conv_lut, b_lut, nb_lut, bit=8, sec_base_bits=3, auto_scale=0):
    # t --> tensor, b --> binary, nb --> non-binary, conv --> integer to MDLNS conversion LUT

    conv_lut_signed = torch.tensor(conv_lut + [-x for x in conv_lut]) # conv_lut_signed  contains all the elements from conv_lut plus their negative counterparts
    # Notice that the above line also converts from conv_lut array to conv_lut_signed tensor
    
    b_lut_expanded = torch.tensor(b_lut + b_lut)
    nb_lut_expanded = torch.tensor(nb_lut + nb_lut)

    if auto_scale == 1:
        t_min, t_max = t.min(), t.max()
        lut_min, lut_max = conv_lut_signed.min(), conv_lut_signed.max()

        # Scale t to fit within the range of conv_lut_signed
        t_scaled = (t - t_min) / (t_max - t_min)
        t_scaled = t_scaled * (lut_max - lut_min) + lut_min

        # Compute differences and get indices of closest LUT entries
        diff = torch.abs(t_scaled.unsqueeze(-1) - conv_lut_signed)
        indices = torch.argmin(diff, dim=-1)  # shape: (B, L)

        # Quantized values using those indices
        t_dequantized = conv_lut_signed[indices]  # shape: (B, L)
        
        # Reconstruct t in the original range
        # Reverse scaling: map from [lut_min, lut_max] back to [t_min, t_max]
        t_reconstructed = (t_dequantized - lut_min) / (lut_max - lut_min)
        t_reconstructed = t_reconstructed * (t_max - t_min) + t_min
    
        # Adding code
        b_dequantized = b_lut_expanded[indices]
        nb_dequantized = nb_lut_expanded[indices]

        
    else:
        
        # Compute differences and get indices of closest LUT entries
        diff = torch.abs(t.unsqueeze(-1) - conv_lut_signed)
        indices = torch.argmin(diff, dim=-1)  # shape: (B, L)

        # Quantized values using those indices - could have been one assignment straight to t_reconstructed
        t_dequantized = conv_lut_signed[indices]  # shape: (B, L)
        t_reconstructed = t_dequantized
        
        # Adding code
        b_dequantized = b_lut_expanded[indices]
        nb_dequantized = nb_lut_expanded[indices]
        


    bin_base_bits = bit - sec_base_bits - 1
    qt = encode_qt_vector(t, b_dequantized, nb_dequantized, bin_base_bits, sec_base_bits)
    #notice that I am passing "t" as the first argument to use it to extract the signs later in the function

    return qt, t_reconstructed
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
    
    # Quantizes a 1D or 2D floating point tensor to LNS representation (PyTorch version).

    # Args:
    #     tensor (torch.Tensor): Input 1D or 2D float32 tensor.
    #     bit (int): Total number of bits (including sign bit).
    #     lns_base (float): Base for logarithm. If 1000, the best base is searched.
    #     exp_bits (int): Number of bits for exponent part.

    # Returns:
    #     base_selected (float): Selected LNS base.
    #     codewords (torch.Tensor): Quantized tensor (codewords with sign and exponent bits).
    #     tensor_dequant (torch.Tensor): Dequantized tensor.
    
    #assert tensor.ndim in [1,2], "Tensor must be 1D or 2D"
    
    tensor = tensor.float()
    sign_bit = (tensor < 0).to(torch.uint8)
    tensor_abs = tensor.abs()

    def compute_qsnr(original, reconstructed):
        signal_power = torch.sum(original**2)
        error_power = torch.sum((original - reconstructed)**2) + 1e-12  # avoid division by zero
        return 10 * torch.log10(signal_power / error_power)

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

    if lns_base == 1000:
        best_qsnr = -float('inf')
        base_candidates = torch.linspace(1.1, 10, 200)
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

# def quantize_per_tensor_lns(tensor, bit=8, lns_base=2, exp_bits=3):
#     """
#     Quantizes a 2D floating point tensor to LNS format (PyTorch version).

#     Args:
#         tensor (torch.Tensor): 2D tensor (float32 or float64).
#         bit (int): Total number of bits for LNS representation.
#         lns_base (float): Base for LNS. If 1000, search for best base.
#         exp_bits (int): Number of bits for exponent.

#     Returns:
#         base_selected (float): The selected base.
#         mantissa_tensor (torch.Tensor): Tensor of mantissas.
#         codewords (torch.Tensor): Tensor of quantized codewords.
#         dequantized (torch.Tensor): Dequantized tensor.
#     """
#     # assert tensor.dim() == 2, "Input tensor must be 2D"

#     device = tensor.device
#     dtype = tensor.dtype
#     mantissa_bits = bit - exp_bits - 1  # 1 bit for sign

#     # Flatten tensor for easier processing
#     tensor_flat = tensor.flatten()
#     sign = (tensor_flat < 0).to(torch.int32)
#     tensor_abs = torch.abs(tensor_flat) + 1e-12  # avoid log(0)



#     # Function to encode LNS
#     # def lns_encode(tensor_abs, base):
#     #     base = torch.tensor(base, device=device, dtype=dtype)
#     #     log_val = torch.log(tensor_abs) / torch.log(base)
#     #     int_log_val = torch.floor(log_val)
#     #     frac_log_val = log_val - int_log_val

#     #     # Quantize exponent
#     #     max_exp = 2 ** exp_bits - 1
#     #     exp = torch.clamp(int_log_val, 0, max_exp).to(torch.int32)

#     #     # Quantize mantissa
#     #     max_mantissa = 2 ** mantissa_bits - 1
#     #     mantissa = torch.clamp((frac_log_val * (max_mantissa + 1)).round(), 0, max_mantissa).to(torch.int32)

#     #     return exp, mantissa

#     # # Function to decode LNS
#     # def lns_decode(sign, exp, mantissa, base):
#     #     base = torch.tensor(base, device=device, dtype=dtype)
#     #     max_mantissa = 2 ** mantissa_bits - 1
#     #     #frac = mantissa.to(dtype) / (max_mantissa + 1)
#     #     frac = mantissa.to(dtype) / max_mantissa
#     #     value = base ** (exp.to(dtype) + frac)
#     #     value = torch.where(sign == 1, -value, value)
#     #     return value

    

#     def lns_encode(tensor_abs, base):
#         base = torch.tensor(base, device=device, dtype=dtype)
#         log_val = torch.log(tensor_abs) / torch.log(base)
#         offset = 2 ** (exp_bits - 1)
#         int_log_val = torch.floor(log_val) + offset

#         max_exp = 2 ** exp_bits - 1
#         exp = torch.clamp(int_log_val, 0, max_exp).to(torch.int32)

#         frac_log_val = log_val - (torch.floor(log_val))
#         max_mantissa = 2 ** mantissa_bits - 1
#         mantissa = torch.clamp((frac_log_val * max_mantissa).round(), 0, max_mantissa).to(torch.int32)

#         return exp, mantissa

#     def lns_decode(sign, exp, mantissa, base):
#         base = torch.tensor(base, device=device, dtype=dtype)
#         max_mantissa = 2 ** mantissa_bits - 1
#         frac = mantissa.to(dtype) / max_mantissa
#         offset = 2 ** (exp_bits - 1)
#         real_exp = exp.to(dtype) - offset
#         value = base ** (real_exp + frac)
#         value = torch.where(sign == 1, -value, value)
#         return value



#     # QSNR calculation
#     # def calculate_qsnr(original, reconstructed):
#     #     signal_power = torch.mean(original ** 2)
#     #     noise_power = torch.mean((original - reconstructed) ** 2)
#     #     qsnr = 10 * torch.log10(signal_power / (noise_power + 1e-12))
#     #     return qsnr

#     # Search for best base if needed
#     if lns_base == 1000:
#         best_qsnr = -float('inf')
#         base_selected = None
#         best_reconstructed = None
#         best_exp, best_mantissa = None, None

#         for candidate_base in torch.linspace(1.1, 5.0, steps=100, device=device):
#             exp, mantissa = lns_encode(tensor_abs, candidate_base.item())
#             reconstructed = lns_decode(sign, exp, mantissa, candidate_base.item())
#             qsnr = calc_qsnr(tensor_flat, reconstructed)
#             if qsnr > best_qsnr:
#                 best_qsnr = qsnr
#                 base_selected = candidate_base.item()
#                 best_reconstructed = reconstructed
#                 best_exp, best_mantissa = exp, mantissa

#         exp = best_exp
#         mantissa = best_mantissa
#         dequantized_flat = best_reconstructed
#     else:
#         base_selected = lns_base
#         exp, mantissa = lns_encode(tensor_abs, base_selected)
#         dequantized_flat = lns_decode(sign, exp, mantissa, base_selected)

#     # Compose codewords: sign | exponent | mantissa
#     codewords = (sign << (bit - 1)) | (exp << mantissa_bits) | mantissa
#     codewords = codewords.view(tensor.shape)
#     mantissa_tensor = mantissa.view(tensor.shape)
#     dequantized = dequantized_flat.view(tensor.shape)

#     return base_selected, mantissa_tensor, codewords, dequantized


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



