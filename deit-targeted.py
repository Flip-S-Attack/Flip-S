import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    DeiTImageProcessor,
    DeiTForImageClassificationWithTeacher,
    BitsAndBytesConfig,
)
import copy
import time
import torchvision
import pickle
import bitsandbytes.functional as F2
from bitstring import Bits

if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))
utils_path=os.path.abspath('../')
import sys
sys.path.append(utils_path)

from utils.load_data import load_split_ImageNet1k_valid
from utils.metrics import imagenet_acc, imagenet_asr
from utils.quant_model import find_all_bnbLinear, replace_with_myLinear, my_8bit_linear

class DataLoaderArguments:
    aux_num = 32
    seed = 0
    batch_size = 64
    shuffle = False
    num_workers = 0
    pin_memory = False
    valdir = '.data/Imagenet/valid'
    split_ratio = 0.1
    
class AttackArguments:
    target_class = 2
    
    topk = 50 # for absmax (or 'Scale Factor')
    topk2 = 50 # for weight
    gamma = 1.
    target_bit = 50
    
def main():
    dataargs = DataLoaderArguments()
    attackrargs = AttackArguments()
    
    model_name = 'facebook/deit-base-distilled-patch16-224'
    processor = DeiTImageProcessor.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeiTForImageClassificationWithTeacher.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        ),
        device_map='cuda:0',
    )
    clean_model = DeiTForImageClassificationWithTeacher.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        ),
        device_map='cuda:0',
    )
    print('[+] Done Load Model')
    
    ##############################################################################################
    # Split dataset
    crossentropyloss = torch.nn.CrossEntropyLoss()
    mseloss = torch.nn.MSELoss()
    val_loader, aux_loader, small_val_loader = load_split_ImageNet1k_valid(dataargs.valdir, aux_num=dataargs.aux_num, seed=dataargs.seed, processor=processor,
                                                         batch_size=dataargs.batch_size, shuffle=dataargs.shuffle, num_workers=dataargs.num_workers, 
                                                         pin_memory=dataargs.pin_memory, split_ratio=dataargs.split_ratio, use_ir=True, use_defalut_valid_batchsize=True)
    normalize = torchvision.transforms.Normalize(mean=processor.image_mean,std=processor.image_std)
    
    #-----------------------Trojan Insertion----------------------------------------------------------------___
    _, modules_to_convert = find_all_bnbLinear(model)
    model, has_been_replaced = replace_with_myLinear(model, modules_to_convert=modules_to_convert, use_our_BFA=True)
    if not has_been_replaced:
        print("[-] Can't find any bnb Linear!")
        exit(0)
    print('[+] Done Replace Model')
    
    target_class = attackrargs.target_class
    load_name = model_name.split('/')[-1]
    with open(f'./trigger/{load_name}-trigger.pkl','rb') as f:
        trigger = pickle.load(f)
    trigger = trigger.cuda()
    mask = torch.zeros(trigger.shape[-2], trigger.shape[-1], dtype=torch.float16).cuda()
    mask[224-48:224,224-48:224] = 1.
    print(f'[+] done load Trigger')
    
    print('========================Before Attack========================')
    acc1 = imagenet_acc(torch.nn.Sequential(normalize, model), small_val_loader, device)
    asr1 = imagenet_asr(model, small_val_loader, trigger, mask, normalize, target_class, device)
    print('========================Start  Attack========================')
    
    topk = attackrargs.topk
    topk2 = attackrargs.topk2
    gamma = attackrargs.gamma
    changed_bit = set()
    base = [2 ** i for i in range(16 - 1, -1, -1)]
    base[0] = -base[0]
    baned_absmax = {}
    baned_weight = {}
    allow_weight = {}
    base = torch.tensor(base,dtype=torch.int16).cuda()
    base2 = [2 ** i for i in range(8 - 1, -1, -1)]
    base2[0] = -base2[0]
    base2 = torch.tensor(base2,dtype=torch.float16).cuda()
    
    for ext_iter in tqdm(range(attackrargs.target_bit+10)):
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            model.zero_grad()
            total_loss = 0.
            
            for batch_idx, (aux_inputs, aux_targets) in enumerate(aux_loader):
                aux_inputs = aux_inputs.to(device)
                aux_targets = aux_targets.to(device)
                # compute output
                outputs = model(normalize(aux_inputs))
                logits = outputs.logits
                # loss = crossentropyloss(logits, aux_targets)
                clean_logits = clean_model(normalize(aux_inputs)).logits
                loss = mseloss(logits, clean_logits)
                
                outputs = model(normalize(aux_inputs.cuda()*(1-mask) + torch.mul(trigger, mask)))
                logits = outputs.logits

                target_loss = crossentropyloss(logits, aux_targets*0+target_class)
                loss += gamma*(target_loss)
                loss = loss / (1 + gamma)
                
                total_loss += loss.data
                loss.backward(retain_graph=True)
            layers = {}
            print(f'[+] ext_epoch {ext_iter}: loss {total_loss/(batch_idx+1)}',flush=True)
            now_loss = total_loss

            # check absmax
            with torch.no_grad():
                for name, layer in model.deit.encoder.named_modules():
                    if isinstance(layer, my_8bit_linear):
                        grad = layer.absmax.grad.data
                        grad = grad.view(-1)
                        
                        grad_abs = grad.abs()
                        if baned_absmax.__contains__(name):
                            for idx in baned_absmax[name]:
                                grad_abs[idx] = -100.
                        
                        now_topk = grad_abs.topk(topk)
                        layers[name] = {'values': grad[now_topk.indices].tolist(), 'indices':now_topk.indices.tolist()}
                    
                all_grad = {}
                for layer in layers:
                    for i, idx in enumerate(layers[layer]['indices']):
                        if baned_absmax.__contains__(layer) and idx in baned_absmax[layer]:
                            continue
                        all_grad['@'.join([layer,str(idx)])] = layers[layer]['values'][i]

                sorted_grad = sorted(all_grad.items(), key = lambda x:abs(x[1]), reverse = True)

                atk_info={}
                for info in sorted_grad[:topk]:
                    layer, idx = info[0].split('@')
                    if not atk_info.__contains__(layer):
                        atk_info[layer] = []
                    atk_info[layer].append((int(idx),info[1]))

                all_loss = {}
                for name, layer in model.deit.encoder.named_modules():
                    if isinstance(layer, my_8bit_linear):
                        if atk_info.__contains__(name):
                            ori_absmax = layer.absmax.detach().clone()
                            
                            for idx, grad in atk_info[name]:
                                now_absmax = ori_absmax[idx]
                                bits = torch.tensor([int(b) for b in Bits(int=int(now_absmax.view(torch.int16)),length=16).bin]).cuda()
                                flag = True
                                changable = []
                                for i in range(16):
                                    old_bit = bits[i].clone()
                                    if old_bit == 0:
                                        new_bit = 1
                                    else:
                                        new_bit = 0
                                    bits[i] = new_bit
                                    new_absmax = bits * base
                                    new_absmax = torch.sum(new_absmax, dim=-1).type(torch.int16).to(now_absmax.device).view(torch.float16)
                                    
                                    if torch.isnan(new_absmax).any() or torch.isinf(new_absmax).any():
                                        bits[i] = old_bit
                                        continue
                                    
                                    if (new_absmax-now_absmax)*grad > 0:
                                        bits[i] = old_bit
                                        continue
                                    
                                    bits[i] = old_bit
                                    
                                    layer.absmax[idx] = new_absmax.clone()
                                    
                                    total_loss = 0.
                                    for batch_idx, (aux_inputs, aux_targets) in enumerate(aux_loader):
                                        aux_inputs = aux_inputs.to(device)
                                        aux_targets = aux_targets.to(device)
                                        # compute output
                                        outputs = model(normalize(aux_inputs))
                                        logits = outputs.logits
                                        # loss = crossentropyloss(logits, aux_targets)
                                        clean_logits = clean_model(normalize(aux_inputs)).logits
                                        loss = mseloss(logits, clean_logits)
                                        
                                        outputs = model(normalize(aux_inputs.cuda()*(1-mask) + torch.mul(trigger, mask)))
                                        logits = outputs.logits

                                        target_loss = crossentropyloss(logits, aux_targets*0+target_class)
                                        loss += gamma*(target_loss)
                                        loss = loss / (1 + gamma)
                                        
                                        total_loss += loss.data
                                        
                                        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                                            break

                                    layer.absmax[idx] = now_absmax.clone()
                                    
                                    if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                                        continue
                                    
                                    best_bit = f'{name}@{idx}@{i}@absmax'
                                    all_loss[best_bit] = total_loss.data
                                    
            # valid absmax
            best_bit = min(all_loss, key=all_loss.get)
            min_loss = all_loss[best_bit]
            skip = False
            if min_loss < now_loss:
                skip = True

            # check weight
            if not skip:
                with torch.no_grad():
                    for name, layer in model.deit.encoder.named_modules():
                        if isinstance(layer, my_8bit_linear):
                            grad = layer.w_int.grad.data
                            grad = grad.view(-1)
                            
                            grad_abs = grad.abs()
                            if baned_weight.__contains__(name):
                                for idx in baned_weight[name]:
                                    grad_abs[idx] = -100.
                            
                            now_topk = grad_abs.topk(topk2)
                            layers[name] = {'values': grad[now_topk.indices].tolist(), 'indices':now_topk.indices.tolist()}
                        
                    all_grad = {}
                    for layer in layers:
                        for i, idx in enumerate(layers[layer]['indices']):
                            if baned_weight.__contains__(layer) and idx in baned_weight[layer]:
                                continue
                            all_grad['@'.join([layer,str(idx)])] = layers[layer]['values'][i]

                    sorted_grad = sorted(all_grad.items(), key = lambda x:abs(x[1]), reverse = True)

                    atk_info={}
                    for info in sorted_grad[:topk2]:
                        layer, idx = info[0].split('@')
                        if not atk_info.__contains__(layer):
                            atk_info[layer] = []
                        atk_info[layer].append((int(idx),info[1]))

                    for name, layer in model.deit.encoder.named_modules():
                        if isinstance(layer, my_8bit_linear):
                            if atk_info.__contains__(name):
                                ori_shape = layer.w_int.shape
                                layer.w_int.data = layer.w_int.data.view(-1)
                                ori_weight = layer.w_int.detach().clone()
                                for idx, grad in atk_info[name]:
                                    now_weight = ori_weight[idx]
                                    bits = torch.tensor([int(b) for b in Bits(int=int(now_weight.type(torch.int8)),length=8).bin]).cuda()
                                    flag = True
                                    changable = []
                                    for i in range(8):
                                        old_bit = bits[i].clone()
                                        if old_bit == 0:
                                            new_bit = 1
                                        else:
                                            new_bit = 0
                                        bits[i] = new_bit
                                        new_weight = bits * base2
                                        new_weight = torch.sum(new_weight, dim=-1).to(now_weight.device).type(torch.float16)
                                        
                                        if (new_weight-now_weight)*grad > 0:
                                            bits[i] = old_bit
                                            continue
                                        
                                        bits[i] = old_bit
                                        
                                        layer.w_int[idx] = new_weight.clone()
                                        layer.w_int.data = layer.w_int.data.view(ori_shape)
                                    
                                        best_bit = f'{name}@{idx}@{i}@weight'
                                        
                                        total_loss = 0.
                                        for batch_idx, (aux_inputs, aux_targets) in enumerate(aux_loader):
                                            aux_inputs = aux_inputs.to(device)
                                            aux_targets = aux_targets.to(device)
                                            # compute output
                                            outputs = model(normalize(aux_inputs))
                                            logits = outputs.logits
                                            # loss = crossentropyloss(logits, aux_targets)
                                            clean_logits = clean_model(normalize(aux_inputs)).logits
                                            loss = mseloss(logits, clean_logits)
                                            
                                            outputs = model(normalize(aux_inputs.cuda()*(1-mask) + torch.mul(trigger, mask)))
                                            logits = outputs.logits

                                            target_loss = crossentropyloss(logits, aux_targets*0+target_class)
                                            loss += gamma*(target_loss)
                                            loss = loss / (1 + gamma)
                                            
                                            total_loss += loss.data
                                    
                                        layer.w_int.data = layer.w_int.data.view(-1)
                                        layer.w_int[idx] = now_weight.clone()
                                    
                                        best_bit  = f'{name}@{idx}@{i}@weight'
                                        all_loss[best_bit] = total_loss.data
                                
                                layer.w_int.data = layer.w_int.data.view(ori_shape)
            
            with torch.no_grad():       
                # select
                best_bit = min(all_loss, key=all_loss.get)
                print(f'[+] change {best_bit}, loss: {all_loss[best_bit]}',flush=True)
                # '{name}@{idx}@{i}@absmax'
                layer_name, idx, i, bit_type = best_bit.split('@')
                idx, i = int(idx), int(i)
                for name, layer in model.deit.encoder.named_modules():
                    if isinstance(layer, my_8bit_linear) and layer_name == name:
                        if bit_type == 'absmax':
                            now_absmax = layer.absmax[idx]
                            bits = torch.tensor([int(b) for b in Bits(int=int(now_absmax.view(torch.int16)),length=16).bin]).cuda()
                            old_bit = bits[i]
                            if old_bit == 0:
                                new_bit = 1
                            else:
                                new_bit = 0
                            bits[i] = new_bit
                            new_absmax = bits * base
                            new_absmax = torch.sum(new_absmax, dim=-1).type(torch.int16).to(now_absmax.device).view(torch.float16)
                            layer.absmax[idx] = new_absmax.clone()
                            
                            start_ban = (idx // (4096//2)) * (4096//2)
                            end_ban = start_ban + (4096//2)
                            end_ban = min(end_ban, len(layer.absmax))
                            for idx in range(start_ban, end_ban):
                                if name not in baned_absmax:
                                    baned_absmax[name] = []
                                baned_absmax[name].append(idx)
                        elif bit_type == 'weight':
                            ori_shape = layer.w_int.shape
                            layer.w_int.data = layer.w_int.data.view(-1)
                            now_weight = layer.w_int[idx]
                            bits = torch.tensor([int(b) for b in Bits(int=int(now_weight.type(torch.int8)),length=8).bin]).cuda()
                            old_bit = bits[i].clone()
                            if old_bit == 0:
                                new_bit = 1
                            else:
                                new_bit = 0
                            bits[i] = new_bit
                            new_weight = bits * base2
                            new_weight = torch.sum(new_weight, dim=-1).to(now_weight.device).type(torch.float16)
                            layer.w_int[idx] = new_weight.clone()
                            
                            start_ban = (idx // (4096)) * 4096
                            end_ban = start_ban + (4096)
                            end_ban = min(end_ban, len(layer.w_int))
                            for idx in range(start_ban, end_ban):
                                if name not in baned_weight:
                                    baned_weight[name] = []
                                baned_weight[name].append(idx)
                            layer.w_int.data = layer.w_int.data.view(ori_shape)
                        else:
                            raise NotImplementedError
                
                try:
                    changed_bit.remove(best_bit)
                    print(f'[-] Revoke Flip {best_bit}')
                    layer_name, idx, i, bit_type = best_bit.split('@')
                    idx, i = int(idx), int(i)
                    for name, layer in model.deit.encoder.named_modules():
                        if isinstance(layer, my_8bit_linear) and layer_name == name:
                            if bit_type == 'absmax':
                                start_ban = (idx // (4096//2)) * (4096//2)
                                end_ban = start_ban + (4096//2)
                                end_ban = min(end_ban, len(layer.absmax))
                                for idx in range(start_ban, end_ban):
                                    baned_absmax[name].remove(idx)
                            elif bit_type == 'weight':
                                start_ban = (idx // (4096)) * 4096
                                end_ban = start_ban + (4096)
                                end_ban = min(end_ban, len(layer.w_int.view(-1)))
                                for idx in range(start_ban, end_ban):
                                    baned_weight[name].remove(idx)
                            else:
                                raise NotImplementedError
                except:
                    changed_bit.add(best_bit)
                    if len(changed_bit) >= attackrargs.target_bit:
                        print('===========================End opt===========================')
                        acc1 = imagenet_acc(torch.nn.Sequential(normalize, model), small_val_loader, device)
                        asr1 = imagenet_asr(model, small_val_loader, trigger, mask, normalize, target_class, device)
                        nbit = len(changed_bit)
                        print(f'[+] Flip {nbit} bit: {changed_bit}',flush=True)
                        exit(0)
                    
    ##############################################################################################
    # End opt
    print('===========================End opt===========================')
    acc1 = imagenet_acc(torch.nn.Sequential(normalize, model), small_val_loader, device)
    asr1 = imagenet_asr(model, small_val_loader, trigger, mask, normalize, target_class, device)
    nbit = len(changed_bit)
    print(f'[+] Flip {nbit} bit: {changed_bit}',flush=True)

if __name__ == '__main__':
    main()
