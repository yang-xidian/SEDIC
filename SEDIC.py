import torch
from model.cldm import ControlLDM
from utils.common import instantiate_from_config, load_full_state_dict
from omegaconf import OmegaConf
from compressai.zoo.image import model_architectures as architectures
from PIL import Image, ImageChops
import cv2
from torchvision import transforms
import math
import pandas as pd
import clip
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
from skimage.transform import resize
from segment_anything import sam_model_registry, SamPredictor
from utils.common import auto_resize, pad
import einops
from utils.sampler import SpacedSampler
from utils.common import instantiate_from_config, load_file_from_url, count_vram_usage
from model.gaussian_diffusion import Diffusion
from typing import overload, Generator, Dict
from utils.common import wavelet_decomposition, wavelet_reconstruction, count_vram_usage
from einops import rearrange
from torch.nn import functional as F

import heapq
from collections import defaultdict, Counter

import os
import sys
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

from grounded_sam_generate_mask import load_image, load_model, get_grounding_output, show_mask, get_mask_data, generate_mask
from segment_anything import (
    sam_model_registry,
    SamPredictor
)


BICUBIC = InterpolationMode.BICUBIC

#model load
MODELS = {
    ### stage_1 model weights
    "bsrnet": "https://github.com/cszn/KAIR/releases/download/v1.0/BSRNet.pth",
    # the following checkpoint is up-to-date, but we use the old version in our paper
    # "swinir_face": "https://github.com/zsyOAOA/DifFace/releases/download/V1.0/General_Face_ffhq512.pth",
    "swinir_face": "https://huggingface.co/lxq007/DiffBIR/resolve/main/face_swinir_v1.ckpt",
    "scunet_psnr": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
    "swinir_general": "https://huggingface.co/lxq007/DiffBIR/resolve/main/general_swinir_v1.ckpt",
    ### stage_2 model weights
    "sd_v21": "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt",
    "v1_face": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_face.pth",
    "v1_general": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_general.pth",
    "v2": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v2.pth"
}

device = 'cuda:0'

def load_model_from_url(url: str) -> Dict[str, torch.Tensor]:
    sd_path = load_file_from_url(url, model_dir="weights")
    sd = torch.load(sd_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if list(sd.keys())[0].startswith("module"):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd

# load model
cldm: ControlLDM = instantiate_from_config(OmegaConf.load('/home/xiaoyang/project/DiffBIR-main/configs/inference/cldm.yaml'))
diffusion: Diffusion = instantiate_from_config(OmegaConf.load("/home/xiaoyang/project/DiffBIR-main/configs/inference/diffusion.yaml"))
#cldm: ControlLDM = instantiate_from_config(OmegaConf.load('./configs/inference/cldm.yaml'))
#diffusion: Diffusion = instantiate_from_config(OmegaConf.load("configs/inference/diffusion.yaml"))
sd = load_model_from_url(MODELS["sd_v21"])
unused =cldm.load_pretrained_sd(sd)
print(f"strictly load pretrained sd_v2.1, unused weights: {unused}")
control_sd = load_model_from_url(MODELS["v2"])
cldm.load_controlnet_from_ckpt(control_sd)
print(f"strictly load controlnet weight")
cldm.eval().to(device)


#load grounded_sam model

config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
grounded_checkpoint = "groundingdino_swint_ogc.pth"

bert_base_uncased_path = None
sam_version = "vit_h"
sam_checkpoint = "sam_vit_h_4b8939.pth"
grounded_model = load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=device)
predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
box_threshold = 0.3
text_threshold = 0.25

# load text

preprocess =  Compose([Resize((224, 224), interpolation=BICUBIC), ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
#seg_model, preprocess = clip.load("CS-ViT-B/16", device=device)
seg_model, preprocess = clip.load("CS-ViT-B/16", device=device)
seg_model.eval()


def divide_integer(num, n):
    quotient = num // n  
    remainder = num % n  
    result = [quotient] * n  

    
    for i in range(remainder):
        result[i] += 1

    return result


def mask_block(mask,num=8,level=0.35):
    tmp=resize(mask, (num, num), mode='reflect')
    tmp[tmp>level]=255
    tmp[tmp<=level]=0
    rp_mat_0=np.array(divide_integer(mask.shape[0], num),dtype='int')
    rp_mat_1=np.array(divide_integer(mask.shape[1], num),dtype='int')
    return tmp.repeat(rp_mat_1,axis=1).repeat(rp_mat_0,axis=0)


def clip_map(img,texts,mask_num=8):
    image = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # CLIP architecture surgery acts on the image encoder
        cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        image_features = seg_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # Prompt ensemble for text features with normalization
        text_features = clip.encode_text_with_prompt_ensemble(seg_model, texts, device)

        # Extract redundant features from an empty string
        redundant_features = clip.encode_text_with_prompt_ensemble(seg_model, [""], device)

        # Apply feature surgery for single text
        similarity = clip.clip_feature_surgery(image_features, text_features, redundant_features)
        similarity_map = clip.get_similarity_map(similarity[:, 1:, :], cv2_img.shape[:2])
        
        mask_0=(similarity_map[0,:,:,0].cpu().numpy() * 255).astype('uint8')
        mask_1=(similarity_map[0,:,:,1].cpu().numpy() * 255).astype('uint8')
        mask_2=(similarity_map[0,:,:,2].cpu().numpy() * 255).astype('uint8')
        mask_0=Image.fromarray(mask_block(mask_0,num=mask_num))
        mask_1=Image.fromarray(mask_block(mask_1,num=mask_num))
        mask_2=Image.fromarray(mask_block(mask_2,num=mask_num))

        return mask_0,mask_1,mask_2

def sam_map(image_path, texts):
    masks = []
    for text in texts:
        mask = generate_mask(grounded_model=grounded_model, sam_model=predictor, image_path=image_path, text_prompt=text, box_threshold=box_threshold, text_threshold=text_threshold, device=device)
        mask = mask.cpu().numpy()
        masks.append(mask)
    mask_0 = masks[0]
    mask_1 = masks[1]
    mask_2 = masks[2]
    mask_0 = Image.fromarray(mask_0)
    mask_1 = Image.fromarray(mask_1)
    mask_2 = Image.fromarray(mask_2)
    return mask_0,mask_1,mask_2








def image_paddle_in(image, num=32):
    
    new_width = ((image.width-1) // num + 1) * num
    new_height = ((image.height-1) // num + 1) * num

    
    new_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))

    
    new_image.paste(image, (0, 0))
    return new_image,image.width,image.height


def image_paddle_out(image, old_width, old_height):

    return image.crop((0,0,old_width,old_height))

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()


# Huffman coding
class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char  
        self.freq = freq  
        self.left = left  
        self.right = right  

    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(text):
    
    frequency = Counter(text)

    
    heap = [HuffmanNode(char, freq) for char, freq in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        
        merged = HuffmanNode(None, left.freq + right.freq, left, right)
        heapq.heappush(heap, merged)

    return heap[0]  

def build_huffman_codes(root):
    
    def build_codes_helper(node, current_code, codes):
        if node is None:
            return

        
        if node.char is not None:
            codes[node.char] = current_code
            return

        
        build_codes_helper(node.left, current_code + "0", codes)
        build_codes_helper(node.right, current_code + "1", codes)

    codes = {}
    build_codes_helper(root, "", codes)
    return codes

def huffman_encode(text, codes):
    return ''.join(codes[char] for char in text)

def huffman_decode(encoded_text, root):
    decoded_text = []
    current_node = root

    for bit in encoded_text:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right

        
        if current_node.char is not None:
            decoded_text.append(current_node.char)
            current_node = root

    return ''.join(decoded_text)

def text_coding(text):
    root = build_huffman_tree(text)
    huffman_codes = build_huffman_codes(root)
    encoded_text = huffman_encode(text, huffman_codes)
    b_word = len(encoded_text)
    decoded_text = huffman_decode(encoded_text, root)

    return decoded_text, b_word
def sr_pipe(img_reference, image, mask=None, pos_prompt="",neg_prompt="low quality, blurry, low-resolution, noisy, unsharp, weird textures",cfg=1.0,steps=40,res=512,better_start=False):

    lq_resized = auto_resize(img_reference, res)
    x = pad(np.array(lq_resized), scale=64)
    

    
    
    #sampler = SpacedSampler(model, var_type="fixed_small")
    sampler = SpacedSampler(diffusion.betas)
    control = torch.tensor(np.stack([x]) / 255.0, dtype=torch.float32, device=device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()
    bs, _, ori_h, ori_w = control.shape
    
    
        

    #control = model.preprocess_model(control)
    cond = cldm.prepare_condition(control, [pos_prompt] * bs)
    uncond = cldm.prepare_condition(control, [neg_prompt] * bs)
    cldm.control_scales = [1.0] * 13
    height, width = control.size(-2), control.size(-1)
    shape = (1, 4, height // 8, width // 8)

    if mask is not None:
        if isinstance(mask, (Image.Image, np.ndarray)):
            mask = [mask]
        if isinstance(mask, list) and isinstance(mask[0], Image.Image):
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask).float()
        mask = mask.to(device)
        mask_latents = torch.nn.functional.interpolate(
            mask, size=(height // 8, width // 8)
        )
        mask_latents = mask_latents.to(device)

        image_origin = auto_resize(image, res)
        image_origin = pad(np.array(image_origin), scale=64)
        image_origin = torch.tensor(np.stack([image_origin])/ 255.0, dtype=torch.float32, device=device).clamp_(0, 1)
        image_origin = einops.rearrange(image_origin, "n h w c -> n c h w").contiguous()
        image_origin = image_origin*2 - 1
        x_origin = cldm.vae_encode(image_origin)
        '''x_origin = (1-mask_latents)* x_origin 
        x_origin = cldm.vae_decode(x_origin)
        print("image_origin",x_origin.shape)
        x_origin = rearrange(x_origin,"n c h w -> n h w c").contiguous()
        image_origin_np = x_origin.cpu().numpy()
        print("image_origin",image_origin_np.shape)
        image_origin_np = (image_origin_np * 255.0).astype(np.uint8)
        image_origin_np = image_origin_np[0]
        Image.fromarray(image_origin_np).resize((old_width, old_height)).save("test_origin.png")'''
        
        #x_origin = cldm.vae_encode(image_origin)

    #x_T = torch.randn(shape, device=device, dtype=torch.float32)
    if better_start:
        # using noised low frequency part of condition as a better start point of 
        # reverse sampling, which can prevent our model from generating noise in 
        # image background.
        image_start = auto_resize(image, res)
        image_start = pad(np.array(image_start), scale=64)
        image_start = torch.tensor(np.stack([image_start]), dtype=torch.float32, device=device)
        image_start = einops.rearrange(image_start, "n h w c -> n c h w").contiguous()
        
        _, low_freq = wavelet_decomposition(image_start)
        
        x_0 = cldm.vae_encode(low_freq)
        #x_0 = cldm.vae_encode(image_start)
        x_T = diffusion.q_sample(
            x_0,
            torch.full((bs, ), diffusion.num_timesteps - 1, dtype=torch.long, device=device),
            torch.randn(x_0.shape, dtype=torch.float32, device=device)
        )
        # print(f"diffusion sqrt_alphas_cumprod: {self.diffusion.sqrt_alphas_cumprod[-1]}")
    else:
        x_T = torch.randn((bs, 4, height // 8, width // 8), dtype=torch.float32, device=device)
    '''z = sampler.sample(
            model=self.cldm, device=self.device, steps=steps, batch_size=bs, x_size=(4, h // 8, w // 8),
            cond=cond, uncond=uncond, cfg_scale=cfg_scale, x_T=x_T, progress=True,
            progress_leave=True, cond_fn=self.cond_fn, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )'''
    
    samples = sampler.sample(
            model=cldm, device=device, steps=steps, batch_size=bs, x_size=(4, height // 8, width // 8),
            cond=cond, uncond=uncond, cfg_scale=cfg, x_T=x_T,progress=True, progress_leave=True, cond_fn=None
        )
    '''samples = sampler.sample(
            steps=steps, shape=shape, cond_img=control,
            positive_prompt=positive_prompt, negative_prompt="Blurry, Low Quality", x_T=x_T,
            cfg_scale=cfg, cond_fn=None,
            color_fix_type="wavelet"
        )'''
    if mask is not None:
        #samples = mask_latents*samples + (1-mask_latents)*x_origin
        samples =  mask_latents*samples+(1-mask_latents)*x_origin 
    x_samples = cldm.vae_decode(samples)
    sample = x_samples[:, :, :ori_h, :ori_w]
    '''x_samples = samples.clamp(0, 1)
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)

    img_sr=Image.fromarray(x_samples[0][:,0:lq_resized.size[0],:])'''
    '''if mask is not None:
        sample = (sample + 1)*mask / 2 + (1-mask)*sample
    else:
        sample = (sample + 1) / 2'''
    
    sample = (sample + 1) / 2
    final_size = (ori_h, ori_w)
    sample = F.interpolate(sample, final_size, mode="bicubic", antialias=True)
    # tensor to image
    sample = rearrange(sample * 255., "n c h w -> n h w c")
    sample = sample.contiguous().clamp(0, 255).to(torch.uint8).cpu().numpy()
    return sample[0]


def load_images(image_folder=""):
    image_info = []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):  
            image_path = os.path.join(image_folder, filename)
            image_info.append({'name': filename, 'path': image_path})
    return image_info



def Multi_stage_image_compression(image_folder="",text_path="",output_folder=""):
    images_info = load_images(image_folder)
    all_bpp = []
    for info in images_info:
        torch.cuda.empty_cache()
        image_name = info['name']
        image_path = info['path']

        img = Image.open(image_path).convert('RGB')
        #Load image text information
        df = pd.read_csv(text_path)
        prompt_value = prompt_value = df.loc[df['name'] == image_name, 'prompt']
        if not prompt_value.empty:
            
            prompt_value = prompt_value.iloc[0]
        else:
            print("not prompt。")
        prompt=prompt_value.split('\n')
        prompt = [element for element in prompt if element != '']
        print(prompt)
        name_0,name_1,name_2=prompt[0].split('.')[0].split(',')
        detail_0,detail_1,detail_2=prompt[1],prompt[2],prompt[3]
        detail_all=prompt[4]
        print("name:0,1,2:",name_0,name_1,name_2)
        print(name_0,":",detail_0)
        print(name_1,":",detail_1)
        print(name_2,":",detail_2)
        print("detail_all:",detail_all)
        mask_0,mask_1,mask_2=sam_map(image_path,[name_0,name_1,name_2])

        #get_reference_image
        arch='cheng2020-attn'
        ckpt_net='./weights/cheng_small.pth.tar'
        checkpoint_path=ckpt_net
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint
        for key in ["network", "state_dict", "model_state_dict"]:
            if key in checkpoint:
                state_dict = checkpoint[key]
        model_cls = architectures[arch]
        comp_net = model_cls.from_state_dict(state_dict).eval().to(device)

        img, old_width, old_height = image_paddle_in(img, 64)
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out_net = comp_net.forward(x)
        out_net['x_hat'].clamp_(0, 1)
        img_reference = transforms.ToPILImage()(out_net['x_hat'].squeeze().cpu())

        img_reference = image_paddle_out(img_reference, old_width, old_height)
        b_image=compute_bpp(out_net)*img.size[0]*img.size[1]
        #print('bpp:',b_image/(img.size[0]*img.size[1]))
        ref_folder = os.path.join(output_folder, 'ref')
        os.makedirs(ref_folder, exist_ok=True)
        img_reference.save(output_folder + 'ref/' + image_name)

        #Multi-stage compressed image restoration
        num_inference_steps=40
        #width=512
        #height=512

        exag=512/max(img_reference.size)
        height=int(img_reference.size[1]*exag/8)*8
        width=int(img_reference.size[0]*exag/8)*8
        print(height,width)

        mask_0=mask_0.resize([width,height])
        mask_1=mask_1.resize([width,height])
        mask_2=mask_2.resize([width,height])

        mask_all=Image.new("RGB", img_reference.size, (255, 255, 255))
        os.makedirs(os.path.join(output_folder, 'Mask0'), exist_ok=True)
        mask_0.convert('RGB').save(output_folder + 'Mask0/' + image_name )
        os.makedirs(os.path.join(output_folder, 'Mask1'), exist_ok=True)
        mask_1.convert('RGB').save(output_folder + 'Mask1/' + image_name )
        os.makedirs(os.path.join(output_folder, 'Mask2'), exist_ok=True)
        mask_2.convert('RGB').save(output_folder + 'Mask2/' + image_name )


        image = img_reference.resize([width,height])
        im_rf = img_reference.resize([width,height])

        using_map = True
        res=512
        if using_map:
            #b_mask = mask_num * mask_num * 3
            #b_word = (len(detail_0) + len(detail_1) + len(detail_2) + len(detail_all)) * 8
            
            decoded_text, b_word = text_coding(prompt_value)
            #bpp = (b_image + b_mask + b_word) / (img.size[0] * img.size[1])
            bpp = (b_image + b_word) / (img.size[0] * img.size[1])
            all_bpp.append(bpp)
            print('bpp=' + str(bpp))
            
            # 1 stage
            image_tmp = sr_pipe(image, image, mask_0, pos_prompt=detail_0, cfg=3.5, steps=5, res=res)
            image_tmp = Image.fromarray(image_tmp)
            image = image_tmp
            '''print("step1:",image_tmp.size)
            print((old_width, old_height))
            image = ImageChops.add(
                ImageChops.multiply(image_tmp, mask_0.convert("RGB")),
                ImageChops.multiply(image, Image.fromarray(255 - np.array(mask_0)).convert("RGB"))
            )'''
            os.makedirs(os.path.join(output_folder, 'Mask0_result'), exist_ok=True)
            image.resize((old_width, old_height)).save(output_folder + 'Mask0_result/' + image_name)

            # 2 stage
            image_tmp = sr_pipe(image, image, mask_1, pos_prompt=detail_1, cfg=3.5, steps=10, res=res)
            image_tmp = Image.fromarray(image_tmp)
            image = image_tmp
            os.makedirs(os.path.join(output_folder, 'Mask1_result'), exist_ok=True)
            image.resize((old_width, old_height)).save(output_folder + 'Mask1_result/' + image_name)
            
            # 3 stage
            image_tmp = sr_pipe(image, image, mask_2, pos_prompt=detail_2, cfg=3.5, steps=10, res=res)
            image_tmp = Image.fromarray(image_tmp)
            image = image_tmp
            os.makedirs(os.path.join(output_folder, 'Mask2_result'), exist_ok=True)
            image.resize((old_width, old_height)).save(output_folder + 'Mask2_result/' + image_name)
            
            # final stage
            image = sr_pipe(im_rf, image, mask=None, pos_prompt=detail_all, cfg=7, steps=50, res=res, better_start=True)
            os.makedirs(os.path.join(output_folder, 'SR'), exist_ok=True)
            Image.fromarray(image).resize((old_width, old_height)).save(output_folder + 'SR/' + image_name)
            print("Processed image:",image_name)
            
        else:  
            b_word = len(detail_all) * 8
            bpp = (b_image + b_word) / (img.size[0] * img.size[1])
            print('bpp=' + str(bpp))

            image = sr_pipe(image, pos_prompt=detail_all, cfg=7, steps=40, res=res)
            Image.fromarray(image).save(output_folder + 'SR/' + image_name)
    avg_bpp = np.mean(all_bpp)
    print("avg_bpp:",avg_bpp)




Multi_stage_image_compression(image_folder='/home/xiaoyang/project/DiffBIR-main/step_vision_image_test',text_path="/home/xiaoyang/project/DiffBIR-main/gpt4_text_step_test.csv",output_folder="/home/xiaoyang/project/DiffBIR-main/test_step_result_10_1/")

