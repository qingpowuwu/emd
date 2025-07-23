import torch
import torch.nn as nn
import numpy as np
import imageio
import os
import cv2
import nvdiffrast.torch as dr 
from arguments.gaussian_options import BaseOptions
from scene.cameras import Camera
from utils.graphics_utils import get_rays_torch
from utils.general_utils import get_expon_lr_func

class SkyCubeMap(nn.Module):
    def __init__(self, cfg: BaseOptions):
        super().__init__()
        
        self.cfg = cfg
        self.sky_resolution = self.cfg.sky_resolution
        eps = 1e-3
        
        if self.cfg.sky_white_background:
            base = torch.ones(6, self.sky_resolution, self.sky_resolution, 3).float().cuda() * (1. - eps)
        else:
            base = torch.zeros(6, self.sky_resolution, self.sky_resolution, 3).float().cuda() + eps
            
        self.sky_cube_map = nn.Parameter(base).requires_grad_(True)    
        
        # # TODO: change hard code here
        max_h, max_w = 640, 960
        if self.cfg.white_background:
            self.sky_color = torch.ones((max_h, max_w, 3)).float().cuda()
        else:
            self.sky_color = torch.zeros((max_h, max_w, 3)).float().cuda()    

    def save_sky_latlong(self):
        sky_latlong = cubemap_to_latlong(self.sky_cube_map, [self.sky_resolution, self.sky_resolution * 2])
        sky_latlong = (sky_latlong.clamp(0., 1.).detach().cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(cfg.model_path, 'sky_latlong.png'), sky_latlong)

    def forward(self, camera: Camera, acc=None, is_train=False):
        # acc: gaussian opacity of foreground model: [1, H, W]
        # mask: [H, W], indicating whether a pixel is covered by forgeground model
        if is_train and hasattr(camera, 'sky_mask'):
            mask = camera.sky_mask.cuda()[0].bool()
            mask[:50, :] = True
        elif acc is not None:
            mask = (1 - acc[0]) > 1e-3
        else:
            mask = None

        # R, T should be in w2c format
        # rays_d: [H, W, 3]
        w2c = camera.world_view_transform.transpose(0, 1)
        H, W, K, R, T = camera.image_height, camera.image_width, camera.intrinsic, w2c[:3, :3], w2c[:3, 3]
        if is_train:
            _, rays_d = get_rays_torch(H, W, K, R, T, perturb=True)
        else:
            _, rays_d = get_rays_torch(H, W, K, R, T, perturb=False)
        
        if mask is None:
            sky_color = dr.texture(self.sky_cube_map[None, ...], rays_d[None, ...],
                               filter_mode='linear', boundary_mode='cube')
            sky_color = sky_color[0].permute(2, 0, 1).clamp(0., 1.) # [3, H, W]
        else:
            if is_train:
                if self.cfg.sky_white_background:
                    sky_color = torch.ones((H, W, 3)).float().cuda() 
                else:
                    sky_color = torch.zeros((H, W, 3)).float().cuda() 
            else:
                sky_color = self.sky_color[:H, :W, :]
                if self.cfg.sky_white_background:
                    torch.fill_(sky_color, 1.)
                else:
                    torch.fill_(sky_color, 0.)

            rays_d = rays_d[mask] # [N, 3]
            sky_color_mask = dr.texture(self.sky_cube_map[None, ...], rays_d[None, None, ...],
                               filter_mode='linear', boundary_mode='cube')

            sky_color_mask = sky_color_mask.squeeze(0).squeeze(0)                 
            sky_color[mask] = sky_color_mask
            sky_color = sky_color.permute(2, 0, 1).clamp(0., 1.) # [3, H, W]
        return sky_color

#----------------------------------------------------------------------------
# Vector operations
#----------------------------------------------------------------------------

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0,1), mode='constant', value=w)

#----------------------------------------------------------------------------
# Cubemap utility functions
#----------------------------------------------------------------------------

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def cube_to_dir(s, x, y):
    if s == 0:   rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2: rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5: rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)

def latlong_to_cubemap(latlong_map, res):
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device='cuda')
    for s in range(6):
        gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                                torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                                indexing='ij')
        v = safe_normalize(cube_to_dir(s, gx, gy))

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
    return cubemap

def cubemap_to_latlong(cubemap, res):
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                            indexing='ij')
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    
    reflvec = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    return dr.texture(cubemap[None, ...], reflvec[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0]