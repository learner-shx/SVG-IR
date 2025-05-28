import torch
import torch.nn as nn
import numpy as np
from arguments import OptimizationParams
from utils.sh_utils import eval_sh, eval_sh_coef
import nvdiffrast.torch as dr
import torch.nn.functional as F


def render_envmap_sg(lgtSGs, viewdirs, batch_size=10000):
    device = lgtSGs.device
    viewdirs = viewdirs.to(device)
    M = lgtSGs.shape[0]
    
    # Precompute the parts of lgtSGs that are reused across all batches
    lgtSGs = lgtSGs.view((1,) * (len(viewdirs.shape) - 1) + (M, 7))
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True))
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., -3:])
    
    # Result tensor to store the output
    result_rgb = []
    
    # Process in batches to reduce memory load
    for i in range(0, viewdirs.shape[0], batch_size):
        viewdirs_batch = viewdirs[i:i + batch_size].unsqueeze(-2)  # [batch_size, 1, 3]
        
        # Expand lgtSGs for the current batch
        lgtSGs_batch = lgtSGs.expand(viewdirs_batch.shape[:-2] + (M, 7))
        lgtSGLobes_batch = lgtSGLobes.expand(viewdirs_batch.shape[:-2] + (M, 3))
        lgtSGLambdas_batch = lgtSGLambdas.expand(viewdirs_batch.shape[:-2] + (M, 1))
        lgtSGMus_batch = lgtSGMus.expand(viewdirs_batch.shape[:-2] + (M, 3))
        
        # Compute RGB for this batch
        dot_product = torch.sum(viewdirs_batch * lgtSGLobes_batch, dim=-1, keepdim=True)
        rgb_batch = lgtSGMus_batch * torch.exp(lgtSGLambdas_batch * (dot_product - 1.))
        rgb_batch = torch.sum(rgb_batch, dim=-2)  # [batch_size, 3]

        # Append result to the list
        result_rgb.append(rgb_batch)
        del viewdirs_batch, lgtSGLobes_batch, lgtSGLambdas_batch, lgtSGMus_batch, dot_product, rgb_batch
        torch.cuda.empty_cache()  # Clear cached memory
    
    # Concatenate all batched results
    result_rgb = torch.cat(result_rgb, dim=0)  # [500000, 3]
    return result_rgb


    viewdirs = viewdirs.to(lgtSGs.device)
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]

    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])
    M = lgtSGs.shape[0]
    lgtSGs = lgtSGs.view([1,] * len(dots_sh) + [M, 7]).expand(dots_sh + [M, 7])

    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True))
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., -3:]) 
    # [..., M, 3]
    rgb = lgtSGMus * torch.exp(lgtSGLambdas * \
        (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = torch.sum(rgb, dim=-2)  # [..., 3]
    return rgb

def compute_envmap(lgtSGs, H, W, upper_hemi=False):
    # same convetion as blender    
    if upper_hemi:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi/2., H), 
                                     torch.linspace(1.0 * np.pi, -1.0 * np.pi, W)])
    else:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), 
                                     torch.linspace(1.0 * np.pi, -1.0 * np.pi, W)])
    viewdirs = torch.stack([torch.cos(theta) * torch.sin(phi), 
                            torch.sin(theta) * torch.sin(phi), 
                            torch.cos(phi)], dim=-1)    # [H, W, 3]
                            
    rgb = render_envmap_sg(lgtSGs, viewdirs)
    envmap = rgb.reshape((H, W, 3))
    return envmap

class DirectLightSG:
    def __init__(self, numLgtSGs=128, H=128):
        self.H = H
        self.W = H * 2
        self.numLgtSGs = numLgtSGs
        self.lgtSGs = nn.Parameter(torch.randn(numLgtSGs, 7).cuda())  # lobe + lambda + mu
        self.lgtSGs.data[..., 3:4] *= 100.
        self.lgtSGs.requires_grad = True
        
    def training_setup(self, training_args: OptimizationParams):
        l = [
            {'params': [self.lgtSGs], 'lr': training_args.env_lr, "name": "env"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def capture(self):
        captured_list = [
            self.lgtSGs,
            self.optimizer.state_dict(),
        ]

        return captured_list

    def restore(self, model_args, training_args,
                is_training=False, restore_optimizer=True):
        pass

    def create_from_ckpt(self, checkpoint_path, restore_optimizer=False):
        (model_args, first_iter) = torch.load(checkpoint_path)
        (self.lgtSGs,
         opt_dict) = model_args[:2]

        if restore_optimizer:
            try:
                self.optimizer.load_state_dict(opt_dict)
            except:
                print("Not loading optimizer state_dict!")

        return first_iter
    

    
    # def direct_light(self, dirs):
    #     shape = dirs.shape
    #     dirs = dirs.reshape(-1, 3)
    #     import pdb;pdb.set_trace()
    #     tu = torch.atan2(dirs[..., 0:1], -dirs[..., 1:2]) / (2 * np.pi) + 0.5
    #     tv = torch.acos(torch.clamp(dirs[..., 2:3], min=-1, max=1)) / (np.pi/2) - 1
        
        
    #     dirs = (dirs.reshape(-1, 3) @ self.to_opengl.T)
    #     tu = torch.atan2(dirs[..., 0:1], -dirs[..., 2:3]) / (2 * np.pi) + 0.5
    #     tv = torch.acos(torch.clamp(dirs[..., 1:2], min=-1, max=1)) / np.pi
    #     texcoord = torch.cat((tu, tv), dim=-1)
    #     # import pdb;pdb.set_trace()
    #     light = dr.texture(self.env, texcoord[None, None, ...], filter_mode='linear')[0, 0]
    #     return light.reshape(*shape).clamp_min(0)
    
    def direct_light(self, dirs, transform=None):
        light_rgbs = render_envmap_sg(self.lgtSGs, dirs)
        return light_rgbs

    # def upsample(self):
    #     self.env = nn.Parameter(F.interpolate(self.env.data.permute(0, 3, 1, 2), scale_factor=2, mode='bilinear', align_corners=True).permute(0, 2, 3, 1).requires_grad_(True))
    #     self.H *= 2
    #     self.W *= 2
        
    #     for group in self.optimizer.param_groups:
    #         stored_state = self.optimizer.state.get(group['params'][0], None)
    #         if stored_state is not None:
    #             stored_state["exp_avg"] = F.interpolate(stored_state["exp_avg"].permute(0, 3, 1, 2), scale_factor=2, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
    #             stored_state["exp_avg_sq"] = F.interpolate(stored_state["exp_avg_sq"].permute(0, 3, 1, 2), scale_factor=2, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)

    #             del self.optimizer.state[group['params'][0]]

    #             group["params"][0] = nn.Parameter(self.env)
    #             self.optimizer.state[group['params'][0]] = stored_state
    #         else:
    #             group["params"][0] = nn.Parameter(self.env)
                

    @property
    def get_env(self):
        # return self.env
        env = compute_envmap(self.lgtSGs, self.H, self.W)
        return F.softplus(env)