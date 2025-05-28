import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import trimesh
import pyexr
import slangtorch
import time
import csv
import numpy as np
from pbgi.bvhhelpers import *
import math
from plyfile import PlyData, PlyElement
# from scene.gaussian_model import GaussianModel
from utils.general_utils import *

class Renderer:
    def __init__(self):
        self.proxy_xyzs = None

        self.m_gen_ele, self.m_morton_codes, self.m_radixsort, self.m_hierarchy, self.m_bounding_box = get_bvh_m()
        self.m_intersect_test = get_intersection_m()
        
        self.hemi_index_buffers = None
        self.uv_buffers = None
        self.hti_indices = None
        self.proxy_rot_mats = None
        class Renderer_indirect_with_index_buffer(torch.autograd.Function):
            @staticmethod
            def forward(ctx, N, ray_directions, view_directions,  albedos, roughnesses, metallics, direct, hemi_index_buffers):
                # irradiance = torch.zeros((N, 3), dtype=torch.float).cuda()
                irradiance = torch.zeros((N, 256, 3), dtype=torch.float32).cuda()

                kernel_with_args = self.m_intersect_test.render_indirect_with_index_buffer(N=N, ray_directions=ray_directions, view_directions=view_directions,
                    albedos=albedos, roughnesses=roughnesses, metallics=metallics, direct=direct, 
                    hit_indices=hemi_index_buffers, irradiance=irradiance)
                kernel_with_args.launchRaw(blockSize=(256, 1, 1), gridSize=((N+255)//256, 256, 1))

                ctx.save_for_backward(ray_directions, view_directions, albedos, roughnesses, metallics, direct, hemi_index_buffers, irradiance)
                return irradiance

            @staticmethod
            def backward(ctx, grad_irradiance):
                ray_directions, view_directions, albedos, roughnesses, metallics, direct, hemi_index_buffers, irradiance = ctx.saved_tensors

                grad_ray_directions = torch.zeros_like(ray_directions).cuda()
                grad_view_directions = torch.zeros_like(view_directions).cuda()

                grad_albedos = torch.zeros_like(albedos).cuda()
                grad_roughnesses = torch.zeros_like(roughnesses).cuda()
                grad_metallics = torch.zeros_like(metallics).cuda()
                grad_direct = torch.zeros_like(direct).cuda()

                grad_hemi_index_buffers = torch.zeros_like(hemi_index_buffers)

                N = albedos.shape[0]

                kernel_with_args=self.m_intersect_test.render_indirect_with_index_buffer.bwd(N=N, ray_directions=(ray_directions, grad_ray_directions), view_directions=(view_directions, grad_view_directions),
                    albedos=(albedos, grad_albedos), roughnesses=(roughnesses, grad_roughnesses), metallics=(metallics, grad_metallics), direct=(direct, grad_direct), hit_indices=hemi_index_buffers,
                    irradiance=(irradiance, grad_irradiance))
                kernel_with_args.launchRaw(blockSize=(256, 1, 1), gridSize=((N+255)//256, 1, 1))

                return None, grad_ray_directions, grad_view_directions, grad_albedos, grad_roughnesses, grad_metallics, grad_direct, grad_hemi_index_buffers
            
        ray_directions = self.get_hemi_directions() # [256, 3]

        self.ray_directions = ray_directions

        self.renderer_indirect_with_index_buffer = Renderer_indirect_with_index_buffer()

        class Renderer_direct(torch.autograd.Function):
            @staticmethod
            def forward(ctx, H, W, view_dirs, normal_map, albedo_map, roughness_map, metallic_map, envmap):
                direct = torch.zeros_like(normal_map)
                debug_results = torch.zeros((normal_map.shape[0], normal_map.shape[1], 9), dtype=torch.float).cuda()
                kernel_with_args = self.m_intersect_test.render_direct(H=H, W=W, view_dirs=view_dirs, 
                                                    normal_map=normal_map, albedo_map=albedo_map, roughness_map=roughness_map, metallic_map=metallic_map,
                                                    envmap=envmap, direct=direct, debug_results=debug_results)
                kernel_with_args.launchRaw(blockSize=(16, 16, 1), gridSize=((H+15)//16, (W+15)//16, 1))
                ctx.save_for_backward(view_dirs, normal_map, albedo_map, roughness_map, metallic_map, envmap, direct, debug_results)
                return direct, debug_results
            
            @staticmethod
            def backward(ctx, grad_direct, grad_debug_results):
                view_dirs, normal_map, albedo_map, roughness_map, metallic_map, envmap, direct, debug_results = ctx.saved_tensors
                grad_view_dirs = torch.zeros_like(view_dirs).cuda()
                grad_normal_map = torch.zeros_like(normal_map).cuda()
                grad_albedo_map = torch.zeros_like(albedo_map).cuda()
                grad_roughness_map = torch.zeros_like(roughness_map).cuda()
                grad_metallic_map = torch.zeros_like(metallic_map).cuda()
                grad_envmap = torch.zeros_like(envmap).cuda()
                grad_direct = grad_direct.contiguous()

                H, W = normal_map.shape[0], normal_map.shape[1]


                kernel_with_args=self.m_intersect_test.render_direct.bwd(H=H, W=W, view_dirs=(view_dirs, grad_view_dirs), 
                    normal_map=(normal_map, grad_normal_map), albedo_map=(albedo_map, grad_albedo_map), roughness_map=(roughness_map, grad_roughness_map), metallic_map=(metallic_map, grad_metallic_map),
                    envmap=(envmap, grad_envmap), direct=(direct, grad_direct), debug_results=debug_results)
                kernel_with_args.launchRaw(blockSize=(16, 16, 1), gridSize=((H+15)//16, (W+15)//16, 1))

                return None, None, grad_view_dirs, grad_normal_map, grad_albedo_map, grad_roughness_map, grad_metallic_map, grad_envmap

        self.renderer_direct = Renderer_direct()


        class Renderer_indexing(torch.autograd.Function):
            @staticmethod
            def forward(ctx, H, W, indirect, proxy_index_map):
                indirect_map = torch.zeros(H, W, 3).cuda()

                kernel_with_args = self.m_intersect_test.render_indexing(H=H, W=W, indirect=indirect, hit_map=proxy_index_map, 
                                                    indirect_map=indirect_map)
                kernel_with_args.launchRaw(blockSize=(16, 16, 1), gridSize=((H+15)//16, (W+15)//16, 1))
                ctx.save_for_backward(indirect, proxy_index_map, indirect_map)
                return indirect_map
            
            @staticmethod
            def backward(ctx, grad_indirect_map):
                indirect , proxy_index_map, indirect_map = ctx.saved_tensors
                grad_indirect = torch.zeros_like(indirect).cuda()
                grad_proxy_index_map = torch.zeros_like(proxy_index_map).cuda()
                grad_indirect_map = grad_indirect_map.contiguous()

                H, W = proxy_index_map.shape[0], proxy_index_map.shape[1]

                kernel_with_args=self.m_intersect_test.render_indexing.bwd(H=H, W=W, indirect=(indirect, grad_indirect), hit_map=proxy_index_map,
                                                                           indirect_map=(indirect_map, grad_indirect_map))
                kernel_with_args.launchRaw(blockSize=(16, 16, 1), gridSize=((H+15)//16, (W+15)//16, 1))

                return None, None, grad_indirect, grad_proxy_index_map

        self.renderer_indexing = Renderer_indexing()

        class Renderer_irradiance(torch.autograd.Function):
            @staticmethod
            def forward(ctx, N, S, ray_directions, envmap, LBVHNode_info, LBVHNode_aabb, centers, scales, rotates, normals, albedos, roughnesses, metallics, opacities, hit_indices, uvs, SHs):
                irradiance = torch.zeros(N, S, 3).cuda()
                kernel_with_args = self.m_intersect_test.render_irradiance(N=N, S=S, ray_directions=ray_directions, envmap=envmap, g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb, 
                                                                           centers=centers, scales=scales, rotates=rotates, 
                                                                           normals=normals, albedos=albedos, roughnesses=roughnesses, metallics=metallics, opacities=opacities, hit_indices=hit_indices, uvs=uvs, SHs=SHs, irradiance=irradiance)
                # kernel_with_args.launchRaw(blockSize=(256, 256, 1), gridSize=((N+255)//256, (256*256 + 255)//256, 1))
                kernel_with_args.launchRaw(blockSize=(256, 1, 1), gridSize=((N+255)//256, S, S))
                ctx.save_for_backward(ray_directions, envmap, LBVHNode_info, LBVHNode_aabb, centers, scales, rotates, normals, albedos, roughnesses, metallics, opacities, hit_indices, uvs, SHs, irradiance)

                return irradiance
            
            @staticmethod
            def backward(ctx, grad_irradiance):
                ray_directions, envmap, LBVHNode_info, LBVHNode_aabb, centers, scales, rotates, normals, albedos, roughnesses, metallics, opacities, hit_indices, uvs, SHs, irradiance = ctx.saved_tensors
                grad_ray_directions = torch.zeros_like(ray_directions).cuda()
                grad_envmap = torch.zeros_like(envmap).cuda()
                grad_LBVHNode_info = torch.zeros_like(LBVHNode_info).cuda()
                grad_LBVHNode_aabb = torch.zeros_like(LBVHNode_aabb).cuda()
                grad_centers = torch.zeros_like(centers).cuda()
                grad_scales = torch.zeros_like(scales).cuda()
                grad_rotates = torch.zeros_like(rotates).cuda()
                grad_normals = torch.zeros_like(normals).cuda()
                grad_albedos = torch.zeros_like(albedos).cuda()
                grad_roughnesses = torch.zeros_like(roughnesses).cuda()
                grad_metallics = torch.zeros_like(metallics).cuda()
                grad_opacities = torch.zeros_like(opacities).cuda()
                grad_hit_indices = torch.zeros_like(hit_indices).cuda()
                grad_uvs = torch.zeros_like(uvs).cuda()
                grad_SHs = torch.zeros_like(SHs).cuda()
                grad_irradiance = grad_irradiance.contiguous()

                N = albedos.shape[0]
                S = hit_indices.shape[1]

                kernel_with_args=self.m_intersect_test.render_irradiance.bwd(N=N, S=S, ray_directions=ray_directions, envmap=(envmap, grad_envmap), 
                                                                             g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb, 
                                                                             centers=centers, scales=scales, rotates=rotates, 
                                                                             normals=normals, albedos=(albedos, grad_albedos), roughnesses=(roughnesses, grad_roughnesses), metallics=(metallics, grad_metallics), opacities=opacities, hit_indices=hit_indices, uvs=uvs, SHs=SHs, irradiance=(irradiance, grad_irradiance))
                # kernel_with_args.launchRaw(blockSize=(256, 256, 1), gridSize=((N+255)//256, (256*256 + 255)//256, 1))
                kernel_with_args.launchRaw(blockSize=(256, 1, 1), gridSize=((N+255)//256, S, S))

                return None, None, grad_ray_directions, grad_envmap, grad_LBVHNode_info, grad_LBVHNode_aabb, grad_centers, grad_scales, grad_rotates, grad_normals, grad_albedos, grad_roughnesses, grad_metallics, grad_opacities, grad_hit_indices, grad_uvs, grad_SHs
        self.renderer_irradiance = Renderer_irradiance()


        class Renderer_irradiance_sample(torch.autograd.Function):
            @staticmethod
            def forward(ctx, N, S, sample_indices, ray_directions, envmap, LBVHNode_info, LBVHNode_aabb, centers, scales, rotates, normals, albedos, roughnesses, metallics, opacities, hit_indices, uvs, SHs):
                irradiance = torch.zeros(N, 3).cuda()
                kernel_with_args = self.m_intersect_test.render_irradiance_sample(N=N, S=S, sample_indices=sample_indices, ray_directions=ray_directions, envmap=envmap, g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb, 
                                                                           centers=centers, scales=scales, rotates=rotates, 
                                                                           normals=normals, albedos=albedos, roughnesses=roughnesses, metallics=metallics, opacities=opacities, hit_indices=hit_indices, uvs=uvs, SHs=SHs, irradiance=irradiance)
                # kernel_with_args.launchRaw(blockSize=(256, 256, 1), gridSize=((N+255)//256, (256*256 + 255)//256, 1))
                kernel_with_args.launchRaw(blockSize=(256, 1, 1), gridSize=((N+255)//256, S, 1))
                ctx.save_for_backward(sample_indices, ray_directions, envmap, LBVHNode_info, LBVHNode_aabb, centers, scales, rotates, normals, albedos, roughnesses, metallics, opacities, hit_indices, uvs, SHs,irradiance)

                return irradiance
            
            @staticmethod
            def backward(ctx, grad_irradiance):
                sample_indices, ray_directions, envmap, LBVHNode_info, LBVHNode_aabb, centers, scales, rotates, normals, albedos, roughnesses, metallics, opacities, hit_indices, uvs, SHs, irradiance = ctx.saved_tensors
                grad_sample_indices = torch.zeros_like(sample_indices).cuda()
                grad_ray_directions = torch.zeros_like(ray_directions).cuda()
                grad_envmap = torch.zeros_like(envmap).cuda()
                grad_LBVHNode_info = torch.zeros_like(LBVHNode_info).cuda()
                grad_LBVHNode_aabb = torch.zeros_like(LBVHNode_aabb).cuda()
                grad_centers = torch.zeros_like(centers).cuda()
                grad_scales = torch.zeros_like(scales).cuda()
                grad_rotates = torch.zeros_like(rotates).cuda()
                grad_normals = torch.zeros_like(normals).cuda()
                grad_albedos = torch.zeros_like(albedos).cuda()
                grad_roughnesses = torch.zeros_like(roughnesses).cuda()
                grad_metallics = torch.zeros_like(metallics).cuda()
                grad_opacities = torch.zeros_like(opacities).cuda()
                grad_hit_indices = torch.zeros_like(hit_indices).cuda()
                grad_uvs = torch.zeros_like(uvs).cuda()
                grad_SHs = torch.zeros_like(SHs).cuda()
                grad_irradiance = grad_irradiance.contiguous()

                N = albedos.shape[0]
                S = hit_indices.shape[1]

                kernel_with_args=self.m_intersect_test.render_irradiance_sample.bwd(N=N, S=S, sample_indices=sample_indices, ray_directions=ray_directions, envmap=(envmap, grad_envmap), 
                                                                             g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb, 
                                                                             centers=centers, scales=scales, rotates=rotates, 
                                                                             normals=normals, albedos=(albedos, grad_albedos), roughnesses=(roughnesses, grad_roughnesses), metallics=(metallics, grad_metallics), opacities=opacities, hit_indices=hit_indices, uvs=uvs, SHs=SHs, irradiance=(irradiance, grad_irradiance))
                # kernel_with_args.launchRaw(blockSize=(256, 256, 1), gridSize=((N+255)//256, (256*256 + 255)//256, 1))
                kernel_with_args.launchRaw(blockSize=(256, 1, 1), gridSize=((N+255)//256, 1, S))

                return None, None, grad_sample_indices, grad_ray_directions, grad_envmap, grad_LBVHNode_info, grad_LBVHNode_aabb, grad_centers, grad_scales, grad_rotates, grad_normals, grad_albedos, grad_roughnesses, grad_metallics, grad_opacities, grad_hit_indices, grad_uvs, grad_SHs
        self.renderer_irradiance_sample = Renderer_irradiance_sample()

        class Renderer_direct_indirect(torch.autograd.Function):
            @staticmethod
            def forward(ctx, N, view_dirs, ray_directions, normals, albedos, roughnesses, metallics, irradiance):
                rgb = torch.zeros(view_dirs.shape[0], 3).cuda()
                kernel_with_args = self.m_intersect_test.render_direct_and_indirect(N=N, view_dirs=view_dirs, ray_directions=ray_directions, normals=normals, albedos=albedos, roughnesses=roughnesses, metallics=metallics, irradiance=irradiance, rgb=rgb)
                kernel_with_args.launchRaw(blockSize=(256, 1, 1), gridSize=((N+255)//256, 256, 1))
                ctx.save_for_backward(view_dirs, ray_directions, normals, albedos, roughnesses, metallics, irradiance, rgb)
                return rgb
            
            @staticmethod
            def backward(ctx, grad_rgb):
                view_dirs, ray_directions, normals, albedos, roughnesses, metallics, irradiance, rgb = ctx.saved_tensors
                grad_view_dirs = torch.zeros_like(view_dirs).cuda()
                grad_ray_directions = torch.zeros_like(ray_directions).cuda()
                grad_normals = torch.zeros_like(normals).cuda()
                grad_albedos = torch.zeros_like(albedos).cuda()
                grad_roughnesses = torch.zeros_like(roughnesses).cuda()
                grad_metallics = torch.zeros_like(metallics).cuda()
                grad_irradiance = torch.zeros_like(irradiance).cuda()
                grad_rgb = grad_rgb.contiguous()

                N = view_dirs.shape[0]

                kernel_with_args=self.m_intersect_test.render_direct_and_indirect.bwd(N=N, view_dirs=(view_dirs, grad_view_dirs), ray_directions=(ray_directions, grad_ray_directions), normals=(normals, grad_normals), albedos=(albedos, grad_albedos), roughnesses=(roughnesses, grad_roughnesses), metallics=(metallics, grad_metallics), irradiance=(irradiance, grad_irradiance), rgb=(rgb, grad_rgb))
                kernel_with_args.launchRaw(blockSize=(256, 1, 1), gridSize=((N+255)//256, 256, 1))

                return None, grad_view_dirs, grad_ray_directions, grad_normals, grad_albedos, grad_roughnesses, grad_metallics, grad_irradiance
        self.renderer_direct_indirect = Renderer_direct_indirect()

        class Renderer_with_SH(torch.autograd.Function):
            @staticmethod
            def forward(ctx, N, ray_directions, view_directions, normals, albedos, roughnesses, metallics, envmap, SHs, hit_indices):
                Le = torch.zeros(N, 256, 3).cuda()

                kernel_with_args = self.m_intersect_test.render_with_SH(N=N, ray_directions=ray_directions, view_directions=view_directions, normals=normals, albedos=albedos, roughnesses=roughnesses, metallics=metallics, envmap=envmap, SHs=SHs, hit_indices=hit_indices, Le=Le)
                kernel_with_args.launchRaw(blockSize=(256, 1, 1), gridSize=((N+255)//256, 256, 1))

                ctx.save_for_backward(ray_directions, view_directions, normals, albedos, roughnesses, metallics, envmap, SHs, hit_indices, Le)
                return Le
            
            @staticmethod
            def backward(ctx, grad_Le):
                ray_directions, view_directions, normals, albedos, roughnesses, metallics, envmap, SHs, hit_indices, Le = ctx.saved_tensors
                grad_ray_directions = torch.zeros_like(ray_directions).cuda()
                grad_view_directions = torch.zeros_like(view_directions).cuda()
                grad_normals = torch.zeros_like(normals).cuda()
                grad_albedos = torch.zeros_like(albedos).cuda()
                grad_roughnesses = torch.zeros_like(roughnesses).cuda()
                grad_metallics = torch.zeros_like(metallics).cuda()
                grad_envmap = torch.zeros_like(envmap).cuda()
                grad_SHs = torch.zeros_like(SHs).cuda()
                grad_hit_indices = torch.zeros_like(hit_indices).cuda()
                grad_Le = grad_Le.contiguous()

                N = albedos.shape[0]

                kernel_with_args=self.m_intersect_test.render_with_SH.bwd(N=N, ray_directions=(ray_directions, grad_ray_directions), view_directions=(view_directions, grad_view_directions), normals=(normals, grad_normals), albedos=(albedos, grad_albedos), roughnesses=(roughnesses, grad_roughnesses), metallics=(metallics, grad_metallics), envmap=(envmap, grad_envmap), SHs=SHs, hit_indices=hit_indices, Le=(Le, grad_Le))
                kernel_with_args.launchRaw(blockSize=(256, 1, 1), gridSize=((N+255)//256, 256, 1))

                return None, grad_ray_directions, grad_view_directions, grad_normals, grad_albedos, grad_roughnesses, grad_metallics, grad_envmap, grad_SHs, grad_hit_indices
        self.renderer_with_SH = Renderer_with_SH()


        class Renderer_direct_with_SH(torch.autograd.Function):
            @staticmethod
            def forward(ctx, N, ray_directions, view_directions, normals, albedos, roughnesses, metallics, envmap, SHs, hit_indices):
                Le = torch.zeros(N, 256, 3).cuda()

                kernel_with_args = self.m_intersect_test.render_direct_with_SH(N=N, ray_directions=ray_directions, view_directions=view_directions, normals=normals, albedos=albedos, roughnesses=roughnesses, metallics=metallics, envmap=envmap, SHs=SHs, hit_indices=hit_indices, Le=Le)
                kernel_with_args.launchRaw(blockSize=(256, 1, 1), gridSize=((N+255)//256, 256, 1))

                ctx.save_for_backward(ray_directions, view_directions, normals, albedos, roughnesses, metallics, envmap, SHs, hit_indices, Le)
                return Le
            
            @staticmethod
            def backward(ctx, grad_Le):
                ray_directions, view_directions, normals, albedos, roughnesses, metallics, envmap, SHs, hit_indices, Le = ctx.saved_tensors
                grad_ray_directions = torch.zeros_like(ray_directions).cuda()
                grad_view_directions = torch.zeros_like(view_directions).cuda()
                grad_normals = torch.zeros_like(normals).cuda()
                grad_albedos = torch.zeros_like(albedos).cuda()
                grad_roughnesses = torch.zeros_like(roughnesses).cuda()
                grad_metallics = torch.zeros_like(metallics).cuda()
                grad_envmap = torch.zeros_like(envmap).cuda()
                grad_SHs = torch.zeros_like(SHs).cuda()
                grad_hit_indices = torch.zeros_like(hit_indices).cuda()
                grad_Le = grad_Le.contiguous()

                N = albedos.shape[0]

                kernel_with_args=self.m_intersect_test.render_direct_with_SH.bwd(N=N, ray_directions=(ray_directions, grad_ray_directions), view_directions=(view_directions, grad_view_directions), normals=(normals, grad_normals), albedos=(albedos, grad_albedos), roughnesses=(roughnesses, grad_roughnesses), metallics=(metallics, grad_metallics), envmap=(envmap, grad_envmap), SHs=SHs, hit_indices=hit_indices, Le=(Le, grad_Le))
                kernel_with_args.launchRaw(blockSize=(256, 1, 1), gridSize=((N+255)//256, 256, 1))

                return None, grad_ray_directions, grad_view_directions, grad_normals, grad_albedos, grad_roughnesses, grad_metallics, grad_envmap, grad_SHs, grad_hit_indices
        self.renderer_direct_with_SH = Renderer_direct_with_SH()

        class Renderer_indirect_with_SH(torch.autograd.Function):
            @staticmethod
            def forward(ctx, N, ray_directions, view_directions, normals, albedos, roughnesses, metallics, envmap, SHs, hit_indices):
                Le = torch.zeros(N, 256, 3).cuda()

                kernel_with_args = self.m_intersect_test.render_indirect_with_SH(N=N, ray_directions=ray_directions, view_directions=view_directions, normals=normals, albedos=albedos, roughnesses=roughnesses, metallics=metallics, envmap=envmap, SHs=SHs, hit_indices=hit_indices, Le=Le)
                kernel_with_args.launchRaw(blockSize=(256, 1, 1), gridSize=((N+255)//256, 256, 1))

                ctx.save_for_backward(ray_directions, view_directions, normals, albedos, roughnesses, metallics, envmap, SHs, hit_indices, Le)
                return Le
            
            @staticmethod
            def backward(ctx, grad_Le):
                ray_directions, view_directions, normals, albedos, roughnesses, metallics, envmap, SHs, hit_indices, Le = ctx.saved_tensors
                grad_ray_directions = torch.zeros_like(ray_directions).cuda()
                grad_view_directions = torch.zeros_like(view_directions).cuda()
                grad_normals = torch.zeros_like(normals).cuda()
                grad_albedos = torch.zeros_like(albedos).cuda()
                grad_roughnesses = torch.zeros_like(roughnesses).cuda()
                grad_metallics = torch.zeros_like(metallics).cuda()
                grad_envmap = torch.zeros_like(envmap).cuda()
                grad_SHs = torch.zeros_like(SHs).cuda()
                grad_hit_indices = torch.zeros_like(hit_indices).cuda()
                grad_Le = grad_Le.contiguous()

                N = albedos.shape[0]

                kernel_with_args=self.m_intersect_test.render_indirect_with_SH.bwd(N=N, ray_directions=(ray_directions, grad_ray_directions), view_directions=(view_directions, grad_view_directions), normals=(normals, grad_normals), albedos=(albedos, grad_albedos), roughnesses=(roughnesses, grad_roughnesses), metallics=(metallics, grad_metallics), envmap=(envmap, grad_envmap), SHs=SHs, hit_indices=hit_indices, Le=(Le, grad_Le))
                kernel_with_args.launchRaw(blockSize=(256, 1, 1), gridSize=((N+255)//256, 256, 1))

                return None, grad_ray_directions, grad_view_directions, grad_normals, grad_albedos, grad_roughnesses, grad_metallics, grad_envmap, grad_SHs, grad_hit_indices
        self.renderer_indirect_with_SH = Renderer_indirect_with_SH()

        class Renderer_irradiance_with_SH(torch.autograd.Function):
            @staticmethod
            def forward(ctx, N, ray_directions, view_directions, normals, albedos, roughnesses, metallics, envmap, SHs, hit_indices):
                Le = torch.zeros(N, 256, 3).cuda()

                kernel_with_args = self.m_intersect_test.render_irradiance_with_SH(N=N, ray_directions=ray_directions, view_directions=view_directions, normals=normals, albedos=albedos, roughnesses=roughnesses, metallics=metallics, envmap=envmap, SHs=SHs, hit_indices=hit_indices, Le=Le)
                kernel_with_args.launchRaw(blockSize=(256, 1, 1), gridSize=((N+255)//256, 256, 1))

                ctx.save_for_backward(ray_directions, view_directions, normals, albedos, roughnesses, metallics, envmap, SHs, hit_indices, Le)
                return Le
            
            @staticmethod
            def backward(ctx, grad_Le):
                ray_directions, view_directions, normals, albedos, roughnesses, metallics, envmap, SHs, hit_indices, Le = ctx.saved_tensors
                grad_ray_directions = torch.zeros_like(ray_directions).cuda()
                grad_view_directions = torch.zeros_like(view_directions).cuda()
                grad_normals = torch.zeros_like(normals).cuda()
                grad_albedos = torch.zeros_like(albedos).cuda()
                grad_roughnesses = torch.zeros_like(roughnesses).cuda()
                grad_metallics = torch.zeros_like(metallics).cuda()
                grad_envmap = torch.zeros_like(envmap).cuda()
                grad_SHs = torch.zeros_like(SHs).cuda()
                grad_hit_indices = torch.zeros_like(hit_indices).cuda()
                grad_Le = grad_Le.contiguous()

                N = albedos.shape[0]

                kernel_with_args=self.m_intersect_test.render_irradiance_with_SH.bwd(N=N, ray_directions=(ray_directions, grad_ray_directions), view_directions=(view_directions, grad_view_directions), normals=(normals, grad_normals), albedos=(albedos, grad_albedos), roughnesses=(roughnesses, grad_roughnesses), metallics=(metallics, grad_metallics), envmap=(envmap, grad_envmap), SHs=SHs, hit_indices=hit_indices, Le=(Le, grad_Le))
                kernel_with_args.launchRaw(blockSize=(256, 1, 1), gridSize=((N+255)//256, 256, 1))

                return None, grad_ray_directions, grad_view_directions, grad_normals, grad_albedos, grad_roughnesses, grad_metallics, grad_envmap, grad_SHs, grad_hit_indices
        self.renderer_irradiance_with_SH = Renderer_irradiance_with_SH()

        class Renderer_with_radiance(torch.autograd.Function):
            @staticmethod
            def forward(ctx, N, ray_directions, view_directions, normals, albedos, roughnesses, metallics, envmap, radiance):
                Le = torch.zeros(N, 256, 3).cuda()

                kernel_with_args = self.m_intersect_test.render_with_radiance(N=N, ray_directions=ray_directions, view_directions=view_directions, normals=normals, albedos=albedos, roughnesses=roughnesses, metallics=metallics, envmap=envmap, radiance=radiance, Le=Le)
                kernel_with_args.launchRaw(blockSize=(256, 1, 1), gridSize=((N+255)//256, 256, 1))

                ctx.save_for_backward(ray_directions, view_directions, normals, albedos, roughnesses, metallics, envmap, radiance, Le)
                return Le
            
            @staticmethod
            def backward(ctx, grad_Le):
                ray_directions, view_directions, normals, albedos, roughnesses, metallics, envmap, radiance, Le = ctx.saved_tensors
                grad_ray_directions = torch.zeros_like(ray_directions).cuda()
                grad_view_directions = torch.zeros_like(view_directions).cuda()
                grad_normals = torch.zeros_like(normals).cuda()
                grad_albedos = torch.zeros_like(albedos).cuda()
                grad_roughnesses = torch.zeros_like(roughnesses).cuda()
                grad_metallics = torch.zeros_like(metallics).cuda()
                grad_envmap = torch.zeros_like(envmap).cuda()
                grad_radiance = torch.zeros_like(radiance).cuda()
                grad_Le = grad_Le.contiguous()

                N = albedos.shape[0]

                kernel_with_args=self.m_intersect_test.render_with_radiance.bwd(N=N, ray_directions=(ray_directions, grad_ray_directions), view_directions=(view_directions, grad_view_directions), normals=(normals, grad_normals), albedos=(albedos, grad_albedos), roughnesses=(roughnesses, grad_roughnesses), metallics=(metallics, grad_metallics), radiance=radiance, envmap=(envmap, grad_envmap), Le=(Le, grad_Le))
                kernel_with_args.launchRaw(blockSize=(256, 1, 1), gridSize=((N+255)//256, 256, 1))

                return None, grad_ray_directions, grad_view_directions, grad_normals, grad_albedos, grad_roughnesses, grad_metallics, grad_envmap, grad_radiance
        self.renderer_with_radiance = Renderer_with_radiance()

    def set_proxy(self, xyzs, scales, rotates, normals):
        self.proxy_xyzs = xyzs
        self.proxy_scales = scales
        self.proxy_rotates = rotates

        self.proxy_normals = normals

    def set_proxy_from_gaussian_model(self, pc):
        xyzs = pc.get_xyz
        scales = pc.get_scaling
        rotates = pc.get_rotation
        normals = pc.get_geo_normal
        opacity = pc.get_opacity

        

        
        # proxy_idx = torch.nonzero(pc.get_opacity > 0.24)[..., 0].int()
        proxy_idx = torch.nonzero(pc.get_opacity > 0.0)[...,0].long()
        
        self.proxy_xyzs = pc.get_xyz
        self.proxy_scales = pc.get_scaling
        # self.proxy_rotates = pc.get_rotation[proxy_idx]
        self.proxy_normals = pc.get_geo_normal
# 
        self.proxy_rotates = rotates
        # self.proxy_normals = normal / (normal.norm(dim=-1, keepdim=True) + 1e-6)

        self.proxy_albedos = pc.get_albedo
        self.proxy_roughnesses = pc.get_roughness
        self.proxy_metallics = pc.get_metallic
        self.proxy_opacity = pc.get_opacity
        self.proxy_features = pc.get_features

        self.proxy_idx = proxy_idx

        indices = torch.arange(proxy_idx.size(0)).int().cuda()
        self.proxy_idx_table = torch.full((xyzs.size(0),), -1, dtype=torch.int32).cuda() # [N]
        self.proxy_idx_table[proxy_idx] = indices.cuda()


        # rot_mats = build_rotation_matrix_from_normal(self.proxy_normals) # [N, 3, 3]
        rot_mats = create_rotation_matrix_from_direction_vector_batch(self.proxy_normals) # [N, 3, 3]

        self.proxy_rot_mats = rot_mats



    def get_hemi_directions(self):
        theta, phi = torch.meshgrid([torch.linspace(0, 1, 16), 
                     torch.linspace(0, 1, 16)], indexing='ij')    # [16, 16]

        theta = 2 * torch.pi * theta.cuda() # (0, 2pi)
        phi = torch.acos(1 - phi).cuda() # (0, pi/2)
        # phi = 0.5 * torch.pi * phi.cuda() # (0, pi/2)

        dir_x = torch.sin(phi) * torch.cos(theta)  # [16, 16]
        dir_y = torch.sin(phi) * torch.sin(theta)
        dir_z = torch.cos(phi)

        dir_x = dir_x.unsqueeze(-1)
        dir_y = dir_y.unsqueeze(-1)
        dir_z = dir_z.unsqueeze(-1)

        ray_directions = torch.cat([dir_x, dir_y, dir_z], axis=2) # [16, 16, 3] 
        ray_directions = ray_directions.reshape(-1, 3) # [256, 3]

        return ray_directions
    
    def calculate_hemi_index_buffers(self): 
        if self.proxy_idx.shape[0] == 0:
            return
        torch.cuda.synchronize()
        time_start_buffer = time.time()
        # build bvh
        LBVHNode_info, LBVHNode_aabb = get_gs_bvh(self.proxy_xyzs, self.proxy_scales, self.proxy_rotates, self.m_gen_ele, self.m_morton_codes, self.m_radixsort, self.m_hierarchy, self.m_bounding_box)
        torch.cuda.synchronize()
        self.LBVHNode_info = LBVHNode_info
        self.LBVHNode_aabb = LBVHNode_aabb
        time_end_buffer = time.time()
        print(f"build bvh cost time = {time_end_buffer - time_start_buffer}")    

        # xyzs [N, 3]
        ray_origins = self.proxy_xyzs.reshape(-1,1,3) # [N, 1, 3]
        ray_origins = ray_origins.repeat(1, 256, 1) # [N, 256, 3]

        ray_directions = self.ray_directions.permute(1, 0).cuda().float()  # [3, 256]
        
        ray_directions = torch.matmul(self.proxy_rot_mats, ray_directions).permute(0,2,1) # [N, 256, 3]

        ret = ray_directions[:, 0, :].squeeze(-1)

        ray_directions = ray_directions.reshape(-1, 3) # [N*256, 3]

        ray_origins = ray_origins.contiguous().reshape(-1,3) # [N*256, 3]
        ray_directions = ray_directions.contiguous().reshape(-1,3) #[N*256, 3]

        num_rays=ray_origins.shape[0]

        # hit = torch.zeros((num_rays, 1), dtype=torch.int).cuda() # [N, 1]
        hit = torch.full((num_rays, 1), -1, dtype=torch.int32).cuda() # [N, 256]
        debug_res = torch.zeros((num_rays, 3), dtype=torch.float).cuda() # [N, 3]

        torch.cuda.synchronize()
        time_start_buffer = time.time()
        self.m_intersect_test.hit_table(num_rays=int(num_rays), rays_o=ray_origins, rays_d=ray_directions,
                    g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb,
                    centers=self.proxy_xyzs, scales=self.proxy_scales, rotates=self.proxy_rotates, colors=self.proxy_normals, opacity=self.proxy_opacity,
                    index_map=hit, debug_result=debug_res)\
        .launchRaw(blockSize=(256, 1, 1), gridSize=((num_rays+255)//256, 1, 1))
        torch.cuda.synchronize()
        time_end_buffer = time.time()
        print(f"index table cost time = {time_end_buffer - time_start_buffer}")
        
        self.hemi_index_buffers = hit.reshape(-1, 256) # [N, 256] proxy index
        return ret # [N, 3]
        
    def render_radiance_with_SH(self):
        if self.proxy_idx.shape[0] == 0:
            return
        torch.cuda.synchronize()
        time_start_buffer = time.time()
        # build bvh
        LBVHNode_info, LBVHNode_aabb = get_gs_bvh(self.proxy_xyzs, self.proxy_scales, self.proxy_rotates, self.m_gen_ele, self.m_morton_codes, self.m_radixsort, self.m_hierarchy, self.m_bounding_box)
        self.LBVHNode_info = LBVHNode_info
        self.LBVHNode_aabb = LBVHNode_aabb

        torch.cuda.synchronize()
        time_end_buffer = time.time()
        print(f"build bvh cost time = {time_end_buffer - time_start_buffer}")    

        # xyzs [N, 3]
        ray_origins = self.proxy_xyzs.reshape(-1,1,3) # [N, 1, 3]
        ray_origins = ray_origins.repeat(1, 256, 1) # [N, 256, 3]

        ray_directions = self.ray_directions.permute(1, 0).cuda().float()  # [3, 256]
        
        ray_directions = torch.matmul(self.proxy_rot_mats, ray_directions).permute(0,2,1) # [N, 256, 3]

        ret = ray_directions[:, 0, :].squeeze(-1)

        num_rays=ray_origins.shape[0]

        self_radiance_buffer = torch.zeros((num_rays, 256, 3), dtype=torch.float).cuda() # [N, 256, 3]

        torch.cuda.synchronize()
        time_start_buffer = time.time()
        self.m_intersect_test.render_radiance_with_SH(N=int(num_rays), ray_directions=ray_directions,
                    g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb,
                    centers=self.proxy_xyzs, scales=self.proxy_scales, rotates=self.proxy_rotates, colors=self.proxy_normals, opacity=self.proxy_opacity,
                    SHs=self.proxy_features, Le=self_radiance_buffer)\
        .launchRaw(blockSize=(256, 1, 1), gridSize=((num_rays+255)//256, 256, 1))
        torch.cuda.synchronize()
        time_end_buffer = time.time()
        print(f"index table cost time = {time_end_buffer - time_start_buffer}")
        
        # self.hemi_index_buffers = hit.reshape(-1, 256) # [N, 256] proxy index
        self.self_radiance_buffer = self_radiance_buffer
        return ret # [N, 3]
    
    def build_bvh(self):
        if self.proxy_idx.shape[0] == 0:
            return
        torch.cuda.synchronize()
        time_start_buffer = time.time()
        # build bvh
        LBVHNode_info, LBVHNode_aabb = get_gs_bvh(self.proxy_xyzs, self.proxy_scales, self.proxy_rotates, self.m_gen_ele, self.m_morton_codes, self.m_radixsort, self.m_hierarchy, self.m_bounding_box)
        self.LBVHNode_info = LBVHNode_info
        self.LBVHNode_aabb = LBVHNode_aabb

        torch.cuda.synchronize()
        time_end_buffer = time.time()
        print(f"build bvh cost time = {time_end_buffer - time_start_buffer}")   

    def render_radiance_with_sampling_SH(self, ray_o, ray_d, cov3D_inv, sample_num=64):
        num_rays=ray_d.shape[0]
        self_radiance_buffer = torch.zeros((num_rays, sample_num, 3), dtype=torch.float).cuda() # [N, 256, 3]
        visibility = torch.ones((num_rays, sample_num, 1), dtype=torch.float).cuda() # [N, 256, 1]

        hit_indices = torch.zeros((num_rays, sample_num, 1), dtype=torch.int).cuda() # [N, 256, 1]
        uvs = torch.zeros((num_rays, sample_num, 2)).float().cuda()

        torch.cuda.synchronize()
        time_start_buffer = time.time()
        self.m_intersect_test.render_radiance_with_sampling_SH(N=int(num_rays), S=sample_num, ray_origins=ray_o, ray_directions=ray_d,
                    g_lbvh_info=self.LBVHNode_info, g_lbvh_aabb=self.LBVHNode_aabb,
                    centers=self.proxy_xyzs, scales=self.proxy_scales, rotates=self.proxy_rotates, colors=self.proxy_normals, opacity=self.proxy_opacity, cov3D_inverse=cov3D_inv,
                    SHs=self.proxy_features, Le=self_radiance_buffer, visibility=visibility, hit_indices=hit_indices, uvs=uvs)\
        .launchRaw(blockSize=(256, 1, 1), gridSize=((num_rays+255)//256, sample_num, 1))
        torch.cuda.synchronize()
        time_end_buffer = time.time()
        print(f"index table cost time = {time_end_buffer - time_start_buffer}")
        
        return self_radiance_buffer, visibility, hit_indices, uvs # [N, 3]
    
    def render_SH(self, ray_o, ray_d, cov3D_inv):
        H = ray_d.shape[0]
        W = ray_d.shape[1]

        render = torch.zeros((H, W, 3), dtype=torch.float).cuda()

        self.m_intersect_test.render_SH(H=H, W=W, ray_origins=ray_o, ray_directions=ray_d,
                                        g_lbvh_info=self.LBVHNode_info, g_lbvh_aabb=self.LBVHNode_aabb,
                                        centers=self.proxy_xyzs, scales=self.proxy_scales, rotates=self.proxy_rotates, colors=self.proxy_normals, opacity=self.proxy_opacity, cov3D_inverse=cov3D_inv,
                                        SHs=self.proxy_features, Le=render)\
        .launchRaw(blockSize=(16, 16, 1), gridSize=((H+15)//16, (W+15)//16, 1))
        
        return render




    def calculate_iradiance_and_occlusion(self, direct): # direct:  [N, 3]
        if self.proxy_idx.shape[0] == 0:
            return
        # build bvh
        LBVHNode_info, LBVHNode_aabb = get_gs_bvh(self.proxy_xyzs, self.proxy_scales, self.proxy_rotates, self.m_gen_ele, self.m_morton_codes, self.m_radixsort, self.m_hierarchy, self.m_bounding_box)

        # xyzs [N, 3]
        ray_origins = self.proxy_xyzs.reshape(-1,1,3) # [N, 1, 3]
        ray_origins = ray_origins.repeat(1, 256, 1) # [N, 256, 3]

        ray_directions = self.get_hemi_directions() # [256, 3]
        ray_directions = ray_directions.permute(1, 0).cuda().float()  # [3, 256]
        
        # rot_mats = build_rotation_matrix_from_normal(self.proxy_normals) # [N, 3, 3]
        rot_mats = create_rotation_matrix_from_direction_vector_batch(self.proxy_normals) # [N, 3, 3]
        rot_mats = torch.rand_like(rot_mats).cuda()

        ray_directions = torch.matmul(rot_mats, ray_directions).permute(0,2,1) # [N, 3, 256]
        ray_directions = ray_directions.permute(0, 2, 1) # [N, 256, 3]
        ray_directions = ray_directions.reshape(-1, 3) # [N*256, 3]

        ray_origins = ray_origins.contiguous().reshape(-1,3) # [N*256, 3]
        ray_directions = ray_directions.contiguous().reshape(-1,3) #[N*256, 3]

        num_rays=ray_origins.shape[0]

        irradiance = torch.zeros((num_rays, 3), dtype=torch.float).cuda() # [N, 3]
        occlusion = torch.zeros((num_rays, 1), dtype=torch.float).cuda() # [N, 1]
        
        self.m_intersect_test.render_indirect(num_rays=int(num_rays), rays_o=ray_origins, rays_d=ray_directions,
                    g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb,
                    centers=self.proxy_xyzs, scales=self.proxy_scales, rotates=self.proxy_rotates, direct=direct,
                    irradiance=irradiance, occlusion=occlusion)\
        .launchRaw(blockSize=(256, 1, 1), gridSize=((num_rays+255)//256, 1, 1))

        # self.hemi_index_buffers = hit.reshape(-1, 256) # [N, 256] proxy index
        return irradiance, occlusion

    def calculate_iradiance_and_occlusion_with_index_buffer(self, direct): # direct:  [N, 3]
        if self.proxy_idx.shape[0] == 0:
            return
        # build bvh
        LBVHNode_info, LBVHNode_aabb = get_gs_bvh(self.proxy_xyzs, self.proxy_scales, self.proxy_rotates, self.m_gen_ele, self.m_morton_codes, self.m_radixsort, self.m_hierarchy, self.m_bounding_box)

        # xyzs [N, 3]
        ray_origins = self.proxy_xyzs.reshape(-1,1,3) # [N, 1, 3]
        ray_origins = ray_origins.repeat(1, 256, 1) # [N, 256, 3]

        ray_directions = self.get_hemi_directions() # [256, 3]
        ray_directions = ray_directions.permute(1, 0).cuda().float()  # [3, 256]
        
        # rot_mats = build_rotation_matrix_from_normal(self.proxy_normals) # [N, 3, 3]
        rot_mats = create_rotation_matrix_from_direction_vector_batch(self.proxy_normals) # [N, 3, 3]

        ray_directions = torch.matmul(rot_mats, ray_directions).permute(0,2,1) # [N, 3, 256]
        ray_directions = ray_directions.permute(0, 2, 1) # [N, 256, 3]
        ray_directions = ray_directions.reshape(-1, 3) # [N*256, 3]

        ray_origins = ray_origins.contiguous().reshape(-1,3) # [N*256, 3]
        ray_directions = ray_directions.contiguous().reshape(-1,3) #[N*256, 3]

        num_rays=ray_origins.shape[0]

        # irradiance = torch.zeros((num_rays, 3), dtype=torch.float).cuda() # [N, 3]
        # occlusion = torch.zeros((num_rays, 1), dtype=torch.float).cuda() # [N, 1]
        
        irradiance, occlusion = self.renderer_with_index_buffer.apply(num_rays, ray_origins, ray_directions, 
                                               LBVHNode_info, LBVHNode_aabb, 
                                               self.proxy_xyzs, self.proxy_scales, self.proxy_rotates, direct, self.hemi_index_buffers.reshape(-1, 1).int())
        # self.m_intersect_test.render_indirect_with_index_buffer(num_rays=int(num_rays), rays_o=ray_origins, rays_d=ray_directions,
        #             g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb,
        #             centers=self.proxy_xyzs, scales=self.proxy_scales, rotates=self.proxy_rotates, direct=direct, hit_indices=self.hemi_index_buffers.reshape(-1, 1).int(),
        #             irradiance=irradiance, occlusion=occlusion)\
        # .launchRaw(blockSize=(256, 1, 1), gridSize=((num_rays+255)//256, 1, 1))

        # self.hemi_index_buffers = hit.reshape(-1, 256) # [N, 256] proxy index
        return irradiance, occlusion
    
    def get_hemi_index_buffer_map(self, main_index_map): # [H, W, 1] -> [H, W, 256]
        if self.proxy_idx.shape[0] == 0:
            return torch.zeros_like(main_index_map).long()
        # main index map is index in all gaussians
        # so convert it to index in proxy gaussians
        proxy_main_index_map = torch.where(main_index_map.detach()==-1, -1, self.proxy_idx_table[main_index_map.detach().int()]).squeeze(-1) # [H, W, 1] -> [H, W, 1]

        hemi_index_buffers_with_miss = torch.cat([self.hemi_index_buffers.detach(), torch.full((256,), -1, dtype=torch.int32).unsqueeze(0).cuda()], dim=0)
        
        # [H, W, 1] , [H, W, 256], [H, W, 256]
        ret = hemi_index_buffers_with_miss[proxy_main_index_map] # [H, W, 256]
        return ret # [H, W, 256]
    
    def get_proxy_index_map(self, main_index_map):
        if self.proxy_idx.shape[0] == 0:
            return torch.zeros_like(main_index_map).long()
        # main index map is index in all gaussians
        # so convert it to index in proxy gaussians
        return torch.where(main_index_map.detach()==-1, -1, self.proxy_idx_table[main_index_map.detach().int()]).squeeze(-1)
    

    
    def render_direct(self, H, W, view_dirs, normal_map, albedo_map, roughness_map, metallic_map, envmap):
        return self.renderer_direct.apply(H, W, view_dirs, normal_map, albedo_map, roughness_map, metallic_map, envmap)
    
    def render_indirect(self, N, view_directions, albedos, roughnesses, metallics, direct):
        return self.renderer_indirect_with_index_buffer.apply(N, self.ray_directions, view_directions, albedos, roughnesses, metallics, direct, self.hemi_index_buffers)
    
    def render_indexing(self, H, W, indirect, proxy_index_map):
        return self.renderer_indexing.apply(H, W, indirect, proxy_index_map)
    
    def render_irradiance(self, N, S, envmap, ray_directions, centers, scales, rotates, normals, albedos, roughnesses, metallics, opacities, SHs):
        # ray_directions = self.ray_directions.permute(1, 0).cuda().float()  # [3, 256]
        # ray_directions = torch.matmul(self.proxy_rot_mats.detach(), ray_directions).permute(0,2,1) # [N, 256, 3]
        return self.renderer_irradiance.apply(N, S, ray_directions, envmap, self.LBVHNode_info, self.LBVHNode_aabb, centers, scales, rotates, normals, albedos, roughnesses, metallics, opacities, self.hemi_index_buffers, self.uv_buffers, SHs)
    
    def render_irradiance_sample(self, N, S, sample_indices, envmap, ray_directions, centers, scales, rotates, normals, albedos, roughnesses, metallics, opacities, SHs):
        # ray_directions = self.ray_directions.permute(1, 0).cuda().float()  # [3, 256]
        # ray_directions = torch.matmul(self.proxy_rot_mats.detach(), ray_directions).permute(0,2,1) # [N, 256, 3]
        return self.renderer_irradiance_sample.apply(N, S, sample_indices, ray_directions, envmap, self.LBVHNode_info, self.LBVHNode_aabb, centers, scales, rotates, normals, albedos, roughnesses, metallics, opacities, self.hemi_index_buffers, self.uv_buffers, SHs)
    
    def render_direct_indirect(self, N, view_dirs, normals, albedos, roughnesses, metallics, irradiance):
        ray_directions = self.ray_directions.permute(1, 0).cuda().float()  # [3, 256]
        ray_directions = torch.matmul(self.proxy_rot_mats, ray_directions).permute(0,2,1) # [N, 256, 3]
        return self.renderer_direct_indirect.apply(N, view_dirs, ray_directions, normals, albedos, roughnesses, metallics, irradiance)
    
    def render_with_SH(self, N, view_directions, normals, albedos, roughnesses, metallics, envmap, SHs):
        ray_directions = self.ray_directions.permute(1, 0).cuda().float()  # [3, 256]
        ray_directions = torch.matmul(self.proxy_rot_mats, ray_directions).permute(0,2,1) # [N, 256, 3]
        return self.renderer_with_SH.apply(N, ray_directions.detach(), view_directions, normals, albedos, roughnesses, metallics, envmap, SHs, self.hemi_index_buffers)
    
    def render_direct_with_SH(self, N, view_directions, normals, albedos, roughnesses, metallics, envmap, SHs):
        ray_directions = self.ray_directions.permute(1, 0).cuda().float()  # [3, 256]
        ray_directions = torch.matmul(self.proxy_rot_mats, ray_directions).permute(0,2,1) # [N, 256, 3]
        return self.renderer_direct_with_SH.apply(N, ray_directions.detach(), view_directions, normals, albedos, roughnesses, metallics, envmap, SHs, self.hemi_index_buffers)
    
    def render_indirect_with_SH(self, N, view_directions, normals, albedos, roughnesses, metallics, envmap, SHs):
        ray_directions = self.ray_directions.permute(1, 0).cuda().float()  # [3, 256]
        ray_directions = torch.matmul(self.proxy_rot_mats, ray_directions).permute(0,2,1) # [N, 256, 3]
        return self.renderer_indirect_with_SH.apply(N, ray_directions.detach(), view_directions, normals, albedos, roughnesses, metallics, envmap, SHs, self.hemi_index_buffers)
    
    def render_irradiance_with_SH(self, N, view_directions, normals, albedos, roughnesses, metallics, envmap, SHs):
        ray_directions = self.ray_directions.permute(1, 0).cuda().float()  # [3, 256]
        ray_directions = torch.matmul(self.proxy_rot_mats, ray_directions).permute(0,2,1) # [N, 256, 3]
        self.radiance_buffer =  self.renderer_irradiance_with_SH.apply(N, ray_directions.detach(), view_directions, normals, albedos, roughnesses, metallics, envmap, SHs, self.hemi_index_buffers)
    
    def render_with_radiance(self, N, view_directions, normals, albedos, roughnesses, metallics, envmap):
        ray_directions = self.ray_directions.permute(1, 0).cuda().float()
        ray_directions = torch.matmul(self.proxy_rot_mats, ray_directions).permute(0,2,1)
        return self.renderer_with_radiance.apply(N, ray_directions.detach(), view_directions, normals, albedos, roughnesses, metallics, envmap, self.self_radiance_buffer)
    
    
