import math
import torch
import numpy as np
import torch.nn.functional as F
from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.sh_utils import eval_sh
from utils.loss_utils import ssim, bilateral_smooth_loss, second_order_edge_aware_loss, tv_loss, first_order_edge_aware_loss, first_order_loss, first_order_edge_aware_norm_loss, cos_loss
from utils.image_utils import psnr, depth2normal, match_depth, normal2curv, resize_image, cross_sample
from utils.graphics_utils import fibonacci_sphere_sampling, rgb_to_srgb, srgb_to_rgb
from .svgss_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import time

def render_view(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                scaling_modifier=1.0, override_color=None, is_training=False, dict_params=None):
    patch_size = [float('inf'), float('inf')]
    direct_light_env_light = dict_params.get("env_light")
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    intrinsic = viewpoint_camera.intrinsics

    # raster_settings = GaussianRasterizationSettings(
    #     image_height=int(viewpoint_camera.image_height),
    #     image_width=int(viewpoint_camera.image_width),
    #     tanfovx=tanfovx,
    #     tanfovy=tanfovy,
    #     cx=float(intrinsic[0, 2]),
    #     cy=float(intrinsic[1, 2]),
    #     bg=bg_color,
    #     scale_modifier=scaling_modifier,
    #     viewmatrix=viewpoint_camera.world_view_transform,
    #     projmatrix=viewpoint_camera.full_proj_transform,
    #     sh_degree=pc.active_sh_degree,
    #     campos=viewpoint_camera.camera_center,
    #     prefiltered=False,
    #     backward_geometry=True,
    #     computer_pseudo_normal=True,
    #     debug=pipe.debug
    # ) 

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        patch_bbox=viewpoint_camera.random_patch(patch_size[0], patch_size[1]),
        prcppoint=viewpoint_camera.prcppoint,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        config=torch.tensor(pc.config).float().cuda()
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.compute_SHs_python:
            dir_pp_normalized = F.normalize(viewpoint_camera.camera_center.repeat(means3D.shape[0], 1) - means3D,
                                            dim=-1)
            shs_view = pc.get_shs.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_shs
    else:
        colors_precomp = override_color

    base_color = pc.get_base_color
    roughness = pc.get_roughness
    normal = pc.get_shading_normal
    # incidents = pc.get_incidents  # incident shs
    incidents = pc.get_radiances  # incident radiance
    # incidents = pc.calc_radiance
    # incidents = torch.zeros_like(incidents)
    viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)

    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_shs.shape[0], 1))
    dir_pp_normalized = F.normalize(dir_pp, dim=-1)

    # time1 = time.time()
    if is_training:
        brdf_color, extra_results = rendering_equation4(
            base_color, roughness, normal, viewdirs, incidents,
            direct_light_env_light, visibility_precompute=pc._visibility_tracing, 
            incident_dirs_precompute=pc._incident_dirs, incident_areas_precompute=pc._incident_areas)
    else:
        chunk_size = 100000
        brdf_color = []
        extra_results = []
        for i in range(0, means3D.shape[0], chunk_size):
            _brdf_color, _extra_results = rendering_equation4(
                base_color[i:i + chunk_size], roughness[i:i + chunk_size], 
                normal[i:i + chunk_size].detach(), viewdirs[i:i + chunk_size], incidents[i:i + chunk_size],
                direct_light_env_light, 
                visibility_precompute=pc._visibility_tracing[i:i + chunk_size], 
                incident_dirs_precompute=pc._incident_dirs[i:i + chunk_size], 
                incident_areas_precompute=pc._incident_areas[i:i + chunk_size])
            brdf_color.append(_brdf_color)
            extra_results.append(_extra_results)
        brdf_color = torch.cat(brdf_color, dim=0)
        extra_results = {k: torch.cat([_extra_results[k] for _extra_results in extra_results], dim=0) for k in extra_results[0]}
        torch.cuda.empty_cache()

    # time2 = time.time()
    # cost_time = time2 - time1

    

    xyz_homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)
    depths = (xyz_homo @ viewpoint_camera.world_view_transform)[:, 2:3]
    depths2 = depths.square()

    # cost_time = 0.0
    if is_training:
        features = torch.cat([
                              extra_results["incident_visibility"].mean(-2),
                              extra_results["local_incident_lights"].mean(-2)], dim=-1)
    else:
        features = torch.cat([
                              extra_results["incident_lights"].mean(-2),
                              extra_results["local_incident_lights"].mean(-2),
                              extra_results["incident_visibility"].mean(-2)], dim=-1)
    # view transmation
    normal = (normal @ viewpoint_camera.world_view_transform[:3,:3])
    normal = normal.transpose(1,2).reshape(normal.shape[0], -1)
    if is_training:
        vfeatures = torch.cat([brdf_color, base_color, normal, roughness,
                              extra_results["diffuse_light"]], dim=-1)
    else:
        vfeatures = torch.cat([brdf_color, base_color, normal, roughness,
                              extra_results["direct"],
                              extra_results["indirect"],], dim=-1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # time1 = time.time()
    (num_rendered, rendered_image, rendered_normal, rendered_opacity, rendered_depth,
     rendered_feature, rendered_vfeature, weights, radii) = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        features=features,
        vfeatures=vfeatures,
    )
    # time2 = time.time()
    # cost_time += time2 - time1
    # print(cost_time)

    torch.cuda.synchronize()
    rendered_feature = rendered_feature / rendered_opacity.clamp_min(1e-5) 
    rendered_vfeature = rendered_vfeature / rendered_opacity.clamp_min(1e-5)
    feature_dict = {}

    def opacity_feilter(r):
        return r * rendered_opacity + (1 - rendered_opacity) * bg_color[:, None, None]
    
    if is_training:
        rendered_visibility, rendered_local_light \
            = rendered_feature.split([1, 3], dim=0)
        feature_dict.update({"local_lights": opacity_feilter(rgb_to_srgb(rendered_local_light)),
                             "visibility": opacity_feilter(rendered_visibility)
                             })
    else:

        rendered_light, rendered_local_light, rendered_visibility \
            = rendered_feature.split([3, 3, 1], dim=0)
        feature_dict.update({"lights": opacity_feilter(rgb_to_srgb(rendered_light)),
                             "local_lights": opacity_feilter(rgb_to_srgb(rendered_local_light)),
                             "visibility": opacity_feilter(rendered_visibility),
                             })
        
    if is_training:
        rendered_pbr, rendered_base_color, rendered_shading_normal, rendered_roughness,\
            rendered_diffuse \
            = rendered_vfeature.split([3, 3, 3, 1, 3], dim=0)
        feature_dict.update({"base_color": opacity_feilter(rgb_to_srgb(rendered_base_color)),
                             "diffuse": opacity_feilter(rgb_to_srgb(rendered_diffuse)),
                             "roughness": opacity_feilter(rendered_roughness),
                             })
    else:
        rendered_pbr, rendered_base_color, rendered_shading_normal, rendered_roughness,\
            rendered_direct_pbr, rendered_indirect_pbr\
            = rendered_vfeature.split([3, 3, 3, 1, 3, 3], dim=0)
        feature_dict.update({
                             "base_color": opacity_feilter(rgb_to_srgb(rendered_base_color)),
                             "direct": rgb_to_srgb(rendered_direct_pbr),
                             "indirect": rgb_to_srgb(rendered_indirect_pbr),
                             "roughness": opacity_feilter(rendered_roughness),
                             })

    pbr = rendered_pbr
    rendered_pbr = pbr * rendered_opacity + (1 - rendered_opacity) * bg_color[:, None, None]

    rendered_pseudo_normal = depth2normal(rendered_depth, viewpoint_camera.image_mask.cuda(), viewpoint_camera)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    results = {"render": rendered_image,
               "depth": rendered_depth,
               "pbr": rgb_to_srgb(rendered_pbr),
               "normal": rendered_shading_normal,
               "opacity": rendered_opacity,
               "depth": rendered_depth,
               "viewspace_points": screenspace_points,
               "pseudo_normal": rendered_pseudo_normal,
               "visibility_filter": radii > 0,
               "radii": radii,
               "num_rendered": num_rendered,
               }

    results.update(feature_dict)
    results["diffuse_light"] = extra_results["diffuse_light"]
    try:
        results["env"] = direct_light_env_light.get_env
    except:
        pass
    
    if not is_training:
        directions = viewpoint_camera.get_world_directions()
        direct_env = direct_light_env_light.direct_light(directions.permute(1, 2, 0)).permute(2, 0, 1)
        results["render_env"] = rendered_image + (1 - rendered_opacity) * rgb_to_srgb(direct_env)
        results["pbr_env"] = rgb_to_srgb(pbr * rendered_opacity + (1 - rendered_opacity) * direct_env)
        results["env_only"] = rgb_to_srgb(direct_env)
        
    return results


def calculate_loss(viewpoint_camera, pc, results, opt, direct_light_env_light, iteration):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    rendered_image = results["render"]
    rendered_depth = results["depth"]
    rendered_normal = results["normal"]
    rendered_pbr = results["pbr"]
    rendered_opacity = results["opacity"]
    rendered_base_color = results["base_color"]
    rendered_roughness = results["roughness"]
    rendered_diffuse = results["diffuse"]
    rendered_local_lights = results["local_lights"]
    image_mask = viewpoint_camera.image_mask.cuda()

    gt_image = viewpoint_camera.original_image.cuda()
    Ll1 = F.l1_loss(rendered_image, gt_image)
    ssim_val = ssim(rendered_image, gt_image)
    tb_dict["l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

    Ll1_pbr = F.l1_loss(rendered_pbr, gt_image)
    ssim_val_pbr = ssim(rendered_pbr, gt_image)
    tb_dict["l1_pbr"] = Ll1_pbr.item()
    tb_dict["ssim_pbr"] = ssim_val_pbr.item()
    tb_dict["psnr_pbr"] = psnr(rendered_pbr, gt_image).mean().item()
    loss_pbr = (1.0 - opt.lambda_dssim) * Ll1_pbr + opt.lambda_dssim * (1.0 - ssim_val_pbr)
    loss = loss + opt.lambda_pbr * loss_pbr


    if 1:
        pool = torch.nn.MaxPool2d(9, stride=1, padding=4)
        d2n = depth2normal(rendered_depth, image_mask, viewpoint_camera)
        loss_mask = (rendered_opacity * (1 - pool(image_mask))).mean()
        loss_surface = cos_loss(rendered_normal, d2n)
        curv_n = normal2curv(rendered_normal, image_mask)
        loss_curv = torch.abs(curv_n).mean() #+ 1 * l1_loss(curv_d2n, 0)

        mono = viewpoint_camera.mono

        opac_ = pc.get_opacity
        opac_mask0 = torch.gt(opac_, 0.01) * torch.le(opac_, 0.5)
        opac_mask1 = torch.gt(opac_, 0.5) * torch.le(opac_, 0.99)
        opac_mask = opac_mask0 * 0.01 + opac_mask1
        loss_opac = (torch.exp(-(opac_ - 0.5)**2 * 20) * opac_mask).mean()

        loss += 0.02 * loss_surface

        
        loss += 0.1 * F.mse_loss(pc.get_normals, torch.zeros_like(pc.get_normals))


    loss_radiance = pc.get_radiance_loss(viewpoint_camera, direct_light_env_light)
    loss = loss + opt.lambda_radiance * loss_radiance


    if opt.lambda_depth > 0:
        gt_depth = viewpoint_camera.depth.cuda()
        image_mask = viewpoint_camera.image_mask.cuda().bool()
        depth_mask = gt_depth > 0
        sur_mask = torch.logical_xor(image_mask, depth_mask)

        loss_depth = F.l1_loss(rendered_depth[~sur_mask], gt_depth[~sur_mask])
        tb_dict["loss_depth"] = loss_depth.item()
        loss = loss + opt.lambda_depth * loss_depth

    if opt.lambda_mask_entropy > 0:
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_mask_entropy = -(image_mask * torch.log(o) + (1 - image_mask) * torch.log(1 - o)).mean()
        tb_dict["loss_mask_entropy"] = loss_mask_entropy.item()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy

    if opt.lambda_normal_render_depth > 0:
        normal_pseudo = results['pseudo_normal']
        image_mask = viewpoint_camera.image_mask.cuda()
        # loss_normal_render_depth = F.mse_loss(
        #     rendered_normal * image_mask, normal_pseudo.detach() * image_mask)
        loss_normal_render_depth = 1 - (rendered_normal * normal_pseudo.detach()).sum(dim=0) * image_mask
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth.item()
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth

    if opt.lambda_normal_mvs_depth > 0:
        gt_depth = viewpoint_camera.depth.cuda()
        depth_mask = (gt_depth > 0).float()
        mvs_normal = viewpoint_camera.normal.cuda()
        # loss_normal_mvs_depth = F.mse_loss(
        #     rendered_normal * depth_mask, mvs_normal * depth_mask)
        loss_normal_mvs_depth = 1 - (rendered_normal * mvs_normal).sum(dim=0) * depth_mask
        tb_dict["loss_normal_mvs_depth"] = loss_normal_mvs_depth.item()
        loss = loss + opt.lambda_normal_mvs_depth * loss_normal_mvs_depth

    if opt.lambda_light > 0:
        diffuse_light = results["diffuse_light"]
        mean_light = diffuse_light.mean(-1, keepdim=True).expand_as(diffuse_light)
        loss_light = F.l1_loss(diffuse_light, mean_light)
        tb_dict["loss_light"] = loss_light.item()
        loss = loss + opt.lambda_light * loss_light

    if opt.lambda_base_color_smooth > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color * image_mask, gt_image * image_mask)
        # loss_base_color_smooth = second_order_edge_aware_loss(rendered_base_color * image_mask, gt_image)
        # loss_base_color_smooth = tv_loss(rendered_base_color * image_mask)
        tb_dict["loss_base_color_smooth"] = loss_base_color_smooth.item()
        loss = loss + opt.lambda_base_color_smooth * loss_base_color_smooth

    if opt.lambda_roughness_smooth > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness * image_mask, gt_image * image_mask)
        # loss_roughness_smooth = second_order_edge_aware_loss(rendered_roughness * image_mask, gt_image)
        # loss_roughness_smooth = tv_loss(rendered_roughness * image_mask)
        tb_dict["loss_roughness_smooth"] = loss_roughness_smooth.item()
        loss = loss + opt.lambda_roughness_smooth * loss_roughness_smooth
    
    if opt.lambda_light_smooth > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_light_smooth = first_order_edge_aware_loss(rendered_diffuse * image_mask, rendered_normal)
        # loss_light_smooth = second_order_edge_aware_loss(rendered_diffuse * image_mask, gt_image)
        tb_dict["loss_light_smooth"] = loss_light_smooth.item()
        loss = loss + opt.lambda_light_smooth * loss_light_smooth
        
    if opt.lambda_env_smooth > 0:
        env = direct_light_env_light.get_env
        loss_env_smooth = tv_loss(env[0].permute(2, 0, 1))
        tb_dict["loss_env_smooth"] = loss_env_smooth.item()
        loss = loss + opt.lambda_env_smooth * loss_env_smooth
    
    if opt.lambda_normal_smooth > 0:
        loss_normal_smooth = second_order_edge_aware_loss(rendered_normal * image_mask, gt_image)
        # loss_normal_smooth = tv_loss(rendered_normal * image_mask)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        loss = loss + opt.lambda_normal_smooth * loss_normal_smooth

    tb_dict["loss"] = loss.item()

    return loss, tb_dict


def render_svgss(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                 scaling_modifier=1.0, override_color=None, opt: OptimizationParams = False,
                 is_training=False, dict_params=None, **kwargs):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    results = render_view(viewpoint_camera, pc, pipe, bg_color,
                          scaling_modifier, override_color, is_training, dict_params)
    iteration = kwargs.get("iteration", 0)
    if is_training:
        loss, tb_dict = calculate_loss(viewpoint_camera, pc, results, opt, direct_light_env_light=dict_params['env_light'], iteration=iteration)
        results["tb_dict"] = tb_dict
        results["loss"] = loss

    results["normal"] = (results["normal"].permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].permute(1,0))).permute(2,0,1)
    results["pseudo_normal"] = (results["pseudo_normal"].permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].permute(1,0))).permute(2,0,1)
    
    return results


def rendering_equation_with_sh(base_color, roughness, normals, viewdirs,
                              incidents, direct_light_env_light=None,
                              visibility_precompute=None, incident_dirs_precompute=None, incident_areas_precompute=None):
    incident_dirs, incident_areas = incident_dirs_precompute, incident_areas_precompute

    deg = int(np.sqrt(incidents.shape[1]) - 1)
    global_incident_lights = direct_light_env_light.direct_light(incident_dirs)
    local_incident_lights = eval_sh(deg, incidents.transpose(1, 2).view(-1, 1, 3, (deg + 1) ** 2), incident_dirs).clamp_min(0)
    
    incident_visibility = visibility_precompute
    global_incident_lights = global_incident_lights * incident_visibility
    incident_lights = local_incident_lights + global_incident_lights

    n_d_i = (normals[:, None] * incident_dirs).sum(-1, keepdim=True).clamp(min=0)
    f_d = base_color[:, None] / np.pi
    f_s = GGX_specular(normals, viewdirs, incident_dirs, roughness, fresnel=0.04)

    transport = incident_lights * incident_areas * n_d_i  # （num_pts, num_sample, 3)
    specular = ((f_s) * transport).mean(dim=-2)
    pbr = ((f_d + f_s) * transport).mean(dim=-2)
    diffuse_light = transport.mean(dim=-2)

    extra_results = {
        "incident_dirs": incident_dirs,
        "incident_lights": incident_lights,
        "local_incident_lights": local_incident_lights,
        "global_incident_lights": global_incident_lights,
        "incident_visibility": incident_visibility,
        "diffuse_light": diffuse_light,
        "specular": specular,
    }
    return pbr, extra_results

def rendering_equation(base_color, roughness, normals, viewdirs,
                              radiance, direct_light_env_light=None,
                              visibility_precompute=None, incident_dirs_precompute=None, incident_areas_precompute=None):
    incident_dirs, incident_areas = incident_dirs_precompute, incident_areas_precompute

    global_incident_lights = direct_light_env_light.direct_light(incident_dirs)
    local_incident_lights = radiance

    # deg = int(np.sqrt(radiance.shape[1]) - 1)
    # global_incident_lights = direct_light_env_light.direct_light(incident_dirs)
    # local_incident_lights = eval_sh(deg, radiance.transpose(1, 2).view(-1, 1, 3, (deg + 1) ** 2), incident_dirs).clamp_min(0)
    
    incident_visibility = visibility_precompute
    global_incident_lights = global_incident_lights * incident_visibility
    incident_lights = local_incident_lights + global_incident_lights

    n_d_i = (normals[:, None] * incident_dirs).sum(-1, keepdim=True).clamp(min=0)
    valid = n_d_i > 0
    f_d = base_color[:, None] / np.pi
    f_s = GGX_specular(normals, viewdirs, incident_dirs, roughness, fresnel=0.04)

    transport = incident_lights * incident_areas * n_d_i  # （num_pts, num_sample, 3)
    specular = ((f_s) * transport).mean(dim=-2)
    pbr = ((f_d + f_s) * transport).mean(dim=-2)
    diffuse_light = transport.mean(dim=-2)

    direct_transport = global_incident_lights * incident_areas * n_d_i
    direct_pbr = ((f_d + f_s) * direct_transport).mean(dim=-2)

    extra_results = {
        "incident_dirs": incident_dirs,
        "incident_lights": incident_lights,
        "local_incident_lights": local_incident_lights,
        "global_incident_lights": global_incident_lights,
        "incident_visibility": incident_visibility,
        "diffuse_light": diffuse_light,
        "specular": specular,
        "direct": direct_pbr,
    }

    return pbr, extra_results

def GGX_specular(
        normal,
        pts2c,
        pts2l,
        roughness,
        fresnel
):
    L = F.normalize(pts2l, dim=-1)  # [nrays, nlights, 3]
    V = F.normalize(pts2c, dim=-1)  # [nrays, 3]
    H = F.normalize((L + V[:, None, :]) / 2.0, dim=-1)  # [nrays, nlights, 3]
    N = F.normalize(normal, dim=-1)  # [nrays, 3]

    NoV = torch.sum(V * N, dim=-1, keepdim=True)  # [nrays, 1]
    N = N * NoV.sign()  # [nrays, 3]

    NoL = torch.sum(N[:, None, :] * L, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1] TODO check broadcast
    NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, 1]
    NoH = torch.sum(N[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]
    VoH = torch.sum(V[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]

    alpha = roughness * roughness  # [nrays, 3]
    alpha2 = alpha * alpha  # [nrays, 3]
    k = (alpha + 2 * roughness + 1.0) / 8.0
    FMi = ((-5.55473) * VoH - 6.98316) * VoH
    frac0 = fresnel + (1 - fresnel) * torch.pow(2.0, FMi)  # [nrays, nlights, 3]
    
    frac = frac0 * alpha2[:, None, :]  # [nrays, 1]
    nom0 = NoH * NoH * (alpha2[:, None, :] - 1) + 1

    nom1 = NoV * (1 - k) + k
    nom2 = NoL * (1 - k[:, None, :]) + k[:, None, :]
    nom = (4 * np.pi * nom0 * nom0 * nom1[:, None, :] * nom2).clamp_(1e-6, 4 * np.pi)
    spec = frac / nom
    return spec

def rendering_equation4(base_color, roughness, normals, viewdirs,
                              radiance, direct_light_env_light=None,
                              visibility_precompute=None, incident_dirs_precompute=None, incident_areas_precompute=None):
    # import time
    # time1 = time.time()
    incident_dirs, incident_areas = incident_dirs_precompute, incident_areas_precompute

    global_incident_lights = direct_light_env_light.direct_light(incident_dirs).clamp(0, 64)
    local_incident_lights = radiance

    # deg = int(np.sqrt(radiance.shape[1]) - 1)
    # global_incident_lights = direct_light_env_light.direct_light(incident_dirs)
    # local_incident_lights = eval_sh(deg, radiance.transpose(1, 2).view(-1, 1, 3, (deg + 1) ** 2), incident_dirs).clamp_min(0)
    
    incident_visibility = visibility_precompute
    global_incident_lights = global_incident_lights * incident_visibility
    incident_lights = local_incident_lights + global_incident_lights

    n_d_i = (normals[:, None] * incident_dirs[:,:,None]).sum(-1, keepdim=True).clamp(min=0) # normal [n, 1, 4, 3] # res [n, 64, 4, 1]
    f_d = base_color[:, None] / np.pi # [n, 1, 12]
    f_s = GGX_specular4(normals, viewdirs, incident_dirs, roughness, fresnel=0.04).squeeze(-1).repeat(1,1,3) # [n, 1, 4, 1]

    transport = incident_lights[:, :, None] * incident_areas[:, :, None] * n_d_i  # （num_pts, num_sample, 4, 3)

    transport = transport.transpose(2, 3).reshape(transport.shape[0], transport.shape[1], -1)
    # transport = transport.repeat_interleave(repeats=4, dim=-1)

    specular = ((f_s) * transport).mean(dim=-2) 
    pbr = ((f_d + f_s) * transport).mean(dim=-2) 
    # time2 = time.time()
    # print(time2 - time1)
    diffuse_light = transport.mean(dim=-2) 



    direct_transport = global_incident_lights[:, :, None] * incident_areas[:, :, None] * n_d_i
    direct_transport = direct_transport.transpose(2, 3).reshape(transport.shape[0], transport.shape[1], -1)
    direct_pbr = ((f_d + f_s) * direct_transport).mean(dim=-2) 

    indirect_transport = local_incident_lights[:, :, None] * incident_areas[:, :, None] * n_d_i
    indirect_transport = indirect_transport.transpose(2, 3).reshape(transport.shape[0], transport.shape[1], -1)
    indirect_pbr = ((f_d + f_s) * indirect_transport).mean(dim=-2)

    extra_results = {
        "incident_dirs": incident_dirs,
        "incident_lights": incident_lights,
        "local_incident_lights": local_incident_lights,
        "global_incident_lights": global_incident_lights,
        "incident_visibility": incident_visibility,
        "diffuse_light": diffuse_light,
        "specular": specular,
        "direct": direct_pbr,
        "indirect": indirect_pbr,
    }
    # time2 = time.time()
    # print(time2 - time1)
    return pbr, extra_results

def GGX_specular4(
        normal,
        pts2c,
        pts2l,
        roughness,
        fresnel
):
    L = F.normalize(pts2l, dim=-1).unsqueeze(-2)  # [nrays, nlights, 1, 3]
    V = F.normalize(pts2c, dim=-1).unsqueeze(-2)   # [nrays, 1, 3]
    H = F.normalize((L + V[:, None, :]) / 2.0, dim=-1)   # [nrays, nlights, 1, 3]
    N = F.normalize(normal, dim=-1)  # [nrays, 4, 3]

    NoV = torch.sum(V * N, dim=-1, keepdim=True)  # [nrays, 4, 1]

    N = N * NoV.sign()  # [nrays, 4, 3]

    NoL = torch.sum(N[:, None, :] * L, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 4, 1] TODO check broadcast
    NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, 4, 1]
    NoH = torch.sum(N[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 4, 1]
    VoH = torch.sum(V[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1, 1]

    roughness = roughness.unsqueeze(1).unsqueeze(-1)
    alpha = roughness*roughness
    alpha2 = alpha * alpha  # [nrays, 3]
    k = (alpha + 2 * roughness + 1.0) / 8.0
    FMi = ((-5.55473) * VoH - 6.98316) * VoH
    frac0 = fresnel + (1 - fresnel) * torch.pow(2.0, FMi)  # [nrays, nlights, 3]
    
    frac = frac0 * alpha2 # [nrays, 1]
    nom0 = NoH * NoH * (alpha2 - 1) + 1

    nom1 = NoV.unsqueeze(1) * (1 - k) + k
    nom2 = NoL * (1 - k) + k
    nom = (4 * np.pi * nom0 * nom0 * nom1 * nom2).clamp_(1e-6, 4 * np.pi)
    spec = frac / nom

    return spec