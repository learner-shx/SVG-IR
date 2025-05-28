#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from arguments import ModelParams
from scene.cameras import Camera
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch
import math
import torch.nn.functional as F

class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0, 4.0, 8.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval,
                                                          debug=args.debug_cuda)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            if "stanford_orb" in args.source_path:
                print("Found keyword stanford_orb, assuming Stanford ORB data set!")
                scene_info = sceneLoadTypeCallbacks["StanfordORB"](args.source_path, args.white_background, args.eval, 
                                                                   debug=args.debug_cuda)
            elif "Synthetic4Relight" in args.source_path:
                print("Found transforms_train.json file, assuming Synthetic4Relight data set!")
                scene_info = sceneLoadTypeCallbacks["Synthetic4Relight"](args.source_path, args.white_background, args.eval,
                                                            debug=args.debug_cuda)
            else:
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, 
                                                               debug=args.debug_cuda)
        elif os.path.exists(os.path.join(args.source_path, "inputs/sfm_scene.json")):
            print("Found sfm_scene.json file, assuming render_relight data set!")
            scene_info = sceneLoadTypeCallbacks["render_relight"](args.source_path, args.white_background, args.eval,
                                                         debug=args.debug_cuda)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                   'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)

        self.scene_info = scene_info

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def get_random_pose(self, camera0, k=4, n=1):
        # get near ref poses,
        # if i0 is None:
        #     i0 = random.randint(0, self.item_num - 1)
        centeri = torch.stack([i.world_view_transform[3, :3].cuda() for i in self.getTrainCameras()])
        center0 = camera0.world_view_transform[3, :3]
        dist = torch.abs(center0 - centeri).norm(2, -1)


        topk_v, topk_i = dist.topk(k=min(k + 1, len(centeri)), dim=0, largest=False)
        # print(topk_v)
        # print(topk_i)
        # # exit()
        topk_i = topk_i[1:].tolist()
        # print(i0, topk_i)
        random.shuffle(topk_i)
        i_n = topk_i[:min(n, k)]

        return self.getTrainCamerasByIdx(i_n)

    def get_bound(self):
        centers = torch.stack([i.camera_center for i in self.getTrainCameras()], 0)
        min_bound = centers.min(0)[0]
        max_bound = centers.max(0)[0]
        return min_bound, max_bound
    
    def visualize_cameras(self):
        points = []
        colors = []
        for i in self.getTrainCameras():
            center = i.camera_center.detach().cpu().numpy()
            # print(center)
            viewDir = i.R[:3, 2].cpu().numpy()
            for j in range(1):
                points.append(center + viewDir * j * 0.1)
                # print(center)
                # print(i.T@i.R)
                # colors.append([1, 1, 1, 1.0] if j == 0 else [0, 0, 0, 0.0])
        import pymeshlab
        import numpy as np
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.array(points)))
        ms.save_current_mesh('test/cameras.ply')
        
    def pass_pose(self, s0, s1):
        c0 = self.getTrainCameras(s0)
        c1 = self.getTrainCameras(s1)
        for i in range(len(c0)):
            c1[i].overwrite_pose(c0[i])
    
    def get_canonical_rays(self, scale: float = 1.0):
        # NOTE: some datasets do not share the same intrinsic (e.g. DTU)
        # get reference camera
        ref_camera: Camera = self.train_cameras[scale][0]
        # TODO: inject intrinsic
        H, W = ref_camera.image_height, ref_camera.image_width
        cen_x = W / 2
        cen_y = H / 2
        tan_fovx = math.tan(ref_camera.FoVx * 0.5)
        tan_fovy = math.tan(ref_camera.FoVy * 0.5)
        focal_x = W / (2.0 * tan_fovx)
        focal_y = H / (2.0 * tan_fovy)

        x, y = torch.meshgrid(
            torch.arange(W),
            torch.arange(H),
            indexing="xy",
        )
        x = x.flatten()  # [H * W]
        y = y.flatten()  # [H * W]
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - cen_x + 0.5) / focal_x,
                    (y - cen_y + 0.5) / focal_y,
                ],
                dim=-1,
            ),
            (0, 1),
            value=1.0,
        )  # [H * W, 3]
        # NOTE: it is not normalized
        return camera_dirs.cuda()