/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, const int S, int VS, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* features,
			const float* vfeatures,
			const float* opacities,
			// float* cutoff,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* prcppoint,
			const float* patchbbox,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* config,
			float* out_color,
			float* out_normal,
			float* out_depth,
			float* out_opac,
			float* out_feature,
			float* out_vfeature,
			float* out_weights,
			int* radii = nullptr,
			bool debug = false);

		static void backward(
			const int P,  int S, int VS, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* features,
			const float* vfeatures,
			const float* colors_precomp,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float* prcppoint,
			const float* patchbbox,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpixcolor,
			const float* dL_dpixnormal,
			const float* dL_dpixdepth,
			const float* dL_dpixopac,
			const float* dL_dpixfeature,
			const float* dL_dpixvfeature,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			// float* dL_dcutoff,
			float* dL_dcolor,
			float* dL_dfeature,
			float* dL_dvfeature,
			float* dL_dnormal,
			float* dL_ddepth,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			float* dL_dviewmat,
			float* dL_dprojmat,
			float* dL_dcampos,
			bool debug,
			float* config);
	};
};

#endif