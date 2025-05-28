#include "mathConstant.slangh"

// world dir to uv
float2 world_to_latlong_map(float3 dir)
{
    float3 p = normalize(dir);
    float2 uv;
    uv.x = atan2(p.x, -p.z) * M_1_2PI + 0.5f;
    uv.y = acos(p.y) * M_1_PI;
    return uv;
}

// uint2 get_envmap_coordinate(float3 dir)
// {
//     uint2 dim = uint2(1024, 512);
//     float2 uv = world_to_latlong_map(dir);

//     return uint2(floor(uv * dim));
// }

[Differentiable]
float3 eval_envmap(DiffTensorView envmap, float3 dir)
{
    float2 uv = no_diff world_to_latlong_map(dir);
    uv = uv - no_diff floor(uv);

    uint2 dim = uint2(16, 32);
    // uint2 dim = uint2(512, 1024);

    uint2 coord0 = no_diff uint2(floor(uv * (dim - 1))) % dim;
    uint2 coord1 = no_diff uint2(floor(uv * (dim - 1 + float2(1, 0)))) % dim;
    uint2 coord2 = no_diff uint2(floor(uv * (dim - 1 + float2(0, 1)))) % dim;
    uint2 coord3 = no_diff uint2(floor(uv * (dim - 1 + float2(1, 1)))) % dim;

    float2 uv0 = float2(coord0) / (dim - 1);
    float2 uv1 = float2(coord1) / (dim - 1);
    float2 uv2 = float2(coord2) / (dim - 1);
    float2 uv3 = float2(coord3) / (dim - 1);

    float3 c0;
    c0.x = envmap[coord0.x, coord0.y, 0];
    c0.y = envmap[coord0.x, coord0.y, 1];
    c0.z = envmap[coord0.x, coord0.y, 2];

    float3 c1;
    c1.x = envmap[coord1.x, coord1.y, 0];
    c1.y = envmap[coord1.x, coord1.y, 1];
    c1.z = envmap[coord1.x, coord1.y, 2];

    float3 c2;
    c2.x = envmap[coord2.x, coord2.y, 0];
    c2.y = envmap[coord2.x, coord2.y, 1];
    c2.z = envmap[coord2.x, coord2.y, 2];

    float3 c3;
    c3.x = envmap[coord3.x, coord3.y, 0];
    c3.y = envmap[coord3.x, coord3.y, 1];
    c3.z = envmap[coord3.x, coord3.y, 2];

    float dx1 = (uv.x - uv0.x);
    float dy1 = (uv.y - uv0.y);
    float dx2 = 1.0f - dx1;
    float dy2 = 1.0f - dy1;

    float3 c = c0 * dx2 * dy2 + c1 * dx1 * dy2 + c2 * dx2 * dy1 + c3 * dx1 * dy1;

    return c;
}

float3 eval_envmap(TensorView<float> envmap, float3 dir)
{
    float2 uv = world_to_latlong_map(dir);
    uv = uv - floor(uv);

    uint2 dim = uint2(16, 32);
    // uint2 dim = uint2(512, 1024);

    uint2 coord0 = uint2(floor(uv * (dim - 1))) % dim;
    uint2 coord1 = uint2(floor(uv * (dim - 1 + float2(1, 0)))) % dim;
    uint2 coord2 = uint2(floor(uv * (dim - 1 + float2(0, 1)))) % dim;
    uint2 coord3 = uint2(floor(uv * (dim - 1 + float2(1, 1)))) % dim;

    float2 uv0 = float2(coord0) / (dim - 1);
    float2 uv1 = float2(coord1) / (dim - 1);
    float2 uv2 = float2(coord2) / (dim - 1);
    float2 uv3 = float2(coord3) / (dim - 1);

    float3 c0;
    c0.x = envmap[coord0.x, coord0.y, 0];
    c0.y = envmap[coord0.x, coord0.y, 1];
    c0.z = envmap[coord0.x, coord0.y, 2];

    float3 c1;
    c1.x = envmap[coord1.x, coord1.y, 0];
    c1.y = envmap[coord1.x, coord1.y, 1];
    c1.z = envmap[coord1.x, coord1.y, 2];

    float3 c2;
    c2.x = envmap[coord2.x, coord2.y, 0];
    c2.y = envmap[coord2.x, coord2.y, 1];
    c2.z = envmap[coord2.x, coord2.y, 2];

    float3 c3;
    c3.x = envmap[coord3.x, coord3.y, 0];
    c3.y = envmap[coord3.x, coord3.y, 1];
    c3.z = envmap[coord3.x, coord3.y, 2];

    float dx1 = (uv.x - uv0.x);
    float dy1 = (uv.y - uv0.y);
    float dx2 = 1.0f - dx1;
    float dy2 = 1.0f - dy1;

    float3 c = c0 * dx2 * dy2 + c1 * dx1 * dy2 + c2 * dx2 * dy1 + c3 * dx1 * dy1;

    return c;
}