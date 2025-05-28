import envmap;
import pbr;
import brdf;
import sh_utils;

#include "mathConstant.slangh"
#define MAX_STACK_SIZE 64

struct GPUBvhNode {
    int index;
    int left;          // pointer to the left child or INVALID_POINTER in case of leaf
    int right;         // pointer to the right child or INVALID_POINTER in case of leaf
    uint primitiveIdx; // custom value that is copied from the input Element or 0 in case of inner node
};

/*
intersection code from https://github.com/maxilevi/raytracer/tree/master
*/

[Differentiable]
bool aabb_hit(float3 rayo, float3 rayd, float t_min, float t_max, float3 bbox_min, float3 bbox_max)
{
    for (int i = 0; i < 3; ++i)
    {
        float ray_d_i = rayd[i];
        if (ray_d_i == 0.f) ray_d_i = 0.000001f;
        float inverse_direction = 1.0 / ray_d_i;
        float t0 = (bbox_min[i] - rayo[i]) * inverse_direction;
        float t1 = (bbox_max[i] - rayo[i]) * inverse_direction;
        if (inverse_direction < 0.0) {
            float tmp = t1;
            t1 = t0;
            t0 = tmp;
        }
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min)
            return false;
    }
    return true;

}

bool triangle_hit(float3 rayo, float3 rayd, float t_min, float t_max, float3 v0, float3 v1, float3 v2, inout float t_hit, inout float3 pos)
{
    float epsilon = 1e-9;
    float3 E1 = v1 - v0;
    float3 E2 = v2 - v0;
    float3 P = cross(rayd, E2);
    float det = dot(E1, P);
    if (det > -epsilon && det < epsilon)
        return false;
    float invDet = 1 / det;

    float3 T = rayo - v0;
    float u = dot(T,P) * invDet;
    if (u < 0 || u > 1) return false;

    float3 Q = cross(T, E1);
    float v = dot(rayd, Q) * invDet;
    if (v < 0 || u + v > 1) return false;

    float t = dot(E2, Q) * invDet;
    t_hit = t;
    pos = rayo + t * rayd;
    return true;
}

[Differentiable]
float determinant(float3x3 M) {
    return M[0][0] * (M[1][1] * M[2][2] - M[2][1] * M[1][2]) -
           M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
           M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
}

[Differentiable]
float3x3 inverse(float3x3 M)
{
    float3x3 inv;
    float invdet = 1.0f / determinant(M);
    inv[0][0] = (M[1][1] * M[2][2] - M[2][1] * M[1][2]) * invdet;
    inv[0][1] = (M[0][2] * M[2][1] - M[0][1] * M[2][2]) * invdet;
    inv[0][2] = (M[0][1] * M[1][2] - M[0][2] * M[1][1]) * invdet;
    inv[1][0] = (M[1][2] * M[2][0] - M[1][0] * M[2][2]) * invdet;
    inv[1][1] = (M[0][0] * M[2][2] - M[0][2] * M[2][0]) * invdet;
    inv[1][2] = (M[1][0] * M[0][2] - M[0][0] * M[1][2]) * invdet;
    inv[2][0] = (M[1][0] * M[2][1] - M[2][0] * M[1][1]) * invdet;
    inv[2][1] = (M[2][0] * M[0][1] - M[0][0] * M[2][1]) * invdet;
    inv[2][2] = (M[0][0] * M[1][1] - M[1][0] * M[0][1]) * invdet;
    return inv;
}


bool ellipse_hit(float3 origin, float3 dir, float3 center, float3x3 rotateMat, float2 scales, float3 normal, float t_min, inout float t_hit, inout float3 pos, inout float2 uv)
{
    // 计算平面法向量
    float3x3 scaleMat = float3x3(scales.x, 0, 0,
                                 0, scales.y, 0,
                                 0, 0, 1);
    float3x3 L = mul(rotateMat, scaleMat);
    float3 normalW = mul(L, float3(0, 0, 1));
    // float3 normalW = normal;
    float denom = dot(normalW, dir);

    if (abs(denom) < 1e-6)
    {
        return false; // 光线与平面平行，或几乎平行
    }

    // 计算光线与平面交点参数 t_hit
    float3 p0l0 = center - origin;
    t_hit = dot(p0l0, normalW) / denom;

    if (t_hit < t_min)
    {
        return false; // 交点在光线的反方向
    }

    // 计算交点位置
    pos = origin + t_hit * dir;

    // 将交点和中心转换到椭圆的本地坐标系
    float3 posM = mul(inverse(rotateMat), pos - center);
    // float3 centerM = mul(inverse(rotateMat), center);
    float2 uv_hit = float2(posM.x / scales.x, posM.y / scales.y);
    if (uv_hit.x < uv_hit.y)
    {
        uv_hit = float2(uv_hit.y, uv_hit.x);
    }

    uv_hit = uv_hit * 0.5 + 0.5;
    uv.x = clamp(uv_hit.x, 0.001f, 0.999f);
    uv.y = clamp(uv_hit.y, 0.001f, 0.999f);

    
    // 检查交点是否在椭圆内
    float disM = sqr(posM.x) / sqr(scales.x) + sqr(posM.y) / sqr(scales.y);

    // if(posM.z < 1e-1)
    // {
    //     return false;
    // }
    if (disM <= 9.0f)
    {
        return true; // 交点在椭圆内
    }
    return false; // 交点在椭圆外
}

// [Differentiable]
bool gaussian_hit(float3 origin, float3 dir, float3 mu, float3x3 cov, inout float t_hit, inout float3 pos, out float d, out float3x3 invcov)
{
    float3x3 covInv = inverse(cov);
    invcov = covInv;
    float3 muDelta = mu - origin;

    t_hit = dot(mul(muDelta, covInv), dir) / dot(mul(dir, covInv), dir);

    float3 hitpos = origin + t_hit * dir;

    float mdistance = dot(mul(hitpos - mu, covInv), hitpos - mu);

    d = mdistance;

    pos = hitpos;

    if (mdistance < 0.1f && t_hit > 1e-6)
    {
        return true;
    }

    return false;
}


float ray_gs_intersects(float3 mean, float[6] cov3D_inverse, float3 ray_o, float3 ray_d)
{
    float3 miu = { mean.x - ray_o.x, mean.y - ray_o.y, mean.z - ray_o.z };
    float t1 = cov3D_inverse[0] * miu.x * ray_d.x + cov3D_inverse[1] * miu.x * ray_d.y + cov3D_inverse[2] * miu.x * ray_d.z +
               cov3D_inverse[1] * miu.y * ray_d.x + cov3D_inverse[3] * miu.y * ray_d.y + cov3D_inverse[4] * miu.y * ray_d.z +
               cov3D_inverse[2] * miu.z * ray_d.x + cov3D_inverse[4] * miu.z * ray_d.y + cov3D_inverse[5] * miu.z * ray_d.z;
    float t2 = cov3D_inverse[0] * ray_d.x * ray_d.x + cov3D_inverse[1] * ray_d.x * ray_d.y + cov3D_inverse[2] * ray_d.x * ray_d.z +
               cov3D_inverse[1] * ray_d.y * ray_d.x + cov3D_inverse[3] * ray_d.y * ray_d.y + cov3D_inverse[4] * ray_d.y * ray_d.z +
               cov3D_inverse[2] * ray_d.z * ray_d.x + cov3D_inverse[4] * ray_d.z * ray_d.y + cov3D_inverse[5] * ray_d.z * ray_d.z;
    return t1 / t2;
}


float gaussian_fn(float3 mean, float3 pos, float[6] cov3D_inverse) 
{
    float3 d = { mean.x - pos.x, mean.y - pos.y, mean.z - pos.z };
    float power = -0.5 * (d.x * d.x * cov3D_inverse[0] + d.y * d.y * cov3D_inverse[3] + d.z * d.z * cov3D_inverse[5] +
                          2 * d.x * d.y * cov3D_inverse[1] + 2 * d.x * d.z * cov3D_inverse[2] + 2 * d.y * d.z * cov3D_inverse[4]);
    return power;
}

// [Differentiable]
float3x3 matrixFromRotationXYZ(float angleX, float angleY, float angleZ)
{
    float c1 = cos(-angleX);
    float c2 = cos(-angleY);
    float c3 = cos(-angleZ);
    float s1 = sin(-angleX);
    float s2 = sin(-angleY);
    float s3 = sin(-angleZ);

    float3x3 m;
    m[0][0] = c2 * c3;
    m[0][1] = c2 * s3;
    m[0][2] = -s2;

    m[1][0] = -c1 * s3 + s1 * s2 * c3;
    m[1][1] = c1 * c3 + s1 * s2 * s3;
    m[1][2] = s1 * c2;

    m[2][0] = s1 * s3 + c1 * s2 * c3;
    m[2][1] = -s1 * c3 + c1 * s2 * s3;
    m[2][2] = c1 * c2;

    return m;
}

// [Differentiable]
float3x3 matrixFromRotationQuaternions(float r0, float r1, float r2, float r3)
{
    float norm = sqrt(r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3 + 0.00000001);

    float3x3 m;

    float r = r0 / norm;
    float x = r1 / norm;
    float y = r2 / norm;
    float z = r3 / norm;

    m[0][0] = 1 - 2 * (y * y + z * z);
    m[0][1] = 2 * (x * y - r * z);
    m[0][2] = 2 * (x * z + r * y);

    m[1][0] = 2 * (x * y + r * z);
    m[1][1] = 1 - 2 * (x * x + z * z);
    m[1][2] = 2 * (y * z - r * x);

    m[2][0] = 2 * (x * z - r * y);
    m[2][1] = 2 * (y * z + r * x);
    m[2][2] = 1 - 2 * (x * x + y * y);

    return m;
}

// [Differentiable]
bool gs_bvh_hit(TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
             TensorView<float> centers, TensorView<float> scales, TensorView<float> rotates, TensorView<float> colors, TensorView<float> opacity, TensorView<float> cov3D_inverse,
             float3 rayo, float3 rayd, float t_min, float t_max, inout float t_hit, inout float3 debug_result, inout int index_hit, inout float2 uv)
{
    GPUBvhNode stack[MAX_STACK_SIZE];
    int count = 0;

    int root_index = 0;
    /* Add root node */
    GPUBvhNode now_node;
    now_node.index = root_index;
    now_node.left = g_lbvh_info[root_index, 0];
    now_node.right = g_lbvh_info[root_index, 1];
    now_node.primitiveIdx = g_lbvh_info[root_index, 2];
    stack[count++] = now_node;
    float closest_so_far = t_max;
    float2 closest_uv = float2(0.f);
    uint closest_index_so_far = 0;
    bool any_hit = false;
    float hit_t = 0.f;

    float2 hit_uv;


    float3 debug_res = float3(0.f);

    while (count > 0) {
        /* Pop the last node from the stack */
        GPUBvhNode node = stack[--count];
        float aabbMinX = g_lbvh_aabb[node.index, 0];
        float aabbMinY = g_lbvh_aabb[node.index, 1];
        float aabbMinZ = g_lbvh_aabb[node.index, 2];
        float aabbMaxX = g_lbvh_aabb[node.index, 3];
        float aabbMaxY = g_lbvh_aabb[node.index, 4];
        float aabbMaxZ = g_lbvh_aabb[node.index, 5];

        if (!aabb_hit(rayo, rayd, t_min, closest_so_far, float3(aabbMinX, aabbMinY, aabbMinZ), float3(aabbMaxX, aabbMaxY, aabbMaxZ)))
            continue;

        if (node.left != 0 && node.right != 0) {
            /* Add both nodes to the stack for further processing */
            GPUBvhNode left_node;
            left_node.index = node.left;
            left_node.left = g_lbvh_info[node.left, 0];
            left_node.right = g_lbvh_info[node.left, 1];
            left_node.primitiveIdx = g_lbvh_info[node.left, 2];

            GPUBvhNode right_node;
            right_node.index = node.right;
            right_node.left = g_lbvh_info[node.right, 0];
            right_node.right = g_lbvh_info[node.right, 1];
            right_node.primitiveIdx = g_lbvh_info[node.right, 2];

            stack[count++] = left_node;
            stack[count++] = right_node;
        } else if (node.left == 0 && node.right == 0)
        {
            /* It's a child node, we should test against the triangles and save the closest */
            // any_hit = true;
            uint now_gs_idx = node.primitiveIdx;

            float now_t_hit = 0.f;
            float3 now_pos_hit = float3(0.f);

            float3 center = float3(centers[now_gs_idx, 0], centers[now_gs_idx, 1], centers[now_gs_idx, 2]);
            float3 scale = float3(scales[now_gs_idx, 0], scales[now_gs_idx, 1], scales[now_gs_idx, 2]);
            float4 rotate = float4(rotates[now_gs_idx, 0], rotates[now_gs_idx, 1], rotates[now_gs_idx, 2], rotates[now_gs_idx, 3]);
            
            float3 color = float3(colors[now_gs_idx, 0], colors[now_gs_idx, 1], colors[now_gs_idx, 2]);

            float opacity = opacity[now_gs_idx];
        
            float3x3 matScale = float3x3(scale.x, 0, 0,
                                         0, scale.y, 0,
                                         0, 0, scale.z);

            // float3x3 matRotate = matrixFromRotationXYZ(rotate.x, rotate.y, rotate.z);
            float3x3 matRotate = matrixFromRotationQuaternions(rotate.x, rotate.y, rotate.z, rotate.w);
            float3x3 R_S = mul(matRotate, matScale);
            float3x3 ST_RT = mul(transpose(matScale), transpose(matRotate));

            float3x3 covariance = mul(R_S, ST_RT);

            float2 scale2 = float2(scales[now_gs_idx, 0], scales[now_gs_idx, 1]);

            // if (any(scale2 > 0.05f))
            // {
            //     continue;
            // }

            // bool hit = triangle_hit(rayo, rayd, t_min, closest_so_far, tri_v0, tri_v1, tri_v2, now_t_hit, now_pos_hit);
            rayd = normalize(rayd);
            float d = 1.0f;
            float3x3 invCov;

            float3 normalized_normal = normalize(color);
            float3 tangent, bitangent;
            branchlessONB(normalized_normal, tangent, bitangent);
            float3 T, B;
            bool valid = orthonormalizeSafe(normalized_normal, float4(tangent, 1.0f), T, B);
            float3x3 TBN = float3x3(T, B, normalized_normal);
            float3x3 invTBN = inverse(TBN);
            // bool hit = gaussian_hit(rayo, rayd, center, covariance, now_t_hit, now_pos_hit, d, invCov);
            // hit = false;
            float2 uv_hit = float2(0.5f);
            bool hit = ellipse_hit(rayo, rayd, center, matRotate, scale2, normalize(color), t_min, now_t_hit, now_pos_hit, uv_hit);
            float[6] cov3d_inv;
            cov3d_inv[0] = cov3D_inverse[now_gs_idx, 0];
            cov3d_inv[1] = cov3D_inverse[now_gs_idx, 1];
            cov3d_inv[2] = cov3D_inverse[now_gs_idx, 2];
            cov3d_inv[3] = cov3D_inverse[now_gs_idx, 3];
            cov3d_inv[4] = cov3D_inverse[now_gs_idx, 4];
            cov3d_inv[5] = cov3D_inverse[now_gs_idx, 5];

            // now_t_hit = ray_gs_intersects(center, cov3d_inv, rayo, rayd);
            // bool hit = now_t_hit > t_min;
            if (now_t_hit < t_min)
            {
                hit = false;
                continue;
            }
                
            
            float power = gaussian_fn(center, now_pos_hit, cov3d_inv);
            if (power > 0.0f)
            {
                hit = false;
                continue;
            }
                
            float alpha = min(0.99, opacity * exp(power));
            if (alpha < 1.0 / 255.0f)
            {
                hit = false;
                continue;
            }


            now_pos_hit = rayo + now_t_hit * rayd;
            
            // float3 normal = mul(matRotate, float3(0, 0, 1));

            // normal.x = normal.x > 0 ? normal.x : 1.0f;
            // normal.y = normal.y > 0 ? normal.y : 1.0f;
            // normal.z = normal.z > 0 ? normal.z : 1.0f;

            // normal = abs(normal);

            // hit = true;
            float dir_dot_normal = dot(rayd, normalize(color));
            // float3 wi = float3(dir_dot_normal, -dir_dot_normal, 0.0f);
            
            // hit = dir_dot_normal < -0.727f && opacity ? hit : false;
            // hit = dir_dot_normal < -0.5f && opacity * exp(power) > 0.1f ? hit : false;
            hit = dir_dot_normal < -0.0f? hit : false;
            // hit = now_t_hit > t_min ? hit : false;

            // float3 normal_param = normalize(color);
            bool update = hit && (now_t_hit < closest_so_far);
            closest_so_far = hit ? min(now_t_hit, closest_so_far) : closest_so_far;
            closest_index_so_far = update ? now_gs_idx : closest_index_so_far;
            closest_uv = update ? uv_hit : closest_uv;
            // closest_index_so_far = now_gs_idx;



            if (hit) {
                any_hit = true;
                hit_t = closest_so_far;
                // debug_res = normal;
                debug_res = float3(1 - alpha);
                hit_uv = closest_uv; 
                // index_hit = closest_index_so_far;
                // hit_pos = now_pos_hit;
                
                // break;
            }
            // hit_pos = tri_v;
        }
    }
    if (any_hit)
    {
        t_hit = hit_t;
        debug_result = debug_res;
        index_hit = closest_index_so_far;
        uv = hit_uv;
    }
    else
    {
        index_hit = -1;
    }
    return any_hit;
}

bool bvh_hit(TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
             TensorView<float> vert, TensorView<int> v_indx,
    float3 rayo, float3 rayd, float t_min, float t_max, inout float t_hit, inout float3 pos)
{
    GPUBvhNode stack[MAX_STACK_SIZE];
    int count = 0;

    int root_index = 0;
    /* Add root node */
    GPUBvhNode now_node;
    now_node.index = root_index;
    now_node.left = g_lbvh_info[root_index, 0];
    now_node.right = g_lbvh_info[root_index, 1];
    now_node.primitiveIdx = g_lbvh_info[root_index, 2];
    stack[count++] = now_node;
    float closest_so_far = t_max;
    bool any_hit = false;
    float hit_t = 0.f;
    float3 hit_pos = float3(0.f);

    while (count > 0) {
        /* Pop the last node from the stack */
        GPUBvhNode node = stack[--count];
        float aabbMinX = g_lbvh_aabb[node.index, 0];
        float aabbMinY = g_lbvh_aabb[node.index, 1];
        float aabbMinZ = g_lbvh_aabb[node.index, 2];
        float aabbMaxX = g_lbvh_aabb[node.index, 3];
        float aabbMaxY = g_lbvh_aabb[node.index, 4];
        float aabbMaxZ = g_lbvh_aabb[node.index, 5];

        if (!aabb_hit(rayo, rayd, t_min, closest_so_far, float3(aabbMinX, aabbMinY, aabbMinZ), float3(aabbMaxX, aabbMaxY, aabbMaxZ)))
            continue;

        if (node.left != 0 && node.right !=0) {
            /* Add both nodes to the stack for further processing */
            GPUBvhNode left_node;
            left_node.index = node.left;
            left_node.left = g_lbvh_info[node.left, 0];
            left_node.right = g_lbvh_info[node.left, 1];
            left_node.primitiveIdx = g_lbvh_info[node.left, 2];

            GPUBvhNode right_node;
            right_node.index = node.right;
            right_node.left = g_lbvh_info[node.right, 0];
            right_node.right = g_lbvh_info[node.right, 1];
            right_node.primitiveIdx = g_lbvh_info[node.right, 2];

            stack[count++] = left_node;
            stack[count++] = right_node;
        } else if (node.left == 0 && node.right == 0)
        {
            /* It's a child node, we should test against the triangles and save the closest */
            //any_hit = true;
            uint now_pri_idx = node.primitiveIdx;
            int3 v_in = int3(v_indx[now_pri_idx, 0], v_indx[now_pri_idx, 1], v_indx[now_pri_idx, 2]);
            float3 tri_v0 = float3(vert[v_in.x, 0], vert[v_in.x, 1], vert[v_in.x, 2]);
            float3 tri_v1 = float3(vert[v_in.y, 0], vert[v_in.y, 1], vert[v_in.y, 2]);
            float3 tri_v2 = float3(vert[v_in.z, 0], vert[v_in.z, 1], vert[v_in.z, 2]);
            float now_t_hit = 0.f;
            float3 now_pos_hit = float3(0.f);
            bool hit = triangle_hit(rayo, rayd, t_min, closest_so_far, tri_v0, tri_v1, tri_v2, now_t_hit, now_pos_hit);
            closest_so_far = hit ? min(now_t_hit, closest_so_far) : closest_so_far;
            if (hit) {
                any_hit = true;
                hit_t = closest_so_far;
                hit_pos = rayo + hit_t * rayd;
            }
            //hit_pos = tri_v;
        }
    }
    if (any_hit)
    {
        t_hit = hit_t;
        pos = hit_pos;
    }
    return any_hit;
}


[PreferRecompute]
bool triangle_hit_with_normal(float3 rayo, float3 rayd, float t_min, float t_max, float3 v0, float3 v1, float3 v2, inout float t_hit, inout float3 pos, inout float3 normal)
{
    float epsilon = 1e-15;
    float3 E1 = v1 - v0;
    float3 E2 = v2 - v0;
    float3 P = cross(rayd, E2);
    float det = dot(E1, P);
    if (det > -epsilon && det < epsilon)
        return false;
    float invDet = 1 / det;

    float3 T = rayo - v0;
    float u = dot(T,P) * invDet;
    if (u < 0 || u > 1) return false;

    float3 Q = cross(T, E1);
    float v = dot(rayd, Q) * invDet;
    if (v < 0 || u + v > 1) return false;

    float t = dot(E2, Q) * invDet;
    t_hit = t;
    pos = rayo + t * rayd;

    float3 face_normal = normalize(cross(E1, E2));
    float r = 1.0 - u - v;
    normal = u * face_normal + v * face_normal + r * face_normal;
    if(dot(-rayd, normal)<0)
    {
        normal = -normal;
    }
    normal = normalize(normal);

    return true;
}
[PreferRecompute]
bool gaussian_hit_with_normal(float3 origin, float3 dir, float3 mu, float3x3 cov, inout float t_hit, inout float3 pos)
{
    float3x3 covInv = inverse(cov);

    float3 muDelta = mu - origin;

    t_hit = dot(mul(muDelta, covInv), dir) / dot(mul(dir, covInv), dir);

    float3 hitpos = origin + t_hit * dir;

    float mdistance = dot(mul(hitpos - mu, covInv), hitpos - mu);

    float d = mdistance;

    pos = origin + t_hit * dir;

    if (mdistance < 0.25f && t_hit > 1e-6)
    {
        return true;
    }

    return false;
}

[PreferRecompute]
bool bvh_hit_with_normal(TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
             TensorView<float> vert, TensorView<int> v_indx,
    float3 rayo, float3 rayd, float t_min, float t_max, inout float t_hit, inout float3 pos, inout float3 normal)
{
    rayd = normalize(rayd);
    GPUBvhNode stack[MAX_STACK_SIZE];
    int count = 0;

    int root_index = 0;
    /* Add root node */
    GPUBvhNode now_node;
    now_node.index = root_index;
    now_node.left = g_lbvh_info[root_index, 0];
    now_node.right = g_lbvh_info[root_index, 1];
    now_node.primitiveIdx = g_lbvh_info[root_index, 2];
    stack[count++] = now_node;
    float closest_so_far = t_max;
    bool any_hit = false;
    float hit_t = 0.f;
    float3 hit_pos = float3(0.f);
    float3 hit_normal = float3(1.f);

    while (count > 0) {
        /* Pop the last node from the stack */
        GPUBvhNode node = stack[--count];
        float aabbMinX = g_lbvh_aabb[node.index, 0];
        float aabbMinY = g_lbvh_aabb[node.index, 1];
        float aabbMinZ = g_lbvh_aabb[node.index, 2];
        float aabbMaxX = g_lbvh_aabb[node.index, 3];
        float aabbMaxY = g_lbvh_aabb[node.index, 4];
        float aabbMaxZ = g_lbvh_aabb[node.index, 5];

        if (!aabb_hit(rayo, rayd, t_min, closest_so_far, float3(aabbMinX, aabbMinY, aabbMinZ), float3(aabbMaxX, aabbMaxY, aabbMaxZ)))
            continue;

        if (node.left != 0 && node.right !=0) {
            /* Add both nodes to the stack for further processing */
            GPUBvhNode left_node;
            left_node.index = node.left;
            left_node.left = g_lbvh_info[node.left, 0];
            left_node.right = g_lbvh_info[node.left, 1];
            left_node.primitiveIdx = g_lbvh_info[node.left, 2];

            GPUBvhNode right_node;
            right_node.index = node.right;
            right_node.left = g_lbvh_info[node.right, 0];
            right_node.right = g_lbvh_info[node.right, 1];
            right_node.primitiveIdx = g_lbvh_info[node.right, 2];

            stack[count++] = left_node;
            stack[count++] = right_node;
        } else if (node.left == 0 && node.right == 0)
        {
            /* It's a child node, we should test against the triangles and save the closest */
            //any_hit = true;
            uint now_pri_idx = node.primitiveIdx;
            int3 v_in = int3(v_indx[now_pri_idx, 0], v_indx[now_pri_idx, 1], v_indx[now_pri_idx, 2]);
            float3 tri_v0 = float3(vert[v_in.x, 0], vert[v_in.x, 1], vert[v_in.x, 2]);
            float3 tri_v1 = float3(vert[v_in.y, 0], vert[v_in.y, 1], vert[v_in.y, 2]);
            float3 tri_v2 = float3(vert[v_in.z, 0], vert[v_in.z, 1], vert[v_in.z, 2]);
            float now_t_hit = 0.f;
            float3 now_pos_hit = float3(0.f);
            float3 now_normal_hit = float3(1.f);
            bool hit = triangle_hit_with_normal(rayo, rayd, t_min, closest_so_far, tri_v0, tri_v1, tri_v2, now_t_hit, now_pos_hit, now_normal_hit);
            closest_so_far = hit ? min(now_t_hit, closest_so_far) : closest_so_far;
            if (hit) {
                any_hit = true;
                hit_t = closest_so_far;
                hit_pos = rayo + hit_t * rayd;
                if(now_t_hit<=closest_so_far)
                    hit_normal = now_normal_hit;
  
            }
            //hit_pos = tri_v;
        }
    }
    if (any_hit)
    {
        t_hit = hit_t;
        pos = hit_pos;
        normal = hit_normal;
    }
    return any_hit;
}

[AutoPyBindCUDA]
[CUDAKernel]
void intersect(int num_rays, TensorView<float> rays_o, TensorView<float> rays_d,
               TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
               TensorView<float> vert, TensorView<int> v_indx,
               TensorView<int> hit_map, TensorView<float> hit_pos_map)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint ray_index = dispatchIdx.x;
    if (ray_index >= num_rays) return;

    float3 rayo = float3(rays_o[ray_index, 0], rays_o[ray_index, 1], rays_o[ray_index, 2]);
    float3 rayd = float3(rays_d[ray_index, 0], rays_d[ray_index, 1], rays_d[ray_index, 2]);
    rayd = normalize(rayd);

    float t_min = 0.f;
    float t_max = 1e9;
    float t_hit = 0.f;
    float3 pos = float3(0.f);
    bool hit = bvh_hit(g_lbvh_info, g_lbvh_aabb, vert, v_indx,
                       rayo, rayd, t_min, t_max, t_hit, pos);
    //float3 normal = float3(0.f);
    //bool hit = bvh_hit_with_normal(g_lbvh_info, g_lbvh_aabb, vert, v_indx,
    //                   rayo, rayd, t_min, t_max, t_hit, pos, normal);

    if (hit)
    {
        hit_map[ray_index, 0] = 1;
        hit_pos_map[ray_index, 0] = pos.x;
        hit_pos_map[ray_index, 1] = pos.y;
        hit_pos_map[ray_index, 2] = pos.z;

        //hit_pos_map[ray_index, 0] = normal.x*0.5+0.5;
        //hit_pos_map[ray_index, 1] = normal.y*0.5+0.5;
        //hit_pos_map[ray_index, 2] = normal.z*0.5+0.5;
    }
}

// [AutoPyBindCUDA]
// [CUDAKernel]
// void gs_intersect(int num_rays, TensorView<float> rays_o, TensorView<float> rays_d,
//                TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
//                TensorView<float> centers, TensorView<float> scales, TensorView<float> rotates, TensorView<float> colors,
//                TensorView<int> hit_map, TensorView<float> hit_pos_map)
// {
//     uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
//     uint ray_index = dispatchIdx.x;
//     if (ray_index >= num_rays) return;

//     float3 rayo = float3(rays_o[ray_index, 0], rays_o[ray_index, 1], rays_o[ray_index, 2]);
//     float3 rayd = float3(rays_d[ray_index, 0], rays_d[ray_index, 1], rays_d[ray_index, 2]);
//     rayd = normalize(rayd);

//     float t_min = 0.f;
//     float t_max = 1e9;
//     float t_hit = 0.f;
//     float3 debug_result = float3(0.f);
//     // bool hit = bvh_hit(g_lbvh_info, g_lbvh_aabb, vert, v_indx,
//     //                    rayo, rayd, t_min, t_max, t_hit, pos);
//     uint index_hit = 0;
//     bool hit = gs_bvh_hit(g_lbvh_info, g_lbvh_aabb, centers, scales, rotates, colors,
//                           rayo, rayd, t_min, t_max, t_hit, debug_result, index_hit);
//     // float3 normal = float3(0.f);
//     // bool hit = bvh_hit_with_normal(g_lbvh_info, g_lbvh_aabb, vert, v_indx,
//     //                    rayo, rayd, t_min, t_max, t_hit, pos, normal);

//     if (hit)
//     {
//         hit_map[ray_index, 0] = 1;
//         hit_pos_map[ray_index, 0] = debug_result.x;
//         hit_pos_map[ray_index, 1] = debug_result.y;
//         hit_pos_map[ray_index, 2] = debug_result.z;

//         // hit_pos_map[ray_index, 0] = normal.x*0.5+0.5;
//         // hit_pos_map[ray_index, 1] = normal.y*0.5+0.5;
//         // hit_pos_map[ray_index, 2] = normal.z*0.5+0.5;
//     }
// }

[AutoPyBindCUDA]
[CUDAKernel]
void hit_table(int num_rays, TensorView<float> rays_o, TensorView<float> rays_d,
               TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
               TensorView<float> centers, TensorView<float> scales, TensorView<float> rotates, TensorView<float> colors, TensorView<float> opacity,
               TensorView<int> index_map, TensorView<float> debug_result)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint ray_index = dispatchIdx.x;
    if (ray_index >= num_rays) return;

    float3 rayo = float3(rays_o[ray_index, 0], rays_o[ray_index, 1], rays_o[ray_index, 2]);
    float3 rayd = float3(rays_d[ray_index, 0], rays_d[ray_index, 1], rays_d[ray_index, 2]);
    rayd = normalize(rayd);

    float t_min = 0.25f;
    // float t_max = 1e9;
    float t_max = 1.5f;
    float t_hit = 0.f;
    float3 debug_res = float3(0.f);
    // bool hit = bvh_hit(g_lbvh_info, g_lbvh_aabb, vert, v_indx,
    //                    rayo, rayd, t_min, t_max, t_hit, pos);
    int index_hit = -1;
    // bool hit = gs_bvh_hit(g_lbvh_info, g_lbvh_aabb, centers, scales, rotates, colors, opacity,
    //                       rayo, rayd, t_min, t_max, t_hit, debug_res, index_hit);
    bool hit = true;
    // float3 normal = float3(0.f);
    // bool hit = bvh_hit_with_normal(g_lbvh_info, g_lbvh_aabb, vert, v_indx,
    //                    rayo, rayd, t_min, t_max, t_hit, pos, normal);

    if (hit)
    {
        index_map[ray_index, 0] = index_hit;
        debug_result[ray_index, 0] = debug_res.x;
        debug_result[ray_index, 1] = debug_res.y;
        debug_result[ray_index, 2] = debug_res.z;
    }
    else
    {
        index_map[ray_index, 0] = -1;
        // debug_result[ray_index, 0] = debug_res.x;
        // debug_result[ray_index, 1] = debug_res.y;
        // debug_result[ray_index, 2] = debug_res.z;
    
    }
}

// input: direct shading [N, 3], 

// [Differentiable]
// [AutoPyBindCUDA]
// [CudaKernel]
// void render_indirect(int num_rays, TensorView<float> rays_o, TensorView<float> rays_d,
//                      TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
//                      TensorView<float> centers, TensorView<float> scales, TensorView<float> rotates, TensorView<float> direct,
//                      TensorView<float> irradiance, TensorView<float> occlusion)
// {
//     uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
//     uint ray_index = dispatchIdx.x;
//     if (ray_index >= num_rays) return;

//     float3 rayo = float3(rays_o[ray_index, 0], rays_o[ray_index, 1], rays_o[ray_index, 2]);
//     float3 rayd = float3(rays_d[ray_index, 0], rays_d[ray_index, 1], rays_d[ray_index, 2]);
//     rayd = normalize(rayd);

//     float t_min = 0.1f;
//     float t_max = 1.0f;
//     float t_hit = 0.f;
//     float3 debug_res = float3(0.f);
//     // bool hit = bvh_hit(g_lbvh_info, g_lbvh_aabb, vert, v_indx,
//     //                    rayo, rayd, t_min, t_max, t_hit, pos);
//     uint index_hit = 0;
//     bool hit = gs_bvh_hit(g_lbvh_info, g_lbvh_aabb, centers, scales, rotates, direct,
//                           rayo, rayd, t_min, t_max, t_hit, debug_res, index_hit);


//     // float3 normal = float3(0.f);
//     // bool hit = bvh_hit_with_normal(g_lbvh_info, g_lbvh_aabb, vert, v_indx,
//     //                    rayo, rayd, t_min, t_max, t_hit, pos, normal);

//     if (hit)
//     {
//         irradiance[uint(ray_index/256), 0] += debug_res.x;
//         irradiance[uint(ray_index/256), 1] += debug_res.y;
//         irradiance[uint(ray_index/256), 2] += debug_res.z;
//         occlusion[uint(ray_index / 256)] += 1;
//     }
// }


// [Differentiable]
// [AutoPyBindCUDA]
// [CudaKernel]
// void render_indirect_with_index_buffer(int num_rays, TensorView<float> rays_o, TensorView<float> rays_d,
//                                        TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
//                                        TensorView<float> centers, TensorView<float> scales, TensorView<float> rotates, DiffTensorView direct, TensorView<int> hit_indices,
//                                        DiffTensorView irradiance, DiffTensorView occlusion)
// {
//     uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
//     uint ray_index = dispatchIdx.x;
//     if (ray_index >= num_rays) return;

//     float3 rayo = float3(rays_o[ray_index, 0], rays_o[ray_index, 1], rays_o[ray_index, 2]);
//     float3 rayd = float3(rays_d[ray_index, 0], rays_d[ray_index, 1], rays_d[ray_index, 2]);
//     rayd = normalize(rayd);

//     float t_min = 0.f;
//     float t_max = 1e9;
//     float t_hit = 0.f;
//     float3 debug_res = float3(0.f);
//     // bool hit = bvh_hit(g_lbvh_info, g_lbvh_aabb, vert, v_indx,
//     //                    rayo, rayd, t_min, t_max, t_hit, pos);
//     uint index_hit = 0;
//     // bool hit = gs_bvh_hit(g_lbvh_info, g_lbvh_aabb, centers, scales, rotates, direct,
//     //                       rayo, rayd, t_min, t_max, t_hit, debug_res, index_hit);
//     index_hit = hit_indices[ray_index, 0];
//     // float3 normal = float3(0.f);
//     // bool hit = bvh_hit_with_normal(g_lbvh_info, g_lbvh_aabb, vert, v_indx,
//     //                    rayo, rayd, t_min, t_max, t_hit, pos, normal);

//     if (index_hit != 0)
//     {
//         irradiance[uint(ray_index / 256), 0] += direct[index_hit, 0];
//         irradiance[uint(ray_index / 256), 1] += direct[index_hit, 1];
//         irradiance[uint(ray_index / 256), 2] += direct[index_hit, 2];
//         occlusion[uint(ray_index / 256)] += 1.0f;
//     }
// }



// input index buffer: [N, 256]
// input direct shading: [N, 3]
// input envmap: [512, 1024, 3]

// input proxy_albedo: [N, 3]
// input proxy_roughness: [N, 1]
// input proxy_metallic: [N, 1]

// input ray_directions: [256, 3]
// input view_directions: [N, 3]

// output irradiance: [N, 3]

// if output irradiance: [N, 256, 3]
// idx.x -> N
// idy -> 256

// ray_directions: [N, 256, 3]


// computing indirect actually
[Differentiable]
[AutoPyBindCUDA]
[CudaKernel]
void render_irradiance(int N, int S, TensorView<float> ray_directions, DiffTensorView envmap,
                       TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
                       TensorView<float> centers, TensorView<float> scales, TensorView<float> rotates,
                       TensorView<float> normals, DiffTensorView albedos, DiffTensorView roughnesses, DiffTensorView metallics, TensorView<float> opacities,
                        TensorView<int> hit_indices, TensorView<float> uvs, TensorView<float> SHs,
                        DiffTensorView irradiance)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    if (dispatchIdx.x >= N || dispatchIdx.y >= S || dispatchIdx.z >= S) return;
    // uint pri_ray_index = uint(floor(dispatchIdx.y / 256));
    uint pri_ray_index = dispatchIdx.y;
    // uint sec_ray_index = dispatchIdx.y % 256;
    uint sec_ray_index = dispatchIdx.z;
    uint gs_index = dispatchIdx.x ;

    float3 pri_ray_dir;
    pri_ray_dir.x = no_diff ray_directions[gs_index, pri_ray_index, 0];
    pri_ray_dir.y = no_diff ray_directions[gs_index, pri_ray_index, 1];
    pri_ray_dir.z = no_diff ray_directions[gs_index, pri_ray_index, 2];

    int hit_index = hit_indices[gs_index, pri_ray_index];

    float3 sh = float3(0.0f);

    float3 pri_normal0;
    float3 pri_normal1;
    float3 pri_normal2;
    float3 pri_normal3;

    pri_normal0.x = no_diff normals[gs_index, 0];
    pri_normal1.x = no_diff normals[gs_index, 1];
    pri_normal2.x = no_diff normals[gs_index, 2];
    pri_normal3.x = no_diff normals[gs_index, 3];

    pri_normal0.y = no_diff normals[gs_index, 4];
    pri_normal1.y = no_diff normals[gs_index, 5];
    pri_normal2.y = no_diff normals[gs_index, 6];
    pri_normal3.y = no_diff normals[gs_index, 7];

    pri_normal0.z = no_diff normals[gs_index, 8];
    pri_normal1.z = no_diff normals[gs_index, 9];
    pri_normal2.z = no_diff normals[gs_index, 10];
    pri_normal3.z = no_diff normals[gs_index, 11];

    float2 pri_uv;
    pri_uv.x = no_diff uvs[gs_index, pri_ray_index, 0];
    pri_uv.y = no_diff uvs[gs_index, pri_ray_index, 1];

    float pri_w0;
    float pri_w1;
    float pri_w2;
    float pri_w3;

    pri_w0 = (1 - pri_uv.x) * (1 - pri_uv.y);
    pri_w1 = pri_uv.x * (1 - pri_uv.y);
    pri_w2 = (1 - pri_uv.x) * pri_uv.y;
    pri_w3 = pri_uv.x * pri_uv.y;

    float3 pri_norma = pri_w0 * pri_normal0 + pri_w1 * pri_normal1 + pri_w2 * pri_normal2 + pri_w3 * pri_normal3;


    if (hit_index == -1)
    {
        // float3 irr = eval_envmap(envmap, pri_ray_dir) / 256;
        float3 irr = float3(0.0f);
        // irradiance.storeOnce(uint3(gs_index, pri_ray_index, 0), irr.x);
        // irradiance.storeOnce(uint3(gs_index, pri_ray_index, 1), irr.y);
        // irradiance.storeOnce(uint3(gs_index, pri_ray_index, 2), irr.z);
        irradiance[gs_index, pri_ray_index, 0] = irr.x;
        irradiance[gs_index, pri_ray_index, 1] = irr.y;
        irradiance[gs_index, pri_ray_index, 2] = irr.z;
    }
    else
    {
        float3 sec_ray_dir;
        sec_ray_dir.x = no_diff ray_directions[hit_index, sec_ray_index, 0];
        sec_ray_dir.y = no_diff ray_directions[hit_index, sec_ray_index, 1];
        sec_ray_dir.z = no_diff ray_directions[hit_index, sec_ray_index, 2];

        sec_ray_dir = normalize(sec_ray_dir);

        int sec_hit_index = hit_indices[hit_index, sec_ray_index];

        float3 normal0;
        float3 normal1;
        float3 normal2;
        float3 normal3;

        normal0.x = no_diff normals[hit_index, 0];
        normal1.x = no_diff normals[hit_index, 1];
        normal2.x = no_diff normals[hit_index, 2];
        normal3.x = no_diff normals[hit_index, 3];

        normal0.y = no_diff normals[hit_index, 4];
        normal1.y = no_diff normals[hit_index, 5];
        normal2.y = no_diff normals[hit_index, 6];
        normal3.y = no_diff normals[hit_index, 7];

        normal0.z = no_diff normals[hit_index, 8];
        normal1.z = no_diff normals[hit_index, 9];
        normal2.z = no_diff normals[hit_index, 10];
        normal3.z = no_diff normals[hit_index, 11];

        float3 albedo0;
        float3 albedo1;
        float3 albedo2;
        float3 albedo3;

        albedo0.x = albedos[hit_index, 0];
        albedo1.x = albedos[hit_index, 1];
        albedo2.x = albedos[hit_index, 2];
        albedo3.x = albedos[hit_index, 3];

        albedo0.y = albedos[hit_index, 4];
        albedo1.y = albedos[hit_index, 5];
        albedo2.y = albedos[hit_index, 6];
        albedo3.y = albedos[hit_index, 7];

        albedo0.z = albedos[hit_index, 8];
        albedo1.z = albedos[hit_index, 9];
        albedo2.z = albedos[hit_index, 10];
        albedo3.z = albedos[hit_index, 11];

        float2 uv;
        uv.x = no_diff uvs[hit_index, sec_ray_index, 0];
        uv.y = no_diff uvs[hit_index, sec_ray_index, 1];

        float w0;
        float w1;
        float w2;
        float w3;

        w0 = (1 - uv.x) * (1 - uv.y);
        w1 = uv.x * (1 - uv.y);
        w2 = (1 - uv.x) * uv.y;
        w3 = uv.x * uv.y;

        float roughness0 = roughnesses[hit_index, 0];
        float roughness1 = roughnesses[hit_index, 1];
        float roughness2 = roughnesses[hit_index, 2];
        float roughness3 = roughnesses[hit_index, 3];


        float metallic = metallics[hit_index, 0];

        float opacity = no_diff opacities[hit_index, 0];

        float3 direct_light;
        direct_light.x = envmap[hit_index, sec_ray_index, 0];
        direct_light.y = envmap[hit_index, sec_ray_index, 1];
        direct_light.z = envmap[hit_index, sec_ray_index, 2];
        float3 irr_res;
        if (sec_hit_index != -1)
        {
            irr_res = float3(0.0, 0.0, 0.0);
        }
        else
        {
            float3 irr0 = shading_brdf_simple(-pri_ray_dir, sec_ray_dir, normal0, albedo0, roughness0) * clamp(dot(normal0, sec_ray_dir), 1e-6, 1.0);
            float3 irr1 = shading_brdf_simple(-pri_ray_dir, sec_ray_dir, normal1, albedo1, roughness1) * clamp(dot(normal1, sec_ray_dir), 1e-6, 1.0);
            float3 irr2 = shading_brdf_simple(-pri_ray_dir, sec_ray_dir, normal2, albedo2, roughness2) * clamp(dot(normal2, sec_ray_dir), 1e-6, 1.0);
            float3 irr3 = shading_brdf_simple(-pri_ray_dir, sec_ray_dir, normal3, albedo3, roughness3) * clamp(dot(normal3, sec_ray_dir), 1e-6, 1.0);

            float3 irr = w0 * irr0 + w1 * irr1 + w2 * irr2 + w3 * irr3;
            // float3 irr = irr0;

            irr_res = irr * direct_light / S;
            // irr_res = sh * opacity / S;
            float test_T = 1.0f;
            sh = eval_sh(SHs, hit_index, pri_ray_dir) / S;
            float t_min = 0.05f;
            float t_max = 0.5f;

            float t_hit = 0.0f;

            float3 rayo = centers[hit_index, 0];
            float3 rayd = sec_ray_dir;

            float3 debug_res = float3(0.0f);
            int index_hit = -1;

            bool done = false;
        }
        
        


        // [ForceUnroll]
        // for (int i = 0; i < 10; i++)
        // // while ((test_T > 0.001f) && (!done))
        // {
        //     if (test_T < 0.001f || done)
        //         break;
        //     bool hit = gs_bvh_hit(g_lbvh_info, g_lbvh_aabb, centers, scales, rotates, centers, opacities,
        //                           rayo, rayd, t_min, t_max, t_hit, debug_res, index_hit);
        //     if (hit)
        //     {
        //         if (test_T < 0.001f)
        //         {
        //             test_T = 0.00099f;
        //             done = true;
        //             break;
        //         }
        //         rayo = rayo + rayd * t_hit;
        //         // t_min = t_hit + 0.005f;
        //         // t_max = t_min + 1.0f;

        //         sh += eval_sh(SHs, index_hit, sec_ray_dir) * no_diff opacities[index_hit] * test_T;
        //         test_T = test_T * (1.0f - no_diff opacities[index_hit]);
        //     }
        //     else
        //     {
        //         done = true;
        //     }
        // }
        

        // float3 SH = eval_sh(SHs, hit_index, pri_ray_dir) * opacity  / (256);

        // irradiance[gs_index, pri_ray_index, 0] += irr_res.x;
        // irradiance[gs_index, pri_ray_index, 1] += irr_res.y;
        // irradiance[gs_index, pri_ray_index, 2] += irr_res.z;
        irradiance[gs_index, pri_ray_index, 0] += irr_res.x;
        irradiance[gs_index, pri_ray_index, 1] += irr_res.y;
        irradiance[gs_index, pri_ray_index, 2] += irr_res.z;
        // irradiance[gs_index, pri_ray_index, 0] += SH.x;
        // irradiance[gs_index, pri_ray_index, 1] += SH.y;
        // irradiance[gs_index, pri_ray_index, 2] += SH.z;

        // irradiance[gs_index, pri_ray_index, 0] += sh.x;
        // irradiance[gs_index, pri_ray_index, 1] += sh.y;
        // irradiance[gs_index, pri_ray_index, 2] += sh.z;
    }
}

[Differentiable]
[AutoPyBindCUDA]
[CudaKernel]
void render_irradiance_sample(int N, int S, TensorView<int> sample_indices, TensorView<float> ray_directions, DiffTensorView envmap,
                              TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
                              TensorView<float> centers, TensorView<float> scales, TensorView<float> rotates,
                              TensorView<float> normals, DiffTensorView albedos, DiffTensorView roughnesses, DiffTensorView metallics, TensorView<float> opacities,
                              TensorView<int> hit_indices, TensorView<float> uvs, TensorView<float> SHs,
                              DiffTensorView irradiance)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    if (dispatchIdx.x >= N || dispatchIdx.y >= S) return;
    // uint pri_ray_index = uint(floor(dispatchIdx.y / 256));
    // uint sec_ray_index = dispatchIdx.y % 256;
    uint sec_ray_index = dispatchIdx.y;
    uint gs_index = dispatchIdx.x;
    uint pri_ray_index = sample_indices[gs_index];

    float3 pri_ray_dir;
    pri_ray_dir.x = no_diff ray_directions[gs_index, pri_ray_index, 0];
    pri_ray_dir.y = no_diff ray_directions[gs_index, pri_ray_index, 1];
    pri_ray_dir.z = no_diff ray_directions[gs_index, pri_ray_index, 2];

    int hit_index = hit_indices[gs_index, pri_ray_index];
    float3 pri_normal0;
    float3 pri_normal1;
    float3 pri_normal2;
    float3 pri_normal3;

    pri_normal0.x = no_diff normals[gs_index, 0];
    pri_normal1.x = no_diff normals[gs_index, 1];
    pri_normal2.x = no_diff normals[gs_index, 2];
    pri_normal3.x = no_diff normals[gs_index, 3];

    pri_normal0.y = no_diff normals[gs_index, 4];
    pri_normal1.y = no_diff normals[gs_index, 5];
    pri_normal2.y = no_diff normals[gs_index, 6];
    pri_normal3.y = no_diff normals[gs_index, 7];

    pri_normal0.z = no_diff normals[gs_index, 8];
    pri_normal1.z = no_diff normals[gs_index, 9];
    pri_normal2.z = no_diff normals[gs_index, 10];
    pri_normal3.z = no_diff normals[gs_index, 11];

    float2 pri_uv;
    pri_uv.x = no_diff uvs[gs_index, pri_ray_index, 0];
    pri_uv.y = no_diff uvs[gs_index, pri_ray_index, 1];

    float pri_w0;
    float pri_w1;
    float pri_w2;
    float pri_w3;

    pri_w0 = (1 - pri_uv.x) * (1 - pri_uv.y);
    pri_w1 = pri_uv.x * (1 - pri_uv.y);
    pri_w2 = (1 - pri_uv.x) * pri_uv.y;
    pri_w3 = pri_uv.x * pri_uv.y;

    float3 pri_norma = pri_w0 * pri_normal0 + pri_w1 * pri_normal1 + pri_w2 * pri_normal2 + pri_w3 * pri_normal3;

    if (hit_index == -1)
    {
        // float3 irr = eval_envmap(envmap, pri_ray_dir) / 256;
        float3 irr = float3(0.0f);
        // irradiance.storeOnce(uint3(gs_index, pri_ray_index, 0), irr.x);
        // irradiance.storeOnce(uint3(gs_index, pri_ray_index, 1), irr.y);
        // irradiance.storeOnce(uint3(gs_index, pri_ray_index, 2), irr.z);
        irradiance[gs_index, pri_ray_index, 0] = irr.x;
        irradiance[gs_index, pri_ray_index, 1] = irr.y;
        irradiance[gs_index, pri_ray_index, 2] = irr.z;
    }
    else
    {
        float3 sec_ray_dir;
        sec_ray_dir.x = no_diff ray_directions[hit_index, sec_ray_index, 0];
        sec_ray_dir.y = no_diff ray_directions[hit_index, sec_ray_index, 1];
        sec_ray_dir.z = no_diff ray_directions[hit_index, sec_ray_index, 2];

        sec_ray_dir = normalize(sec_ray_dir);

        int sec_hit_index = hit_indices[hit_index, sec_ray_index];

        float3 normal0;
        float3 normal1;
        float3 normal2;
        float3 normal3;

        normal0.x = no_diff normals[hit_index, 0];
        normal1.x = no_diff normals[hit_index, 1];
        normal2.x = no_diff normals[hit_index, 2];
        normal3.x = no_diff normals[hit_index, 3];

        normal0.y = no_diff normals[hit_index, 4];
        normal1.y = no_diff normals[hit_index, 5];
        normal2.y = no_diff normals[hit_index, 6];
        normal3.y = no_diff normals[hit_index, 7];

        normal0.z = no_diff normals[hit_index, 8];
        normal1.z = no_diff normals[hit_index, 9];
        normal2.z = no_diff normals[hit_index, 10];
        normal3.z = no_diff normals[hit_index, 11];

        float3 albedo0;
        float3 albedo1;
        float3 albedo2;
        float3 albedo3;

        albedo0.x = albedos[hit_index, 0];
        albedo1.x = albedos[hit_index, 1];
        albedo2.x = albedos[hit_index, 2];
        albedo3.x = albedos[hit_index, 3];

        albedo0.y = albedos[hit_index, 4];
        albedo1.y = albedos[hit_index, 5];
        albedo2.y = albedos[hit_index, 6];
        albedo3.y = albedos[hit_index, 7];

        albedo0.z = albedos[hit_index, 8];
        albedo1.z = albedos[hit_index, 9];
        albedo2.z = albedos[hit_index, 10];
        albedo3.z = albedos[hit_index, 11];

        float2 uv;
        uv.x = no_diff uvs[hit_index, sec_ray_index, 0];
        uv.y = no_diff uvs[hit_index, sec_ray_index, 1];

        float w0;
        float w1;
        float w2;
        float w3;

        w0 = (1 - uv.x) * (1 - uv.y);
        w1 = uv.x * (1 - uv.y);
        w2 = (1 - uv.x) * uv.y;
        w3 = uv.x * uv.y;

        float roughness = roughnesses[hit_index, 0];

        float metallic = metallics[hit_index, 0];

        float opacity = no_diff opacities[hit_index, 0];

        float3 direct_light;
        direct_light.x = envmap[hit_index, sec_ray_index, 0];
        direct_light.y = envmap[hit_index, sec_ray_index, 1];
        direct_light.z = envmap[hit_index, sec_ray_index, 2];
        float3 irr_res;
        if (sec_hit_index != -1)
        {
            irr_res = float3(0.0, 0.0, 0.0);
        }
        else
        {
            float3 irr0 = shading_brdf_simple(-pri_ray_dir, sec_ray_dir, normal0, albedo0, roughness);
            float3 irr1 = shading_brdf_simple(-pri_ray_dir, sec_ray_dir, normal1, albedo1, roughness);
            float3 irr2 = shading_brdf_simple(-pri_ray_dir, sec_ray_dir, normal2, albedo2, roughness);
            float3 irr3 = shading_brdf_simple(-pri_ray_dir, sec_ray_dir, normal3, albedo3, roughness);

            float3 irr = w0 * irr0 + w1 * irr1 + w2 * irr2 + w3 * irr3;

            irr_res = irr * direct_light / S;
            // irr_res = sh * opacity / S;
            float test_T = 1.0f;
            float3 sh = eval_sh(SHs, hit_index, pri_ray_dir);
            float t_min = 0.05f;
            float t_max = 0.5f;

            float t_hit = 0.0f;

            float3 rayo = centers[hit_index, 0];
            float3 rayd = sec_ray_dir;

            float3 debug_res = float3(0.0f);
            int index_hit = -1;

            bool done = false;
        }
        float3 sh = float3(0.0f);

        // [ForceUnroll]
        // for (int i = 0; i < 10; i++)
        // // while ((test_T > 0.001f) && (!done))
        // {
        //     if (test_T < 0.001f || done)
        //         break;
        //     bool hit = gs_bvh_hit(g_lbvh_info, g_lbvh_aabb, centers, scales, rotates, centers, opacities,
        //                           rayo, rayd, t_min, t_max, t_hit, debug_res, index_hit);
        //     if (hit)
        //     {
        //         if (test_T < 0.001f)
        //         {
        //             test_T = 0.00099f;
        //             done = true;
        //             break;
        //         }
        //         rayo = rayo + rayd * t_hit;
        //         // t_min = t_hit + 0.005f;
        //         // t_max = t_min + 1.0f;

        //         sh += eval_sh(SHs, index_hit, sec_ray_dir) * no_diff opacities[index_hit] * test_T;
        //         test_T = test_T * (1.0f - no_diff opacities[index_hit]);
        //     }
        //     else
        //     {
        //         done = true;
        //     }
        // }

        // float3 SH = eval_sh(SHs, hit_index, pri_ray_dir) * opacity  / (256);

        // irradiance[gs_index, pri_ray_index, 0] += irr_res.x;
        // irradiance[gs_index, pri_ray_index, 1] += irr_res.y;
        // irradiance[gs_index, pri_ray_index, 2] += irr_res.z;
        irradiance[gs_index, 0] += irr_res.x;
        irradiance[gs_index, 1] += irr_res.y;
        irradiance[gs_index, 2] += irr_res.z;
        // irradiance[gs_index, pri_ray_index, 0] += SH.x;
        // irradiance[gs_index, pri_ray_index, 1] += SH.y;
        // irradiance[gs_index, pri_ray_index, 2] += SH.z;

        // irradiance[gs_index, pri_ray_index, 0] += sh.x;
        // irradiance[gs_index, pri_ray_index, 1] += sh.y;
        // irradiance[gs_index, pri_ray_index, 2] += sh.z;
    }
}




// input ray_directions: [N, 256, 3]
// normal: [N, 3]
// albedo: [N, 3]
// roughness: [N, 1]
// metallic: [N, 1]
// irradiance: [N, 256, 3]


// output rgb: [N, 3]
[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void render_direct_and_indirect(int N, DiffTensorView view_dirs, DiffTensorView ray_directions,
                   DiffTensorView normals, DiffTensorView albedos, DiffTensorView roughnesses, DiffTensorView metallics, DiffTensorView irradiance,
                   DiffTensorView rgb)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    if (dispatchIdx.x >= N || dispatchIdx.y >= 256) return;

    uint gs_index = dispatchIdx.x;
    uint ray_index = dispatchIdx.y;

    float3 normal;
    normal.x = normals[gs_index, 0];
    normal.y = normals[gs_index, 1];
    normal.z = normals[gs_index, 2];

    float3 albedo;
    albedo.x = albedos[gs_index, 0];
    albedo.y = albedos[gs_index, 1];
    albedo.z = albedos[gs_index, 2];

    float roughness = roughnesses[gs_index, 0];

    float metallic = metallics[gs_index, 0];

    float3 view_dir;
    view_dir.x = no_diff view_dirs[gs_index, 0];
    view_dir.y = no_diff view_dirs[gs_index, 1];
    view_dir.z = no_diff view_dirs[gs_index, 2];

    float3 ray_direction;
    ray_direction.x = no_diff ray_directions[gs_index, ray_index, 0];
    ray_direction.y = no_diff ray_directions[gs_index, ray_index, 1];
    ray_direction.z = no_diff ray_directions[gs_index, ray_index, 2];

    float3 Li;
    Li.x = irradiance[gs_index, ray_index, 0];
    Li.y = irradiance[gs_index, ray_index, 1];
    Li.z = irradiance[gs_index, ray_index, 2];

    float3 brdf = shading_brdf(-view_dir, ray_direction, normal, albedo, roughness, metallic);

    float3 result = brdf * Li / 256;

    rgb[gs_index, 0] += result.x;
    rgb[gs_index, 1] += result.y;
    rgb[gs_index, 2] += result.z;
}




[Differentiable]
[AutoPyBindCUDA]
[CudaKernel]
void render_indirect_with_index_buffer(int N, DiffTensorView ray_directions, DiffTensorView view_directions,
                                       DiffTensorView albedos, DiffTensorView roughnesses, DiffTensorView metallics, 
                                       DiffTensorView direct, TensorView<int> hit_indices,
                                       DiffTensorView irradiance)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    if (dispatchIdx.x >= N || dispatchIdx.y >= 256) return;
    uint ray_index = dispatchIdx.y;
    uint gs_index = dispatchIdx.x;


    float3 viewd_local;
    viewd_local.x = no_diff view_directions[gs_index, 0];
    viewd_local.y = no_diff view_directions[gs_index, 1];
    viewd_local.z = no_diff view_directions[gs_index, 2];

    float3 albedo;
    albedo.x = albedos[gs_index, 0];
    albedo.y = albedos[gs_index, 1];
    albedo.z = albedos[gs_index, 2];

    float roughness = roughnesses[gs_index, 0];

    float metallic = metallics[gs_index, 0];

    float3 rayd_local;
    rayd_local.x = no_diff ray_directions[ray_index, 0];
    rayd_local.y = no_diff ray_directions[ray_index, 1];
    rayd_local.z = no_diff ray_directions[ray_index, 2];

    int hit = hit_indices[gs_index, ray_index];

    float3 angle_res = hit == -1 ? float3(0.0f) : float3(direct[hit, 0], direct[hit, 1], direct[hit, 2]) / 255;
    // float3 angle_res = hit == -1 ? float3(0.0f) : float3(1.0f);
    // float3 angle_res = hit == -1 ? float3(0.0f) : float3(direct[gs_index, 0], direct[gs_index, 1], direct[gs_index, 2]);
    // float3 angle_res = albedo;
    // float3 angle_res = float3(1.0f);

    // irradiance[gs_index, 0] += angle_res.x;
    // irradiance[gs_index, 1] += angle_res.y;
    // irradiance[gs_index, 2] += angle_res.z;
    // irradiance[gs_index, ray_index, 0] = angle_res.x;
    // irradiance[gs_index, ray_index, 1] = angle_res.y;
    // irradiance[gs_index, ray_index, 2] = angle_res.z;


    // irradiance_res += angle_res;


    irradiance.storeOnce(uint3(gs_index, ray_index, 0), angle_res.x);
    irradiance.storeOnce(uint3(gs_index, ray_index, 1), angle_res.y);
    irradiance.storeOnce(uint3(gs_index, ray_index, 2), angle_res.z);
}


// view: cam -> gaussian

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void render_direct(int H, int W, DiffTensorView view_dirs,
            DiffTensorView normal_map, DiffTensorView albedo_map, DiffTensorView roughness_map, DiffTensorView metallic_map, DiffTensorView envmap, 
            DiffTensorView direct, TensorView<float> debug_results)
{


    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    if (dispatchIdx.x >= H || dispatchIdx.y >= W) return;

    float3 background = float3(0.0f, 0.0f, 0.0f);
    uint pixx = dispatchIdx.x;
    uint pixy = dispatchIdx.y;

    float3 normal;
    normal.x = normal_map[pixx, pixy, 0];
    normal.y = normal_map[pixx, pixy, 1];
    normal.z = normal_map[pixx, pixy, 2];

    float3 albedo;
    albedo.x = albedo_map[pixx, pixy, 0];
    albedo.y = albedo_map[pixx, pixy, 1];
    albedo.z = albedo_map[pixx, pixy, 2];

    float roughness = roughness_map[pixx, pixy, 0];

    float metallic = metallic_map[pixx, pixy, 0];

    float3 view_dir;
    view_dir.x = no_diff view_dirs[pixx, pixy, 0];
    view_dir.y = no_diff view_dirs[pixx, pixy, 1];
    view_dir.z = no_diff view_dirs[pixx, pixy, 2];

    float3 wi = shading_wi(view_dir, normal, albedo, roughness, metallic, envmap);
    float3 pbr_result = shading_direct(-view_dir, normal, albedo, roughness, metallic, envmap);
    // float3 wo = normalize(view_dir);
    // float3 normalized_normal = normalize(normal);
    // float3 wi = normalize(2 * dot(wo, normalized_normal) * normalized_normal - wo);

    // float3 pbr_result = albedo * evalBRDF(wi, wo, normalized_normal, roughness, 1 - metallic, metallic);

    float3 wo = shading_wo(view_dir, normal, albedo, roughness, metallic, envmap);
    float3 specular = shading_specular(view_dir, normal, albedo, roughness, metallic, envmap);


    direct.storeOnce(uint3(pixx, pixy, 0), pbr_result.x);
    direct.storeOnce(uint3(pixx, pixy, 1), pbr_result.y);
    direct.storeOnce(uint3(pixx, pixy, 2), pbr_result.z);

    // direct[pixx, pixy, 0] = pbr_result.x;
    // direct[pixx, pixy, 1] = pbr_result.y;
    // direct[pixx, pixy, 2] = pbr_result.z;

    debug_results[pixx, pixy, 0] = wi.x;
    debug_results[pixx, pixy, 1] = wi.y;
    debug_results[pixx, pixy, 2] = wi.z;

    debug_results[pixx, pixy, 3] = wo.x;
    debug_results[pixx, pixy, 4] = wo.y;
    debug_results[pixx, pixy, 5] = wo.z;

    debug_results[pixx, pixy, 6] = specular.x;
    debug_results[pixx, pixy, 7] = specular.y;
    debug_results[pixx, pixy, 8] = specular.z;

    

}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void render_indexing(int H, int W, DiffTensorView indirect, TensorView<int> hit_map,
                   DiffTensorView indirect_map)
{

    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    if (dispatchIdx.x >= H || dispatchIdx.y >= W) return;

    uint pixx = dispatchIdx.x;
    uint pixy = dispatchIdx.y;

    // float3 indirect_res = float3(0.0f, 0.0f, 0.0f);
    // int index = hit_map[pixx, pixy];
    uint index = abs(hit_map.load(uint3(pixx, pixy, 0)));

    float indirectx = indirect[uint(index), 0];
    float indirecty = indirect[uint(index), 1];
    float indirectz = indirect[uint(index), 2];

    // indirect_res.x = indirectx;
    // indirect_res.y = indirecty;
    // indirect_res.z = indirectz;
    // indirect_res.y = indirect[uint(index), 1];
    // indirect_res.z = indirect[uint(index), 2];
    // if (index != -1)
    // {

    // }
    // else
    // {
    //     indirect_res = float3(0.0f, 0.0f, 0.0f);
    // }

    indirect_map.storeOnce(uint3(pixx, pixy, 0), indirectx);
    indirect_map.storeOnce(uint3(pixx, pixy, 1), indirecty);
    indirect_map.storeOnce(uint3(pixx, pixy, 2), indirectz);
}


[Differentiable]
[AutoPyBindCUDA]
[CudaKernel]
void render_with_SH(int N, DiffTensorView ray_directions, DiffTensorView view_directions,
                                       DiffTensorView normals, DiffTensorView albedos, DiffTensorView roughnesses, DiffTensorView metallics,
                                       DiffTensorView envmap, TensorView<float> SHs, TensorView<int> hit_indices, 
                                       DiffTensorView Le)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    if (dispatchIdx.x >= N || dispatchIdx.y >= 256) return;
    uint ray_index = dispatchIdx.y;
    uint gs_index = dispatchIdx.x;

    float3 view_dir;
    view_dir.x = no_diff view_directions[gs_index, 0];
    view_dir.y = no_diff view_directions[gs_index, 1];
    view_dir.z = no_diff view_directions[gs_index, 2];

    view_dir = normalize(view_dir);

    float3 normal;
    normal.x = normals[gs_index, 0];
    normal.y = normals[gs_index, 1];
    normal.z = normals[gs_index, 2];

    float3 albedo;
    albedo.x = albedos[gs_index, 0];
    albedo.y = albedos[gs_index, 1];
    albedo.z = albedos[gs_index, 2];

    float roughness = roughnesses[gs_index, 0];

    float metallic = metallics[gs_index, 0];

    float3 ray_direction;
    ray_direction.x = no_diff ray_directions[gs_index, ray_index, 0];
    ray_direction.y = no_diff ray_directions[gs_index, ray_index, 1];
    ray_direction.z = no_diff ray_directions[gs_index, ray_index, 2];

    ray_direction = normalize(ray_direction);

    int hit = hit_indices[gs_index, ray_index];

    float3 Li = hit == -1 ? eval_envmap(envmap, -view_dir) : eval_sh(SHs, hit, ray_direction);

    float3 rgb = shading_brdf(-view_dir, ray_direction, normal, albedo, roughness, metallic) * Li;

    // float3 rgb = float3(1.0f);
    // Le.store(uint2(gs_index, 0), Le[gs_index, 0] + rgb.x);
    // Le.store(uint2(gs_index, 1), Le[gs_index, 1] + rgb.y);
    // Le.store(uint2(gs_index, 2), Le[gs_index, 2] + rgb.z);
    Le.storeOnce(uint3(gs_index, ray_index, 0), rgb.x);
    Le.storeOnce(uint3(gs_index, ray_index, 1), rgb.y);
    Le.storeOnce(uint3(gs_index, ray_index, 2), rgb.z);
}

[Differentiable]
[AutoPyBindCUDA]
[CudaKernel]
void render_direct_with_SH(int N, DiffTensorView ray_directions, DiffTensorView view_directions,
                    DiffTensorView normals, DiffTensorView albedos, DiffTensorView roughnesses, DiffTensorView metallics,
                    DiffTensorView envmap, TensorView<float> SHs, TensorView<int> hit_indices,
                    DiffTensorView Le)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    if (dispatchIdx.x >= N || dispatchIdx.y >= 256) return;
    uint ray_index = dispatchIdx.y;
    uint gs_index = dispatchIdx.x;

    float3 view_dir;
    view_dir.x = no_diff view_directions[gs_index, 0];
    view_dir.y = no_diff view_directions[gs_index, 1];
    view_dir.z = no_diff view_directions[gs_index, 2];

    view_dir = normalize(view_dir);

    float3 normal;
    normal.x = normals[gs_index, 0];
    normal.y = normals[gs_index, 1];
    normal.z = normals[gs_index, 2];

    float3 albedo;
    albedo.x = albedos[gs_index, 0];
    albedo.y = albedos[gs_index, 1];
    albedo.z = albedos[gs_index, 2];

    float roughness = roughnesses[gs_index, 0];

    float metallic = metallics[gs_index, 0];

    float3 ray_direction;
    ray_direction.x = no_diff ray_directions[gs_index, ray_index, 0];
    ray_direction.y = no_diff ray_directions[gs_index, ray_index, 1];
    ray_direction.z = no_diff ray_directions[gs_index, ray_index, 2];

    ray_direction = normalize(ray_direction);

    int hit = hit_indices[gs_index, ray_index];

    float3 Li = hit == -1 ? eval_envmap(envmap, ray_direction) : float3(0.0f);

    float3 rgb = shading_brdf(-view_dir, ray_direction, normal, albedo, roughness, metallic);

    // float3 rgb = float3(1.0f);
    // Le.store(uint2(gs_index, 0), Le[gs_index, 0] + rgb.x);
    // Le.store(uint2(gs_index, 1), Le[gs_index, 1] + rgb.y);
    // Le.store(uint2(gs_index, 2), Le[gs_index, 2] + rgb.z);
    Le.storeOnce(uint3(gs_index, ray_index, 0), rgb.x);
    Le.storeOnce(uint3(gs_index, ray_index, 1), rgb.y);
    Le.storeOnce(uint3(gs_index, ray_index, 2), rgb.z);
}

[Differentiable]
[AutoPyBindCUDA]
[CudaKernel]
void render_indirect_with_SH(int N, DiffTensorView ray_directions, DiffTensorView view_directions,
                    DiffTensorView normals, DiffTensorView albedos, DiffTensorView roughnesses, DiffTensorView metallics,
                    DiffTensorView envmap, TensorView<float> SHs, TensorView<int> hit_indices,
                    DiffTensorView Le)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    if (dispatchIdx.x >= N || dispatchIdx.y >= 256) return;
    uint ray_index = dispatchIdx.y;
    uint gs_index = dispatchIdx.x;

    float3 view_dir;
    view_dir.x = no_diff view_directions[gs_index, 0];
    view_dir.y = no_diff view_directions[gs_index, 1];
    view_dir.z = no_diff view_directions[gs_index, 2];

    view_dir = normalize(view_dir);

    float3 normal;
    normal.x = normals[gs_index, 0];
    normal.y = normals[gs_index, 1];
    normal.z = normals[gs_index, 2];

    float3 albedo;
    albedo.x = albedos[gs_index, 0];
    albedo.y = albedos[gs_index, 1];
    albedo.z = albedos[gs_index, 2];

    float roughness = roughnesses[gs_index, 0];

    float metallic = metallics[gs_index, 0];

    float3 ray_direction;
    ray_direction.x = no_diff ray_directions[gs_index, ray_index, 0];
    ray_direction.y = no_diff ray_directions[gs_index, ray_index, 1];
    ray_direction.z = no_diff ray_directions[gs_index, ray_index, 2];

    ray_direction = normalize(ray_direction);

    int hit = hit_indices[gs_index, ray_index];

    // float3 Li = hit == -1 ? float3(0.0f) : eval_sh(SHs, hit, ray_direction);

    // // float3 rgb = shading_brdf(-view_dir, ray_direction, normal, albedo, roughness, metallic) * Li;
    // float3 rgb = eval_sh(SHs, gs_index, -view_dir);

    float3 Li = hit == -1 ? eval_envmap(envmap, ray_direction) : float3(0.0f);

    float3 rgb = Li;

    // float3 rgb = float3(1.0f);
    // Le.store(uint2(gs_index, 0), Le[gs_index, 0] + rgb.x);
    // Le.store(uint2(gs_index, 1), Le[gs_index, 1] + rgb.y);
    // Le.store(uint2(gs_index, 2), Le[gs_index, 2] + rgb.z);
    Le.storeOnce(uint3(gs_index, ray_index, 0), rgb.x);
    Le.storeOnce(uint3(gs_index, ray_index, 1), rgb.y);
    Le.storeOnce(uint3(gs_index, ray_index, 2), rgb.z);
}


[AutoPyBindCUDA]
[CudaKernel]
void render_radiance_with_SH(int N, TensorView<float> ray_directions,
                               TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
                               TensorView<float> centers, TensorView<float> scales, TensorView<float> rotates, TensorView<float> colors, TensorView<float> opacity,
                               TensorView<float> SHs,
                               TensorView<float> Le)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    if (dispatchIdx.x >= N || dispatchIdx.y >= 256) return;
    uint ray_index = dispatchIdx.y;
    uint gs_index = dispatchIdx.x;


    float3 ray_direction;
    ray_direction.x = ray_directions[gs_index, ray_index, 0];
    ray_direction.y = ray_directions[gs_index, ray_index, 1];
    ray_direction.z = ray_directions[gs_index, ray_index, 2];

    ray_direction = normalize(ray_direction);
    int index_hit = -1;

    float3 xyz;
    xyz.x = centers[gs_index, 0];
    xyz.y = centers[gs_index, 1];
    xyz.z = centers[gs_index, 2];

    float test_T = 1.0f;

    float t_min = 0.25f;
    // float t_max = 1e9;
    float t_max = 1.5f;
    float t_hit = 0.f;

    float3 debug_res;
    bool done = false;
    float3 rayo = xyz;
    float3 rayd = ray_direction;

    float3 sh = float3(0.0f);

    
    while ((test_T > 0.001f) && (!done))
    {
        // bool hit = gs_bvh_hit(g_lbvh_info, g_lbvh_aabb, centers, scales, rotates, colors, opacity,
        //                       rayo, rayd, t_min, t_max, t_hit, debug_res, index_hit);
        bool hit = true;
        hit = index_hit == gs_index ? false : hit;
        if (hit)
        {
            if (test_T < 0.001f)
            {
                test_T = 0.00099f;
                done = true;
                break;
            }
            rayo = rayo + rayd * t_hit;
            // t_min = t_hit + 0.005f;
            // t_max = t_min + 1.0f;

            sh += eval_sh(SHs, index_hit, ray_direction) * opacity[index_hit] * test_T;
            test_T = test_T * (1.0f - opacity[index_hit]);
            // done = true;
        }
        else
        {
            done = true;
        }
    }
    

    // float3 Li = hit == -1 ? float3(0.0f) : eval_sh(SHs, hit, ray_direction);

    float3 rgb = sh;

    // float3 rgb = float3(1.0f);
    // Le.store(uint2(gs_index, 0), Le[gs_index, 0] + rgb.x);
    // Le.store(uint2(gs_index, 1), Le[gs_index, 1] + rgb.y);
    // Le.store(uint2(gs_index, 2), Le[gs_index, 2] + rgb.z);
    // Le.storeOnce(uint3(gs_index, ray_index, 0), rgb.x);
    // Le.storeOnce(uint3(gs_index, ray_index, 1), rgb.y);
    // Le.storeOnce(uint3(gs_index, ray_index, 2), rgb.z);

    Le[gs_index, ray_index, 0] = rgb.x;
    Le[gs_index, ray_index, 1] = rgb.y;
    Le[gs_index, ray_index, 2] = rgb.z;
}


// colors actually means normals
[AutoPyBindCUDA]
[CudaKernel]
void render_radiance_with_sampling_SH(int N,int S, TensorView<float> ray_origins, TensorView<float> ray_directions,
                             TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
                             TensorView<float> centers, TensorView<float> scales, TensorView<float> rotates, TensorView<float> colors, TensorView<float> opacity, TensorView<float> cov3D_inverse,
                             TensorView<float> SHs,
                             TensorView<float> Le, TensorView<float> visibility, TensorView<int> hit_indices, TensorView<float> uvs)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    if (dispatchIdx.x >= N || dispatchIdx.y >= S) return;
    uint ray_index = dispatchIdx.y;
    uint gs_index = dispatchIdx.x;

    float3 ray_direction;
    ray_direction.x = ray_directions[gs_index, ray_index, 0];
    ray_direction.y = ray_directions[gs_index, ray_index, 1];
    ray_direction.z = ray_directions[gs_index, ray_index, 2];

    ray_direction = normalize(ray_direction);
    int index_hit = -1;
    float2 uv = float2(0.0f);

    float3 xyz;
    xyz.x = ray_origins[gs_index, 0];
    xyz.y = ray_origins[gs_index, 1];
    xyz.z = ray_origins[gs_index, 2];

    float test_T = 1.0f;

    float t_min = 0.042f;
    float t_max = 0.2f;
    // float t_max = 1.5f;
    float t_hit = 0.f;

    float3 debug_res;
    bool done = false;
    float3 rayo = xyz;
    float3 rayd = ray_direction;

    float3 sh = float3(0.0f);
    bool visibable = true;

    float3 normal;
    normal.x = colors[gs_index, 0];
    normal.y = colors[gs_index, 1];
    normal.z = colors[gs_index, 2];

    int first_hit = -1;
    float2 first_uv = float2(0.0f);

    while ((test_T > 0.001f) && (!done))
    {
        bool hit = gs_bvh_hit(g_lbvh_info, g_lbvh_aabb, centers, scales, rotates, colors, opacity, cov3D_inverse,
                              rayo, rayd, t_min, t_max, t_hit, debug_res, index_hit, uv);
        hit = index_hit == gs_index ? false : hit;
        
        if (hit)
        {
            if (test_T < 0.001f)
            {
                test_T = 0.00099f;
                // test_T = 0.0f;
                done = true;
                break;
            }

            if (first_hit == -1)
            {
                first_hit = index_hit;
                first_uv = uv;
                t_min = 0.01f;
            }
            float3 hit_gs_pos;
            hit_gs_pos.x = centers[index_hit, 0];
            hit_gs_pos.y = centers[index_hit, 1];
            hit_gs_pos.z = centers[index_hit, 2];

            float3 sh_dir = hit_gs_pos - rayo;
            sh_dir = normalize(sh_dir);
            rayo = rayo + rayd * t_hit;
            // t_min = t_hit + 0.005f;
            // t_max = t_min + 1.0f;

            sh += eval_sh(SHs, index_hit, sh_dir) * (1-debug_res.x) * test_T;
            test_T = test_T * debug_res.x;
            if (test_T < 0.2f)
            {
                visibable = false;
                // test_T = 0.0f;
            }
            // done = true;
        }
        else
        {
            done = true;
        }
    }

    sh = clamp(sh, 0.0f, 10.0f);
    // float3 rgb = pow(sh, 1 / 2.2);
    float3 rgb = sh;

    visibility[gs_index, ray_index, 0] = visibable ? test_T : 0.0f;

    Le[gs_index, ray_index, 0] = rgb.x;
    Le[gs_index, ray_index, 1] = rgb.y;
    Le[gs_index, ray_index, 2] = rgb.z;

    hit_indices[gs_index, ray_index, 0] = first_hit;

    uvs[gs_index, ray_index, 0] = first_uv.x;
    uvs[gs_index, ray_index, 1] = first_uv.y;
}

// colors actually means normals
[AutoPyBindCUDA]
[CudaKernel]
void render_SH(int H, int W, TensorView<float> ray_origins, TensorView<float> ray_directions,
                TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
                TensorView<float> centers, TensorView<float> scales, TensorView<float> rotates, TensorView<float> colors, TensorView<float> opacity, TensorView<float> cov3D_inverse,
                TensorView<float> SHs,
                TensorView<float> Le)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    if (dispatchIdx.x >= H || dispatchIdx.y >= W) return;
    uint pixx = dispatchIdx.x;
    uint pixy = dispatchIdx.y;

    float3 ray_origin = float3(0.0, 0.0, 4.0f);
    // ray_origin.x = ray_origins[0];
    // ray_origin.y = ray_origins[1];
    // ray_origin.z = ray_origins[2];

    float3 ray_direction;
    ray_direction.x = ray_directions[pixx, pixy, 0];
    ray_direction.y = ray_directions[pixx, pixy, 1];
    ray_direction.z = ray_directions[pixx, pixy, 2];

    ray_direction = normalize(ray_direction);
    int index_hit = -1;


    float test_T = 1.0f;

    float t_min = 0.01f;
    float t_max = 1e9;
    // float t_max = 1.5f;
    float t_hit = 0.f;

    float3 debug_res;
    bool done = false;
    float3 rayo = ray_origin;
    float3 rayd = ray_direction;

    float3 sh = float3(0.0f);
    bool visibable = true;

    float2 uv;
    int first_hit = -1;

    while ((test_T > 0.001f) && (!done))
    {
        bool hit = gs_bvh_hit(g_lbvh_info, g_lbvh_aabb, centers, scales, rotates, colors, opacity, cov3D_inverse,
                              rayo, rayd, t_min, t_max, t_hit, debug_res, index_hit, uv);

        if (hit)
        {
            if (test_T < 0.001f)
            {
                test_T = 0.00099f;
                // test_T = 0.0f;
                done = true;
                break;
            }

            if (first_hit == -1)
            {
                first_hit = index_hit;
            }

            // float opc = clamp(debug_res.x, 0.001, 0.999);
            float opc = debug_res.x;


            rayo = rayo + rayd * t_hit;
            // t_min = t_hit + 0.005f;
            // t_max = t_min + 1.0f;

            sh += eval_sh(SHs, index_hit, ray_direction) * (1-opc) * test_T;
            // sh += eval_sh(SHs, index_hit, ray_direction);
            test_T = test_T * debug_res.x;
            if (test_T < 0.9f)
            {
                visibable = false;
                // test_T = 0.0f;
            }

            // done = true;
        }
        else
        {
            done = true;
        }
    }

    sh = clamp(sh, 0.0f, 2.0f);
    // float3 rgb = pow(sh, 1 / 2.2);
    float3 rgb = sh;
    if (first_hit == -1)
    {
        rgb += float3(0.3f);
    }
    // rgb += 0.3f;

    Le[pixx, pixy, 0] = rgb.x;
    Le[pixx, pixy, 1] = rgb.y;
    Le[pixx, pixy, 2] = rgb.z;

    // Le[pixx, pixy, 0] = ray_direction.x;
    // Le[pixx, pixy, 1] = ray_direction.y;
    // Le[pixx, pixy, 2] = ray_direction.z;

    // Le[pixx, pixy, 0] = ray_origin.x;
    // Le[pixx, pixy, 1] = ray_origin.y;
    // Le[pixx, pixy, 2] = ray_origin.z;
}

[Differentiable]
[AutoPyBindCUDA]
[CudaKernel]
void render_irradiance_with_SH(int N, int S, DiffTensorView ray_directions, DiffTensorView view_directions,
                               DiffTensorView normals, DiffTensorView albedos, DiffTensorView roughnesses, DiffTensorView metallics,
                               DiffTensorView envmap, TensorView<float> SHs, TensorView<int> hit_indices,
                               DiffTensorView Le)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    if (dispatchIdx.x >= N || dispatchIdx.y >= 256) return;
    uint ray_index = dispatchIdx.y;
    uint gs_index = dispatchIdx.x;

    float3 view_dir;
    view_dir.x = no_diff view_directions[gs_index, 0];
    view_dir.y = no_diff view_directions[gs_index, 1];
    view_dir.z = no_diff view_directions[gs_index, 2];

    view_dir = normalize(view_dir);

    float3 normal;
    normal.x = normals[gs_index, 0];
    normal.y = normals[gs_index, 1];
    normal.z = normals[gs_index, 2];

    float3 albedo;
    albedo.x = albedos[gs_index, 0];
    albedo.y = albedos[gs_index, 1];
    albedo.z = albedos[gs_index, 2];

    float roughness = roughnesses[gs_index, 0];

    float metallic = metallics[gs_index, 0];

    float3 ray_direction;
    ray_direction.x = no_diff ray_directions[gs_index, ray_index, 0];
    ray_direction.y = no_diff ray_directions[gs_index, ray_index, 1];
    ray_direction.z = no_diff ray_directions[gs_index, ray_index, 2];

    ray_direction = normalize(ray_direction);

    int hit = hit_indices[gs_index, ray_index];

    float3 Li = hit == -1 ? float3(0.0f) : eval_sh(SHs, hit, ray_direction);

    float3 rgb = Li;

    // float3 rgb = float3(1.0f);
    // Le.store(uint2(gs_index, 0), Le[gs_index, 0] + rgb.x);
    // Le.store(uint2(gs_index, 1), Le[gs_index, 1] + rgb.y);
    // Le.store(uint2(gs_index, 2), Le[gs_index, 2] + rgb.z);
    Le.storeOnce(uint3(gs_index, ray_index, 0), rgb.x);
    Le.storeOnce(uint3(gs_index, ray_index, 1), rgb.y);
    Le.storeOnce(uint3(gs_index, ray_index, 2), rgb.z);
}

[Differentiable]
[AutoPyBindCUDA]
[CudaKernel]
void render_with_radiance(int N, DiffTensorView ray_directions, DiffTensorView view_directions,
                          DiffTensorView normals, DiffTensorView albedos, DiffTensorView roughnesses, DiffTensorView metallics,
                          DiffTensorView envmap, TensorView<float> radiance,
                          DiffTensorView Le)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    if (dispatchIdx.x >= N || dispatchIdx.y >= 256) return;
    uint ray_index = dispatchIdx.y;
    uint gs_index = dispatchIdx.x;

    float3 view_dir;
    view_dir.x = no_diff view_directions[gs_index, 0];
    view_dir.y = no_diff view_directions[gs_index, 1];
    view_dir.z = no_diff view_directions[gs_index, 2];

    view_dir = normalize(view_dir);

    float3 normal;
    normal.x = normals[gs_index, 0];
    normal.y = normals[gs_index, 1];
    normal.z = normals[gs_index, 2];
    normal = normalize(normal);

    float3 albedo;
    albedo.x = albedos[gs_index, 0];
    albedo.y = albedos[gs_index, 1];
    albedo.z = albedos[gs_index, 2];

    float roughness = roughnesses[gs_index, 0];

    float metallic = metallics[gs_index, 0];

    float3 ray_direction;
    ray_direction.x = no_diff ray_directions[gs_index, ray_index, 0];
    ray_direction.y = no_diff ray_directions[gs_index, ray_index, 1];
    ray_direction.z = no_diff ray_directions[gs_index, ray_index, 2];

    ray_direction = normalize(ray_direction);

    // int hit = hit_indices[gs_index, ray_index];
    float3 Li;
    Li.x = no_diff radiance[gs_index, ray_index, 0];
    Li.y = no_diff radiance[gs_index, ray_index, 1];
    Li.z = no_diff radiance[gs_index, ray_index, 2];

    Li = Li * 4 * M_PI * dot(normal, ray_direction);
    // if (any(Li == 0.0f))
    if (true)
    {
        Li = eval_envmap(envmap, ray_direction) * 2 * M_PI * dot(normal, ray_direction);
        // Li = float3(0.0f);
    }

    float3 rgb = shading_brdf(-view_dir, ray_direction, normal, albedo, roughness, metallic) * Li;
    // float3 rgb = Li;
    // float3 rgb = ray_direction;

    // float3 rgb = float3(1.0f);
    // Le.store(uint2(gs_index, 0), Le[gs_index, 0] + rgb.x);
    // Le.store(uint2(gs_index, 1), Le[gs_index, 1] + rgb.y);
    // Le.store(uint2(gs_index, 2), Le[gs_index, 2] + rgb.z);
    Le.storeOnce(uint3(gs_index, ray_index, 0), rgb.x);
    Le.storeOnce(uint3(gs_index, ray_index, 1), rgb.y);
    Le.storeOnce(uint3(gs_index, ray_index, 2), rgb.z);
}

// computing indirect actually
[Differentiable]
[AutoPyBindCUDA]
[CudaKernel]
void radiance_loss(int N, DiffTensorView ray_directions, DiffTensorView view_directions, DiffTensorView envmap,
                       DiffTensorView normals, DiffTensorView albedos, DiffTensorView roughnesses, DiffTensorView metallics,
                       TensorView<int> hit_indices,
                       DiffTensorView irradiance)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    if (dispatchIdx.x >= N || dispatchIdx.y >= 256 || dispatchIdx.z >= 256) return;
    uint pri_ray_index = dispatchIdx.y;
    uint sec_ray_index = dispatchIdx.z;
    uint gs_index = dispatchIdx.x;

    float3 pri_ray_dir;
    pri_ray_dir.x = no_diff ray_directions[gs_index, pri_ray_index, 0];
    pri_ray_dir.y = no_diff ray_directions[gs_index, pri_ray_index, 1];
    pri_ray_dir.z = no_diff ray_directions[gs_index, pri_ray_index, 2];

    int hit_index = hit_indices[gs_index, pri_ray_index];

    if (hit_index == -1)
    {
        // float3 irr = eval_envmap(envmap, pri_ray_dir) / 256;
        float3 irr = float3(0.0f);
        irradiance.storeOnce(uint3(gs_index, pri_ray_index, 0), irr.x);
        irradiance.storeOnce(uint3(gs_index, pri_ray_index, 1), irr.y);
        irradiance.storeOnce(uint3(gs_index, pri_ray_index, 2), irr.z);
    }
    else
    {
        float3 sec_ray_dir;
        sec_ray_dir.x = no_diff ray_directions[hit_index, sec_ray_index, 0];
        sec_ray_dir.y = no_diff ray_directions[hit_index, sec_ray_index, 1];
        sec_ray_dir.z = no_diff ray_directions[hit_index, sec_ray_index, 2];

        int sec_hit_index = hit_indices[hit_index, sec_ray_index];

        float3 normal;
        normal.x = normals[hit_index, 0];
        normal.y = normals[hit_index, 1];
        normal.z = normals[hit_index, 2];

        float3 albedo;
        albedo.x = albedos[hit_index, 0];
        albedo.y = albedos[hit_index, 1];
        albedo.z = albedos[hit_index, 2];

        float roughness = roughnesses[hit_index, 0];

        float metallic = metallics[hit_index, 0];

        float3 irr_sec = sec_hit_index == -1 ? eval_envmap(envmap, sec_ray_dir) : float3(0.0f);

        float3 irr = shading_brdf(-pri_ray_dir, sec_ray_dir, normal, albedo, roughness, metallic);

        float3 irr_res = irr * irr_sec / 256;

        irradiance[gs_index, pri_ray_index, 0] += irr_res.x;
        irradiance[gs_index, pri_ray_index, 1] += irr_res.y;
        irradiance[gs_index, pri_ray_index, 2] += irr_res.z;
    }
}