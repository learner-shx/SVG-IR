#include "mathConstant.slangh"
import envmap;
import brdf;
[Differentiable]
float3 sqr(float3 x)
{
    return x * x;
}

[Differentiable]
float sqr(float x)
{
    return x * x;
}

[Differentiable]
float smithLambda(float3 v, float roughness)
{
    float2 alpha = float2(roughness, roughness);
    float cosThetaSq = v.z * v.z;
    float sinThetaSq = max(1.0f - cosThetaSq, 0.0f);
    float tanThetaSq = sinThetaSq / cosThetaSq;
    if (isinf(tanThetaSq)) return 0.0f;

    float2 cosSinPhiSq = sinThetaSq == 0.0f ? float2(1.0f, 0.0f) : float2(v.x * v.x, v.y * v.y) / sinThetaSq;
    float alphaSq = dot(cosSinPhiSq, alpha * alpha);
    float ret = 1.0f + alphaSq * tanThetaSq > 0.0f ? sqrt(1.0f + alphaSq * tanThetaSq) * 0.5f : 0.0f;
    return ret;
}

[Differentiable]
float G1(float3 w, float roughness)
{
    return 1.0f / (1.0f + smithLambda(w, roughness));
}

[Differentiable]
float G2(float3 wi, float3 wo, float roughness)
{
    return 1.0f / (1.0f + smithLambda(wo, roughness) + smithLambda(wi, roughness));
}

[Differentiable]
float evalNDF(float3 h, float roughness)
{
    float2 alpha = float2(roughness, roughness);
    float cosThetaSq = h.z * h.z;
    float sinThetaSq = max(1.0f - cosThetaSq, 0.0f);
    float tanThetaSq = sinThetaSq / cosThetaSq;
    if (isinf(tanThetaSq)) return 0.0f;

    float cosThetaQu = cosThetaSq * cosThetaSq;
    if (cosThetaQu < 1e-9f) return 0.0f;

    float2 cosSinPhiSq = sinThetaSq == 0.0f ? float2(1.0f, 0.0f) : float2(h.x * h.x, h.y * h.y) / sinThetaSq;
    float2 alphaSq = cosSinPhiSq / (alpha * alpha);
    float e = tanThetaSq * (alphaSq.x + alphaSq.y);
    return 1.0f / (M_PI * alpha.x * alpha.y * cosThetaQu * sqr(1.0f + e));
}

[Differentiable]
float3 evalFresnel(float woDotH, float3 f0)
{
    return f0 + (1.0f - f0) * pow(1.0f - woDotH, 5.0f);
}


// wiLocal: light direction in local space
// woLocal: view direction in local space

[Differentiable]
float3 eval(float3 wiLocal, float3 woLocal, float3 albedo, float roughness, float metallics)
{
    float3 wi = normalize(wiLocal);
    float3 wo = normalize(woLocal);
    float3 h = normalize(wi + wo);
    float G = G2(wi, wo, roughness);
    float D = evalNDF(h, roughness);

    float3 f0 = (1-metallics) * 0.04f + metallics * albedo;

    float woDotH = abs(dot(wo, h));
    float3 F = evalFresnel(woDotH, f0);
    if (wo.z < 1e-4f)
        return float3(1e-4f);

    // return float3(woDotH);
    // return float3(1.0f);
    return D * G * F / (4.0f * wo.z + 1e-5f); // multiplied cos(l, n)
}

void branchlessONB(const float3 n, out float3 b1, out float3 b2)
{
    // can't use just `sign` because we need 0 to go into -1/+1, but not 0
    float sign = (n.z >= 0 ? 1 : -1);
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;
    b1 = float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    b2 = float3(b, sign + n.y * n.y * a, -n.y);
}
[Differentiable]
float3 d_sg(float r, float3 cos)
{
    float r2 = clamp(r * r, 1e-7f, 10.0f);
    float amp = 1 / (r2 * M_PI);
    float sharp = 2 / r2;
    return amp * exp(sharp * (cos-1));
}

[Differentiable]
float3 v_schlick_ggx(float r, float3 cos)
{
    float one_plus_r = 1.0f + r;
    float r2 = one_plus_r * one_plus_r / 8.0f;
    return 0.5 / clamp((cos * (1 - r2) + r2), 1e-7f, 10.0f);

}

[Differentiable]
float3 eval_specular(float3 wiLocal, float3 woLocal, float3 albedo, float roughness, float metallics)
{
    float3 wi = normalize(wiLocal);
    float3 wo = normalize(woLocal);
    float3 h = normalize(wi + wo);

    float3 h_d_n = saturate(h.z);
    float3 h_d_o = saturate(dot(h, wo));
    float3 n_d_i = saturate(wi.z);
    float3 n_d_o = saturate(wo.z);

    float3 D = d_sg(roughness, h_d_n);

    float3 F0 = 0.04 * (1 - metallics) + albedo * metallics;
    float3 F = F0 + (1 - F0) * pow(1 - n_d_o, 5);

    float3 V = v_schlick_ggx(roughness, n_d_i) * v_schlick_ggx(roughness, n_d_o);

    return D * F * V;
}

// float3 perp_stark(float3 u)
// {
//     // TODO: Validate this and look at numerical precision etc. Are there better ways to do it?
//     float3 a = abs(u);
//     uint uyx = (a.x - a.y) < 0 ? 1 : 0;
//     uint uzx = (a.x - a.z) < 0 ? 1 : 0;
//     uint uzy = (a.y - a.z) < 0 ? 1 : 0;
//     uint xm = uyx & uzx;
//     uint ym = (1 ^ xm) & uzy;
//     uint zm = 1 ^ (xm | ym); // 1 ^ (xm & ym)
//     float3 v = normalize(cross(u, float3(xm, ym, zm)));
//     return v;
// }

[Differentiable]
bool orthonormalizeSafe(const float3 N, const float4 tangentW, out float3 T, out float3 B)
{
    // Check that tangent space exists and can be safely orthonormalized.
    // Otherwise invent a tanget frame based on the normal.
    // We check that:
    //  - Tangent exists, this is indicated by a sign (w) that is +-1.0.
    //  - It has nonzero length. Zeros can occur due to interpolation or bad assets.
    //  - It is not parallel to the normal. This can occur due to normal mapping or bad assets.
    //  - It does not have NaNs. These will propagate and trigger the fallback.

    float NdotT = dot(tangentW.xyz, N);
    bool nonParallel = abs(NdotT) < 0.9999f;
    bool nonZero = dot(tangentW.xyz, tangentW.xyz) > 0.f;
    bool valid = abs(tangentW.w) == 1.0f && nonZero && nonParallel;

    if (valid)
    {
        T = normalize(tangentW.xyz - N * NdotT);
        B = cross(N, T) * tangentW.w;
    }
    else
    {
        T = no_diff perp_stark(N);
        B = cross(N, T);
    }

    return valid;
}

[Differentiable]
float3 shading_direct(float3 view_dir, float3 normal, float3 albedo, float roughness, float metallics, DiffTensorView envmap)
{

    float3 normalized_normal = normalize(normal);
    float3 tangent, bitangent;
    branchlessONB(normalized_normal, tangent, bitangent);
    // float3x3 TBN = float3x3(tangent, bitangent, normal);
    float3 T, B;
    bool valid = orthonormalizeSafe(normalized_normal, float4(tangent, 1.0f), T, B);
    float3x3 TBN = float3x3(T, B, normalized_normal);
    if (!valid)
        return float3(0.0f);

    if (length(normal) < 0.168f)
        return float3(0.0f);

    float3 wo = normalize(view_dir);
    // float3 wi = 2 * dot(wo, normal) * normal - wo;
    float3 wi = normalize(2 * dot(wo, normalized_normal) * normalized_normal - wo);

    float3 wiLocal = normalize(mul(TBN, wi));
    float3 woLocal = normalize(mul(TBN, wo));

    float3 light_intensity = eval_envmap(envmap, wi);

    // float3 light_intensity = float3(1.0f);
    float cosTheta = wiLocal.z;
    if (cosTheta < 1e-4f)
        return float3(0.0f);
    float3 specular = eval(wiLocal, woLocal, albedo, roughness, metallics) * light_intensity;

    float3 diffuse = eval_envmap(envmap, normal) * albedo;


    diffuse = diffuse * cosTheta;
        
    // float3 diffuse = eval_envmap(envmap, normal) * albedo * abs(wiLocal.z) / M_PI;
    
    float3 irradiance = diffuse + specular;
    // float3 irradiance = albedo * evalBRDF(wi, wo, normal, roughness, 1.0f, 1.0f) * light_intensity;

    // float3 h = normalize(wiLocal + woLocal);
    // return diffuse;

    return irradiance;
}

[Differentiable]
float3 shading_brdf(float3 view_dir, float3 light_dir, float3 normal, float3 albedo, float roughness, float metallics)
{

    float3 normalized_normal = normalize(normal);
    float3 tangent, bitangent;
    branchlessONB(normalized_normal, tangent, bitangent);
    // float3x3 TBN = float3x3(tangent, bitangent, normal);
    float3 T, B;
    bool valid = orthonormalizeSafe(normalized_normal, float4(tangent, 1.0f), T, B);
    float3x3 TBN = float3x3(T, B, normalized_normal);
    if (!valid)
        return float3(0.0f);

    if (length(normal) < 0.168f)
        return float3(0.0f);

    float3 wo = normalize(view_dir);
    // float3 wi = 2 * dot(wo, normal) * normal - wo;
    float3 wi = normalize(light_dir);

    float3 wiLocal = normalize(mul(TBN, wi));
    float3 woLocal = normalize(mul(TBN, wo));


    // float3 light_intensity = float3(1.0f);
    // float cosTheta = wiLocal.z;
    // if (cosTheta < 1e-4f)
    //     return float3(0.0f);

    // float3 specular = eval(wiLocal, woLocal, albedo, roughness, metallics);
    float3 specular = eval_specular(wiLocal, woLocal, albedo, roughness, metallics);

    float3 diffuse = (1 - metallics) * albedo * M_1_PI;


    // float3 diffuse = eval_envmap(envmap, normal) * albedo * abs(wiLocal.z) / M_PI;

    // float3 brdf = diffuse + specular;
    // float3 brdf = diffuse + specular;
    float3 brdf = specular + diffuse;
    // float3 irradiance = albedo * evalBRDF(wi, wo, normal, roughness, 1.0f, 1.0f) * light_intensity;

    // float3 h = normalize(wiLocal + woLocal);
    // return diffuse;

    return brdf;
}

[Differentiable]
float3 shading_brdf_simple(float3 view_dir, float3 light_dir, float3 normal, float3 albedo, float roughness)
{

    float3 normalized_normal = normalize(normal);

    float3 wo = normalize(view_dir);
    // float3 wi = 2 * dot(wo, normal) * normal - wo;
    float3 wi = normalize(light_dir);


    float3 L = wi;
    float3 V = wo;
    float3 H = normalize(V + L);
    float3 N = normalized_normal;

    float NoV = dot(N, V);
    if (NoV < 0.0f)
        NoV = -NoV;

    float NoL = clamp(dot(N, L), 1e-6, 1);
    NoV = clamp(dot(N, V), 1e-6, 1);
    float NoH = clamp(dot(N, H), 1e-6, 1);
    float VoH = clamp(dot(V, H), 1e-6, 1);

    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float k = (alpha + 2.0f * roughness + 1.0f) / 8.0f;

    float3 FMi = (-5.55473f * VoH - 6.98316f) * VoH;
    float3 frac0 = 0.04f + (1 - 0.04f) * pow(2.0, FMi);

    float3 frac = frac0 * alpha2;

    float3 nom0 = NoH * NoH * (alpha2 - 1.0f) + 1.0f;
    float3 nom1 = NoV * (1.0f - k) + k;
    float3 nom2 = NoL * (1.0f - k) + k;
    float3 nom = clamp(M_4PI * nom0 * nom0 * nom1 * nom2, 1e-6, M_4PI);

    float3 specular = frac / nom;

    float3 diffuse = albedo * M_1_PI;

    float3 brdf = specular + diffuse;


    return brdf;
}

[Differentiable]
float3 shading_brdf_local(float3 view_dir_local, float3 light_dir_local, float3 albedo, float roughness, float metallics)
{

    float3 woLocal = normalize(-view_dir_local);
    // float3 wi = 2 * dot(wo, normal) * normal - wo;
    float3 wiLocal = normalize(light_dir_local);


    // float3 light_intensity = float3(1.0f);
    float cosTheta = wiLocal.z;
    if (cosTheta < 1e-4f)
        return float3(0.0f);
    float3 specular = eval(wiLocal, woLocal, albedo, roughness, metallics);

    float3 diffuse = albedo * M_1_PI * cosTheta;

    // float3 diffuse = eval_envmap(envmap, normal) * albedo * abs(wiLocal.z) / M_PI;

    float3 brdf = diffuse + specular;
    // float3 irradiance = albedo * evalBRDF(wi, wo, normal, roughness, 1.0f, 1.0f) * light_intensity;

    // float3 h = normalize(wiLocal + woLocal);
    // return diffuse;

    return brdf;
}

[Differentiable]
float3 shading_wo(float3 view_dir, float3 normal, float3 albedo, float roughness, float metallics, DiffTensorView envmap)
{
    float3 normalized_normal = normalize(normal);
    float3 tangent, bitangent;
    branchlessONB(normalized_normal, tangent, bitangent);
    // float3x3 TBN = float3x3(tangent, bitangent, normal);
    float3 T, B;
    bool valid = orthonormalizeSafe(normalized_normal, float4(tangent, 1.0f), T, B);
    float3x3 TBN = float3x3(T, B, normalized_normal);
    if (!valid)
        return float3(0.0f);

    if (length(normal) < 0.168f)
        return float3(0.0f);

    float3 wo = normalize(view_dir);
    // float3 wi = 2 * dot(wo, normal) * normal - wo;
    float3 wi = normalize(2 * dot(wo, normalized_normal) * normalized_normal - wo);

    float3 wiLocal = normalize(mul(TBN, wi));
    float3 woLocal = normalize(mul(TBN, wo));

    float3 light_intensity = eval_envmap(envmap, wi);
    // float3 light_intensity = float3(1.0f);
    float3 specular = eval(woLocal, wiLocal, albedo, roughness, metallics) * light_intensity;
    float3 diffuse = eval_envmap(envmap, normal) * albedo;

    return wo;
}

[Differentiable]
float3 shading_wi(float3 view_dir, float3 normal, float3 albedo, float roughness, float metallics, DiffTensorView envmap)
{
    float3 normalized_normal = normalize(normal);
    float3 tangent, bitangent;
    branchlessONB(normalized_normal, tangent, bitangent);
    // float3x3 TBN = float3x3(tangent, bitangent, normal);
    float3 T, B;
    bool valid = orthonormalizeSafe(normalized_normal, float4(tangent, 1.0f), T, B);
    float3x3 TBN = float3x3(T, B, normalized_normal);
    if (!valid)
        return float3(0.0f);

    if (length(normal) < 0.168f)
        return float3(0.0f);

    float3 wo = normalize(view_dir);
    // float3 wi = 2 * dot(wo, normal) * normal - wo;
    float3 wi = normalize(2 * dot(wo, normalized_normal) * normalized_normal - wo);

    float3 wiLocal = normalize(mul(TBN, wi));
    float3 woLocal = normalize(mul(TBN, wo));

    float3 light_intensity = eval_envmap(envmap, wi);
    // float3 light_intensity = float3(1.0f);
    float3 specular = eval(woLocal, wiLocal, albedo, roughness, metallics) * light_intensity;
    float3 diffuse = eval_envmap(envmap, normal) * albedo;

    return wi;
}

[Differentiable]
float3 shading_specular(float3 view_dir, float3 normal, float3 albedo, float roughness, float metallics, DiffTensorView envmap)
{
    float3 normalized_normal = normalize(normal);
    float3 tangent, bitangent;
    branchlessONB(normalized_normal, tangent, bitangent);
    // float3x3 TBN = float3x3(tangent, bitangent, normal);
    float3 T, B;
    bool valid = orthonormalizeSafe(normalized_normal, float4(tangent, 1.0f), T, B);
    float3x3 TBN = float3x3(T, B, normalized_normal);
    if (!valid)
        return float3(0.0f);

    if (length(normal) < 0.168f)
        return float3(0.0f);

    float3 wo = normalize(view_dir);
    // float3 wi = 2 * dot(wo, normal) * normal - wo;
    float3 wi = normalize(2 * dot(wo, normalized_normal) * normalized_normal - wo);

    float3 wiLocal = normalize(mul(TBN, wi));
    float3 woLocal = normalize(mul(TBN, wo));

    float3 light_intensity = eval_envmap(envmap, wi);
    // float3 light_intensity = float3(1.0f);
    float3 specular = eval(woLocal, wiLocal, albedo, roughness, metallics) * light_intensity;
    float3 diffuse = eval_envmap(envmap, normal) * albedo;

    // float3 h = normalize(wiLocal + woLocal);
    return specular;
}

