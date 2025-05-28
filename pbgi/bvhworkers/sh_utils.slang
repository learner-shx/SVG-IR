// eval sh degree = 3
// sh: [M, 16, 3]
[Differentiable]
float3 eval_sh(TensorView<float> SHs, uint gs_index, float3 dir)
{
    dir = normalize(dir);
    float C0 = 0.28209479177387814;
    float C1 = 0.4886025119029199;

    float C2[5];
    C2[0] = 1.0925484305920792;
    C2[1] = -1.0925484305920792;
    C2[2] = 0.31539156525252005;
    C2[3] = -1.0925484305920792;
    C2[4] = 0.5462742152960396;

    float C3[7];
    C3[0] = -0.5900435899266435;
    C3[1] = 2.890611442640554;
    C3[2] = -0.4570457994644658;
    C3[3] = 0.3731763325901154;
    C3[4] = -0.4570457994644658;
    C3[5] = 1.445305721320277;
    C3[6] = -0.5900435899266435;

    float3 sh[16];
    sh[0] = float3(no_diff SHs[gs_index, 0, 0], no_diff SHs[gs_index, 0, 1], no_diff SHs[gs_index, 0, 2]);
    sh[1] = float3(no_diff SHs[gs_index, 1, 0], no_diff SHs[gs_index, 1, 1], no_diff SHs[gs_index, 1, 2]);
    sh[2] = float3(no_diff SHs[gs_index, 2, 0], no_diff SHs[gs_index, 2, 1], no_diff SHs[gs_index, 2, 2]);
    sh[3] = float3(no_diff SHs[gs_index, 3, 0], no_diff SHs[gs_index, 3, 1], no_diff SHs[gs_index, 3, 2]);
    sh[4] = float3(no_diff SHs[gs_index, 4, 0], no_diff SHs[gs_index, 4, 1], no_diff SHs[gs_index, 4, 2]);
    sh[5] = float3(no_diff SHs[gs_index, 5, 0], no_diff SHs[gs_index, 5, 1], no_diff SHs[gs_index, 5, 2]);
    sh[6] = float3(no_diff SHs[gs_index, 6, 0], no_diff SHs[gs_index, 6, 1], no_diff SHs[gs_index, 6, 2]);
    sh[7] = float3(no_diff SHs[gs_index, 7, 0], no_diff SHs[gs_index, 7, 1], no_diff SHs[gs_index, 7, 2]);
    sh[8] = float3(no_diff SHs[gs_index, 8, 0], no_diff SHs[gs_index, 8, 1], no_diff SHs[gs_index, 8, 2]);
    sh[9] = float3(no_diff SHs[gs_index, 9, 0], no_diff SHs[gs_index, 9, 1], no_diff SHs[gs_index, 9, 2]);
    sh[10] = float3(no_diff SHs[gs_index, 10, 0], no_diff SHs[gs_index, 10, 1], no_diff SHs[gs_index, 10, 2]);
    sh[11] = float3(no_diff SHs[gs_index, 11, 0], no_diff SHs[gs_index, 11, 1], no_diff SHs[gs_index, 11, 2]);
    sh[12] = float3(no_diff SHs[gs_index, 12, 0], no_diff SHs[gs_index, 12, 1], no_diff SHs[gs_index, 12, 2]);
    sh[13] = float3(no_diff SHs[gs_index, 13, 0], no_diff SHs[gs_index, 13, 1], no_diff SHs[gs_index, 13, 2]);
    sh[14] = float3(no_diff SHs[gs_index, 14, 0], no_diff SHs[gs_index, 14, 1], no_diff SHs[gs_index, 14, 2]);
    sh[15] = float3(no_diff SHs[gs_index, 15, 0], no_diff SHs[gs_index, 15, 1], no_diff SHs[gs_index, 15, 2]);


    float3 result;
    result = C0 * sh[0];
    result = result - C1 * dir.y * sh[1] + C1 * dir.z * sh[2] - C1 * dir.x * sh[3];
    result = result + 
             C2[0] * dir.x * dir.y * sh[4] + 
             C2[1] * dir.y * dir.z * sh[5] + 
             C2[2] * (2.0 * dir.z * dir.z - dir.x * dir.x - dir.y * dir.y) * sh[6] + 
             C2[3] * dir.x * dir.z * sh[7] +
             C2[4] * (dir.x * dir.x - dir.y * dir.y) * sh[8];
    result = result + 
             C3[0] * dir.y * (3.0 * dir.x * dir.x - dir.y * dir.y) * sh[9] + 
             C3[1] * dir.x * dir.y * dir.z * sh[10] + 
             C3[2] * dir.y * (4.0 * dir.z * dir.z - dir.x * dir.x - dir.y * dir.y) * sh[11] + 
             C3[3] * dir.z * (2.0 * dir.z * dir.z - 3.0 * dir.x * dir.x - 3.0 * dir.y * dir.y) * sh[12] + 
             C3[4] * dir.x * (4.0 * dir.z * dir.z - dir.x * dir.x - dir.y * dir.y) * sh[13] + 
             C3[5] * dir.z * (dir.x * dir.x - dir.y * dir.y) * sh[14] + 
             C3[6] * dir.x * (dir.x * dir.x - 3.0 * dir.y * dir.y) * sh[15];

    result = result + float3(0.5, 0.5, 0.5);

    // result = saturate(sh[0] + 0.5);

    return result;
}