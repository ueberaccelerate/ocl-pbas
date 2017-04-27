#define IS_BOUNDS(val, bound) ((val) >= (0) && (val) < (bound))

#define SK_SIZE 3

constant float min_R = 18.f;
constant int min_index = 2;
constant float R_inc_dec = 0.05f;
constant int R_scale = 5;
constant int R_lower = 10;
constant int T_lower = 2;
constant float T_dec = 0.05f;
constant int T_inc = 1;
constant int T_upper = 200;

constant int model_size = 20;

constant float alpha = 10.f;
constant float beta = 1.f;

constant float Gy[SK_SIZE * SK_SIZE] = {-1.f, -2.f, -1.f, 0.f, 0.f,
                                        0.f,  1.f,  2.f,  1.f};
constant float Gx[SK_SIZE * SK_SIZE] = {-1.f, 0.f,  1.f, -2.f, 0.f,
                                        2.f,  -1.f, 0.f, 1.f};

uint lfsr113_Bits(global uint *seed)
{

  // enum { A = 4294883355U };
  // uint x = (*seed).x, c = (*seed).y;  // Unpack the state
  // uint res = x^c;                     // Calculate the result
  // uint hi = mul_hi(x, A);              // Step the RNG
  // x = x*A + c;
  // c = hi + (x<c);
  //*seed = (uint2)(x, c);               // Pack the state back up
  // return res;                       // Return the next result

  *seed = (*seed * 1103515245U + 12345U) & 0x7fffffffU;
  return *seed;

  // unsigned int z1 = *seed, z2 = 1313245, z3 = *seed, z4 = 12345;
  // unsigned int b;
  // b = ((z1 << 6) ^ z1) >> 13;
  // z1 = ((z1 & 4294967294U) << 18) ^ b;
  // b = ((z2 << 2) ^ z2) >> 27;
  // z2 = ((z2 & 4294967288U) << 2) ^ b;
  // b = ((z3 << 13) ^ z3) >> 21;
  // z3 = ((z3 & 4294967280U) << 7) ^ b;
  // b = ((z4 << 3) ^ z4) >> 12;
  // z4 = ((z4 & 4294967168U) << 13) ^ b;
  //*seed = abs((z1 ^ z2 ^ z3 ^ z4));
  // return *seed;
}

kernel void fill_T_R(global float *oT, global float *oR, const uint width,
                     const uint height, const uint T_lower, const uint R_lower)
{
  int ii = get_global_id(0); // == get_global_id(0);
  int jj = get_global_id(1); // == get_global_id(1);
  int2 coord = (int2)(get_global_id(0), get_global_id(1));
  /* if (IS_BOUNDS(coord.x, width) && IS_BOUNDS(coord.y, height)) */
  {
    oT[jj * width + ii] = T_lower;
    oR[jj * width + ii] = R_lower;
  }
}
kernel void fill_model(global float2 *feature, const uint width,
                       const uint height, const uint model_index,
                       global float2 *model)
{
  int ii = get_global_id(0); // == get_global_id(0);
  int jj = get_global_id(1); // == get_global_id(1);
  int2 coord = (int2)(get_global_id(0), get_global_id(1));

  const float2 feature_val = feature[jj * width + ii];
  model[jj * width * model_size + ii * model_size + model_index] = feature_val;
}
kernel void average(global float2 *Im, const uint width, const uint height,
                    global float *avrg_i)
{
  int ii = get_global_id(0); // == get_global_id(0);
  int jj = get_global_id(1); // == get_global_id(1);
  int2 coords = (int2)(ii, jj);

  /* if (IS_BOUNDS(ii * jj, width * height)) */
  {
    *avrg_i += Im[jj * width + ii].y;
  }
}

kernel void magnitude(global uchar *src, const uint width, const uint height,
                      global float2 *des)
{
  int ii = get_global_id(0); // == get_global_id(0);
  int jj = get_global_id(1); // == get_global_id(1);
  int2 coords = (int2)(ii, jj);

  float gx_val = 0.f;
  float gy_val = 0.f;

  const int half_sk_size = SK_SIZE / 2;
  for (int fi = -half_sk_size; fi <= half_sk_size; fi++)
  {

    for (int fj = -half_sk_size; fj <= half_sk_size; fj++)
    {
      const int skpos = (jj + fj) * width + ii + fi;
      if (IS_BOUNDS(ii + fi, width) && IS_BOUNDS(jj + fj, height))
      {
        const float src_val = src[skpos] / 256.f;
        gx_val +=
            src_val * Gx[(fi + half_sk_size) * SK_SIZE + fj + half_sk_size];
        gy_val +=
            src_val * Gy[(fi + half_sk_size) * SK_SIZE + fj + half_sk_size];
      }
    }
  }

  /* if (IS_BOUNDS(ii * jj, width * height)) */
  {
    const float gxx = gx_val * gx_val;
    const float gyy = gy_val * gy_val;

    const float mag = sqrt(gxx + gyy);
    const float i_val = src[jj * width + ii];
    des[jj * width + ii].x = i_val;
    des[jj * width + ii].y = mag;
  }
}

inline float pbas_distance(const float I_i, const float I_m, const float B_i,
                           const float B_m, const float alpha,
                           const float avarage_m)
{
  const float Im_diff = abs_diff((int)(I_m), (int)(B_m));
  const float Ii_diff = abs_diff((int)(I_i), (int)(B_i));

  const float res = (alpha / avarage_m) * Im_diff + Ii_diff;
  return res;
}

kernel void pbas_part1(global float2 *feature, const uint width,
                       const uint height, global float *R,
                       const uint total_model_index, global float *D,
                       global float2 *model, global uint *index_r,
                       const float average_mag)
{
  int ii = get_global_id(0);
  int jj = get_global_id(1);

  int2 coords = (int2)(ii, jj);
  int index_d = 0; // index_r[jj * width + ii];
  int i = 0;
  while (index_d < min_index && i < total_model_index)
  {
    const float2 I_val = feature[jj * width + ii];
    const float r_val = R[jj * width + ii];
    const float2 B_val = model[jj * width * model_size + ii * model_size + i];

    const float diff =
        pbas_distance(I_val.x, I_val.y, B_val.x, B_val.y, alpha, average_mag);

    if (diff < r_val)
    {
      if (diff < min_R)
      {
        D[jj * width * model_size + ii * model_size + i] = diff;
      }
      index_d++;
    }

    index_r[jj * width + ii] = index_d;
    ++i;
  }
}

// NOTE model in range [0 ... model_size]
kernel void pbas_part2(global float2 *feature, const uint width,
                       const uint height, global float *R, global float *T,
                       global uint *index_r, const uint min_index,
                       const uint index_l, const uint model_size,
                       global uchar *mask, global float *avrg_d,
                       global uint *rand_n, global float2 *model,
                       global float *D)
{

  int ii = get_global_id(0); // == get_global_id(0);
  int jj = get_global_id(1); // == get_global_id(1);

  int2 coords = (int2)(ii, jj);

  float avr = 1.f;
  uchar color = 0;
  if (index_r[jj * width + ii] >= min_index)
  {
    color = 0;

    if (index_l == model_size)
    {
      const int pos_b = jj * width + ii;
      //
      float ratio = ceil((float)T_upper / T[jj * width + ii]);
      const float I_i = feature[pos_b].x;
      const float I_m = feature[pos_b].y;

      int rand_T = lfsr113_Bits(&rand_n[jj * width + ii]) % T_upper;

      if (rand_T < ratio)
      {
        /* if (IS_BOUNDS(ii * jj, width * height)) */
        {

          const uint random_model_index =
              lfsr113_Bits(&rand_n[jj * width + ii]) % model_size;

          const int pos_model =
              jj * width * model_size + ii * model_size + random_model_index;
          model[pos_model] = (float2)(I_i, I_m);

          for (int i = 0; i < model_size; ++i)
          {
            avr += D[jj * width * model_size + ii * model_size + i];
          }
          avr /= model_size;
        }
      }
      rand_T = lfsr113_Bits(&rand_n[jj * width + ii]) % T_upper;
      if (rand_T < ratio)
      {
        const uint random_model_index =
            lfsr113_Bits(&rand_n[jj * width + ii]) % model_size;
        const int n_y = -1 + lfsr113_Bits(&rand_n[jj * width + ii]) % 3;
        const int n_x = -1 + lfsr113_Bits(&rand_n[jj * width + ii]) % 3;

        if ((jj + n_y) >= 0 && (jj + n_y) < height && (ii + n_x) >= 0 &&
            (ii + n_x) < width)
        {
          const int pos_b = (jj + n_y) * width * model_size +
                            (ii + n_x) * model_size + random_model_index;
          model[pos_b] = feature[(jj + n_y) * width + (ii + n_x)];
        }
      }
    }
  }
  else
  {
    color = 255;
  }
  /* if (IS_BOUNDS(ii * jj, width * height)) */
  {
    mask[jj * width + ii] = color;
    avrg_d[jj * width + ii] = avr;
  }
}

kernel void update_T_R(global uchar *mask, const uint width, const uint height,
                       global float *R, global float *T, global float *avrg_d)
{
  int ii = get_global_id(0); // == get_global_id(0);
  int jj = get_global_id(1); // == get_global_id(1);
  int2 coords = (int2)(ii, jj);

  float R_val = R[jj * width + ii];
  float T_val = T[jj * width + ii];
  const uint pos = jj * width + ii;

  const uchar color = mask[pos];
  const float avr = avrg_d[pos];

  if (R_val > avr * R_scale)
  {
    R_val = R_val * (1.f - R_inc_dec);
  }
  else
  {
    R_val = R_val * (1.f + R_inc_dec);
  }

  if (R_val < R_lower)
    R_val = (float)R_lower;

  if (color == 0)
  {
    T_val += (T_dec / avr);
  }
  else
  {
    T_val -= (T_inc / avr);
  }

  if (T_val < T_lower)
    T_val = T_lower;
  else if (T_val > T_upper)
    T_val = T_upper;

  R[jj * width + ii] = R_val;
  T[jj * width + ii] = T_val;
}
