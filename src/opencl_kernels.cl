#define IS_BOUNDS(val, bound) ((val) >= (0) && (val) < (bound))

#define SK_SIZE 3

struct ImageInfo
{
  uint width;
  uint height;
};
struct PBASParameter
{
  struct ImageInfo imageInfo;
  uint modelSize;
  uint minModels;
  uint T_lower;
  uint R_lower;
  uint min_index;
  uint R_scale;
  uint T_inc;
  uint T_upper;
  float min_R;
  float R_inc_dec;
  float T_dec;
  float alpha;
  float beta;
};

constant float Gy[SK_SIZE * SK_SIZE] = {-1.f, -2.f, -1.f, 0.f, 0.f,
                                        0.f,  1.f,  2.f,  1.f};
constant float Gx[SK_SIZE * SK_SIZE] = {-1.f, 0.f,  1.f, -2.f, 0.f,
                                        2.f,  -1.f, 0.f, 1.f};

uint lfsr113_Bits(global uint *seed)
{
  *seed = (*seed * 1103515245U + 12345U) & 0x7fffffffU;
  return *seed;
}

kernel void fill_T_R(global float *oT, global float *oR,
                     const struct PBASParameter parameters)
{
  int ii = get_global_id(0); // == get_global_id(0);
  int jj = get_global_id(1); // == get_global_id(1);
  int2 coord = (int2)(get_global_id(0), get_global_id(1));
  const uint width = parameters.imageInfo.width;
  const uint height = parameters.imageInfo.height;
  {
    oT[jj * width + ii] = parameters.T_lower;
    oR[jj * width + ii] = parameters.R_lower;
  }
}
kernel void fill_model(global float2 *feature, global float2 *model,
                       const uint model_index,
                       const struct PBASParameter parameters)
{
  int ii = get_global_id(0); // == get_global_id(0);
  int jj = get_global_id(1); // == get_global_id(1);
  int2 coord = (int2)(get_global_id(0), get_global_id(1));
  const uint width = parameters.imageInfo.width;
  const uint height = parameters.imageInfo.height;
  const uint model_size = parameters.modelSize;

  const float2 feature_val = feature[jj * width + ii];
  model[jj * width * model_size + ii * model_size + model_index] = feature_val;
}
kernel void average(global float2 *Im, const struct PBASParameter parameters,
                    global float *avrg_i)
{
  int ii = get_global_id(0); // == get_global_id(0);
  int jj = get_global_id(1); // == get_global_id(1);
  int2 coords = (int2)(ii, jj);
  const uint width = parameters.imageInfo.width;
  *avrg_i += Im[jj * width + ii].y;
}

kernel void magnitude(global uchar *src, global float2 *des,
                      const struct PBASParameter parameters)
{
  int ii = get_global_id(0);
  int jj = get_global_id(1);
  int2 coords = (int2)(ii, jj);

  const uint width = parameters.imageInfo.width;
  const uint height = parameters.imageInfo.height;

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

  const float gxx = gx_val * gx_val;
  const float gyy = gy_val * gy_val;

  const float mag = sqrt(gxx + gyy);
  const float i_val = src[jj * width + ii];
  des[jj * width + ii].x = i_val;
  des[jj * width + ii].y = mag;
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

kernel void pbas(global float2 *feature, global float *R, global float *T,
                 global float *D, global float2 *model, global uchar *mask,
                 global float *avrg_d, global uint *rand_n,
                 const uint total_model_index, const float average_mag,
                 const struct PBASParameter parameters)
{
  int ii = get_global_id(0);
  int jj = get_global_id(1);

  int2 coords = (int2)(ii, jj);
  const uint width = parameters.imageInfo.width;
  const uint height = parameters.imageInfo.height;
  const uint min_index = parameters.minModels;
  const uint model_size = parameters.modelSize;
  const float alpha = parameters.alpha;
  const float beta = parameters.beta;
  const uint T_upper = parameters.T_upper;
  const float min_R = parameters.min_R;
  int index_r = 0;
  int curri = 0;
  while (index_r < min_index && curri < total_model_index)
  {
    const float2 I_val = feature[jj * width + ii];
    const float r_val = R[jj * width + ii];
    const float2 B_val = model[jj * width * model_size + ii * model_size + curri];

    const float diff =
        pbas_distance(I_val.x, I_val.y, B_val.x, B_val.y, alpha, average_mag);
    if (diff < r_val)
    {
      if (diff < min_R)
      {
        D[jj * width * model_size + ii * model_size + curri] = diff;
      }
      ++index_r;
    }

    ++curri;
  }

  float avr = 1.f;
  uchar color = 0;
  if (index_r >= min_index)
  {
    color = 0;
    if (total_model_index == parameters.modelSize)
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
  mask[jj * width + ii] = color;
  avrg_d[jj * width + ii] = avr;
}

kernel void update_T_R(global uchar *mask, global float *R, global float *T,
                       global float *avrg_d,
                       const struct PBASParameter parameters)
{
  int ii = get_global_id(0); // == get_global_id(0);
  int jj = get_global_id(1); // == get_global_id(1);
  int2 coords = (int2)(ii, jj);

  const uint width = parameters.imageInfo.width;
  const uint height = parameters.imageInfo.height;
  const float R_scale = parameters.R_scale;
  const float R_inc_dec = parameters.R_inc_dec;
  const float R_lower = parameters.R_lower;
  const float T_dec = parameters.T_dec;
  const float T_lower = parameters.T_lower;
  const float T_upper = parameters.T_upper;
  const float T_inc = parameters.T_inc;
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
