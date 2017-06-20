#define IS_BOUNDS(val, bound) ((val) >= (0) && (val) < (bound))

#define SK_SIZE 3
#define DEBUG

/**************************************************************************
**
**  svd3
**
** Quick singular value decomposition as described by:
** A. McAdams, A. Selle, R. Tamstorf, J. Teran and E. Sifakis,
** Computing the Singular Value Decomposition of 3x3 matrices
** with minimal branching and elementary floating point operations,
**  University of Wisconsin - Madison technical report TR1690, May 2011
**
**	OPTIMIZED GPU VERSION
** 	Implementation by: Eric Jang
**
**  13 Apr 2014
**
**************************************************************************/

#define _gamma 5.828427124 // FOUR_GAMMA_SQUARED = sqrt(8)+3;
#define _cstar 0.923879532 // cos(pi/8)
#define _sstar 0.3826834323 // sin(p/8)
#define EPSILON 1e-6


inline float accurateSqrt(float x)
{
  return x * rsqrt(x);
}

inline void condSwap(bool c, float *X, float *Y)
{
  // used in step 2
  float Z = *X;
  *X = c ? (*Y) : (*X);
  *Y = c ? (Z) : (*Y);
}

inline float dist2(float x, float y, float z)
{
  return x*x + y*y + z*z;
}

inline void condNegSwap(bool c, float *X, float *Y)
{
  // used in step 2 and 3
  float Z = -(*X);
  *X = c ? (*Y) : (*X);
  *Y = c ? Z : (*Y);
}

// matrix multiplication M = A * B
void multAB(
  float a11, float a12, float a13,
  float a21, float a22, float a23,
  float a31, float a32, float a33,
  //
  float b11, float b12, float b13,
  float b21, float b22, float b23,
  float b31, float b32, float b33,
  //
  float *m11, float *m12, float *m13,
  float *m21, float *m22, float *m23,
  float *m31, float *m32, float *m33)
{
  *m11 = a11*b11 + a12*b21 + a13*b31; *m12 = a11*b12 + a12*b22 + a13*b32; *m13 = a11*b13 + a12*b23 + a13*b33;
  *m21 = a21*b11 + a22*b21 + a23*b31; *m22 = a21*b12 + a22*b22 + a23*b32; *m23 = a21*b13 + a22*b23 + a23*b33;
  *m31 = a31*b11 + a32*b21 + a33*b31; *m32 = a31*b12 + a32*b22 + a33*b32; *m33 = a31*b13 + a32*b23 + a33*b33;
}

// matrix multiplication M = Transpose[A] * B
void multAtB(
  float a11, float a12, float a13,
  float a21, float a22, float a23,
  float a31, float a32, float a33,
  //
  float b11, float b12, float b13,
  float b21, float b22, float b23,
  float b31, float b32, float b33,
  //
  float *m11, float *m12, float *m13,
  float *m21, float *m22, float *m23,
  float *m31, float *m32, float *m33)
{
  *m11 = a11*b11 + a21*b21 + a31*b31; *m12 = a11*b12 + a21*b22 + a31*b32; *m13 = a11*b13 + a21*b23 + a31*b33;
  *m21 = a12*b11 + a22*b21 + a32*b31; *m22 = a12*b12 + a22*b22 + a32*b32; *m23 = a12*b13 + a22*b23 + a32*b33;
  *m31 = a13*b11 + a23*b21 + a33*b31; *m32 = a13*b12 + a23*b22 + a33*b32; *m33 = a13*b13 + a23*b23 + a33*b33;
}

void quatToMat3(const float * qV,
  float *m11, float *m12, float *m13,
  float *m21, float *m22, float *m23,
  float *m31, float *m32, float *m33
)
{
  float w = qV[3];
  float x = qV[0];
  float y = qV[1];
  float z = qV[2];

  float qxx = x*x;
  float qyy = y*y;
  float qzz = z*z;
  float qxz = x*z;
  float qxy = x*y;
  float qyz = y*z;
  float qwx = w*x;
  float qwy = w*y;
  float qwz = w*z;

  *m11 = 1 - 2 * (qyy + qzz); *m12 = 2 * (qxy - qwz);     *m13 = 2 * (qxz + qwy);
  *m21 = 2 * (qxy + qwz);     *m22 = 1 - 2 * (qxx + qzz); *m23 = 2 * (qyz - qwx);
  *m31 = 2 * (qxz - qwy);     *m32 = 2 * (qyz + qwx);     *m33 = 1 - 2 * (qxx + qyy);
}

void approximateGivensQuaternion(float a11, float a12, float a22, float *ch, float *sh)
{
  /*
  * Given givens angle computed by approximateGivensAngles,
  * compute the corresponding rotation quaternion.
  */
  *ch = 2 * (a11 - a22);
  *sh = a12;
  const float lch = *ch;
  const float lsh = *sh;
  const bool b = (_gamma*lsh*lsh) < (lch*lch);
  const float w = rsqrt(lch*lch + lsh*lsh);
  *ch = b ? w*lch : _cstar;
  *sh = b ? w*lsh : _sstar;
}

void jacobiConjugation(const int x, const int y, const int z,
  float *s11,
  float *s21, float *s22,
  float *s31, float *s32, float *s33,
  float *qV)
{
  float ch, sh;
  approximateGivensQuaternion(*s11, *s21, *s22, &ch, &sh);

  const float scale = ch*ch + sh*sh;
  const float a = (ch*ch - sh*sh) / scale;
  const float b = (2 * sh*ch) / scale;

  // make temp copy of S
  float _s11 = *s11;
  float _s21 = *s21; float _s22 = *s22;
  float _s31 = *s31; float _s32 = *s32; float _s33 = *s33;

  // perform conjugation S = Q'*S*Q
  // Q already implicitly solved from a, b
  *s11 = a*(a*_s11 + b*_s21) + b*(a*_s21 + b*_s22);
  *s21 = a*(-b*_s11 + a*_s21) + b*(-b*_s21 + a*_s22);	*s22 = -b*(-b*_s11 + a*_s21) + a*(-b*_s21 + a*_s22);
  *s31 = a*_s31 + b*_s32;								*s32 = -b*_s31 + a*_s32; *s33 = _s33;

  // update cumulative rotation qV
  float tmp[3];
  tmp[0] = qV[0] * sh;
  tmp[1] = qV[1] * sh;
  tmp[2] = qV[2] * sh;
  sh *= qV[3];

  qV[0] *= ch;
  qV[1] *= ch;
  qV[2] *= ch;
  qV[3] *= ch;

  // (x,y,z) corresponds to ((0,1,2),(1,2,0),(2,0,1))
  // for (p,q) = ((0,1),(1,2),(0,2))
  qV[z] += sh;
  qV[3] -= tmp[z]; // w
  qV[x] += tmp[y];
  qV[y] -= tmp[x];

  // re-arrange matrix for next iteration
  _s11 = *s22;
  _s21 = *s32; _s22 = *s33;
  _s31 = *s21; _s32 = *s31; _s33 = *s11;
  *s11 = _s11;
  *s21 = _s21; *s22 = _s22;
  *s31 = _s31; *s32 = _s32; *s33 = _s33;

}

// finds transformation that diagonalizes a symmetric matrix
void jacobiEigenanlysis( // symmetric matrix
  float *s11,
  float *s21, float *s22,
  float *s31, float *s32, float *s33,
  // quaternion representation of V
  float * qV)
{
  qV[3] = 1; qV[0] = 0; qV[1] = 0; qV[2] = 0; // follow same indexing convention as GLM
  for (int i = 0; i < 3; i++)
  {
    // we wish to eliminate the maximum off-diagonal element
    // on every iteration, but cycling over all 3 possible rotations
    // in fixed order (p,q) = (1,2) , (2,3), (1,3) still retains
    //  asymptotic convergence
    jacobiConjugation(0, 1, 2, s11, s21, s22, s31, s32, s33, qV); // p,q = 0,1
    jacobiConjugation(1, 2, 0, s11, s21, s22, s31, s32, s33, qV); // p,q = 1,2
    jacobiConjugation(2, 0, 1, s11, s21, s22, s31, s32, s33, qV); // p,q = 0,2
  }
}

void sortSingularValues(// matrix that we want to decompose
  float *b11, float *b12, float *b13,
  float *b21, float *b22, float *b23,
  float *b31, float *b32, float *b33,
  // sort V simultaneously
  float *v11, float *v12, float *v13,
  float *v21, float *v22, float *v23,
  float *v31, float *v32, float *v33)
{
  float rho1 = dist2(*b11, *b21, *b31);
  float rho2 = dist2(*b12, *b22, *b32);
  float rho3 = dist2(*b13, *b23, *b33);
  bool c;
  c = rho1 < rho2;
  condNegSwap(c, b11, b12); condNegSwap(c, v11, v12);
  condNegSwap(c, b21, b22); condNegSwap(c, v21, v22);
  condNegSwap(c, b31, b32); condNegSwap(c, v31, v32);
  condSwap(c, &rho1, &rho2);
  c = rho1 < rho3;
  condNegSwap(c, b11, b13); condNegSwap(c, v11, v13);
  condNegSwap(c, b21, b23); condNegSwap(c, v21, v23);
  condNegSwap(c, b31, b33); condNegSwap(c, v31, v33);
  condSwap(c, &rho1, &rho3);
  c = rho2 < rho3;
  condNegSwap(c, b12, b13); condNegSwap(c, v12, v13);
  condNegSwap(c, b22, b23); condNegSwap(c, v22, v23);
  condNegSwap(c, b32, b33); condNegSwap(c, v32, v33);
}

void QRGivensQuaternion(float a1, float a2, float *ch, float *sh)
{
  // a1 = pivot point on diagonal
  // a2 = lower triangular entry we want to annihilate
  const float epsilon = 1e-6;
  const float rho = accurateSqrt(a1*a1 + a2*a2);

  *sh = rho > epsilon ? a2 : 0;

  *ch = fabs(a1) + fmax(rho, epsilon);
  bool b = a1 < 0;
  condSwap(b, sh, ch);
  const float lch = *ch;
  const float lsh = *sh;
  float w = rsqrt(lch*lch + lsh*lsh);
  *ch *= w;
  *sh *= w;
}

void QRDecomposition(// matrix that we want to decompose
  float b11, float b12, float b13,
  float b21, float b22, float b23,
  float b31, float b32, float b33,
  // output Q
  float *q11, float *q12, float *q13,
  float *q21, float *q22, float *q23,
  float *q31, float *q32, float *q33,
  // output R
  float *r11, float *r12, float *r13,
  float *r21, float *r22, float *r23,
  float *r31, float *r32, float *r33)
{
  float ch1, sh1, ch2, sh2, ch3, sh3;
  float a, b;

  // first givens rotation (ch,0,0,sh)
  QRGivensQuaternion(b11, b21, &ch1, &sh1);
  a = 1 - 2 * sh1*sh1;
  b = 2 * ch1*sh1;
  // apply B = Q' * B
  *r11 = a*b11 + b*b21;  *r12 = a*b12 + b*b22;  *r13 = a*b13 + b*b23;
  *r21 = -b*b11 + a*b21; *r22 = -b*b12 + a*b22; *r23 = -b*b13 + a*b23;
  *r31 = b31;            *r32 = b32;            *r33 = b33;

  // second givens rotation (ch,0,-sh,0)
  QRGivensQuaternion(*r11, *r31, &ch2, &sh2);
  a = 1 - 2 * sh2*sh2;
  b = 2 * ch2*sh2;
  // apply B = Q' * B;
  b11 = a*(*r11) + b*(*r31);  b12 = a*(*r12) + b*(*r32);  b13 = a*(*r13) + b*(*r33);
  b21 = *r21;                 b22 = *r22;                 b23 = *r23;
  b31 = -b*(*r11) + a*(*r31); b32 = -b*(*r12) + a*(*r32); b33 = -b*(*r13) + a*(*r33);

  // third givens rotation (ch,sh,0,0)
  QRGivensQuaternion(b22, b32, &ch3, &sh3);
  a = 1 - 2 * sh3*sh3;
  b = 2 * ch3*sh3;
  // R is now set to desired value
  *r11 = b11;               *r12 = b12;             *r13 = b13;
  *r21 = a*b21 + b*b31;     *r22 = a*b22 + b*b32;   *r23 = a*b23 + b*b33;
  *r31 = -b*b21 + a*b31;    *r32 = -b*b22 + a*b32;  *r33 = -b*b23 + a*b33;

  // construct the cumulative rotation Q=Q1 * Q2 * Q3
  // the number of floating point operations for three quaternion multiplications
  // is more or less comparable to the explicit form of the joined matrix.
  // certainly more memory-efficient!
  float sh12 = sh1*sh1;
  float sh22 = sh2*sh2;
  float sh32 = sh3*sh3;

  *q11 = (-1 + 2 * sh12)*(-1 + 2 * sh22);
  *q12 = 4 * ch2*ch3*(-1 + 2 * sh12)*sh2*sh3 + 2 * ch1*sh1*(-1 + 2 * sh32);
  *q13 = 4 * ch1*ch3*sh1*sh3 - 2 * ch2*(-1 + 2 * sh12)*sh2*(-1 + 2 * sh32);

  *q21 = 2 * ch1*sh1*(1 - 2 * sh22);
  *q22 = -8 * ch1*ch2*ch3*sh1*sh2*sh3 + (-1 + 2 * sh12)*(-1 + 2 * sh32);
  *q23 = -2 * ch3*sh3 + 4 * sh1*(ch3*sh1*sh3 + ch1*ch2*sh2*(-1 + 2 * sh32));

  *q31 = 2 * ch2*sh2;
  *q32 = 2 * ch3*(1 - 2 * sh22)*sh3;
  *q33 = (-1 + 2 * sh22)*(-1 + 2 * sh32);
}

void svd(// input A
  float a11, float a12, float a13,
  float a21, float a22, float a23,
  float a31, float a32, float a33,
  // output U
  float *u11, float *u12, float *u13,
  float *u21, float *u22, float *u23,
  float *u31, float *u32, float *u33,
  // output S
  float *s11, float *s12, float *s13,
  float *s21, float *s22, float *s23,
  float *s31, float *s32, float *s33,
  // output V
  float *v11, float *v12, float *v13,
  float *v21, float *v22, float *v23,
  float *v31, float *v32, float *v33)
{
  // normal equations matrix
  float ATA11, ATA12, ATA13;
  float ATA21, ATA22, ATA23;
  float ATA31, ATA32, ATA33;

  //printf("a = %f %f %f %f %f %f %f %f %f\n", a11, a12, a13, a21, a22, a23, a31, a32, a33);

  multAtB(
    a11, a12, a13, a21, a22, a23, a31, a32, a33,
    a11, a12, a13, a21, a22, a23, a31, a32, a33,
    &ATA11, &ATA12, &ATA13, &ATA21, &ATA22, &ATA23, &ATA31, &ATA32, &ATA33);
  // printf("ATA = %f %f %f %f %f %f %f %f %f\n", ATA11, ATA12, ATA13, ATA21, ATA22, ATA23, ATA31, ATA32, ATA33);
   // symmetric eigenalysis
  float qV[4];
  jacobiEigenanlysis(&ATA11, &ATA21, &ATA22, &ATA31, &ATA32, &ATA33, &qV);
  //printf("qV = %f %f %f %f\n", qV[0], qV[1], qV[2], qV[3]);

  quatToMat3(qV, v11, v12, v13, v21, v22, v23, v31, v32, v33);
  //printf("v = %f %f %f %f %f %f %f %f %f\n", *v11, *v12, *v13, *v21, *v22, *v23, *v31, *v32, *v33);

  float b11, b12, b13;
  float b21, b22, b23;
  float b31, b32, b33;
  multAB(a11, a12, a13, a21, a22, a23, a31, a32, a33,
    *v11, *v12, *v13, *v21, *v22, *v23, *v31, *v32, *v33,
    &b11, &b12, &b13, &b21, &b22, &b23, &b31, &b32, &b33);
  //return;

  //printf("b = %f %f %f %f %f %f %f %f %f\n", b11, b12, b13, b21, b22, b23, b31, b32, b33);


  // sort singular values and find V
  sortSingularValues(&b11, &b12, &b13, &b21, &b22, &b23, &b31, &b32, &b33,
    v11, v12, v13, v21, v22, v23, v31, v32, v33);

  //printf("b =\n %f %f %f \n%f %f %f\n %f %f %f\n", b11, b12, b13, b21, b22, b23, b31, b32, b33);
  //printf("v =\n %f %f %f \n%f %f %f\n %f %f %f\n", *v11, *v12, *v13, *v21, *v22, *v23, *v31, *v32, *v33);

  // QR decomposition
  QRDecomposition(b11, b12, b13, b21, b22, b23, b31, b32, b33,
    u11, u12, u13, u21, u22, u23, u31, u32, u33,
    s11, s12, s13, s21, s22, s23, s31, s32, s33
  );
  //printf("SingValues =\n %f %f %f \n%f %f %f\n %f %f %f\n", *s11, *s12, *s13, *s21, *s22, *s23, *s31, *s32, *s33);
}



struct ImageInfo
{
  uint width;
  uint height;
  uint channels;
};
struct PBASParameter
{
  struct ImageInfo imageInfo;
  uint modelSize;
  uint minModels;
  uint T_lower;
  uint R_lower;
  uint R_scale;
  uint T_inc;
  uint T_upper;
  float min_R;
  float R_inc_dec;
  float T_dec;
  float alpha;
  float beta;
  float avrg_mag;
};

constant float Gy[SK_SIZE * SK_SIZE] = { -1.f, -2.f, -1.f, 0.f, 0.f,
                                        0.f,  1.f,  2.f,  1.f };
constant float Gx[SK_SIZE * SK_SIZE] = { -1.f, 0.f,  1.f, -2.f, 0.f,
                                        2.f,  -1.f, 0.f, 1.f };

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
  const uint channels = parameters.imageInfo.channels;
  if (channels == 0)
  {
    oT[jj * width + ii] = parameters.T_lower;
    oR[jj * width + ii] = parameters.R_lower;
  }
  else
  {
    oT[jj * width*channels + ii*channels] = parameters.T_lower;
    oT[jj * width*channels + ii*channels] = parameters.T_lower;
    oT[jj * width*channels + ii*channels] = parameters.T_lower;

    oR[jj * width*channels + ii*channels] = parameters.R_lower;
    oR[jj * width*channels + ii*channels] = parameters.R_lower;
    oR[jj * width*channels + ii*channels] = parameters.R_lower;

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
float abs_diff_f(float f1, float f2)
{
  return f1 > f2 ? f1 - f2 : f2 - f1;
}
kernel void magnitude(global uchar *src, global float2 *des,
  global struct PBASParameter *parameters)
{
  int ii = get_global_id(0);
  int jj = get_global_id(1);
  int2 coords = (int2)(ii, jj);
  const uint width = parameters->imageInfo.width;
  const uint height = parameters->imageInfo.height;

  //// output U
  float u11, u12, u13;
  float u21, u22, u23;
  float u31, u32, u33;
  // output S
  float s11, s12, s13;
  float s21, s22, s23;
  float s31, s32, s33;
  // output V
  float v11, v12, v13;
  float v21, v22, v23;
  float v31, v32, v33;


  float src11 = src[(jj - 1) * width + (ii - 1)], src12 = src[(jj - 1) * width + (ii - 0)], src13 = src[(jj - 1) * width + (ii + 1)];
  float src21 = src[(jj - 0) * width + (ii - 1)], src22 = src[(jj - 0) * width + (ii - 0)], src23 = src[(jj - 0) * width + (ii + 1)];
  float src31 = src[(jj + 1) * width + (ii - 1)], src32 = src[(jj + 1) * width + (ii - 0)], src33 = src[(jj + 1) * width + (ii + 1)];

  {
    svd(
      src11, src12, src13,
      src21, src22, src23,
      src31, src32, src33,

      &u11, &u12, &u13,
      &u21, &u22, &u23,
      &u31, &u32, &u33,

      &s11, &s12, &s13,
      &s21, &s12, &s23,
      &s31, &s12, &s33,

      &v11, &v12, &v13,
      &v21, &v22, &v23,
      &v31, &v32, &v33
    );
  }

  const float i_val = src[jj * width + ii];

  des[jj * width + ii].x = i_val;


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

  //des[jj * width + ii].y = 1.f/ (1.f+mag);



  //last//des[jj * width + ii].y = mag / (1.f + ((s12 / s11) + (s33 / s11)));

  des[jj * width + ii].y = mag / (1.f + ((s12 / s11) + (s33 / s11)));

  //des[jj * width + ii].y = (((s12 / s11) + (s33 / s11)) / (1.f + mag))*256;

  //des[jj * width + ii].y = (mag*((s12 / s11) + (s33 / s11)));
  //des[jj * width + ii].y = (mag*((s12 / s11) + (s33 / s11)));
}
#ifdef DEBUG
kernel void read_featureY(global float2 *features, global float *outArray,
  const struct PBASParameter parameters)
{
  int ii = get_global_id(0);
  int jj = get_global_id(1);
  int2 coords = (int2)(ii, jj);
  const uint width = parameters.imageInfo.width;
  const uint height = parameters.imageInfo.height;


  const float i_val = features[jj * width + ii].y;
  outArray[jj * width + ii] = i_val;
}
#endif

inline float pbas_distance(const float I_i, const float I_m, const float B_i,
  const float B_m, const float alpha,
  const float avarage_m)
{
  const float Im_diff = (alpha / avarage_m) *((I_m - B_m)*(I_m - B_m));
  const float Ii_diff = ((I_i - B_i)*(I_i - B_i));

  const float res = /*(alpha / avarage_m) * Im_diff +*/ sqrt(Ii_diff + Im_diff);
  return res;
}

kernel void pbas(global float2 *feature, global float *R, global float *T,
  global float *D, global float2 *model, global uchar *mask,
  global float *avrg_d, global uint *rand_n,
  const uint total_model_index,
  global float *model_out,
  global struct PBASParameter *parameters)
{
  int ii = get_global_id(0);
  int jj = get_global_id(1);

  int2 coords = (int2)(ii, jj);
  const uint width = parameters->imageInfo.width;
  const uint height = parameters->imageInfo.height;
  const float R_inc_dec = parameters->R_inc_dec;
  const uint min_index = parameters->minModels;
  const uint model_size = parameters->modelSize;
  const float alpha = parameters->alpha;
  const float beta = parameters->beta;
  const uint T_upper = parameters->T_upper;
  const float min_R = parameters->min_R;

  int index_r = 0;
  int curri = 0;
  float avrg_mag = 1.f;
  float model_out_val = 0.f;
  for (int i = 0; i < total_model_index; ++i)
  {
    model_out_val += model[jj * width * model_size + ii * model_size + curri].x;
    avrg_mag += model[jj * width * model_size + ii * model_size + curri].y;
  }
  model_out_val /= model_size;
  model_out[jj * width + ii] = model_out_val;
  avrg_mag /= total_model_index;

  while (index_r < min_index && curri < total_model_index)
  {
    const float2 I_val = feature[jj * width + ii];
    const float r_val = R[jj * width + ii];
    const float2 B_val = model[jj * width * model_size + ii * model_size + curri];

    const float diff =
      pbas_distance(I_val.x, I_val.y, B_val.x, B_val.y, alpha, avrg_mag);

    int lbp_dist = 0;
    for(int y = -1; y < 2; y++)
    { 
      for (int x = -1; x < 2; x++)
      {
        int distI = feature[(jj + y) * width + (ii + x)].y >= I_val.y;
        int distB = model[(jj + y) * width * model_size + (ii + x) * model_size + curri].y >= B_val.y;

        if (distI != distB)
          lbp_dist++;

      }
    }



    if (diff < r_val && lbp_dist <= 8)
    {
      if (diff < min_R)
      {
        D[jj * width * model_size + ii * model_size + curri] = diff;
      }

      ++index_r;
    }

    ++curri;
  }

  uchar color = 0;

  if (index_r >= min_index)
  {
    color = 0;

    if (total_model_index == model_size)
    {
      const int pos_b = jj * width + ii;

      float ratio = T[jj * width + ii] / T_upper;

      const float I_i = feature[pos_b].x;
      const float I_m = feature[pos_b].y;

      uint rand_val = lfsr113_Bits(&rand_n[jj * width + ii]) % T_upper;
      float rand_T = (float)rand_val / T_upper;

      if (rand_T > ratio)
      {
        const uint random_model_index =
          lfsr113_Bits(&rand_n[jj * width + ii]) % model_size;

        const int pos_model =
          jj * width * model_size + ii * model_size + random_model_index;
        //model[pos_model] = (float2)((I_i + model_out_val)*0.5f, (I_m + avrg_mag)*0.5f);
        model[pos_model] = (float2)((I_i )*1.f, (I_m )*1.f);

        float avr = 0.f;
        for (int i = 0; i < total_model_index; ++i)
        {
          avr += D[jj * width * model_size + ii * model_size + i];
        }
        avr /= total_model_index;

        avrg_d[jj * width + ii] = avr;
      }
  
      rand_val = lfsr113_Bits(&rand_n[jj * width + ii]) % T_upper;
      rand_T = (float)rand_val / T_upper;
      if (rand_T > ratio)
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
          //model[pos_b].x = (feature[(jj + n_y) * width + (ii + n_x)].x + model_out_val)*0.5f;
          model[pos_b].x = (feature[(jj + n_y) * width + (ii + n_x)].x)*1.0f;
          //model[pos_b].y = (feature[(jj + n_y) * width + (ii + n_x)].y + avrg_mag)*0.5f;
          model[pos_b].y = (feature[(jj + n_y) * width + (ii + n_x)].y)*1.0f;
        }

      }
    }
  }
  else
  {
    color = 255;

  }
  mask[jj * width + ii] = color;
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
  const float T_inc = (float)parameters.T_inc;
  float R_val = R[jj * width + ii];
  float T_val = T[jj * width + ii];
  const uint pos = jj * width + ii;

  const uchar color = mask[pos];
  const float avr = avrg_d[pos];

  if (R_val > avr * R_scale)
  {
    R_val *= (1.f - R_inc_dec);
  }
  else
  {
    R_val *= (1.f + R_inc_dec);
  }

  if (R_val < R_lower)
    R_val = (float)R_lower;

  if (R_val > T_upper)
    R_val = (float)T_upper;

  if (color == 255)
  {
    T_val += (25 / (avr + 1.f));
  }
  else
  {
    T_val -= (T_dec / (avr + 1.f));
  }

  if (T_val < T_lower)
    T_val = T_lower;
  else if (T_val > T_upper)
    T_val = T_upper;

  R[jj * width + ii] = R_val;
  T[jj * width + ii] = T_val;
}