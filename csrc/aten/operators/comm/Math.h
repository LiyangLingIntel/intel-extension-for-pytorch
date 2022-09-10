#pragma once

#include <ATen/NumericUtils.h>
#include <c10/macros/Macros.h>
#include <c10/util/MathConstants.h>
#include <c10/util/complex.h>
#include <c10/util/math_compat.h>
#include <oneapi/dpl/cmath>
#include <oneapi/dpl/complex>
#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/utility>
#include "AccumulateType.h"
#include "General.h"
#include "Numerics.h"

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
DPCPP_BOTH static inline scalar_t zeta(scalar_t _x, scalar_t _q) {
  using acc_t = acc_type<scalar_t>;
  const acc_t MACHEP = acc_t{1.11022302462515654042E-16};
  constexpr acc_t zero = acc_t{0.0};
  constexpr acc_t half = acc_t{0.5};
  constexpr acc_t one = acc_t{1.0};
  static const acc_t A[] = {
      12.0,
      -720.0,
      30240.0,
      -1209600.0,
      47900160.0,
      -1.8924375803183791606e9, /*1.307674368e12/691*/
      7.47242496e10,
      -2.950130727918164224e12, /*1.067062284288e16/3617*/
      1.1646782814350067249e14, /*5.109094217170944e18/43867*/
      -4.5979787224074726105e15, /*8.028576626982912e20/174611*/
      1.8152105401943546773e17, /*1.5511210043330985984e23/854513*/
      -7.1661652561756670113e18 /*1.6938241367317436694528e27/236364091*/
  };
  acc_t x = static_cast<acc_t>(_x);
  acc_t q = static_cast<acc_t>(_q);

  int i = 0;
  acc_t a, b, k, s, t, w;
  if (x == one) {
    return std::numeric_limits<scalar_t>::infinity();
  }

  if (x < one) {
    return std::numeric_limits<scalar_t>::quiet_NaN();
  }

  if (q <= zero) {
    if (q == sycl::floor(q)) {
      return std::numeric_limits<scalar_t>::infinity();
    }
    if (x != sycl::floor(x)) {
      return std::numeric_limits<scalar_t>::quiet_NaN();
    }
  }

  s = sycl::pow(q, -x);
  a = q;
  i = 0;
  b = zero;
  while ((i < 9) || (a <= acc_t{9.0})) {
    i += 1;
    a += one;
    b = sycl::pow(a, -x);
    s += b;
    if ((-MACHEP * s < b) && (b < MACHEP * s)) {
      return static_cast<scalar_t>(s);
    }
  };

  w = a;
  s += b * w / (x - one);
  s -= half * b;
  a = one;
  k = zero;
  for (int i = 0; i < 12; i++) {
    a *= x + k;
    b /= w;
    t = a * b / A[i];
    s = s + t;
    t = sycl::fabs(t / s);
    if (t < MACHEP) {
      return static_cast<scalar_t>(s);
    }
    k += one;
    a *= x + k;
    b /= w;
    k += one;
  }
  return static_cast<scalar_t>(s);
}

template <typename scalar_t>
static inline DPCPP_BOTH scalar_t calc_digamma(scalar_t in) {
  // [C++ Standard Reference: Gamma Function]
  // https://en.cppreference.com/w/cpp/numeric/math/tgamma
  using accscalar_t = acc_type<scalar_t>;
  static const double PI_f64 = 3.14159265358979323846;
  const accscalar_t PSI_10 = 2.25175258906672110764;
  const accscalar_t A[] = {
      8.33333333333333333333E-2,
      -2.10927960927960927961E-2,
      7.57575757575757575758E-3,
      -4.16666666666666666667E-3,
      3.96825396825396825397E-3,
      -8.33333333333333333333E-3,
      8.33333333333333333333E-2,
  };

  accscalar_t x = static_cast<accscalar_t>(in);
  if (x == 0) {
    // As per C++ standard for gamma related functions and SciPy,
    // If the argument is ±0, ±∞ is returned
    return std::copysign(static_cast<scalar_t>(INFINITY), -x);
  }

  bool x_is_integer = x == ::trunc(x);
  accscalar_t result = 0;
  if (x < 0) {
    if (x_is_integer) {
      // As per C++ standard for gamma related functions and SciPy,
      // If the argument is a negative integer, NaN is returned
      return static_cast<scalar_t>(NAN);
    }
    // Extracts the fractional part of x as r, since tan(pi * r) is more
    // numerically accurate than tan(pi * x). While these operations are
    // mathematically equivalent since both x and r are in radians and tan() has
    // a periodicity of pi, in practice the computation of pi * x is a source of
    // error (when |x| > 1).
    double q, r;
    r = sycl::modf(static_cast<double>(x), &q);
    result = static_cast<accscalar_t>(-PI_f64 / sycl::tan(PI_f64 * r));
    x = 1 - x;
  }

  while (x < 10) {
    result -= 1 / x;
    x += 1;
  }
  if (x == 10) {
    return static_cast<scalar_t>(result + PSI_10);
  }

  accscalar_t y = 0;
  if (x < 1.0e17f) {
    accscalar_t z = 1 / (x * x);

    accscalar_t polevl_result = 0;
    for (int i = 0; i <= 6; i++) {
      polevl_result = polevl_result * z + A[i];
    }
    y = z * polevl_result;
  }

  return static_cast<scalar_t>(
      sycl::log(x) - (static_cast<accscalar_t>(0.5) / x) - y + result);
}

template <typename scalar_t>
static inline DPCPP_BOTH scalar_t calc_trigamma(scalar_t in) {
  using accscalar_t = acc_type<scalar_t>;
  const accscalar_t PI = 3.14159265358979323846;
  accscalar_t x = static_cast<accscalar_t>(in);
  accscalar_t sign = +1;
  accscalar_t result = 0;
  if (x < 0.5f) {
    sign = -1;
    accscalar_t sin_pi_x = sycl::sin(PI * x);
    result -= (PI * PI) / (sin_pi_x * sin_pi_x);
    x = 1 - x;
  }
  for (int i = 0; i < 6; ++i) {
    result += 1 / (x * x);
    x += 1;
  }
  const accscalar_t one = static_cast<scalar_t>(1);
  const accscalar_t ixx = 1 / (x * x);
  result += (1 + 1 / (2 * x) +
             ixx * (one / 6 - ixx * (one / 30 - ixx * (one / 42)))) /
      x;
  return static_cast<scalar_t>(sign * result);
}

template <typename scalar_t>
static inline DPCPP_BOTH scalar_t calc_gcd(scalar_t a_in, scalar_t b_in) {
  scalar_t a = sycl::abs(a_in);
  scalar_t b = sycl::abs(b_in);
  while (a != 0) {
    scalar_t c = a;
    a = b % a;
    b = c;
  }
  return b;
}

/*
 * For licensing information and documentation, please refer to the the cpu
 * implementation located in "ATen/native/Math.h".
 */
template <typename scalar_t>
static inline DPCPP_BOTH scalar_t
chbevl(scalar_t _x, const scalar_t array[], size_t len) {
  static_assert(
      !std::is_same<scalar_t, Half>() && !std::is_same<scalar_t, BFloat16>(),
      "don't instantiate with low precision type");
  using accscalar_t = acc_type<scalar_t>;

  accscalar_t x = static_cast<accscalar_t>(_x);
  accscalar_t b0, b1, b2;

  b0 = static_cast<accscalar_t>(array[0]);
  b1 = 0;

  for (size_t i = 1; i < len; ++i) {
    b2 = b1;
    b1 = b0;
    b0 = x * b1 - b2 + static_cast<accscalar_t>(array[i]);
  }

  return static_cast<scalar_t>(0.5 * (b0 - b2));
}

/*
 * For licensing information and documentation, please refer to the the cpu
 * implementation located in "ATen/native/Math.h".
 */
template <typename T>
DPCPP_BOTH inline std::tuple<const T*, size_t> chebyshev_coefficients_i0e_A() {
  /* Chebyshev coefficients for exp(-x) I0(x)
   * in the interval [0,8].
   *
   * lim(x->0){ exp(-x) I0(x) } = 1.
   */
  static const T coefficients[] = {
      -4.41534164647933937950E-18, 3.33079451882223809783E-17,
      -2.43127984654795469359E-16, 1.71539128555513303061E-15,
      -1.16853328779934516808E-14, 7.67618549860493561688E-14,
      -4.85644678311192946090E-13, 2.95505266312963983461E-12,
      -1.72682629144155570723E-11, 9.67580903537323691224E-11,
      -5.18979560163526290666E-10, 2.65982372468238665035E-9,
      -1.30002500998624804212E-8,  6.04699502254191894932E-8,
      -2.67079385394061173391E-7,  1.11738753912010371815E-6,
      -4.41673835845875056359E-6,  1.64484480707288970893E-5,
      -5.75419501008210370398E-5,  1.88502885095841655729E-4,
      -5.76375574538582365885E-4,  1.63947561694133579842E-3,
      -4.32430999505057594430E-3,  1.05464603945949983183E-2,
      -2.37374148058994688156E-2,  4.93052842396707084878E-2,
      -9.49010970480476444210E-2,  1.71620901522208775349E-1,
      -3.04682672343198398683E-1,  6.76795274409476084995E-1};

  return std::make_tuple(coefficients, 30);
}

template <typename T>
DPCPP_BOTH inline std::tuple<const T*, size_t> chebyshev_coefficients_i0e_B() {
  /* Chebyshev coefficients for exp(-x) sqrt(x) I0(x)
   * in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I0(x) } = 1/sqrt(2pi).
   */
  static const T coefficients[] = {
      -7.23318048787475395456E-18, -4.83050448594418207126E-18,
      4.46562142029675999901E-17,  3.46122286769746109310E-17,
      -2.82762398051658348494E-16, -3.42548561967721913462E-16,
      1.77256013305652638360E-15,  3.81168066935262242075E-15,
      -9.55484669882830764870E-15, -4.15056934728722208663E-14,
      1.54008621752140982691E-14,  3.85277838274214270114E-13,
      7.18012445138366623367E-13,  -1.79417853150680611778E-12,
      -1.32158118404477131188E-11, -3.14991652796324136454E-11,
      1.18891471078464383424E-11,  4.94060238822496958910E-10,
      3.39623202570838634515E-9,   2.26666899049817806459E-8,
      2.04891858946906374183E-7,   2.89137052083475648297E-6,
      6.88975834691682398426E-5,   3.36911647825569408990E-3,
      8.04490411014108831608E-1};

  return std::make_tuple(coefficients, 25);
}

template <typename scalar_t>
static inline DPCPP_BOTH scalar_t calc_i0(scalar_t _x) {
  static_assert(
      !std::is_same<scalar_t, Half>() && !std::is_same<scalar_t, BFloat16>(),
      "don't instantiate with low precision type");
  // Upcast input for numerical accuracy purposes
  // Needed for accurate results if input is bfloat16 or float16
  scalar_t x = sycl::abs(_x);

  if (x <= scalar_t{8.0}) {
    auto coeff_pair = chebyshev_coefficients_i0e_A<scalar_t>();
    auto A = std::get<0>(coeff_pair);
    auto len = std::get<1>(coeff_pair);
    scalar_t y = (x / scalar_t{2.0}) - scalar_t{2.0};
    return (sycl::exp(x) * chbevl(y, A, len));
  }

  auto coeff_pair = chebyshev_coefficients_i0e_B<scalar_t>();
  auto B = std::get<0>(coeff_pair);
  auto len = std::get<1>(coeff_pair);
  return (
      sycl::exp(x) * chbevl(scalar_t{32.0} / x - scalar_t{2.0}, B, len) /
      sycl::sqrt(x));
}

/*
 * This function is derived from the implementation of the i0e function in the
 * Cephes Math Library. See note [3-Clause BSD License for the Cephes Math
 * Library].
 *
 * Computes an approximation of the exponentially scaled zeroth order modified
 * Bessel function of the first kind. The approximation is actually two
 * (sub)approximations, both using a Chebyshev polynomial expansion. One
 * approximates the function over [0, 8], and the other over (8, infinity). This
 * function takes the absolute value of all inputs to convert them into the
 * domain of the approximation.
 */
template <typename scalar_t>
static inline DPCPP_BOTH scalar_t calc_i0e(scalar_t _x) {
  scalar_t x = sycl::abs(_x);

  if (x <= scalar_t{8.0}) {
    auto coeff_pair = chebyshev_coefficients_i0e_A<scalar_t>();
#ifdef SYCL_DEVICE_ONLY
    auto A = dpl::get<0>(coeff_pair);
    auto len = dpl::get<1>(coeff_pair);
#else
    auto A = std::get<0>(coeff_pair);
    auto len = std::get<1>(coeff_pair);
#endif
    scalar_t y = (x / scalar_t{2.0}) - scalar_t{2.0};
    return chbevl(y, A, len);
  }

  auto coeff_pair = chebyshev_coefficients_i0e_B<scalar_t>();
#ifdef SYCL_DEVICE_ONLY
  auto B = dpl::get<0>(coeff_pair);
  auto len = dpl::get<1>(coeff_pair);
#else
  auto B = std::get<0>(coeff_pair);
  auto len = std::get<1>(coeff_pair);
#endif
  return chbevl(scalar_t{32.0} / x - scalar_t{2.0}, B, len) / sycl::sqrt(x);
}

// Upcast bfloat16 input to float for numerical accuracy purposes
static inline c10::BFloat16 calc_i0e(c10::BFloat16 a) {
  return calc_i0e(static_cast<float>(a));
}

template <typename T>
DPCPP_BOTH inline typename std::enable_if<
    std::is_same<double, T>::value,
    std::tuple<const T*, size_t>>::type
chebyshev_coefficients_i1e_A() {
  /* Chebyshev coefficients for exp(-x) I1(x)
   * in the interval [0,8].
   *
   * lim(x->0){ exp(-x) I1(x) / x } = 1/2.
   */
  static const T coefficients[] = {
      2.77791411276104639959E-18, -2.11142121435816608115E-17,
      1.55363195773620046921E-16, -1.10559694773538630805E-15,
      7.60068429473540693410E-15, -5.04218550472791168711E-14,
      3.22379336594557470981E-13, -1.98397439776494371520E-12,
      1.17361862988909016308E-11, -6.66348972350202774223E-11,
      3.62559028155211703701E-10, -1.88724975172282928790E-9,
      9.38153738649577178388E-9,  -4.44505912879632808065E-8,
      2.00329475355213526229E-7,  -8.56872026469545474066E-7,
      3.47025130813767847674E-6,  -1.32731636560394358279E-5,
      4.78156510755005422638E-5,  -1.61760815825896745588E-4,
      5.12285956168575772895E-4,  -1.51357245063125314899E-3,
      4.15642294431288815669E-3,  -1.05640848946261981558E-2,
      2.47264490306265168283E-2,  -5.29459812080949914269E-2,
      1.02643658689847095384E-1,  -1.76416518357834055153E-1,
      2.52587186443633654823E-1};

  return std::make_tuple(coefficients, 29);
}

template <typename T>
DPCPP_BOTH inline typename std::
    enable_if<std::is_same<float, T>::value, std::tuple<const T*, size_t>>::type
    chebyshev_coefficients_i1e_A() {
  /* Chebyshev coefficients for exp(-x) I1(x)
   * in the interval [0,8].
   *
   * lim(x->0){ exp(-x) I1(x) / x } = 1/2.
   */
  static const T coeff[] = {
      9.38153738649577178388E-9f,
      -4.44505912879632808065E-8f,
      2.00329475355213526229E-7f,
      -8.56872026469545474066E-7f,
      3.47025130813767847674E-6f,
      -1.32731636560394358279E-5f,
      4.78156510755005422638E-5f,
      -1.61760815825896745588E-4f,
      5.12285956168575772895E-4f,
      -1.51357245063125314899E-3f,
      4.15642294431288815669E-3f,
      -1.05640848946261981558E-2f,
      2.47264490306265168283E-2f,
      -5.29459812080949914269E-2f,
      1.02643658689847095384E-1f,
      -1.76416518357834055153E-1f,
      2.52587186443633654823E-1f};
  return std::make_tuple(coeff, 17);
};

template <typename T>
DPCPP_BOTH inline typename std::enable_if<
    std::is_same<double, T>::value,
    std::tuple<const T*, size_t>>::type
chebyshev_coefficients_i1e_B() {
  /* Chebyshev coefficients for exp(-x) sqrt(x) I1(x)
   * in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I1(x) } = 1/sqrt(2pi).
   */
  static const T coefficients[] = {
      7.51729631084210481353E-18,  4.41434832307170791151E-18,
      -4.65030536848935832153E-17, -3.20952592199342395980E-17,
      2.96262899764595013876E-16,  3.30820231092092828324E-16,
      -1.88035477551078244854E-15, -3.81440307243700780478E-15,
      1.04202769841288027642E-14,  4.27244001671195135429E-14,
      -2.10154184277266431302E-14, -4.08355111109219731823E-13,
      -7.19855177624590851209E-13, 2.03562854414708950722E-12,
      1.41258074366137813316E-11,  3.25260358301548823856E-11,
      -1.89749581235054123450E-11, -5.58974346219658380687E-10,
      -3.83538038596423702205E-9,  -2.63146884688951950684E-8,
      -2.51223623787020892529E-7,  -3.88256480887769039346E-6,
      -1.10588938762623716291E-4,  -9.76109749136146840777E-3,
      7.78576235018280120474E-1};

  return std::make_tuple(coefficients, 25);
}

template <typename T>
DPCPP_BOTH inline typename std::
    enable_if<std::is_same<float, T>::value, std::tuple<const T*, size_t>>::type
    chebyshev_coefficients_i1e_B() {
  /* Chebyshev coefficients for exp(-x) sqrt(x) I1(x)
   * in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I1(x) } = 1/sqrt(2pi).
   */
  static const T coeff[] = {
      -3.83538038596423702205E-9f,
      -2.63146884688951950684E-8f,
      -2.51223623787020892529E-7f,
      -3.88256480887769039346E-6f,
      -1.10588938762623716291E-4f,
      -9.76109749136146840777E-3f,
      7.78576235018280120474E-1f};

  return std::make_tuple(coeff, 7);
};

template <typename scalar_t>
static inline DPCPP_BOTH scalar_t calc_i1(scalar_t _x) {
  const auto x = sycl::abs(_x);
  if (x <= scalar_t{8.0}) {
    auto coeff_pair = chebyshev_coefficients_i1e_A<scalar_t>();
    auto A = std::get<0>(coeff_pair);
    auto len = std::get<1>(coeff_pair);
    scalar_t y = x / scalar_t{2.0} - scalar_t{2.0};
    const scalar_t out = sycl::exp(x) * x * chbevl(y, A, len);
    return (_x < scalar_t{0.0}) ? -out : out;
  }

  auto coeff_pair = chebyshev_coefficients_i1e_B<scalar_t>();
  auto B = std::get<0>(coeff_pair);
  auto len = std::get<1>(coeff_pair);
  const scalar_t out =
      (sycl::exp(x) * chbevl(scalar_t{32.0} / x - scalar_t{2.0}, B, len)) /
      sycl::sqrt(x);
  return (_x < scalar_t{0.0}) ? -out : out;
}

template <typename scalar_t>
static inline DPCPP_BOTH scalar_t calc_i1e(scalar_t _x) {
  const auto x = sycl::abs(_x);
  if (x <= scalar_t{8.0}) {
    auto coeff_pair = chebyshev_coefficients_i1e_A<scalar_t>();
    auto A = std::get<0>(coeff_pair);
    auto len = std::get<1>(coeff_pair);
    const scalar_t y = x / scalar_t{2.0} - scalar_t{2.0};
    const scalar_t out = chbevl(y, A, len) * x;
    return (_x < scalar_t{0.0}) ? -out : out;
  }

  auto coeff_pair = chebyshev_coefficients_i1e_B<scalar_t>();
  auto B = std::get<0>(coeff_pair);
  auto len = std::get<1>(coeff_pair);
  const scalar_t out =
      chbevl(scalar_t{32.0} / x - scalar_t{2.0}, B, len) / sycl::sqrt(x);
  return (_x < scalar_t{0.0}) ? -out : out;
}

template <typename scalar_t>
static inline DPCPP_BOTH scalar_t calc_polygamma(scalar_t x, int n) {
  // already blocked if n <= 1
  const auto one = scalar_t{1};
  return ((n % 2) ? one : -one) *
      sycl::exp(std::lgamma(static_cast<scalar_t>(n) + one)) *
      zeta<scalar_t>(static_cast<scalar_t>(n + 1), x);
}
/*
 * This function is derived from the implementation of the digamma function in
 * the Cephes Math Library. See note [3-Clause BSD License for the Cephes Math
 * Library].
 *
 * Evaluates polynomial of degree N:
 *
 *                     2          N
 * y  =  C  + C x + C x  +...+ C x
 *        0    1     2          N
 *
 * Coefficients are stored in reverse order:
 *
 * coef[0] = C  , ..., coef[N] = C  .
 *            N                   0
 */
template <typename T>
static inline DPCPP_BOTH T polevl(const T x, const T A[], size_t len) {
  T result = 0;
  for (size_t i = 0; i <= len; i++) {
    result = result * x + A[i];
  }
  return result;
}

/*
 * This function is derived from the implementation of the ndtri function in the
 * Cephes Math Library. See note [3-Clause BSD License for the Cephes Math
 * Library].
 *
 * Computes the argument, x, for which the area under the Gaussian probability
 * density function (integrated from minus infinity to x) is equal to y.
 */
template <typename scalar_t>
static inline DPCPP_BOTH scalar_t calc_ndtri(scalar_t y0) {
  /* sqrt(2pi) */
  constexpr scalar_t s2pi = 2.50662827463100050242E0;
  constexpr scalar_t one = 1;
  constexpr scalar_t zero = 0;

  /* approximation for 0 <= |y - 0.5| <= 3/8 */
  static const scalar_t P0[5] = {
      -5.99633501014107895267E1,
      9.80010754185999661536E1,
      -5.66762857469070293439E1,
      1.39312609387279679503E1,
      -1.23916583867381258016E0,
  };
  static const scalar_t Q0[9] = {
      1.00000000000000000000E0,
      1.95448858338141759834E0,
      4.67627912898881538453E0,
      8.63602421390890590575E1,
      -2.25462687854119370527E2,
      2.00260212380060660359E2,
      -8.20372256168333339912E1,
      1.59056225126211695515E1,
      -1.18331621121330003142E0,
  };

  /* Approximation for interval z = sqrt(-2 log y ) between 2 and 8
   * i.e., y between exp(-2) = .135 and exp(-32) = 1.27e-14.
   */
  static const scalar_t P1[9] = {
      4.05544892305962419923E0,
      3.15251094599893866154E1,
      5.71628192246421288162E1,
      4.40805073893200834700E1,
      1.46849561928858024014E1,
      2.18663306850790267539E0,
      -1.40256079171354495875E-1,
      -3.50424626827848203418E-2,
      -8.57456785154685413611E-4,
  };
  static const scalar_t Q1[9] = {
      1.00000000000000000000E0,
      1.57799883256466749731E1,
      4.53907635128879210584E1,
      4.13172038254672030440E1,
      1.50425385692907503408E1,
      2.50464946208309415979E0,
      -1.42182922854787788574E-1,
      -3.80806407691578277194E-2,
      -9.33259480895457427372E-4,
  };

  /* Approximation for interval z = sqrt(-2 log y ) between 8 and 64
   * i.e., y between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
   */

  static const scalar_t P2[9] = {
      3.23774891776946035970E0,
      6.91522889068984211695E0,
      3.93881025292474443415E0,
      1.33303460815807542389E0,
      2.01485389549179081538E-1,
      1.23716634817820021358E-2,
      3.01581553508235416007E-4,
      2.65806974686737550832E-6,
      6.23974539184983293730E-9,
  };
  static const scalar_t Q2[9] = {
      1.00000000000000000000E0,
      6.02427039364742014255E0,
      3.67983563856160859403E0,
      1.37702099489081330271E0,
      2.16236993594496635890E-1,
      1.34204006088543189037E-2,
      3.28014464682127739104E-4,
      2.89247864745380683936E-6,
      6.79019408009981274425E-9,
  };

  if (y0 == zero) {
    return Numerics<scalar_t>::lower_bound();
  }
  if (y0 == one) {
    return Numerics<scalar_t>::upper_bound();
  }
  if (y0 < zero || y0 > one) {
    return std::numeric_limits<scalar_t>::quiet_NaN();
  }
  bool code = true;
  scalar_t y = y0;
  if (y > one - scalar_t{0.13533528323661269189}) { /* 0.135... = exp(-2) */
    y = one - y;
    code = false;
  }
  if (y > scalar_t{0.13533528323661269189}) {
    y = y - scalar_t{0.5};
    const scalar_t y2 = y * y;
    scalar_t x = y + y * (y2 * polevl(y2, P0, 4) / polevl(y2, Q0, 8));
    return (x * s2pi);
  }

  scalar_t x = ::sqrt(scalar_t{-2.0} * ::log(y));
  const scalar_t x0 = x - ::log(x) / x;

  const scalar_t z = one / x;
  scalar_t x1;
  if (x < scalar_t{8.0}) /* y > exp(-32) = 1.2664165549e-14 */
  {
    x1 = z * polevl(z, P1, 8) / polevl(z, Q1, 8);
  } else {
    x1 = z * polevl(z, P2, 8) / polevl(z, Q2, 8);
  }
  x = x0 - x1;
  if (code) {
    x = -x;
  }
  return x;
}
// Upcast bfloat16 input to float for numerical accuracy purposes
static inline c10::BFloat16 calc_ndtri(c10::BFloat16 a) {
  return calc_ndtri(static_cast<float>(a));
}

template <typename T>
static inline DPCPP_BOTH T erfcx_y100(T y100) {
  switch (static_cast<int>(y100)) {
    case 0: {
      T t = 2 * y100 - 1;
      return 0.70878032454106438663e-3 +
          (0.71234091047026302958e-3 +
           (0.35779077297597742384e-5 +
            (0.17403143962587937815e-7 +
             (0.81710660047307788845e-10 +
              (0.36885022360434957634e-12 + 0.15917038551111111111e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 1: {
      T t = 2 * y100 - 3;
      return 0.21479143208285144230e-2 +
          (0.72686402367379996033e-3 +
           (0.36843175430938995552e-5 +
            (0.18071841272149201685e-7 +
             (0.85496449296040325555e-10 +
              (0.38852037518534291510e-12 + 0.16868473576888888889e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 2: {
      T t = 2 * y100 - 5;
      return 0.36165255935630175090e-2 +
          (0.74182092323555510862e-3 +
           (0.37948319957528242260e-5 +
            (0.18771627021793087350e-7 +
             (0.89484715122415089123e-10 +
              (0.40935858517772440862e-12 + 0.17872061464888888889e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 3: {
      T t = 2 * y100 - 7;
      return 0.51154983860031979264e-2 +
          (0.75722840734791660540e-3 +
           (0.39096425726735703941e-5 +
            (0.19504168704300468210e-7 +
             (0.93687503063178993915e-10 +
              (0.43143925959079664747e-12 + 0.18939926435555555556e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 4: {
      T t = 2 * y100 - 9;
      return 0.66457513172673049824e-2 +
          (0.77310406054447454920e-3 +
           (0.40289510589399439385e-5 +
            (0.20271233238288381092e-7 +
             (0.98117631321709100264e-10 +
              (0.45484207406017752971e-12 + 0.20076352213333333333e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 5: {
      T t = 2 * y100 - 11;
      return 0.82082389970241207883e-2 +
          (0.78946629611881710721e-3 +
           (0.41529701552622656574e-5 +
            (0.21074693344544655714e-7 +
             (0.10278874108587317989e-9 +
              (0.47965201390613339638e-12 + 0.21285907413333333333e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 6: {
      T t = 2 * y100 - 13;
      return 0.98039537275352193165e-2 +
          (0.80633440108342840956e-3 +
           (0.42819241329736982942e-5 +
            (0.21916534346907168612e-7 +
             (0.10771535136565470914e-9 +
              (0.50595972623692822410e-12 + 0.22573462684444444444e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 7: {
      T t = 2 * y100 - 15;
      return 0.11433927298290302370e-1 +
          (0.82372858383196561209e-3 +
           (0.44160495311765438816e-5 +
            (0.22798861426211986056e-7 +
             (0.11291291745879239736e-9 +
              (0.53386189365816880454e-12 + 0.23944209546666666667e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 8: {
      T t = 2 * y100 - 17;
      return 0.13099232878814653979e-1 +
          (0.84167002467906968214e-3 +
           (0.45555958988457506002e-5 +
            (0.23723907357214175198e-7 +
             (0.11839789326602695603e-9 +
              (0.56346163067550237877e-12 + 0.25403679644444444444e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 9: {
      T t = 2 * y100 - 19;
      return 0.14800987015587535621e-1 +
          (0.86018092946345943214e-3 +
           (0.47008265848816866105e-5 +
            (0.24694040760197315333e-7 +
             (0.12418779768752299093e-9 +
              (0.59486890370320261949e-12 + 0.26957764568888888889e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 10: {
      T t = 2 * y100 - 21;
      return 0.16540351739394069380e-1 +
          (0.87928458641241463952e-3 +
           (0.48520195793001753903e-5 +
            (0.25711774900881709176e-7 +
             (0.13030128534230822419e-9 +
              (0.62820097586874779402e-12 + 0.28612737351111111111e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 11: {
      T t = 2 * y100 - 23;
      return 0.18318536789842392647e-1 +
          (0.89900542647891721692e-3 +
           (0.50094684089553365810e-5 +
            (0.26779777074218070482e-7 +
             (0.13675822186304615566e-9 +
              (0.66358287745352705725e-12 + 0.30375273884444444444e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 12: {
      T t = 2 * y100 - 25;
      return 0.20136801964214276775e-1 +
          (0.91936908737673676012e-3 +
           (0.51734830914104276820e-5 +
            (0.27900878609710432673e-7 +
             (0.14357976402809042257e-9 +
              (0.70114790311043728387e-12 + 0.32252476000000000000e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 13: {
      T t = 2 * y100 - 27;
      return 0.21996459598282740954e-1 +
          (0.94040248155366777784e-3 +
           (0.53443911508041164739e-5 +
            (0.29078085538049374673e-7 +
             (0.15078844500329731137e-9 +
              (0.74103813647499204269e-12 + 0.34251892320000000000e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 14: {
      T t = 2 * y100 - 29;
      return 0.23898877187226319502e-1 +
          (0.96213386835900177540e-3 +
           (0.55225386998049012752e-5 +
            (0.30314589961047687059e-7 +
             (0.15840826497296335264e-9 +
              (0.78340500472414454395e-12 + 0.36381553564444444445e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 15: {
      T t = 2 * y100 - 31;
      return 0.25845480155298518485e-1 +
          (0.98459293067820123389e-3 +
           (0.57082915920051843672e-5 +
            (0.31613782169164830118e-7 +
             (0.16646478745529630813e-9 +
              (0.82840985928785407942e-12 + 0.38649975768888888890e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 16: {
      T t = 2 * y100 - 33;
      return 0.27837754783474696598e-1 +
          (0.10078108563256892757e-2 +
           (0.59020366493792212221e-5 +
            (0.32979263553246520417e-7 +
             (0.17498524159268458073e-9 +
              (0.87622459124842525110e-12 + 0.41066206488888888890e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 17: {
      T t = 2 * y100 - 35;
      return 0.29877251304899307550e-1 +
          (0.10318204245057349310e-2 +
           (0.61041829697162055093e-5 +
            (0.34414860359542720579e-7 +
             (0.18399863072934089607e-9 +
              (0.92703227366365046533e-12 + 0.43639844053333333334e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 18: {
      T t = 2 * y100 - 37;
      return 0.31965587178596443475e-1 +
          (0.10566560976716574401e-2 +
           (0.63151633192414586770e-5 +
            (0.35924638339521924242e-7 +
             (0.19353584758781174038e-9 +
              (0.98102783859889264382e-12 + 0.46381060817777777779e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 19: {
      T t = 2 * y100 - 39;
      return 0.34104450552588334840e-1 +
          (0.10823541191350532574e-2 +
           (0.65354356159553934436e-5 +
            (0.37512918348533521149e-7 +
             (0.20362979635817883229e-9 +
              (0.10384187833037282363e-11 + 0.49300625262222222221e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 20: {
      T t = 2 * y100 - 41;
      return 0.36295603928292425716e-1 +
          (0.11089526167995268200e-2 +
           (0.67654845095518363577e-5 +
            (0.39184292949913591646e-7 +
             (0.21431552202133775150e-9 +
              (0.10994259106646731797e-11 + 0.52409949102222222221e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 21: {
      T t = 2 * y100 - 43;
      return 0.38540888038840509795e-1 +
          (0.11364917134175420009e-2 +
           (0.70058230641246312003e-5 +
            (0.40943644083718586939e-7 +
             (0.22563034723692881631e-9 +
              (0.11642841011361992885e-11 + 0.55721092871111111110e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 22: {
      T t = 2 * y100 - 45;
      return 0.40842225954785960651e-1 +
          (0.11650136437945673891e-2 +
           (0.72569945502343006619e-5 +
            (0.42796161861855042273e-7 +
             (0.23761401711005024162e-9 +
              (0.12332431172381557035e-11 + 0.59246802364444444445e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 23: {
      T t = 2 * y100 - 47;
      return 0.43201627431540222422e-1 +
          (0.11945628793917272199e-2 +
           (0.75195743532849206263e-5 +
            (0.44747364553960993492e-7 +
             (0.25030885216472953674e-9 +
              (0.13065684400300476484e-11 + 0.63000532853333333334e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 24: {
      T t = 2 * y100 - 49;
      return 0.45621193513810471438e-1 +
          (0.12251862608067529503e-2 +
           (0.77941720055551920319e-5 +
            (0.46803119830954460212e-7 +
             (0.26375990983978426273e-9 +
              (0.13845421370977119765e-11 + 0.66996477404444444445e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 25: {
      T t = 2 * y100 - 51;
      return 0.48103121413299865517e-1 +
          (0.12569331386432195113e-2 +
           (0.80814333496367673980e-5 +
            (0.48969667335682018324e-7 +
             (0.27801515481905748484e-9 +
              (0.14674637611609884208e-11 + 0.71249589351111111110e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 26: {
      T t = 2 * y100 - 53;
      return 0.50649709676983338501e-1 +
          (0.12898555233099055810e-2 +
           (0.83820428414568799654e-5 +
            (0.51253642652551838659e-7 +
             (0.29312563849675507232e-9 +
              (0.15556512782814827846e-11 + 0.75775607822222222221e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 27: {
      T t = 2 * y100 - 55;
      return 0.53263363664388864181e-1 +
          (0.13240082443256975769e-2 +
           (0.86967260015007658418e-5 +
            (0.53662102750396795566e-7 +
             (0.30914568786634796807e-9 +
              (0.16494420240828493176e-11 + 0.80591079644444444445e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 28: {
      T t = 2 * y100 - 57;
      return 0.55946601353500013794e-1 +
          (0.13594491197408190706e-2 +
           (0.90262520233016380987e-5 +
            (0.56202552975056695376e-7 +
             (0.32613310410503135996e-9 +
              (0.17491936862246367398e-11 + 0.85713381688888888890e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 29: {
      T t = 2 * y100 - 59;
      return 0.58702059496154081813e-1 +
          (0.13962391363223647892e-2 +
           (0.93714365487312784270e-5 +
            (0.58882975670265286526e-7 +
             (0.34414937110591753387e-9 +
              (0.18552853109751857859e-11 + 0.91160736711111111110e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 30: {
      T t = 2 * y100 - 61;
      return 0.61532500145144778048e-1 +
          (0.14344426411912015247e-2 +
           (0.97331446201016809696e-5 +
            (0.61711860507347175097e-7 +
             (0.36325987418295300221e-9 +
              (0.19681183310134518232e-11 + 0.96952238400000000000e-14 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 31: {
      T t = 2 * y100 - 63;
      return 0.64440817576653297993e-1 +
          (0.14741275456383131151e-2 +
           (0.10112293819576437838e-4 +
            (0.64698236605933246196e-7 +
             (0.38353412915303665586e-9 +
              (0.20881176114385120186e-11 + 0.10310784480000000000e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 32: {
      T t = 2 * y100 - 65;
      return 0.67430045633130393282e-1 +
          (0.15153655418916540370e-2 +
           (0.10509857606888328667e-4 +
            (0.67851706529363332855e-7 +
             (0.40504602194811140006e-9 +
              (0.22157325110542534469e-11 + 0.10964842115555555556e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 33: {
      T t = 2 * y100 - 67;
      return 0.70503365513338850709e-1 +
          (0.15582323336495709827e-2 +
           (0.10926868866865231089e-4 +
            (0.71182482239613507542e-7 +
             (0.42787405890153386710e-9 +
              (0.23514379522274416437e-11 + 0.11659571751111111111e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 34: {
      T t = 2 * y100 - 69;
      return 0.73664114037944596353e-1 +
          (0.16028078812438820413e-2 +
           (0.11364423678778207991e-4 +
            (0.74701423097423182009e-7 +
             (0.45210162777476488324e-9 +
              (0.24957355004088569134e-11 + 0.12397238257777777778e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 35: {
      T t = 2 * y100 - 71;
      return 0.76915792420819562379e-1 +
          (0.16491766623447889354e-2 +
           (0.11823685320041302169e-4 +
            (0.78420075993781544386e-7 +
             (0.47781726956916478925e-9 +
              (0.26491544403815724749e-11 + 0.13180196462222222222e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 36: {
      T t = 2 * y100 - 73;
      return 0.80262075578094612819e-1 +
          (0.16974279491709504117e-2 +
           (0.12305888517309891674e-4 +
            (0.82350717698979042290e-7 +
             (0.50511496109857113929e-9 +
              (0.28122528497626897696e-11 + 0.14010889635555555556e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 37: {
      T t = 2 * y100 - 75;
      return 0.83706822008980357446e-1 +
          (0.17476561032212656962e-2 +
           (0.12812343958540763368e-4 +
            (0.86506399515036435592e-7 +
             (0.53409440823869467453e-9 +
              (0.29856186620887555043e-11 + 0.14891851591111111111e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 38: {
      T t = 2 * y100 - 77;
      return 0.87254084284461718231e-1 +
          (0.17999608886001962327e-2 +
           (0.13344443080089492218e-4 +
            (0.90900994316429008631e-7 +
             (0.56486134972616465316e-9 +
              (0.31698707080033956934e-11 + 0.15825697795555555556e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 39: {
      T t = 2 * y100 - 79;
      return 0.90908120182172748487e-1 +
          (0.18544478050657699758e-2 +
           (0.13903663143426120077e-4 +
            (0.95549246062549906177e-7 +
             (0.59752787125242054315e-9 +
              (0.33656597366099099413e-11 + 0.16815130613333333333e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 40: {
      T t = 2 * y100 - 81;
      return 0.94673404508075481121e-1 +
          (0.19112284419887303347e-2 +
           (0.14491572616545004930e-4 +
            (0.10046682186333613697e-6 +
             (0.63221272959791000515e-9 +
              (0.35736693975589130818e-11 + 0.17862931591111111111e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 41: {
      T t = 2 * y100 - 83;
      return 0.98554641648004456555e-1 +
          (0.19704208544725622126e-2 +
           (0.15109836875625443935e-4 +
            (0.10567036667675984067e-6 +
             (0.66904168640019354565e-9 +
              (0.37946171850824333014e-11 + 0.18971959040000000000e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 42: {
      T t = 2 * y100 - 85;
      return 0.10255677889470089531e0 +
          (0.20321499629472857418e-2 +
           (0.15760224242962179564e-4 +
            (0.11117756071353507391e-6 +
             (0.70814785110097658502e-9 +
              (0.40292553276632563925e-11 + 0.20145143075555555556e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 43: {
      T t = 2 * y100 - 87;
      return 0.10668502059865093318e0 +
          (0.20965479776148731610e-2 +
           (0.16444612377624983565e-4 +
            (0.11700717962026152749e-6 +
             (0.74967203250938418991e-9 +
              (0.42783716186085922176e-11 + 0.21385479360000000000e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 44: {
      T t = 2 * y100 - 89;
      return 0.11094484319386444474e0 +
          (0.21637548491908170841e-2 +
           (0.17164995035719657111e-4 +
            (0.12317915750735938089e-6 +
             (0.79376309831499633734e-9 +
              (0.45427901763106353914e-11 + 0.22696025653333333333e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 45: {
      T t = 2 * y100 - 91;
      return 0.11534201115268804714e0 +
          (0.22339187474546420375e-2 +
           (0.17923489217504226813e-4 +
            (0.12971465288245997681e-6 +
             (0.84057834180389073587e-9 +
              (0.48233721206418027227e-11 + 0.24079890062222222222e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 46: {
      T t = 2 * y100 - 93;
      return 0.11988259392684094740e0 +
          (0.23071965691918689601e-2 +
           (0.18722342718958935446e-4 +
            (0.13663611754337957520e-6 +
             (0.89028385488493287005e-9 +
              (0.51210161569225846701e-11 + 0.25540227111111111111e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 47: {
      T t = 2 * y100 - 95;
      return 0.12457298393509812907e0 +
          (0.23837544771809575380e-2 +
           (0.19563942105711612475e-4 +
            (0.14396736847739470782e-6 +
             (0.94305490646459247016e-9 +
              (0.54366590583134218096e-11 + 0.27080225920000000000e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 48: {
      T t = 2 * y100 - 97;
      return 0.12941991566142438816e0 +
          (0.24637684719508859484e-2 +
           (0.20450821127475879816e-4 +
            (0.15173366280523906622e-6 +
             (0.99907632506389027739e-9 +
              (0.57712760311351625221e-11 + 0.28703099555555555556e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 49: {
      T t = 2 * y100 - 99;
      return 0.13443048593088696613e0 +
          (0.25474249981080823877e-2 +
           (0.21385669591362915223e-4 +
            (0.15996177579900443030e-6 +
             (0.10585428844575134013e-8 +
              (0.61258809536787882989e-11 + 0.30412080142222222222e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 50: {
      T t = 2 * y100 - 101;
      return 0.13961217543434561353e0 +
          (0.26349215871051761416e-2 +
           (0.22371342712572567744e-4 +
            (0.16868008199296822247e-6 +
             (0.11216596910444996246e-8 +
              (0.65015264753090890662e-11 + 0.32210394506666666666e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 51: {
      T t = 2 * y100 - 103;
      return 0.14497287157673800690e0 +
          (0.27264675383982439814e-2 +
           (0.23410870961050950197e-4 +
            (0.17791863939526376477e-6 +
             (0.11886425714330958106e-8 +
              (0.68993039665054288034e-11 + 0.34101266222222222221e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 52: {
      T t = 2 * y100 - 105;
      return 0.15052089272774618151e0 +
          (0.28222846410136238008e-2 +
           (0.24507470422713397006e-4 +
            (0.18770927679626136909e-6 +
             (0.12597184587583370712e-8 +
              (0.73203433049229821618e-11 + 0.36087889048888888890e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 53: {
      T t = 2 * y100 - 107;
      return 0.15626501395774612325e0 +
          (0.29226079376196624949e-2 +
           (0.25664553693768450545e-4 +
            (0.19808568415654461964e-6 +
             (0.13351257759815557897e-8 +
              (0.77658124891046760667e-11 + 0.38173420035555555555e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 54: {
      T t = 2 * y100 - 109;
      return 0.16221449434620737567e0 +
          (0.30276865332726475672e-2 +
           (0.26885741326534564336e-4 +
            (0.20908350604346384143e-6 +
             (0.14151148144240728728e-8 +
              (0.82369170665974313027e-11 + 0.40360957457777777779e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 55: {
      T t = 2 * y100 - 111;
      return 0.16837910595412130659e0 +
          (0.31377844510793082301e-2 +
           (0.28174873844911175026e-4 +
            (0.22074043807045782387e-6 +
             (0.14999481055996090039e-8 +
              (0.87348993661930809254e-11 + 0.42653528977777777779e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 56: {
      T t = 2 * y100 - 113;
      return 0.17476916455659369953e0 +
          (0.32531815370903068316e-2 +
           (0.29536024347344364074e-4 +
            (0.23309632627767074202e-6 +
             (0.15899007843582444846e-8 +
              (0.92610375235427359475e-11 + 0.45054073102222222221e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 57: {
      T t = 2 * y100 - 115;
      return 0.18139556223643701364e0 +
          (0.33741744168096996041e-2 +
           (0.30973511714709500836e-4 +
            (0.24619326937592290996e-6 +
             (0.16852609412267750744e-8 +
              (0.98166442942854895573e-11 + 0.47565418097777777779e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 58: {
      T t = 2 * y100 - 117;
      return 0.18826980194443664549e0 +
          (0.35010775057740317997e-2 +
           (0.32491914440014267480e-4 +
            (0.26007572375886319028e-6 +
             (0.17863299617388376116e-8 +
              (0.10403065638343878679e-10 + 0.50190265831111111110e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 59: {
      T t = 2 * y100 - 119;
      return 0.19540403413693967350e0 +
          (0.36342240767211326315e-2 +
           (0.34096085096200907289e-4 +
            (0.27479061117017637474e-6 +
             (0.18934228504790032826e-8 +
              (0.11021679075323598664e-10 + 0.52931171733333333334e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 60: {
      T t = 2 * y100 - 121;
      return 0.20281109560651886959e0 +
          (0.37739673859323597060e-2 +
           (0.35791165457592409054e-4 +
            (0.29038742889416172404e-6 +
             (0.20068685374849001770e-8 +
              (0.11673891799578381999e-10 + 0.55790523093333333334e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 61: {
      T t = 2 * y100 - 123;
      return 0.21050455062669334978e0 +
          (0.39206818613925652425e-2 +
           (0.37582602289680101704e-4 +
            (0.30691836231886877385e-6 +
             (0.21270101645763677824e-8 +
              (0.12361138551062899455e-10 + 0.58770520160000000000e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 62: {
      T t = 2 * y100 - 125;
      return 0.21849873453703332479e0 +
          (0.40747643554689586041e-2 +
           (0.39476163820986711501e-4 +
            (0.32443839970139918836e-6 +
             (0.22542053491518680200e-8 +
              (0.13084879235290858490e-10 + 0.61873153262222222221e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 63: {
      T t = 2 * y100 - 127;
      return 0.22680879990043229327e0 +
          (0.42366354648628516935e-2 +
           (0.41477956909656896779e-4 +
            (0.34300544894502810002e-6 +
             (0.23888264229264067658e-8 +
              (0.13846596292818514601e-10 + 0.65100183751111111110e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 64: {
      T t = 2 * y100 - 129;
      return 0.23545076536988703937e0 +
          (0.44067409206365170888e-2 +
           (0.43594444916224700881e-4 +
            (0.36268045617760415178e-6 +
             (0.25312606430853202748e-8 +
              (0.14647791812837903061e-10 + 0.68453122631111111110e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 65: {
      T t = 2 * y100 - 131;
      return 0.24444156740777432838e0 +
          (0.45855530511605787178e-2 +
           (0.45832466292683085475e-4 +
            (0.38352752590033030472e-6 +
             (0.26819103733055603460e-8 +
              (0.15489984390884756993e-10 + 0.71933206364444444445e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 66: {
      T t = 2 * y100 - 133;
      return 0.25379911500634264643e0 +
          (0.47735723208650032167e-2 +
           (0.48199253896534185372e-4 +
            (0.40561404245564732314e-6 +
             (0.28411932320871165585e-8 +
              (0.16374705736458320149e-10 + 0.75541379822222222221e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 67: {
      T t = 2 * y100 - 135;
      return 0.26354234756393613032e0 +
          (0.49713289477083781266e-2 +
           (0.50702455036930367504e-4 +
            (0.42901079254268185722e-6 +
             (0.30095422058900481753e-8 +
              (0.17303497025347342498e-10 + 0.79278273368888888890e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 68: {
      T t = 2 * y100 - 137;
      return 0.27369129607732343398e0 +
          (0.51793846023052643767e-2 +
           (0.53350152258326602629e-4 +
            (0.45379208848865015485e-6 +
             (0.31874057245814381257e-8 +
              (0.18277905010245111046e-10 + 0.83144182364444444445e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 69: {
      T t = 2 * y100 - 139;
      return 0.28426714781640316172e0 +
          (0.53983341916695141966e-2 +
           (0.56150884865255810638e-4 +
            (0.48003589196494734238e-6 +
             (0.33752476967570796349e-8 +
              (0.19299477888083469086e-10 + 0.87139049137777777779e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 70: {
      T t = 2 * y100 - 141;
      return 0.29529231465348519920e0 +
          (0.56288077305420795663e-2 +
           (0.59113671189913307427e-4 +
            (0.50782393781744840482e-6 +
             (0.35735475025851713168e-8 +
              (0.20369760937017070382e-10 + 0.91262442613333333334e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 71: {
      T t = 2 * y100 - 143;
      return 0.30679050522528838613e0 +
          (0.58714723032745403331e-2 +
           (0.62248031602197686791e-4 +
            (0.53724185766200945789e-6 +
             (0.37827999418960232678e-8 +
              (0.21490291930444538307e-10 + 0.95513539182222222221e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 72: {
      T t = 2 * y100 - 145;
      return 0.31878680111173319425e0 +
          (0.61270341192339103514e-2 +
           (0.65564012259707640976e-4 +
            (0.56837930287837738996e-6 +
             (0.40035151353392378882e-8 +
              (0.22662596341239294792e-10 + 0.99891109760000000000e-13 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 73: {
      T t = 2 * y100 - 147;
      return 0.33130773722152622027e0 +
          (0.63962406646798080903e-2 +
           (0.69072209592942396666e-4 +
            (0.60133006661885941812e-6 +
             (0.42362183765883466691e-8 +
              (0.23888182347073698382e-10 + 0.10439349811555555556e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 74: {
      T t = 2 * y100 - 149;
      return 0.34438138658041336523e0 +
          (0.66798829540414007258e-2 +
           (0.72783795518603561144e-4 +
            (0.63619220443228800680e-6 +
             (0.44814499336514453364e-8 +
              (0.25168535651285475274e-10 + 0.10901861383111111111e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 75: {
      T t = 2 * y100 - 151;
      return 0.35803744972380175583e0 +
          (0.69787978834882685031e-2 +
           (0.76710543371454822497e-4 +
            (0.67306815308917386747e-6 +
             (0.47397647975845228205e-8 +
              (0.26505114141143050509e-10 + 0.11376390933333333333e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 76: {
      T t = 2 * y100 - 153;
      return 0.37230734890119724188e0 +
          (0.72938706896461381003e-2 +
           (0.80864854542670714092e-4 +
            (0.71206484718062688779e-6 +
             (0.50117323769745883805e-8 +
              (0.27899342394100074165e-10 + 0.11862637614222222222e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 77: {
      T t = 2 * y100 - 155;
      return 0.38722432730555448223e0 +
          (0.76260375162549802745e-2 +
           (0.85259785810004603848e-4 +
            (0.75329383305171327677e-6 +
             (0.52979361368388119355e-8 +
              (0.29352606054164086709e-10 + 0.12360253370666666667e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 78: {
      T t = 2 * y100 - 157;
      return 0.40282355354616940667e0 +
          (0.79762880915029728079e-2 +
           (0.89909077342438246452e-4 +
            (0.79687137961956194579e-6 +
             (0.55989731807360403195e-8 +
              (0.30866246101464869050e-10 + 0.12868841946666666667e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 79: {
      T t = 2 * y100 - 159;
      return 0.41914223158913787649e0 +
          (0.83456685186950463538e-2 +
           (0.94827181359250161335e-4 +
            (0.84291858561783141014e-6 +
             (0.59154537751083485684e-8 +
              (0.32441553034347469291e-10 + 0.13387957943111111111e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 80: {
      T t = 2 * y100 - 161;
      return 0.43621971639463786896e0 +
          (0.87352841828289495773e-2 +
           (0.10002929142066799966e-3 +
            (0.89156148280219880024e-6 +
             (0.62480008150788597147e-8 +
              (0.34079760983458878910e-10 + 0.13917107176888888889e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 81: {
      T t = 2 * y100 - 163;
      return 0.45409763548534330981e0 +
          (0.91463027755548240654e-2 +
           (0.10553137232446167258e-3 +
            (0.94293113464638623798e-6 +
             (0.65972492312219959885e-8 +
              (0.35782041795476563662e-10 + 0.14455745872000000000e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 82: {
      T t = 2 * y100 - 165;
      return 0.47282001668512331468e0 +
          (0.95799574408860463394e-2 +
           (0.11135019058000067469e-3 +
            (0.99716373005509038080e-6 +
             (0.69638453369956970347e-8 +
              (0.37549499088161345850e-10 + 0.15003280712888888889e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 83: {
      T t = 2 * y100 - 167;
      return 0.49243342227179841649e0 +
          (0.10037550043909497071e-1 +
           (0.11750334542845234952e-3 +
            (0.10544006716188967172e-5 +
             (0.73484461168242224872e-8 +
              (0.39383162326435752965e-10 + 0.15559069118222222222e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 84: {
      T t = 2 * y100 - 169;
      return 0.51298708979209258326e0 +
          (0.10520454564612427224e-1 +
           (0.12400930037494996655e-3 +
            (0.11147886579371265246e-5 +
             (0.77517184550568711454e-8 +
              (0.41283980931872622611e-10 + 0.16122419680000000000e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 85: {
      T t = 2 * y100 - 171;
      return 0.53453307979101369843e0 +
          (0.11030120618800726938e-1 +
           (0.13088741519572269581e-3 +
            (0.11784797595374515432e-5 +
             (0.81743383063044825400e-8 +
              (0.43252818449517081051e-10 + 0.16692592640000000000e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 86: {
      T t = 2 * y100 - 173;
      return 0.55712643071169299478e0 +
          (0.11568077107929735233e-1 +
           (0.13815797838036651289e-3 +
            (0.12456314879260904558e-5 +
             (0.86169898078969313597e-8 +
              (0.45290446811539652525e-10 + 0.17268801084444444444e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 87: {
      T t = 2 * y100 - 175;
      return 0.58082532122519320968e0 +
          (0.12135935999503877077e-1 +
           (0.14584223996665838559e-3 +
            (0.13164068573095710742e-5 +
             (0.90803643355106020163e-8 +
              (0.47397540713124619155e-10 + 0.17850211608888888889e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 88: {
      T t = 2 * y100 - 177;
      return 0.60569124025293375554e0 +
          (0.12735396239525550361e-1 +
           (0.15396244472258863344e-3 +
            (0.13909744385382818253e-5 +
             (0.95651595032306228245e-8 +
              (0.49574672127669041550e-10 + 0.18435945564444444444e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 89: {
      T t = 2 * y100 - 179;
      return 0.63178916494715716894e0 +
          (0.13368247798287030927e-1 +
           (0.16254186562762076141e-3 +
            (0.14695084048334056083e-5 +
             (0.10072078109604152350e-7 +
              (0.51822304995680707483e-10 + 0.19025081422222222222e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 90: {
      T t = 2 * y100 - 181;
      return 0.65918774689725319200e0 +
          (0.14036375850601992063e-1 +
           (0.17160483760259706354e-3 +
            (0.15521885688723188371e-5 +
             (0.10601827031535280590e-7 +
              (0.54140790105837520499e-10 + 0.19616655146666666667e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 91: {
      T t = 2 * y100 - 183;
      return 0.68795950683174433822e0 +
          (0.14741765091365869084e-1 +
           (0.18117679143520433835e-3 +
            (0.16392004108230585213e-5 +
             (0.11155116068018043001e-7 +
              (0.56530360194925690374e-10 + 0.20209663662222222222e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 92: {
      T t = 2 * y100 - 185;
      return 0.71818103808729967036e0 +
          (0.15486504187117112279e-1 +
           (0.19128428784550923217e-3 +
            (0.17307350969359975848e-5 +
             (0.11732656736113607751e-7 +
              (0.58991125287563833603e-10 + 0.20803065333333333333e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 93: {
      T t = 2 * y100 - 187;
      return 0.74993321911726254661e0 +
          (0.16272790364044783382e-1 +
           (0.20195505163377912645e-3 +
            (0.18269894883203346953e-5 +
             (0.12335161021630225535e-7 +
              (0.61523068312169087227e-10 + 0.21395783431111111111e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 94: {
      T t = 2 * y100 - 189;
      return 0.78330143531283492729e0 +
          (0.17102934132652429240e-1 +
           (0.21321800585063327041e-3 +
            (0.19281661395543913713e-5 +
             (0.12963340087354341574e-7 +
              (0.64126040998066348872e-10 + 0.21986708942222222222e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 95: {
      T t = 2 * y100 - 191;
      return 0.81837581041023811832e0 +
          (0.17979364149044223802e-1 +
           (0.22510330592753129006e-3 +
            (0.20344732868018175389e-5 +
             (0.13617902941839949718e-7 +
              (0.66799760083972474642e-10 + 0.22574701262222222222e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 96: {
      T t = 2 * y100 - 193;
      return 0.85525144775685126237e0 +
          (0.18904632212547561026e-1 +
           (0.23764237370371255638e-3 +
            (0.21461248251306387979e-5 +
             (0.14299555071870523786e-7 +
              (0.69543803864694171934e-10 + 0.23158593688888888889e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 97: {
      T t = 2 * y100 - 195;
      return 0.89402868170849933734e0 +
          (0.19881418399127202569e-1 +
           (0.25086793128395995798e-3 +
            (0.22633402747585233180e-5 +
             (0.15008997042116532283e-7 +
              (0.72357609075043941261e-10 + 0.23737194737777777778e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 98: {
      T t = 2 * y100 - 197;
      return 0.93481333942870796363e0 +
          (0.20912536329780368893e-1 +
           (0.26481403465998477969e-3 +
            (0.23863447359754921676e-5 +
             (0.15746923065472184451e-7 +
              (0.75240468141720143653e-10 + 0.24309291271111111111e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
    case 99: {
      T t = 2 * y100 - 199;
      return 0.97771701335885035464e0 +
          (0.22000938572830479551e-1 +
           (0.27951610702682383001e-3 +
            (0.25153688325245314530e-5 +
             (0.16514019547822821453e-7 +
              (0.78191526829368231251e-10 + 0.24873652355555555556e-12 * t) *
                  t) *
                 t) *
                t) *
               t) *
          t;
    }
  }
  // we only get here if y = 1, i.e. |x| < 4*eps, in which case
  // erfcx is within 1e-15 of 1..
  return 1.0;
}

template <typename scalar_t>
static inline DPCPP_BOTH scalar_t calc_erfcx(scalar_t x) {
  if (at::_isnan(x)) {
    return x;
  }

  if (x >= 0) {
    if (x > 50) { // continued-fraction expansion is faster
      const scalar_t ispi = 0.56418958354775628694807945156; // 1 / sqrt(pi)
      if (x > 5e7) { // 1-term expansion, important to avoid overflow
        return ispi / x;
      }
      /* 5-term expansion (rely on compiler for CSE), simplified from:
                ispi / (x+0.5/(x+1/(x+1.5/(x+2/x))))  */
      return ispi * ((x * x) * (x * x + 4.5) + 2) /
          (x * ((x * x) * (x * x + 5) + 3.75));
    }
    return erfcx_y100(400 / (4 + x));
  } else {
    if (x < -26.7) {
      return Numerics<scalar_t>::upper_bound();
    } else if (x < -6.1) {
      return 2 * exp(x * x);
    } else {
      return 2 * exp(x * x) - erfcx_y100(400 / (4 - x));
    }
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at
