#pragma once

#include <cmath>
#include <iostream>

template < typename T, int ... n >
struct tensor;

template < typename T, int m, int ... n >
struct tensor< T, m, n ... > {
  template < typename I, typename ... JKL >
  constexpr auto & operator()(I i, JKL ... jkl) { return values[i](jkl ...); }
  template < typename I, typename ... JKL >
  constexpr auto & operator()(I i, JKL ... jkl) const { return values[i](jkl ...); }

  constexpr auto & operator[](int i) { return values[i]; }
  constexpr const auto & operator[](int i) const { return values[i]; }
  tensor < T, n ... > values[m];
};

template < typename T, int m >
struct tensor< T, m > {
  template < typename S >
  constexpr auto & operator()(S i) { return values[i]; }
  template < typename S >
  constexpr auto & operator()(S i) const { return values[i]; }
  constexpr auto & operator[](int i) { return values[i]; }
  constexpr const auto & operator[](int i) const { return values[i]; }
  T values[m];
};

template < int n >
using vec = tensor<double, n>;

using vec2 = tensor<double, 2>;
using vec3 = tensor<double, 3>;
using mat3 = tensor<double, 3,3>;

using vec2f = tensor<float, 2>;
using vec3f = tensor<float, 3>;
using vec4f = tensor<float, 4>;
using mat2f = tensor<float, 2,2>;
using mat3f = tensor<float, 3,3>;
using mat4x2f = tensor<float, 4,2>;
using mat2x4f = tensor<float, 2,4>;

template <typename T, int... n>
auto operator==(const tensor<T, n...> & a, const tensor<T, n...> & b) {
  for (int i = 0; i < (n * ...); i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

template <typename T, int... n>
auto operator*=(const tensor<T, n...> & a, const T b) {
  for (int i = 0; i < (n * ...); i++) {
    ((T*)a.values)[i] *= b;
  }
}

template <typename T, int... n>
auto operator*(const tensor<T, n...> a, const T b) {
  tensor<T, n...> c{};
  for (int i = 0; i < (n * ...); i++) {
    ((T*)c.values)[i] = ((T*)a.values)[i] * b;
  }
  return c;
}

template <typename T, int... n>
auto operator*(const double a, const tensor<T, n...>& b) {
  tensor<T, n...> c{};
  for (int i = 0; i < (n * ...); i++) {
    ((T*)c.values)[i] = a * ((T*)b.values)[i];
  }
  return c;
}

template <typename T, int... n>
auto operator*(const tensor<T, n...>& A, const tensor<T, n...>& B) {
  tensor<T, n...> C{};
  for (int i = 0; i < (n * ...); i++) {
    ((T*)C.values)[i] = ((T*)A.values)[i] * ((T*)B.values)[i];
  }
  return C;
}

template <typename T, int... n>
auto operator/(const tensor<T, n...> a, const double b) {
  tensor<T, n...> c{};
  for (int i = 0; i < (n * ...); i++) {
    ((T*)c.values)[i] = ((T*)a.values)[i] / b;
  }
  return c;
}

template <typename T, int... n>
auto operator/(const T a, const tensor<T, n...>& b) {
  tensor<T, n...> c{};
  for (int i = 0; i < (n * ...); i++) {
    ((T*)c.values)[i] = a / ((T*)b.values)[i];
  }
  return c;
}

template <typename T, int n>
auto operator/(const tensor<T,n> & a, const tensor<T, n>& b) {
  tensor<T, n> c{};
  for (int i = 0; i < n; i++) {
    c[i] = a[i] / b[i];
  }
  return c;
}

template <typename T, int... n>
auto operator+(const tensor<T, n...>& A, const tensor<T, n...>& B) {
  tensor<T, n...> C{};
  for (int i = 0; i < (n * ...); i++) {
    ((T*)C.values)[i] = ((T*)A.values)[i] + ((T*)B.values)[i];
  }
  return C;
}

template <typename T, int... n>
auto operator-(const tensor<T,n...>& A) {
  tensor<T,n...> minusA{};
  for (int i = 0; i < (n * ...); i++) {
    ((T*)minusA.values)[i] = -((T*)A.values)[i];
  }
  return minusA;
}

template <typename T, int... n>
auto operator-(const tensor<T,n...>& A, const tensor<T,n...>& B) {
  tensor<T,n...> C{};
  for (int i = 0; i < (n * ...); i++) {
    ((T*)C.values)[i] = ((T*)A.values)[i] - ((T*)B.values)[i];
  }
  return C;
}

template <typename T, int... n>
auto operator+=(tensor<T,n...>& A, const tensor<T,n...>& B) {
  for (int i = 0; i < (n * ...); i++) {
    ((T*)A.values)[i] += ((T*)B.values)[i];
  }
}

template <typename T, int... n>
auto operator-=(tensor<T,n...>& A, const tensor<T,n...>& B) {
  for (int i = 0; i < (n * ...); i++) {
    ((T*)A.values)[i] -= ((T*)B.values)[i];
  }
}

template <typename T, int dim>
constexpr tensor<T, dim, dim> Identity() {
  tensor<T, dim, dim> I{};
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      I(i, j) = (i == j);
    }
  }
  return I;
}

template <typename T, int dim>
constexpr tensor<T, dim, dim> eye = Identity<T, dim>();

template <typename T, int dim>
constexpr tensor<T, dim, dim> sym(tensor<T, dim, dim> A) {
  tensor<T, dim, dim> A_sym{};
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      A_sym(i, j) = 0.5 * (A(i, j) + A(j, i));
    }
  }
  return A_sym;
}

template <typename T, int dim>
constexpr T tr(tensor<T, dim, dim> A) {
  T trA = 0.0;
  for (int i = 0; i < dim; i++) {
    trA += A(i, i);
  }
  return trA;
}

template < typename T >
constexpr T det(const tensor<T, 2, 2> A) {
  return A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
}

template < typename T >
constexpr T det(const tensor<T, 3, 3>& A) {
  return A(0, 0) * A(1, 1) * A(2, 2) + A(0, 1) * A(1, 2) * A(2, 0) +
         A(0, 2) * A(1, 0) * A(2, 1) - A(0, 0) * A(1, 2) * A(2, 1) -
         A(0, 1) * A(1, 0) * A(2, 2) - A(0, 2) * A(1, 1) * A(2, 0);
}

template <typename T, int n>
constexpr tensor<T, n> linear_solve(tensor<T, n, n> A,
                                    tensor<T, n> b) {
  constexpr auto abs = [](T x) { return (x < 0) ? -x : x; };

  tensor<T, n> x{};

  for (int i = 0; i < n; i++) {
    // Search for maximum in this column
    T max_val = abs(A(i, i));

    int max_row = i;
    for (int j = i + 1; j < n; j++) {
      if (abs(A(j, i)) > max_val) {
        max_val = abs(A(j, i));
        max_row = j;
      }
    }

    // Swap maximum row with current row
    T tmp = b(max_row);
    b(max_row) = b(i);
    b(i) = tmp;
    for (int j = 0; j < n; j++) {
      tmp = A(max_row, j);
      A(max_row, j) = A(i, j);
      A(i, j) = tmp;
    }

    // zero entries below in this column
    for (int j = i + 1; j < n; j++) {
      T c = -A(j, i) / A(i, i);

      for (int k = i + 1; k < n; k++) {
        A(j, k) += c * A(i, k);
      }
      b(j) += c * b(i);
      A(j, i) = 0;
    }
  }

  // Solve equation Ax=b for an upper triangular matrix A
  for (int i = n - 1; i >= 0; i--) {
    x(i) = b(i) / A(i, i);
    for (int j = i - 1; j >= 0; j--) {
      b(j) -= A(j, i) * x(i);
    }
  }

  return x;
}

template < typename T >
constexpr tensor<T, 2, 2> inv(const tensor<T, 2, 2>& A) {
  T inv_detA(1.0 / det(A));

  tensor<T, 2, 2> invA{};

  invA(0, 0) = A(1, 1) * inv_detA;
  invA(0, 1) = -A(0, 1) * inv_detA;
  invA(1, 0) = -A(1, 0) * inv_detA;
  invA(1, 1) = A(0, 0) * inv_detA;

  return invA;
}

template < typename T >
constexpr tensor<T, 3, 3> inv(const tensor<T, 3, 3>& A) {
  T inv_detA(1.0 / det(A));

  tensor<T, 3, 3> invA{};

  invA(0, 0) = (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1)) * inv_detA;
  invA(0, 1) = (A(0, 2) * A(2, 1) - A(0, 1) * A(2, 2)) * inv_detA;
  invA(0, 2) = (A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1)) * inv_detA;
  invA(1, 0) = (A(1, 2) * A(2, 0) - A(1, 0) * A(2, 2)) * inv_detA;
  invA(1, 1) = (A(0, 0) * A(2, 2) - A(0, 2) * A(2, 0)) * inv_detA;
  invA(1, 2) = (A(0, 2) * A(1, 0) - A(0, 0) * A(1, 2)) * inv_detA;
  invA(2, 0) = (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0)) * inv_detA;
  invA(2, 1) = (A(0, 1) * A(2, 0) - A(0, 0) * A(2, 1)) * inv_detA;
  invA(2, 2) = (A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0)) * inv_detA;

  return invA;
}

template <typename T, int n>
constexpr tensor<T, n, n> inv(const tensor<T, n, n>& A) {
  tensor<T, n, n> invA{};
  for (int j = 0; j < n; j++) {
    tensor<T, n> e_j{}; e_j[j] = 1.0;
    auto col = linear_solve(A, e_j);
    for (int i = 0; i < n; i++) {
      invA(i, j) = col(i);
    }
  }
  return invA;
}

template <typename T, int dim>
constexpr tensor<T, dim, dim> dev(tensor<T, dim, dim> A) {
  T delta = tr(A) / dim;
  tensor<T, dim, dim> A_dev{};
  for (int i = 0; i < dim; i++) {
    A_dev(i, i) -= delta;
  }
  return A_dev;
}

template <typename T, int... n>
constexpr T norm_squared(tensor<T,n...> u) {
  T norm_sq = 0.0;
  for (int i = 0; i < (n * ...); i++) {
    norm_sq += ((T*)u.values)[i] * ((T*)u.values)[i];
  }
  return norm_sq;
}

template <typename T, int... n>
constexpr T norm(tensor<T,n...> u) {
  return sqrt(norm_squared(u));
}

template <typename T>
constexpr auto normalize(T u) {
  float norm_u = norm(u);
  if (norm_u != 0) {
    return u / norm_u;
  } else {
    return T{};
  }
}

// vector-vector products
template <typename T>
constexpr tensor<T,3> cross(tensor<T, 3> u, tensor<T, 3> v) {
  return tensor<T,3>{
    u[1] * v[2] - u[2] * v[1], 
    u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0] 
  };
}

template <typename T, int m, int n>
auto dot(const tensor<T, m, n>& A, const tensor<T, n>& b) {
  tensor<T, m> Ab{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      Ab(i) += A(i, j) * b(j);
    }
  }
  return Ab;
}

template <typename T, int m, int n>
auto dot(const tensor<T, m>& a, const tensor<T, m, n>& B) {
  tensor<T, m> aB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      aB(i) += a(j) * B(j, i);
    } 
  }
  return aB;
}

template <typename T, int m>
constexpr T dot(tensor<T, m> u, tensor<T, m> v) {
  T uTv = 0.0;
  for (int i = 0; i < m; i++) {
    uTv += u(i) * v(i);
  }
  return uTv;
}

template <typename T, int m, int n, int p>
constexpr auto dot(tensor<T, m, n> A, tensor<T, n, p> B) {
  tensor<T,m,p> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < p; j++) {
      for (int k = 0; k < n; k++) {
        AB(i,j) += A(i,k) * B(k,j);
      }
    }
  }
  return AB;
}

template <typename T, int m, int n>
constexpr tensor<T, n, m> transpose(tensor<T, m, n> A) {
  tensor<T, n, m> AT{};
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      AT(i, j) = A(j, i);
    }
  }
  return AT;
}

template <typename T, int m, int n>
constexpr tensor<T, m, n> outer(tensor<T, m> u, tensor<T, n> v) {
  tensor<T, m, n> uvT{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      uvT(i, j) = u(i) * v(j);
    }
  }
  return uvT;
}

template < typename T >
inline tensor<T, 2> xy(const tensor<T, 3> & xyz) {
  return {xyz[0], xyz[1]};
}

template < typename T, int n >
inline tensor<T, n> abs(const tensor<T, n> & v) {
  tensor<T,n> absv{};
  for (int i = 0; i < n; i++) {
    absv[i] = std::abs(v[i]);
  }
  return absv;
}

template < typename T, int n >
inline tensor<T, n> min(const tensor<T, n> & v, T value) {
  tensor<T,n> out{};
  for (int i = 0; i < n; i++) {
    out[i] = std::min(v[i], value);
  }
  return out;
}

template < typename T, int n >
inline tensor<T, n> min(T value, const tensor<T, n> & v) { return min(v, value); }

template < typename T, int n >
inline tensor<T, n> max(const tensor<T, n> & v, T value) {
  tensor<T,n> out{};
  for (int i = 0; i < n; i++) {
    out[i] = std::max(v[i], value);
  }
  return out;
}

template < typename T, int n >
inline tensor<T, n> max(T value, const tensor<T, n> & v) { return max(v, value); }

template < typename T >
T clamp(const T & value, const T & lower, const T & upper) {
  return std::max(lower, std::min(value, upper));
}

template < typename T, int n >
inline tensor<T, n> clamp(const tensor<T, n> & v, T lower, T upper) {
  tensor<T,n> out{};
  for (int i = 0; i < n; i++) {
    out[i] = clamp(v[i], lower, upper);
  }
  return out;
}

template < typename T >
inline tensor<T, 3, 3> axis_angle_rotation(const tensor<T, 3> & omega) {

  T norm_omega = norm(omega);

  if (fabs(norm_omega) < 0.000001f) {

    return Identity< T, 3 >();

  } else {

    auto u = omega / norm_omega;

    T c = cos(norm_omega);
    T s = sin(norm_omega);
    T g = 1 - c;

    return tensor<T,3,3>{{
      {
        u[0]*u[0]*g + c,
        u[0]*u[1]*g - u[2]*s,
        u[0]*u[2]*g + u[1]*s
      },{
        u[1]*u[0]*g + u[2]*s,
        u[1]*u[1]*g + c,
        u[1]*u[2]*g - u[0]*s
      },{
        u[2]*u[0]*g - u[1]*s,
        u[2]*u[1]*g + u[0]*s,
        u[2]*u[2]*g + c
      }
    }};

  }

}

template < typename T >
inline tensor < T, 3, 3 > R3_basis(const tensor<T, 3> & n) {
  T s = (n[2] >= 0.0f) ? 1.0f : -1.0f;
  T a = -1.0f / (s + n[2]); 
  T b = n[0] * n[1] * a;
  return tensor < T, 3, 3 >{{
    {1.0f + s * n[0] * n[0] * a, b, n[0]},
    {s * b, s + n[1] * n[1] * a, n[1]},
    {-s * n[0], -n[1], n[2]}
  }};
}

template < typename T, int n >
std::ostream & operator<<(std::ostream & out, tensor<T, n> v) {
  out << '{' << v(0);
  for (int i = 1; i < n; i++) {
    out << ", " << v(i);
  }
  out << '}';
  return out;
}

template < typename T, int m, int n >
std::ostream & operator<<(std::ostream & out, tensor<T, m, n> A) {
  out << '{' << '\n';
  for (int i = 0; i < m; i++) {
    out << "  {" << A(i,0);
    for (int j = 1; j < n; j++) {
      out << ", " << A(i,j);
    }
    out << '}' << '\n';
  }
  out << '}';
  return out;
}

