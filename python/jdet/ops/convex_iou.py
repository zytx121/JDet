import jittor as jt

HEADER=r"""
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N) {
    int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int max_block_num = 65000;
    return std::min(optimal_block_num, max_block_num);
}

#define MAXN 100
#define NMAX 512
__device__ const double EPS = 1E-8;

__device__ inline int sig(double d) { return (d > EPS) - (d < -EPS); }

struct Point {
  double x, y;
  __device__ Point() {}
  __device__ Point(double x, double y) : x(x), y(y) {}
};

__device__ inline bool point_same(Point& a, Point& b) {
  return sig(a.x - b.x) == 0 && sig(a.y - b.y) == 0;
}

__device__ inline void swap1(Point* a, Point* b) {
  Point temp;
  temp.x = a->x;
  temp.y = a->y;

  a->x = b->x;
  a->y = b->y;

  b->x = temp.x;
  b->y = temp.y;
}

__device__ inline void reverse1(Point* a, const int n) {
  for (int i = 0; i < (n - 1) / 2.0; i++) {
    Point* j = &(a[i]);
    Point* k = &(a[n - 1 - i]);
    swap1(j, k);
  }
}

__device__ inline double cross(Point o, Point a, Point b) {
  return (a.x - o.x) * (b.y - o.y) - (b.x - o.x) * (a.y - o.y);
}

__device__ inline double dis(Point a, Point b) {
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}
__device__ inline double area(Point* ps, int n) {
  ps[n] = ps[0];
  double res = 0;
  for (int i = 0; i < n; i++) {
    res += ps[i].x * ps[i + 1].y - ps[i].y * ps[i + 1].x;
  }
  return res / 2.0;
}
__device__ inline double polygon_area_grad(Point* ps, int n,
                                           int* polygon_to_pred_index,
                                           int n_pred, double* grad_C) {
  ps[n] = ps[0];
  double partion_grad[4 * 30 + 2];
  double res = 0;
  for (int i = 0; i < n; i++) {
    res += ps[i].x * ps[i + 1].y - ps[i].y * ps[i + 1].x;
    partion_grad[i * 4 + 2] = ps[i + 1].y;
    partion_grad[i * 4 + 3] = -ps[i + 1].x;
    if (i != n - 1) {
      partion_grad[i * 4 + 4] = -ps[i].y;
      partion_grad[i * 4 + 5] = ps[i].x;
    } else {
      partion_grad[0] = -ps[i].y;
      partion_grad[1] = ps[i].x;
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n_pred; j++) {
      if (i == polygon_to_pred_index[j]) {
        grad_C[2 * polygon_to_pred_index[j + n_pred]] =
            (partion_grad[i * 4] + partion_grad[i * 4 + 2]) / 2;
        break;
      }
    }
    for (int j = 0; j < n_pred; j++) {
      if (i == polygon_to_pred_index[j]) {
        grad_C[2 * polygon_to_pred_index[j + n_pred] + 1] =
            (partion_grad[i * 4 + 1] + partion_grad[i * 4 + 1 + 2]) / 2;
        break;
      }
    }
  }

  return res / 2.0;
}


__device__ inline void Jarvis(Point* in_poly, int& n_poly) {
  Point p_max, p_k;
  int max_index, k_index;
  int Stack[NMAX] = {}, top1, top2;
  double sign;
  Point right_point[10], left_point[10];

  for (int i = 0; i < n_poly; i++) {
    if (in_poly[i].y < in_poly[0].y ||
        in_poly[i].y == in_poly[0].y && in_poly[i].x < in_poly[0].x) {
      Point* j = &(in_poly[0]);
      Point* k = &(in_poly[i]);
      swap1(j, k);
    }
    if (i == 0) {
      p_max = in_poly[0];
      max_index = 0;
    }
    if (in_poly[i].y > p_max.y ||
        in_poly[i].y == p_max.y && in_poly[i].x > p_max.x) {
      p_max = in_poly[i];
      max_index = i;
    }
  }

  if (max_index == 0) {
    max_index = 1;
    p_max = in_poly[max_index];
  }

  k_index = 0, Stack[0] = 0, top1 = 0;
  while (k_index != max_index) {
    p_k = p_max;
    k_index = max_index;
    for (int i = 1; i < n_poly; i++) {
      sign = cross(in_poly[Stack[top1]], in_poly[i], p_k);
      if ((sign > 0) || ((sign == 0) && (dis(in_poly[Stack[top1]], in_poly[i]) >
                                         dis(in_poly[Stack[top1]], p_k)))) {
        p_k = in_poly[i];
        k_index = i;
      }
    }
    top1++;
    Stack[top1] = k_index;
  }
  for (int i = 0; i <= top1; i++) right_point[i] = in_poly[Stack[i]];

  k_index = 0, Stack[0] = 0, top2 = 0;

  while (k_index != max_index) {
    p_k = p_max;
    k_index = max_index;
    for (int i = 1; i < n_poly; i++) {
      sign = cross(in_poly[Stack[top2]], in_poly[i], p_k);
      if ((sign < 0) || (sign == 0) && (dis(in_poly[Stack[top2]], in_poly[i]) >
                                        dis(in_poly[Stack[top2]], p_k))) {
        p_k = in_poly[i];
        k_index = i;
      }
    }
    top2++;
    Stack[top2] = k_index;
  }
  for (int i = top2 - 1; i >= 0; i--) left_point[i] = in_poly[Stack[i]];

  for (int i = 0; i < top1 + top2; i++) {
    if (i <= top1) {
      in_poly[i] = right_point[i];
    } else {
      in_poly[i] = left_point[top2 - (i - top1)];
    }
  }
  n_poly = top1 + top2;
}

__device__ inline double intersectAreaPoly(Point* ps1, int n1, Point* ps2,
                                           int n2, double* grad_C) {
  Point polygon[MAXN];
  int n = n1 + n2, n_poly = 0;
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n - n1; j++) {
      if (point_same(ps1[i], ps2[j])) {
        for (int k = j; k < n - n1 - 1; k++) {
          ps2[k] = ps2[k + 1];
        }
        n2--;
        break;
      }
    }
  }
  n_poly = n1 + n2;
  for (int i = 0; i < n_poly; i++) {
    if (i < n1) {
      polygon[i] = ps1[i];
    } else {
      polygon[i] = ps2[i - n1];
    }
  }

  Jarvis(polygon, n_poly);

  int polygon_to_pred_index[18] = {-1, -1, -1, -1, -1, -1, -1, -1, -1,
                                   -1, -1, -1, -1, -1, -1, -1, -1, -1};
  int n_pred = 0;
  for (int i = 0; i < n_poly; i++) {
    for (int j = 0; j < n1; j++) {
      if (polygon[i].x == ps1[j].x && polygon[i].y == ps1[j].y) {
        polygon_to_pred_index[n_pred] = i;
        polygon_to_pred_index[n_pred + n1] = j;
        n_pred += 1;
        break;
      }
    }
  }
  if (n_pred == 0) {
    double polygon_area = fabs(area(polygon, n_poly));
    for (int i = 0; i < 18; i++) {
      grad_C[i] = 0.0;
    }
    return polygon_area;
  } else {
    double polygon_area =
        polygon_area_grad(polygon, n_poly, polygon_to_pred_index, n1, grad_C);
    if (polygon_area < 0) {
      for (int i = 0; i < 18; i++) {
        grad_C[i] = -grad_C[i];
      }
    }
    return fabs(polygon_area);
  }
}

// convex_find and get the polygon_index_box_index
__device__ inline void Jarvis_and_index(Point* in_poly, int& n_poly,
                                        int* points_to_convex_ind) {
  int n_input = n_poly;
  Point input_poly[20];
  for (int i = 0; i < n_input; i++) {
    input_poly[i].x = in_poly[i].x;
    input_poly[i].y = in_poly[i].y;
  }
  Point p_max, p_k;
  int max_index, k_index;
  int Stack[20], top1, top2;
  double sign;
  Point right_point[10], left_point[10];

  for (int i = 0; i < n_poly; i++) {
    if (in_poly[i].y < in_poly[0].y ||
        in_poly[i].y == in_poly[0].y && in_poly[i].x < in_poly[0].x) {
      Point* j = &(in_poly[0]);
      Point* k = &(in_poly[i]);
      swap1(j, k);
    }
    if (i == 0) {
      p_max = in_poly[0];
      max_index = 0;
    }
    if (in_poly[i].y > p_max.y ||
        in_poly[i].y == p_max.y && in_poly[i].x > p_max.x) {
      p_max = in_poly[i];
      max_index = i;
    }
  }
  if (max_index == 0) {
    max_index = 1;
    p_max = in_poly[max_index];
  }

  k_index = 0, Stack[0] = 0, top1 = 0;
  while (k_index != max_index) {
    p_k = p_max;
    k_index = max_index;
    for (int i = 1; i < n_poly; i++) {
      sign = cross(in_poly[Stack[top1]], in_poly[i], p_k);
      if ((sign > 0) || ((sign == 0) && (dis(in_poly[Stack[top1]], in_poly[i]) >
                                         dis(in_poly[Stack[top1]], p_k)))) {
        p_k = in_poly[i];
        k_index = i;
      }
    }
    top1++;
    Stack[top1] = k_index;
  }
  for (int i = 0; i <= top1; i++) {
    right_point[i] = in_poly[Stack[i]];
  }

  k_index = 0, Stack[0] = 0, top2 = 0;

  while (k_index != max_index) {
    p_k = p_max;
    k_index = max_index;
    for (int i = 1; i < n_poly; i++) {
      sign = cross(in_poly[Stack[top2]], in_poly[i], p_k);
      if ((sign < 0) || (sign == 0) && (dis(in_poly[Stack[top2]], in_poly[i]) >
                                        dis(in_poly[Stack[top2]], p_k))) {
        p_k = in_poly[i];
        k_index = i;
      }
    }
    top2++;
    Stack[top2] = k_index;
  }

  for (int i = top2 - 1; i >= 0; i--) {
    left_point[i] = in_poly[Stack[i]];
  }

  for (int i = 0; i < top1 + top2; i++) {
    if (i <= top1) {
      in_poly[i] = right_point[i];
    } else {
      in_poly[i] = left_point[top2 - (i - top1)];
    }
  }
  n_poly = top1 + top2;
  for (int i = 0; i < n_poly; i++) {
    for (int j = 0; j < n_input; j++) {
      if (point_same(in_poly[i], input_poly[j])) {
        points_to_convex_ind[i] = j;
        break;
      }
    }
  }
}

__device__ inline int lineCross(Point a, Point b, Point c, Point d, Point& p) {
  double s1, s2;
  s1 = cross(a, b, c);
  s2 = cross(a, b, d);
  if (sig(s1) == 0 && sig(s2) == 0) return 2;
  if (sig(s2 - s1) == 0) return 0;
  p.x = (c.x * s2 - d.x * s1) / (s2 - s1);
  p.y = (c.y * s2 - d.y * s1) / (s2 - s1);
  return 1;
}

__device__ inline void polygon_cut(Point* p, int& n, Point a, Point b) {
  Point pp[MAXN];
  int m = 0;
  p[n] = p[0];
  for (int i = 0; i < n; i++) {
    if (sig(cross(a, b, p[i])) > 0) {
      pp[m] = p[i];
      m++;
    }
    if (sig(cross(a, b, p[i])) != sig(cross(a, b, p[i + 1]))) {
      lineCross(a, b, p[i], p[i + 1], pp[m]);
      m++;
    }
  }
  n = 0;
  for (int i = 0; i < m; i++) {
    if (!i || !(point_same(pp[i], pp[i - 1]))) {
      p[n] = pp[i];
      n++;
    }
  }

  while (n > 1 && point_same(p[n - 1], p[0])) n--;
}

__device__ inline double intersectArea(Point a, Point b, Point c, Point d) {
  Point o(0, 0);
  int s1 = sig(cross(o, a, b));
  int s2 = sig(cross(o, c, d));
  if (s1 == 0 || s2 == 0) return 0.0;
  if (s1 == -1) {
    Point* i = &a;
    Point* j = &b;
    swap1(i, j);
  }
  if (s2 == -1) {
    Point* i = &c;
    Point* j = &d;
    swap1(i, j);
  }
  Point p[10] = {o, a, b};
  int n = 3;

  polygon_cut(p, n, o, c);
  polygon_cut(p, n, c, d);
  polygon_cut(p, n, d, o);
  double res = area(p, n);
  if (s1 * s2 == -1) res = -res;
  return res;
}
__device__ inline double intersectAreaO(Point* ps1, int n1, Point* ps2,
                                        int n2) {
  if (area(ps1, n1) < 0) reverse1(ps1, n1);
  if (area(ps2, n2) < 0) reverse1(ps2, n2);
  ps1[n1] = ps1[0];
  ps2[n2] = ps2[0];
  double res = 0;
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      res += intersectArea(ps1[i], ps1[i + 1], ps2[j], ps2[j + 1]);
    }
  }
  return res;
}

template <typename T>
__device__ inline float devrIoU(T const* const p, T const* const q) {
  Point ps1[MAXN], ps2[MAXN];
  Point convex[MAXN];
  for (int i = 0; i < 9; i++) {
    convex[i].x = (double)p[i * 2];
    convex[i].y = (double)p[i * 2 + 1];
  }
  int n_convex = 9;
  int points_to_convex_ind[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
  Jarvis_and_index(convex, n_convex, points_to_convex_ind);
  int n1 = n_convex;
  for (int i = 0; i < n1; i++) {
    ps1[i].x = (double)convex[i].x;
    ps1[i].y = (double)convex[i].y;
  }
  int n2 = 4;
  for (int i = 0; i < n2; i++) {
    ps2[i].x = (double)q[i * 2];
    ps2[i].y = (double)q[i * 2 + 1];
  }
  double inter_area = intersectAreaO(ps1, n1, ps2, n2);
  double S_pred = area(ps1, n1);
  double union_area = fabs(S_pred) + fabs(area(ps2, n2)) - inter_area;
  double iou = inter_area / union_area;
  return (float)iou;
}

template <typename T>
__global__ void convex_iou_cuda_kernel(const int ex_n_boxes,
                                       const int gt_n_boxes, const T* ex_boxes,
                                       const T* gt_boxes, T* iou) {
  CUDA_1D_KERNEL_LOOP(index, ex_n_boxes) {
    const T* cur_box = ex_boxes + index * 18;
    for (int i = 0; i < gt_n_boxes; i++) {
      iou[index * gt_n_boxes + i] = devrIoU(cur_box, gt_boxes + i * 8);
    }
  }
}
"""

def convex_iou(pointsets, polygons):
    assert jt.flags.use_cuda
    N, K = pointsets.size(0), polygons.size(0)
    ious = jt.zeros((N, K), dtype=pointsets.dtype)
    src = f"""
    const int output_size = {ious.numel()};
    const int num_pointsets = {pointsets.size(0)};
    const int num_polygons = {polygons.size(0)};
    convex_iou_cuda_kernel<<<GET_BLOCKS(output_size), THREADS_PER_BLOCK >>>(
                    num_pointsets, num_polygons, in0_p, in1_p, out0_p);
    """
    return jt.code(ious.shape,ious.dtype,[pointsets,polygons],cuda_header=HEADER,cuda_src=src)

def test():
    import numpy as np
    np_pointsets = np.asarray([[
        1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 1.0,
        3.0, 3.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.5, 1.5
    ],
    [
        1.5, 1.5, 2.5, 2.5, 1.5, 2.5, 2.5, 1.5, 1.5,
        3.5, 3.5, 1.5, 2.5, 3.5, 3.5, 2.5, 2.0, 2.0
    ]])

    np_polygons = np.asarray([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0],
                            [1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 1.0]])

    np_expected_iou = np.asarray([[0.2857, 0.8750], [0.0588, 0.4286]])


    pointsets = jt.array(np_pointsets).float()
    polygons = jt.array(np_polygons).float()

    assert np.allclose(
        convex_iou(pointsets, polygons).numpy(),
        np_expected_iou,
        atol=1e-4)


if __name__ == "__main__":
    jt.flags.use_cuda=1
    test()