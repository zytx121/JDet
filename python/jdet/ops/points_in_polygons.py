import jittor as jt

HEADER=r"""
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N, const int num_threads = THREADS_PER_BLOCK) {
  int optimal_block_num = (N + num_threads - 1) / num_threads;
  int max_block_num = 4096;
  return min(optimal_block_num, max_block_num);
}

struct point {
  float x, y;
};

template <typename scalar_t>
__global__ void points_in_polygons_forward_cuda_kernel(
    const int nthreads, const scalar_t *vertex1, const scalar_t *vertex2,
    const int rows, const int cols, scalar_t *inside_flag) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int row = index / cols;
    int col = index % cols;

    const scalar_t *offset_vertex1 = vertex1 + row * 2;
    const scalar_t *offset_vertex2 = vertex2 + col * 8;

    point point_[1];
    point polygon[4];

    point_[0].x = offset_vertex1[0];
    point_[0].y = offset_vertex1[1];

    polygon[0].x = offset_vertex2[0];
    polygon[0].y = offset_vertex2[1];
    polygon[1].x = offset_vertex2[2];
    polygon[1].y = offset_vertex2[3];
    polygon[2].x = offset_vertex2[4];
    polygon[2].y = offset_vertex2[5];
    polygon[3].x = offset_vertex2[6];
    polygon[3].y = offset_vertex2[7];

    int nCross = 0;
    int i, j;
    float sx, sy, tx, ty, px, py, x;
    for (i = 0, j = 3; i < 4; j = i, i++) {
      sx = polygon[i].x;
      sy = polygon[i].y;
      tx = polygon[j].x;
      ty = polygon[j].y;

      px = point_[0].x;
      py = point_[0].y;

      if (py < min(sy, ty)) continue;
      if (py > max(sy, ty)) continue;

      if ((sx == px && sy == py) || (tx == px && ty == py)) {
        break;
      } else {
        if ((sy < py && ty >= py) || (sy >= py && ty < py)) {
          x = sx + (py - sy) * (tx - sx) / (ty - sy);
          if (x == px) {
            break;
          }
          if (x > px) {
            nCross++;
          }
        }
      }
    }
    if (nCross % 2 == 1) {
      inside_flag[index] = 1.0;
    } else {
      inside_flag[index] = 0.0;
    }
    return;
  }
}
"""

def points_in_polygons(pointsets, polygons):
    assert jt.flags.use_cuda
    N, K = pointsets.size(0), polygons.size(0)
    output = jt.zeros((N, K), dtype=pointsets.dtype)
    src = f"""
    const int output_size = {output.numel()};
    const int num_pointsets = {pointsets.size(0)};
    const int num_polygons = {polygons.size(0)};
    points_in_polygons_forward_cuda_kernel<<<GET_BLOCKS(output_size), THREADS_PER_BLOCK >>>(
                    output_size, in0_p, in1_p, num_pointsets, num_polygons, out0_p);
    """
    return jt.code(output.shape,output.dtype,[pointsets,polygons],cuda_header=HEADER,cuda_src=src)

def test():
    import numpy as np
    np_pointsets = np.array([[300., 300.], [400., 400.], [100., 100], [300, 250],
                       [100, 0]])
    np_polygons = np.array([[200., 200., 400., 400., 500., 200., 400., 100.],
                         [400., 400., 500., 500., 600., 300., 500., 200.],
                         [300., 300., 600., 700., 700., 700., 700., 100.]])
    np_expected_iou = np.array([[0., 0., 0.], [0., 0., 1.], [0., 0., 0.],
                                [1., 0., 0.], [0., 0., 0.]])


    pointsets = jt.array(np_pointsets).float()
    polygons = jt.array(np_polygons).float()

    assert np.allclose(
        points_in_polygons(pointsets, polygons).numpy(),
        np_expected_iou,
        atol=1e-4)

if __name__ == "__main__":
    jt.flags.use_cuda=1
    test()
