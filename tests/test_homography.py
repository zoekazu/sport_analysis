import numpy as np
import pytest
import sport_analysis as sp


class TestHomoGraphyTransformer():

    @pytest.fixture(
        params=(
            [
                np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
                np.array([[0, 5], [5, 10], [10, 5], [5, 0]])
            ],
            [
                np.array([[0, 5], [5, 10], [10, 5], [5, 0]]),
                np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
            ]
        )
    )
    def src_dst_pts(self, request):
        return request.param

    @pytest.fixture
    def instance_(self, src_dst_pts):
        return sp.HomograpyTransformer(src_dst_pts[0], src_dst_pts[1])

    def test_warp_to_dst_pts(self, src_dst_pts, instance_):
        warped_dst_pts = instance_.warp_to_dst_pts(src_dst_pts[0])
        assert np.allclose(src_dst_pts[1], warped_dst_pts)

    def test_warp_to_src_pts(self, src_dst_pts, instance_):
        warped_src_pts = instance_.warp_to_src_pts(src_dst_pts[1])
        assert np.allclose(src_dst_pts[0], warped_src_pts)

    def test_warp_pts_recurcive(self, src_dst_pts, instance_):
        warped_dst_pts = instance_.warp_to_src_pts(src_dst_pts[0])
        warped_src_pts = instance_.warp_to_dst_pts(warped_dst_pts)
        assert np.allclose(src_dst_pts[0], warped_src_pts)

        warped_src_pts = instance_.warp_to_dst_pts(src_dst_pts[1])
        warped_dst_pts = instance_.warp_to_src_pts(warped_src_pts)
        assert np.allclose(src_dst_pts[1], warped_dst_pts)


if __name__ == "__main__":
    pytest.main([__file__])
