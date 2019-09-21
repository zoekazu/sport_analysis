import typing as tp

import cv2
import numpy as np


class HomograpyTransformer():

    def __init__(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        *,
        homograpy_kwargs: tp.Dict[str, tp.Any] = {
            "method": cv2.RANSAC, "ransacReprojThreshold": 3.0}
    ) -> None:

        self._check_points_for_homograpy_calucuation(src_pts, dst_pts)

        self._homograpy_matrix = self._calc_homography_matrix(
            src_pts, dst_pts, homograpy_kwargs)
        self._inversed_homograpy_matrix = np.linalg.pinv(
            self._homograpy_matrix)

    def _check_points_for_homograpy_calucuation(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
    ) -> None:
        """Check the constructor's input was correct.

        Parameters
        ----------
        src_pts : np.ndarray
            Ndarray of source points.
        dst_pts : np.ndarray
            Ndarray of destination points.

        Raises
        ------
        ValueError
            Dimension was incorrect.
        ValueError
            Shape was incorrect.
        ValueError
            Length of source and destination was not the same.
        """
        src_pts_shape = src_pts.shape
        dst_pts_shape = dst_pts.shape

        for shape_ in [src_pts_shape, dst_pts_shape]:
            if len(shape_) != 2:
                raise ValueError(
                    'Dimension of points for calibrating '
                    + 'homograpy matrix must be 2'
                    + f'Your input was {len(shape_)} dimenstion.')
            if shape_[1] != 2:
                raise ValueError(
                    'Image position should be x-y coordination'
                    + f'Your input was {shape_[1]}.')

        if src_pts_shape[0] != dst_pts_shape[0]:
            raise ValueError(
                'The length of point array must be the same'
                + 'between source and destination.')

    def _calc_homography_matrix(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        homograpy_kwargs: tp.Dict[str, tp.Any]
    ) -> np.ndarray:
        """Caluculate homograpy matrix.

        Parameters
        ----------
        src_pts : np.ndarray
            Ndarray of source points.
        dst_pts : np.ndarray
            Ndarray of destination points.
        homograpy_kwargs : tp.Dict[str, tp.Any]
            Kwargs for `cv2.findHomography`

        Returns
        -------
        np.ndarray
            Homograpy matrix.
        """
        homograpy_matrix, _ = cv2.findHomography(
            src_pts, dst_pts, **homograpy_kwargs)
        return homograpy_matrix

    def _warp_to_img(
        self,
        img: np.ndarray,
        img_size: tp.Tuple[int, int],
        matrix: np.ndarray,
        warp_perspective_kwargs: tp.Dict[str, tp.Any] = {}
    ) -> np.ndarray:
        """Warp to the specific image with matrix.

        Parameters
        ----------
        img : np.ndarray
            Input image.
        img_size_hw : tp.Tuple[int, int]
            Image size of destination image.
            The order of it is here:
                - height
                - width
        matrix : np.ndarray
            Conversion matrix.
        warp_perspective_kwargs : tp.Dict[str, tp.Any], optional
            Kwargs for `cv2.warpPerspective`, by default {}

        Returns
        -------
        np.ndarray
            Warped image.
        """
        return cv2.warpPerspective(
            np.float32(img),
            matrix,
            img_size,
            **warp_perspective_kwargs)

    def warp_to_dst_img(
        self,
        src_img: np.ndarray,
        dst_img_size_hw: tp.Tuple[int, int],
        warp_perspective_kwargs: tp.Dict[str, tp.Any] = {}
    ) -> np.ndarray:
        """Warp the source image to the destination image coordinate.

        Parameters
        ----------
        src_img : np.ndarray
            Source image.
        dst_img_size_hw : tp.Tuple[int, int]
            Image size of destination image.
            The order of it is here:
                - height
                - width
        warp_perspective_kwargs : tp.Dict[str, tp.Any], optional
            Kwargs for `cv2.warpPerspective`, by default {}

        Returns
        -------
        np.ndarray
            Warped image in the destination coordinate.

        Examples
        --------
        >>> import numpy as np
        >>> src_pts = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
        >>> dst_pts = np.array([[0, 5], [5, 10], [10,  5], [5,  0]])
        >>> # ( 0, 0)  --- ( 0, 10)           --- (5, 0)  ---
        >>> #   |      ---    |             |                 |
        >>> #   |      ---    |     ==>>  (5, 0)            (0, 5)
        >>> #   |      ---    |             |                 |
        >>> # (10, 0)  --- (10, 10)           --- (5, 5)  ---
        >>> img = np.arange(0, 25).reshape(5, 5, 1)
        >>> dst_img_size = (10, 10)
        >>> homograpy_transformer = HomograpyTransformer(src_pts, dst_pts)
        >>> homograpy_transformer.warp_to_dst_img(img, dst_img_size)
        array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  2.,  8., 14.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  6., 12., 18., 24.,  0.,  0.,  0.,  0.,  0.],
               [ 0., 10., 16., 22.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0., 20.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]], dtype=float32)
        """
        return self._warp_to_img(
            np.float32(src_img),
            dst_img_size_hw,
            self._homograpy_matrix,
            **warp_perspective_kwargs)

    def warp_to_src_img(
        self,
        dst_img: np.ndarray,
        src_img_size: tp.Tuple[int, int],
        warp_perspective_kwargs: tp.Dict[str, tp.Any] = {}
    ) -> np.ndarray:
        """Warp the destination image to the source image coordinate.

        Parameters
        ----------
        dst_img : np.ndarray
            Source image.
        src_img_size_hw : tp.Tuple[int, int]
            Image size of source image.
            The order of it is here:
                - height
                - width
        warp_perspective_kwargs : tp.Dict[str, tp.Any], optional
            Kwargs for `cv2.warpPerspective`, by default {}

        Returns
        -------
        np.ndarray
            Warped image in the source coordinate.

        Examples
        --------
        >>> import numpy as np
        >>> src_pts = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
        >>> dst_pts = np.array([[0, 5], [5, 10], [10,  5], [5,  0]])
        >>> # ( 0, 0)  --- ( 0, 10)           --- (5, 0)  ---
        >>> #   |      ---    |             |                 |
        >>> #   |      ---    |     ==>>  (5, 0)            (0, 5)
        >>> #   |      ---    |             |                 |
        >>> # (10, 0)  --- (10, 10)           --- (5, 5)  ---
        >>> img = np.arange(0, 25).reshape(5, 5, 1)
        >>> dst_img_size = (5, 5)
        >>> homograpy_transformer = HomograpyTransformer(src_pts, dst_pts)
        >>> homograpy_transformer.warp_to_src_img(img, dst_img_size)
        array([[ 0.  , 10.25, 21.  , 19.  , 17.  ],
               [ 0.  ,  0.  , 10.75, 22.  , 20.  ],
               [ 0.  ,  0.  ,  0.  , 11.25, 23.  ],
               [ 0.  ,  0.  ,  0.  ,  0.  , 11.75],
               [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ]], dtype=float32)
        """
        return self._warp_to_img(
            np.float32(dst_img),
            src_img_size,
            self._inversed_homograpy_matrix,
            **warp_perspective_kwargs)

    def warp_to_dst_pts(
        self,
        src_pts: np.ndarray
    ) -> np.ndarray:
        """Warp the source points to the destination points.

        Parameters
        ----------
        src_pts : np.ndarray
            Points of source coordinate.

        Returns
        -------
        np.ndarray
            Points of destination coordinate.

        Examples
        --------
        >>> import numpy as np
        >>> src_pts = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
        >>> dst_pts = np.array([[0, 5], [5, 10], [10,  5], [5,  0]])
        >>> # ( 0, 0)  --- ( 0, 10)           --- (5, 0)  ---
        >>> #   |      ---    |             |                 |
        >>> #   |      ---    |     ==>>  (5, 0)            (0, 5)
        >>> #   |      ---    |             |                 |
        >>> # (10, 0)  --- (10, 10)           --- (5, 5)  ---
        >>> homograpy_transformer = HomograpyTransformer(src_pts, dst_pts)
        >>> homograpy_transformer.warp_to_dst_pts(src_pts)
        array([[ 1.9860274e-15,  5.0000000e+00],
               [ 5.0000000e+00,  1.0000000e+01],
               [ 1.0000000e+01,  5.0000000e+00],
               [ 5.0000000e+00, -1.7763568e-15]], dtype=float32)
        """
        return cv2.perspectiveTransform(
            np.float32(src_pts[np.newaxis, :, :]),
            self._homograpy_matrix
        ).reshape(-1, 2)

    def warp_to_dst_pt(
        self,
        src_pt: np.ndarray
    ) -> np.ndarray:
        """Warp the source point to the destination point.

        Parameters
        ----------
        src_pt : np.ndarray
            Point of source coordinate.

        Returns
        -------
        np.ndarray
            Point of destination coordinate.

        Examples
        --------
        >>> import numpy as np
        >>> src_pts = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
        >>> dst_pts = np.array([[0, 5], [5, 10], [10,  5], [5,  0]])
        >>> # ( 0, 0)  --- ( 0, 10)           --- (5, 0)  ---
        >>> #   |      ---    |             |                 |
        >>> #   |      ---    |     ==>>  (5, 0)            (0, 5)
        >>> #   |      ---    |             |                 |
        >>> # (10, 0)  --- (10, 10)           --- (5, 5)  ---
        >>> homograpy_transformer = HomograpyTransformer(src_pts, dst_pts)
        >>> homograpy_transformer.warp_to_dst_pt(src_pts[0])
        array([1.9860274e-15, 5.0000000e+00], dtype=float32)
        """
        return self.warp_to_dst_pts(
            src_pt[np.newaxis, :]
        ).reshape(-1)

    def warp_to_src_pts(
        self,
        dst_pts: np.ndarray
    ) -> np.ndarray:
        """Warp the destination points to the source points.

        Parameters
        ----------
        dst_pts : np.ndarray
            Points of destination coordinate.

        Returns
        -------
        np.ndarray
            Points of source coordinate.

        Examples
        --------
        >>> import numpy as np
        >>> src_pts = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
        >>> dst_pts = np.array([[0, 5], [5, 10], [10,  5], [5,  0]])
        >>> # ( 0, 0)  --- ( 0, 10)           --- (5, 0)  ---
        >>> #   |      ---    |             |                 |
        >>> #   |      ---    |     ==>>  (5, 0)            (0, 5)
        >>> #   |      ---    |             |                 |
        >>> # (10, 0)  --- (10, 10)           --- (5, 5)  ---
        >>> homograpy_transformer = HomograpyTransformer(src_pts, dst_pts)
        >>> homograpy_transformer.warp_to_src_pts(dst_pts)
        array([[-2.6645353e-15, -4.4408921e-15],
               [-6.2172489e-15,  1.0000000e+01],
               [ 1.0000000e+01,  1.0000000e+01],
               [ 1.0000000e+01,  0.0000000e+00]], dtype=float32)
        """
        return cv2.perspectiveTransform(
            np.float32(dst_pts[np.newaxis, :, :]),
            self._inversed_homograpy_matrix
        ).reshape(-1, 2)

    def warp_to_src_pt(
        self,
        dst_pt: np.ndarray
    ) -> np.ndarray:
        """Warp the destination point to source point.

        Parameters
        ----------
        des_pt : np.ndarray
            Point of destination coordinate.

        Returns
        -------
        np.ndarray
            Point of source coordinate.

        Examples
        --------
        >>> import numpy as np
        >>> src_pts = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
        >>> dst_pts = np.array([[0, 5], [5, 10], [10,  5], [5,  0]])
        >>> # ( 0, 0)  --- ( 0, 10)           --- (5, 0)  ---
        >>> #   |      ---    |             |                 |
        >>> #   |      ---    |     ==>>  (5, 0)            (0, 5)
        >>> #   |      ---    |             |                 |
        >>> # (10, 0)  --- (10, 10)           --- (5, 5)  ---
        >>> homograpy_transformer = HomograpyTransformer(src_pts, dst_pts)
        >>> homograpy_transformer.warp_to_src_pt(dst_pts[0])
        array([-2.6645353e-15, -4.4408921e-15], dtype=float32)
        """
        return self.warp_to_src_pts(
            dst_pt[np.newaxis, :]
        ).reshape(-1)

    @property
    def matrix(self) -> np.ndarray:
        """Get homograpy matrix

        Returns
        -------
        np.ndarray
            Homograpy matrix.

        Examples
        --------
        >>> import numpy as np
        >>> src_pts = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
        >>> dst_pts = np.array([[0, 5], [5, 10], [10,  5], [5,  0]])
        >>> homograpy_transformer = HomograpyTransformer(src_pts, dst_pts)
        >>> homograpy_transformer.matrix
        array([[ 5.00000000e-01,  5.00000000e-01,  1.98602732e-15],
               [-5.00000000e-01,  5.00000000e-01,  5.00000000e+00],
               [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        """
        return self._homograpy_matrix

    @property
    def inversed_matrix(self) -> np.ndarray:
        """Get inversed homograpy matrix

        Returns
        -------
        np.ndarray
            Inversed homograpy matrix.

        Examples
        --------
        >>> import numpy as np
        >>> src_pts = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
        >>> dst_pts = np.array([[0, 5], [5, 10], [10,  5], [5,  0]])
        >>> homograpy_transformer = HomograpyTransformer(src_pts, dst_pts)
        >>> homograpy_transformer.inversed_matrix
        array([[ 1.00000000e+00, -1.00000000e+00,  5.00000000e+00],
               [ 1.00000000e+00,  1.00000000e+00, -5.00000000e+00],
               [ 8.06101499e-17, -1.21399194e-16,  1.00000000e+00]])
        """
        return self._inversed_homograpy_matrix

    def __eq__(self, other: object) -> bool:
        return np.allclose(self.matrix, other.matrix)

    def __repr__(self) -> str:
        return (
            f"Homograpy matrix: {self.matrix} \n"
            + f"Inversed homograpy matrix: {self.inversed_matrix} "
        )


if __name__ == "__main__":
    import doctest
    doctest.testmod()
