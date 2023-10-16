"""Statistics quantifying turbulence based on instantaneous velocity fields"""
from dataclasses import dataclass
import functools
from typing import Callable, Optional

from numba import jit
import numpy as np
from multiprocessing import Pool
import enum


class Neighbourhood(enum.Enum):
    CUBE = "cube"
    BALL = "ball"

    def __str__(self) -> str:
        return self.value


class RelativeTo(enum.Enum):
    AVERAGE = "average"
    CENTRE = "centre"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class LAADType:
    neighbourhood_shape: Neighbourhood
    relative_to: RelativeTo

    def __str__(self) -> str:
        return f"(neighbourhood_shape={self.neighbourhood_shape}, relative_to={self.relative_to})"


class Axis(enum.Enum):
    X = "x"
    Y = "y"
    Z = "z"


@jit(nopython=True)
def _spatial_average(flow_field: np.ndarray) -> np.ndarray:
    return np.full_like(flow_field, np.nanmean(flow_field))


@jit(nopython=True)
def _spatial_fluctuation(flow_field: np.ndarray) -> np.ndarray:
    return flow_field - _spatial_average(flow_field)


@jit(nopython=True)
def _window(
    flow_field: np.ndarray, index: tuple[int, int, int], window_size: int
) -> np.ndarray:
    i, j, k = index
    return flow_field[
        max(0, i - window_size) : min(i + window_size, flow_field.shape[0]) + 1,
        max(0, j - window_size) : min(j + window_size, flow_field.shape[1]) + 1,
        max(0, k - window_size) : min(k + window_size, flow_field.shape[2]) + 1,
    ]


@jit(nopython=True)
def _window_center(
    shape: tuple[int, int, int], index: tuple[int, int, int], window_size: int
) -> tuple[int, int, int]:
    i, j, k = index
    lower_bound_x = max(0, i - window_size)
    lower_bound_y = max(0, j - window_size)
    lower_bound_z = max(0, k - window_size)
    return (i - lower_bound_x, j - lower_bound_y, k - lower_bound_z)


def _window_average(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    window_size: int,
    index: tuple[int, int, int],
    print_index: bool = False,
) -> tuple[tuple[int, int, int], float]:
    if print_index:
        print(index)

    uw = _window(u, index, window_size)
    vw = _window(v, index, window_size)
    ww = _window(w, index, window_size)
    # Ignore the return type, as mypy does not understand that the second
    # value is a float and not an np.ndarray
    return index, np.linalg.norm(
        np.array([np.mean(uw), np.mean(vw), np.mean(ww)])
    )  # type: ignore


def local_average_fluctuation(
    u: np.ndarray, v: np.ndarray, w: np.ndarray, window_size: int, n_proc: int = 1
) -> np.ndarray:
    """Compute the local average of the (global) spatial velocity fluctuation

    Args:
        u (np.ndarray): The velocity field in the ...-direction in m/s
        v (np.ndarray): The velocity field in the ...-direction in m/s
        w (np.ndarray): The velocity field in the ...-direction in m/s
        window_size (int): The window size, described as half the width of the square
            surrounding each point.
        n_proc (int, optional): Number of processes to use. Defaults to 1.

    Returns:
        np.ndarray: _description_
    """
    result = np.empty_like(u)

    u_fluct = _spatial_fluctuation(u)
    v_fluct = _spatial_fluctuation(v)
    w_fluct = _spatial_fluctuation(w)
    window_average_partial = functools.partial(
        _window_average, u_fluct, v_fluct, w_fluct, window_size, print_index=True
    )

    average_fluctuations = []

    # Use a pool of processes if requested
    if n_proc > 1:
        with Pool(n_proc) as pool:
            average_fluctuations = pool.map(window_average_partial, np.ndindex(u.shape))
    # Otherwise just use a for loop to prevent the overhead of creating a process
    else:
        for index in np.ndindex(u.shape):
            average_fluctuations.append(window_average_partial(index))

    for index, value in average_fluctuations:
        result[index] = value

    return result


def _scale_fields(
    u: np.ndarray, v: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Scale the velocity field components to be in the range [-1, 1]
    u_max = max(np.abs(np.nanmax(u)), np.abs(np.nanmin(u)))
    v_max = max(np.abs(np.nanmax(v)), np.abs(np.nanmin(v)))
    w_max = max(np.abs(np.nanmax(w)), np.abs(np.nanmin(w)))

    scale_factor = max(u_max, v_max, w_max)

    return u / scale_factor, v / scale_factor, w / scale_factor


def _slice_indices(
    shape: tuple[int, int, int], slice_axis: Optional[Axis], slice_value: Optional[int]
) -> np.ndarray:
    if slice_axis is None:
        indices = np.fromiter((x for x in np.ndindex(shape)), dtype=np.dtype((int, 3)))
    elif slice_axis is Axis.X:
        indices = np.fromiter(
            (x for x in np.ndindex(shape) if x[0] == slice_value),
            dtype=np.dtype((int, 3)),
        )
    elif slice_axis is Axis.Y:
        indices = np.fromiter(
            (x for x in np.ndindex(shape) if x[1] == slice_value),
            dtype=np.dtype((int, 3)),
        )
    elif slice_axis is Axis.Z:
        indices = np.fromiter(
            (x for x in np.ndindex(shape) if x[2] == slice_value),
            dtype=np.dtype((int, 3)),
        )

    return indices


def _result_shape(
    shape: tuple[int, int, int], slice_axis: Optional[Axis]
) -> tuple[int, int] | tuple[int, int, int]:
    if slice_axis is None:
        return shape
    elif slice_axis is Axis.X:
        return (shape[1], shape[2])
    elif slice_axis is Axis.Y:
        return (shape[0], shape[2])
    elif slice_axis is Axis.Z:
        return (shape[0], shape[1])

    raise ValueError(f"Unknown value for axis: {slice_axis}")


def _assemble_chunks(
    slice_axis: Optional[Axis],
    chunks: list[np.ndarray],
    chunk_values: list[np.ndarray],
    result_shape: tuple[int, int] | tuple[int, int, int],
) -> np.ndarray:
    result = np.empty(result_shape)

    if slice_axis is None:
        for chunk, values in zip(chunks, chunk_values):
            for index, value in zip(chunk, values):
                result[index[0], index[1], index[2]] = value
    elif slice_axis is Axis.X:
        for chunk, values in zip(chunks, chunk_values):
            for index, value in zip(chunk, values):
                result[index[1], index[2]] = value
    elif slice_axis is Axis.Y:
        for chunk, values in zip(chunks, chunk_values):
            for index, value in zip(chunk, values):
                result[index[0], index[2]] = value
    elif slice_axis is Axis.Z:
        for chunk, values in zip(chunks, chunk_values):
            for index, value in zip(chunk, values):
                result[index[0], index[1]] = value
    return result


def _execute_mp(
    func: Callable[[np.ndarray], np.ndarray],
    shape: tuple[int, int, int],
    n_proc: int,
    slice_axis: Optional[Axis] = None,
    slice_value: Optional[int] = None,
) -> np.ndarray:
    with Pool(n_proc) as pool:
        indices = _slice_indices(shape, slice_axis, slice_value)
        chunks = np.array_split(indices, n_proc)
        chunk_values = pool.map(func, chunks)

        result_shape = _result_shape(shape, slice_axis)

        result = _assemble_chunks(slice_axis, chunks, chunk_values, result_shape)

    return result

@jit(nopython=True)
def _index_inner_product_at_index(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    window_size: int,
    relative_to: str,
    ball_shaped: bool,
    indices: np.ndarray,
) -> np.ndarray:
    result = np.empty(indices.shape[0])
    window_size_squared = window_size * window_size
    for i_index, index in enumerate(indices):
        tmp = 0.0
        global_index = (index[0], index[1], index[2])

        # Skip this index if it is outside the geometry, or otherwise invalid
        if (
            np.isnan(u[global_index])
            or np.isnan(v[global_index])
            or np.isnan(w[global_index])
        ):
            result[i_index] = np.nan
            continue

        u_w = _window(u, index, window_size)
        v_w = _window(v, index, window_size)
        w_w = _window(w, index, window_size)
        center = _window_center(u.shape, index, window_size)

        if relative_to == "average":
            u_ref = np.nanmean(u_w)
            v_ref = np.nanmean(v_w)
            w_ref = np.nanmean(w_w)
        elif relative_to == "centre":
            u_ref = u_w[center]
            v_ref = v_w[center]
            w_ref = w_w[center]

        count = 0

        for i in range(0, u_w.shape[0]):
            for j in range(0, u_w.shape[1]):
                for k in range(0, u_w.shape[2]):
                    other_index = i, j, k
                    other_u = u_w[other_index]
                    other_v = v_w[other_index]
                    other_w = w_w[other_index]

                    if np.isnan(other_u) or np.isnan(other_v) or np.isnan(other_w):
                        continue

                    if other_index == center:
                        continue

                    # When considering a ball instead of a cube, reject when outside of the ball
                    if ball_shaped:
                        d0 = i - center[0]
                        d1 = j - center[1]
                        d2 = k - center[2]
                        if d0 * d0 + d1 * d1 + d2 * d2 > window_size_squared:
                            continue

                    tmp += u_ref * other_u + v_ref * other_v + w_ref * other_w
                    count += 1
        result[i_index] = tmp / count
    return result


def index_inner_product(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    window_size: int,
    laad_type: LAADType,
    n_proc: int = 1,
    slice_axis: Optional[Axis] = None,
    slice_value: Optional[int] = None,
) -> np.ndarray:
    if u.shape != v.shape or u.shape != w.shape:
        raise ValueError("Velocity fields must all have the same shape")

    if slice_axis is not None and slice_value is None:
        raise ValueError("Slice value must be specified when slice axis is not None")

    # Scale the velocity field components to be in the range [-1, 1]
    u_scaled, v_scaled, w_scaled = _scale_fields(u, v, w)

    relative_to = laad_type.relative_to.value
    ball_shaped = laad_type.neighbourhood_shape is Neighbourhood.BALL
    func = functools.partial(
        _index_inner_product_at_index,
        u_scaled,
        v_scaled,
        w_scaled,
        window_size,
        relative_to,
        ball_shaped,
    )

    # Ignore type checking in below statement, as mypy does not understand that u is three-dimensional
    result = _execute_mp(func, u.shape, n_proc, slice_axis, slice_value)  # type:ignore

    return result


@jit(nopython=True)
def _index_vec_diff_at_index(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    window_size: int,
    relative_to: str,
    ball_shaped: bool,
    indices: np.ndarray,
) -> np.ndarray:
    result = np.empty(indices.shape[0])
    window_size_squared = window_size * window_size
    for i_index, index in enumerate(indices):
        tmp = 0.0
        global_index = (index[0], index[1], index[2])

        # Skip this index if it is outside the geometry, or otherwise invalid
        if (
            np.isnan(u[global_index])
            or np.isnan(v[global_index])
            or np.isnan(w[global_index])
        ):
            result[i_index] = np.nan
            continue

        u_w = _window(u, index, window_size)
        v_w = _window(v, index, window_size)
        w_w = _window(w, index, window_size)
        center = _window_center(u.shape, index, window_size)

        if relative_to == "average":
            u_ref = np.nanmean(u_w)
            v_ref = np.nanmean(v_w)
            w_ref = np.nanmean(w_w)
        elif relative_to == "centre":
            u_ref = u_w[center]
            v_ref = v_w[center]
            w_ref = w_w[center]

        count = 0

        for i in range(0, u_w.shape[0]):
            for j in range(0, u_w.shape[1]):
                for k in range(0, u_w.shape[2]):
                    other_index = i, j, k
                    other_u = u_w[other_index]
                    other_v = v_w[other_index]
                    other_w = w_w[other_index]

                    if np.isnan(other_u) or np.isnan(other_v) or np.isnan(other_w):
                        continue

                    if relative_to == "center" and other_index == center:
                        continue

                    # When considering a ball instead of a cube, reject when outside of the ball
                    if ball_shaped:
                        d0 = i - center[0]
                        d1 = j - center[1]
                        d2 = k - center[2]
                        if d0 * d0 + d1 * d1 + d2 * d2 > window_size_squared:
                            continue

                    diff_x = u_ref - other_u
                    diff_y = v_ref - other_v
                    diff_z = w_ref - other_w
                    tmp += np.sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z)
                    count += 1
        result[i_index] = tmp / count
    return result


def index_vec_diff(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    window_size: int,
    laad_type: LAADType,
    n_proc: int = 1,
    slice_axis: Optional[Axis] = None,
    slice_value: Optional[int] = None,
) -> np.ndarray:
    if u.shape != v.shape or u.shape != w.shape:
        raise ValueError("Velocity fields must all have the same shape")

    if slice_axis is not None and slice_value is None:
        raise ValueError("Slice value must be specified when slice axis is not None")

    # Scale the velocity field components to be in the range [-1, 1]
    u_scaled, v_scaled, w_scaled = _scale_fields(u, v, w)

    relative_to = laad_type.relative_to.value
    ball_shaped = laad_type.neighbourhood_shape is Neighbourhood.BALL
    func = functools.partial(
        _index_vec_diff_at_index,
        u_scaled,
        v_scaled,
        w_scaled,
        window_size,
        relative_to,
        ball_shaped,
    )

    # Ignore type checking in below statement, as mypy does not understand that u is three-dimensional
    result = _execute_mp(func, u.shape, n_proc, slice_axis, slice_value)  # type:ignore

    return result

@jit(nopython=True)
def _laad_at_index(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    window_size: int,
    relative_to: str,
    ball_shaped: bool,
    indices: np.ndarray,
) -> np.ndarray:
    result = np.empty(indices.shape[0])
    window_size_squared = window_size * window_size
    for i_index, index in enumerate(indices):
        tmp = 0.0
        global_index = (index[0], index[1], index[2])

        # Skip this index if it is outside the geometry, or otherwise invalid
        if (
            np.isnan(u[global_index])
            or np.isnan(v[global_index])
            or np.isnan(w[global_index])
        ):
            result[i_index] = np.nan
            continue

        u_w = _window(u, index, window_size)
        v_w = _window(v, index, window_size)
        w_w = _window(w, index, window_size)
        center = _window_center(u.shape, index, window_size)

        if relative_to == "average":
            u_ref = np.nanmean(u_w)
            v_ref = np.nanmean(v_w)
            w_ref = np.nanmean(w_w)
        elif relative_to == "centre":
            u_ref = u_w[center]
            v_ref = v_w[center]
            w_ref = w_w[center]

        norm_ref = np.sqrt(u_ref * u_ref + v_ref * v_ref + w_ref * w_ref)

        count = 0

        for i in range(0, u_w.shape[0]):
            for j in range(0, u_w.shape[1]):
                for k in range(0, u_w.shape[2]):
                    other_index = i, j, k
                    other_u = u_w[other_index]
                    other_v = v_w[other_index]
                    other_w = w_w[other_index]

                    if np.isnan(other_u) or np.isnan(other_v) or np.isnan(other_w):
                        continue

                    if other_index == center:
                        continue

                    # When considering a ball instead of a cube, reject when outside of the ball
                    if ball_shaped:
                        d0 = i - center[0]
                        d1 = j - center[1]
                        d2 = k - center[2]
                        if d0 * d0 + d1 * d1 + d2 * d2 > window_size_squared:
                            continue

                    other_norm = np.sqrt(
                        other_u * other_u + other_v * other_v + other_w * other_w
                    )

                    cos_angle = (
                        u_ref * other_u + v_ref * other_v + w_ref * other_w
                    ) / (norm_ref * other_norm)

                    if cos_angle < -1:
                        cos_angle = -1
                    elif cos_angle > 1:
                        cos_angle = 1

                    tmp += np.arccos(cos_angle)
                    # tmp += np.abs(norm_av - other_norm)
                    count += 1
        max_value_one_cell = np.pi
        result[i_index] = tmp / (
            count * max_value_one_cell
        )  # Normalise result to be in [0, 1]
        # result[i_index] = tmp / count
    return result


def laad(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    window_size: int,
    laad_type: LAADType,
    n_proc: int = 1,
    slice_axis: Optional[Axis] = None,
    slice_value: Optional[int] = None,
) -> np.ndarray:
    """Compute the cross-correlation coefficient for every point in the domain.

    Args:
        u: The velocity field in the ...-direction in m/s
        v: The velocity field in the ...-direction in m/s
        w: The velocity field in the ...-direction in m/s

    Returns:
        np.ndarray: A 3-dimensional array of the same dimensions as u, v, and w, describing the
        cross-correlation coefficient at each point
    """
    if u.shape != v.shape or u.shape != w.shape:
        raise ValueError("Velocity fields must all have the same shape")

    if slice_axis is not None and slice_value is None:
        raise ValueError("Slice value must be specified when slice axis is not None")

    # Scale the velocity field components to be in the range [-1, 1]
    u_scaled, v_scaled, w_scaled = _scale_fields(u, v, w)

    relative_to = laad_type.relative_to.value
    ball_shaped = laad_type.neighbourhood_shape is Neighbourhood.BALL
    func = functools.partial(
        _laad_at_index,
        u_scaled,
        v_scaled,
        w_scaled,
        window_size,
        relative_to,
        ball_shaped,
    )

    # Ignore type checking in below statement, as mypy does not understand that u is three-dimensional
    result = _execute_mp(func, u.shape, n_proc, slice_axis, slice_value)  # type:ignore

    return result
