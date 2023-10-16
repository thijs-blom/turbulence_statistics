#! /usr/bin/env python3
import argparse
import meshio
import numpy as np
from pathlib import Path


def convert_flow_field(
    vtu_path: Path, nx: int, ny: int, nz: int, u_label: str, v_label: str, w_label: str
):
    if not vtu_path.exists():
        raise ValueError(f"File {vtu_path} does not exist")

    mesh = meshio.read(vtu_path)

    if nx * ny * nz != mesh.points.size:
        raise ValueError(
            "Product of nx, ny, nz should equal the number of points in the VTU file"
        )

    for label in [u_label, v_label, w_label]:
        if label not in mesh.point_data:
            raise ValueError(
                f"Label {label} is not present in {vtu_path}. "
                "Is the label correct, and has the point data been included in the data file?"
            )

    u = mesh.point_data[u_label].reshape((nx, ny, nz), order="F")
    v = mesh.point_data[v_label].reshape((nx, ny, nz), order="F")
    w = mesh.point_data[w_label].reshape((nx, ny, nz), order="F")

    return u, v, w


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="vtu_to_numpy", description="COMSOL VTU to numpy converter"
    )
    parser.add_argument("vtu_file", type=Path)
    parser.add_argument("nx", type=int)
    parser.add_argument("ny", type=int)
    parser.add_argument("nz", type=int)
    parser.add_argument("--u_label", type=str, default="u")
    parser.add_argument("--v_label", type=str, default="v")
    parser.add_argument("--w_label", type=str, default="w")
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Path prefix for output files. "
        "Generates prefix_u.npy, prefix_v.npy, prefix_w.npy."
        "If not prefix is given, the files u.npy, v.npy, w.npy are generated",
    )

    args = parser.parse_args()

    u, v, w = convert_flow_field(
        args.vtu_file,
        args.nx,
        args.ny,
        args.nz,
        args.u_label,
        args.v_label,
        args.w_label,
    )

    p = Path()
    np.save(p / f"{args.prefix}_u.npy", u)
    np.save(p / f"{args.prefix}_v.npy", v)
    np.save(p / f"{args.prefix}_w.npy", w)
