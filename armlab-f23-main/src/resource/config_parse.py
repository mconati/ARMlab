import os

import numpy as np
import numpy.typing as npt


def parse_dh_param_file(dh_config_file: str) -> npt.NDArray[np.float64]:
    assert dh_config_file is not None
    f_line_contents = None
    with open(dh_config_file, "r") as f:
        f_line_contents = f.readlines()

    assert f.closed
    assert f_line_contents is not None
    # maybe not the most efficient/clean/etc. way to do this, but should only have to be done once so NBD
    dh_params = np.asarray([line.rstrip().split(",") for line in f_line_contents[1:]])
    dh_params = dh_params.astype(float)
    return dh_params


def parse_pox_param_file(
    pox_config_file: str,
) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]]:
    assert pox_config_file is not None
    assert os.path.exists(pox_config_file)

    f_line_contents = []
    with open(pox_config_file, "r") as f:
        for line in f:
            if not line.startswith("#"):
                tokens = [float(x) for x in line.strip().split()]
                f_line_contents.append(tokens)

    M = np.array(f_line_contents[:4], dtype=np.float64)
    s_lst = []

    for params in f_line_contents[4:]:
        s_lst.append(np.array(params, dtype=np.float64))

    return M, s_lst
