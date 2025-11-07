import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


@pytest.mark.parametrize(
    "loc, scale",
    [
        (0.0, 1.0),  # standard_normal
        (42.0, 0.0),  # degenerate distribution
        (0.0, 1e-16),  # low variance
    ],
)
def test_groupby_skew_equivalence(loc, scale):
    # Test that that groupby skew method (which uses libgroupby.group_skew)
    #  matches the results of operating group-by-group (which uses nanops.nanskew)
    nrows = 1000
    ngroups = 3
    ncols = 2
    nan_frac = 0.05

    rng = np.random.default_rng(2)
    arr = rng.normal(loc, scale, (nrows, ncols))
    arr[rng.random(nrows) < nan_frac] = np.nan

    df = pd.DataFrame(arr)
    grps = np.random.default_rng(2).integers(0, ngroups, size=nrows)
    gb = df.groupby(grps)

    result = gb.skew()

    grpwise = [grp.skew().to_frame(i).T for i, grp in gb]
    expected = pd.concat(grpwise, axis=0)
    expected.index = expected.index.astype(result.index.dtype)  # 32bit builds
    tm.assert_frame_equal(result, expected)
