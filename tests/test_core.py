"""Unit tests against values from ecotraj 1.2.2, obtained by black-box calls in R
(``Rscript``; ecotraj used purely as a numerical oracle). Matrices are
column-major as returned by ``as.numeric(as.matrix(.))``.

Dataset: 12 states in 2D, trajectories A, B, C of 4 surveys each:
  xy <- matrix(c(0,0, 1,0.5, 2.2,0.4, 3,1.5, 0.5,1, 1.2,1.8, 1.5,2.5, 2.6,2.2,
                 2,-1, 1.4,-0.5, 1.7,0.3, 0.9,0.9), ncol=2, byrow=TRUE)
  ent <- rep(c("A","B","C"), each=4); sv <- rep(1:4,3); tm <- c(0,1,2.5,3, 0,2,3,5, 1,2,3,4)
  x <- defineTrajectories(dist(xy), ent, sv, tm)
"""
import numpy as np
import pytest

from ecotrajpy import (
    clr, define_trajectories, directionality_coords, increment_split, is_metric,
    segment_distances, signflip_path, signflip_test, trajectory_angles,
    trajectory_convergence, trajectory_directionality, trajectory_distances,
    trajectory_internal_variation, trajectory_lengths, trajectory_metrics,
    trajectory_speeds,
)

nan = np.nan
XY = np.array([[0, 0], [1, 0.5], [2.2, 0.4], [3, 1.5], [0.5, 1], [1.2, 1.8], [1.5, 2.5],
               [2.6, 2.2], [2, -1], [1.4, -0.5], [1.7, 0.3], [0.9, 0.9]])
ENT = list("AAAABBBBCCCC")
SV = [1, 2, 3, 4] * 3
TM = [0, 1, 2.5, 3, 0, 2, 3, 5, 1, 2, 3, 4]
TOL = 1e-6


def F(v, shape):
    return np.array(v, dtype=float).reshape(shape, order="F")


@pytest.fixture(scope="module")
def x():
    return define_trajectories.from_coords(XY, ENT, SV, TM)


@pytest.fixture(scope="module")
def xs():
    return define_trajectories.from_coords(XY, ENT, SV)


def x_():
    return define_trajectories.from_coords(XY, ENT, SV, TM)


def _seg_cos(xx):
    ini = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10])
    v = XY[ini + 1] - XY[ini]
    v = v / np.linalg.norm(v, axis=1)[:, None]
    return v @ v.T


def close(a, b, tol=TOL):
    np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=tol, rtol=0)


def test_lengths(x):
    # ecotraj::trajectoryLengths(x)
    r = trajectory_lengths(x)
    assert list(r.columns) == ["S1", "S2", "S3", "Path"] and list(r.index) == ["A", "B", "C"]
    close(r.values, F([1.1180339887, 1.0630145813, 0.7810249676, 1.2041594579, 0.7615773106, 0.8544003745, 1.3601470509, 1.1401754251, 1.0000000000, 3.6823404975, 2.9647673170, 2.6354253421], (3, 4)))


def test_lengths_relative(x):
    # ecotraj::trajectoryLengths(x, relativeToInitial=TRUE)
    r = trajectory_lengths(x, relative_to_initial=True)
    assert list(r.columns) == ["Lt1_t2", "Lt1_t3", "Lt1_t4", "Path"]
    close(r.values, F([1.1180339887, 1.0630145813, 0.7810249676, 2.2360679775, 1.8027756377, 1.3341664064, 3.3541019662, 2.4186773245, 2.1954498400, 6.7082039325, 5.2844675435, 4.3106412140], (3, 4)))


def test_speeds(x):
    # ecotraj::trajectorySpeeds(x)
    close(trajectory_speeds(x).values, F([1.1180339887, 0.5315072906, 0.7810249676, 0.8027729719, 0.7615773106, 0.8544003745, 2.7202941017, 0.5700877125, 1.0000000000, 1.2274468325, 0.5929534634, 0.8784751140], (3, 4)))


def test_angles(x):
    # ecotraj::trajectoryAngles(x)  (columns S1-S2, S2-S3, mean, sd, rho)
    r = trajectory_angles(x)
    assert list(r.columns) == ["S1-S2", "S2-S3", "mean", "sd", "rho"]
    close(r.values, F([31.3286928678, 17.9873346521, 70.7504741273, 58.7362683056, 82.0565281894, 73.6861475737, 45.0324805867, 50.0219314207, 72.2183108505, 0.2403313055, 0.5747964761, 0.0256199851, 0.9715334609, 0.8477279617, 0.9996718620], (3, 5)))
    # stats=FALSE
    assert list(trajectory_angles(x, stats=False).columns) == ["S1-S2", "S2-S3"]


def test_angles_all(x):
    # ecotraj::trajectoryAngles(x, all=TRUE)  -> A1..A4 (triplets 123,124,134,234), mean, sd, rho
    r = trajectory_angles(x, all=True)
    assert list(r.columns) == ["A1", "A2", "A3", "A4", "mean", "sd", "rho"]
    close(r.values, F([31.3286928678, 17.9873346521, 70.7504741273, 0.0000000000, 32.8686789334, 30.5406048497, 43.6677801461, 71.5650511771, 40.1354855622, 58.7362683056, 82.0565281894, 73.6861475737, 33.7010235511, 51.1780970691, 53.8085669410, 0.3795302738, 0.4699278014, 0.3305386186, 0.9305107832, 0.8954614838, 0.9468374109], (3, 7)))


def test_angles_relative_to_initial(x):
    # ecotraj::trajectoryAngles(x, relativeToInitial=TRUE)
    r = trajectory_angles(x, relative_to_initial=True)
    assert list(r.columns) == ["t1-S2", "t1-S3", "mean", "sd", "rho"]
    close(r.values, F([31.3286928678, 17.9873346521, 70.7504741273, 43.6677801461, 71.5650511771, 40.1354855622, 37.4982365070, 44.7761929146, 55.4429798448, 0.1077831645, 0.4765173863, 0.2687813556, 0.9942082321, 0.8926734792, 0.9645228956], (3, 5)))


def test_directionality(x):
    # ecotraj::trajectoryDirectionality(x)
    close(trajectory_directionality(x).values, [0.8214573464, 0.7083979224, 0.7148708000])


def test_internal_variation(x):
    # ecotraj::trajectoryInternalVariation(x)  (ss_1..ss_4, internal_ss, internal_variance)
    r = trajectory_internal_variation(x)
    assert list(r.columns) == ["ss_1", "ss_2", "ss_3", "ss_4", "internal_ss", "internal_variance"]
    close(r.values, F([2.7625000000, 1.6681250000, 1.1056250000, 0.3125000000, 0.0681250000, 0.1906250000, 0.4625000000, 0.3931250000, 0.1806250000, 2.9125000000, 1.4281250000, 1.3106250000, 6.4500000000, 3.5575000000, 2.7875000000, 2.1500000000, 1.1858333333, 0.9291666667], (3, 6)))
    # relativeContributions=TRUE
    r = trajectory_internal_variation(x, relative_contributions=True)
    assert list(r.columns)[0] == "rc_1"
    close(r.values, F([0.4282945736, 0.4689037245, 0.3966367713, 0.0484496124, 0.0191496838, 0.0683856502, 0.0717054264, 0.1105059733, 0.0647982063, 0.4515503876, 0.4014406184, 0.4701793722, 6.4500000000, 3.5575000000, 2.7875000000, 2.1500000000, 1.1858333333, 0.9291666667], (3, 6)))


def test_metrics(x):
    # ecotraj::trajectoryMetrics(x)[, -1]
    r = trajectory_metrics(x)
    assert list(r.columns) == ["trajectory", "n", "t_start", "t_end", "duration", "length",
                               "mean_speed", "mean_angle", "directionality", "internal_ss",
                               "internal_variance"]
    close(r.drop(columns="trajectory").values, F([4.0000000000, 4.0000000000, 4.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 3.0000000000, 5.0000000000, 4.0000000000, 3.0000000000, 5.0000000000, 3.0000000000, 3.6823404975, 2.9647673170, 2.6354253421, 1.2274468325, 0.5929534634, 0.8784751140, 45.0324805867, 50.0219314207, 72.2183108505, 0.8214573464, 0.7083979224, 0.7148708000, 6.4500000000, 3.5575000000, 2.7875000000, 2.1500000000, 1.1858333333, 0.9291666667], (3, 10)))


_DS_R = F([0.0000000000, 1.2041594579, 2.2360679775, 1.3152946438, 2.1633307653, 2.9154759474, 1.8543375968, 1.4866068747, 1.3780339887, 1.2041594579, 0.0000000000, 1.3601470509, 1.6744831457, 2.0346142564, 2.0610863454, 2.2811924193, 0.9633275663, 1.4641594579, 2.2360679775, 1.3601470509, 0.0000000000, 1.8248287591, 1.8012758241, 1.8439088915, 2.5612496950, 1.7691806013, 1.7691806013, 1.3152946438, 1.6744831457, 1.8248287591, 0.0000000000, 1.0630145813, 1.8027756377, 2.5000000000, 1.7492855685, 1.3668811387, 2.1633307653, 2.0346142564, 1.8012758241, 1.0630145813, 0.0000000000, 1.1401754251, 3.0016662040, 2.3086792761, 1.7088007491, 2.9154759474, 2.0610863454, 1.8439088915, 1.8027756377, 1.1401754251, 0.0000000000, 3.7016281719, 2.9206032043, 2.8489761742, 1.8543375968, 2.2811924193, 2.5612496950, 2.5000000000, 3.0016662040, 3.7016281719, 0.0000000000, 0.8544003745, 1.4866068747, 1.4866068747, 0.9633275663, 1.7691806013, 1.7492855685, 2.3086792761, 2.9206032043, 0.8544003745, 0.0000000000, 1.0000000000, 1.3780339887, 1.4641594579, 1.7691806013, 1.3668811387, 1.7088007491, 2.8489761742, 1.4866068747, 1.0000000000, 0.0000000000], (9, 9))


def test_segment_distances_directed_nonobtuse():
    # ecotraj::segmentDistances(x)$Dseg. For segment pairs whose angle is < 90 deg
    # the directed-segment distance equals the Hausdorff distance (exact match);
    # the obtuse-pair penalty is covered by the full test below.
    r = segment_distances(x_()), segment_distances(x_(), "Hausdorff")
    assert r[0]["Dseg"].index[0] == "A[1-2]" and r[0]["Dseg"].index[-1] == "C[3-4]"
    ang = _seg_cos(x_()) >= 0
    close(r[0]["Dseg"].values[ang], _DS_R[ang])


def test_segment_distances_directed_full():
    close(segment_distances(x_())["Dseg"].values, _DS_R)


def test_segment_distances_hausdorff(x):
    # ecotraj::segmentDistances(x, "Hausdorff") and its Dini/Dfin/Dinifin
    r = segment_distances(x, "Hausdorff")
    close(r["Dseg"].values, F([0.0000000000, 1.2041594579, 2.2360679775, 1.3152946438, 2.1633307653, 2.9154759474, 1.8027756377, 1.4866068747, 1.2727922061, 1.2041594579, 0.0000000000, 1.3601470509, 1.6744831457, 2.0346142564, 2.0610863454, 1.4117731575, 0.9633275663, 0.5099019514, 2.2360679775, 1.3601470509, 0.0000000000, 1.8248287591, 1.8012758241, 1.8439088915, 2.5612496950, 1.7691806013, 1.7691806013, 1.3152946438, 1.6744831457, 1.8248287591, 0.0000000000, 1.0630145813, 1.8027756377, 2.5000000000, 1.7492855685, 1.3640452592, 2.1633307653, 2.0346142564, 1.8012758241, 1.0630145813, 0.0000000000, 1.1401754251, 3.0016662040, 2.3086792761, 1.7088007491, 2.9154759474, 2.0610863454, 1.8439088915, 1.8027756377, 1.1401754251, 0.0000000000, 3.2557641192, 2.9206032043, 2.0698569256, 1.8027756377, 1.4117731575, 2.5612496950, 2.5000000000, 3.0016662040, 3.2557641192, 0.0000000000, 0.8544003745, 1.4866068747, 1.4866068747, 0.9633275663, 1.7691806013, 1.7492855685, 2.3086792761, 2.9206032043, 0.8544003745, 0.0000000000, 1.0000000000, 1.2727922061, 0.5099019514, 1.7691806013, 1.3640452592, 1.7088007491, 2.0698569256, 1.4866068747, 1.0000000000, 0.0000000000], (9, 9)))
    close(r["Dini"].values, F([0.0000000000, 1.1180339887, 2.2360679775, 1.1180339887, 2.1633307653, 2.9154759474, 2.2360679775, 1.4866068747, 1.7262676502, 1.1180339887, 0.0000000000, 1.2041594579, 0.7071067812, 1.3152946438, 2.0615528128, 1.8027756377, 1.0770329614, 0.7280109889, 2.2360679775, 1.2041594579, 0.0000000000, 1.8027756377, 1.7204650534, 2.2135943621, 1.4142135624, 1.2041594579, 0.5099019514, 1.1180339887, 0.7071067812, 1.8027756377, 0.0000000000, 1.0630145813, 1.8027756377, 2.5000000000, 1.7492855685, 1.3892443989, 2.1633307653, 1.3152946438, 1.7204650534, 1.0630145813, 0.0000000000, 0.7615773106, 2.9120439557, 2.3086792761, 1.5811388301, 2.9154759474, 2.0615528128, 2.2135943621, 1.8027756377, 0.7615773106, 0.0000000000, 3.5355339059, 3.0016662040, 2.2090722034, 2.2360679775, 1.8027756377, 1.4142135624, 2.5000000000, 2.9120439557, 3.5355339059, 0.0000000000, 0.7810249676, 1.3341664064, 1.4866068747, 1.0770329614, 1.2041594579, 1.7492855685, 2.3086792761, 3.0016662040, 0.7810249676, 0.0000000000, 0.8544003745, 1.7262676502, 0.7280109889, 0.5099019514, 1.3892443989, 1.5811388301, 2.2090722034, 1.3341664064, 0.8544003745, 0.0000000000], (9, 9)))
    close(r["Dfin"].values, F([0.0000000000, 1.2041594579, 2.2360679775, 1.3152946438, 2.0615528128, 2.3345235060, 1.0770329614, 0.7280109889, 0.4123105626, 1.2041594579, 0.0000000000, 1.3601470509, 1.7204650534, 2.2135943621, 1.8439088915, 1.2041594579, 0.5099019514, 1.3928388277, 2.2360679775, 1.3601470509, 0.0000000000, 1.8248287591, 1.8027756377, 0.8062257748, 2.5612496950, 1.7691806013, 2.1840329668, 1.3152946438, 1.7204650534, 1.8248287591, 0.0000000000, 0.7615773106, 1.4560219779, 2.3086792761, 1.5811388301, 0.9486832981, 2.0615528128, 2.2135943621, 1.8027756377, 0.7615773106, 0.0000000000, 1.1401754251, 3.0016662040, 2.2090722034, 1.7088007491, 2.3345235060, 1.8439088915, 0.8062257748, 1.4560219779, 1.1401754251, 0.0000000000, 2.9546573405, 2.1023796042, 2.1400934559, 1.0770329614, 1.2041594579, 2.5612496950, 2.3086792761, 3.0016662040, 2.9546573405, 0.0000000000, 0.8544003745, 1.4866068747, 0.7280109889, 0.5099019514, 1.7691806013, 1.5811388301, 2.2090722034, 2.1023796042, 0.8544003745, 0.0000000000, 1.0000000000, 0.4123105626, 1.3928388277, 2.1840329668, 0.9486832981, 1.7088007491, 2.1400934559, 1.4866068747, 1.0000000000, 0.0000000000], (9, 9)))
    close(r["Dinifin"].values, F([1.1180339887, 0.0000000000, 1.2041594579, 0.7071067812, 1.3152946438, 2.0615528128, 1.8027756377, 1.0770329614, 0.7280109889, 2.2360679775, 1.2041594579, 0.0000000000, 1.8027756377, 1.7204650534, 2.2135943621, 1.4142135624, 1.2041594579, 0.5099019514, 3.3541019662, 2.2360679775, 1.3601470509, 2.5495097568, 1.8248287591, 1.8027756377, 2.6925824036, 2.5612496950, 1.7691806013, 2.1633307653, 1.3152946438, 1.7204650534, 1.0630145813, 0.0000000000, 0.7615773106, 2.9120439557, 2.3086792761, 1.5811388301, 2.9154759474, 2.0615528128, 2.2135943621, 1.8027756377, 0.7615773106, 0.0000000000, 3.5355339059, 3.0016662040, 2.2090722034, 3.4058772732, 2.3345235060, 1.8439088915, 2.4186773245, 1.4560219779, 1.1401754251, 3.2557641192, 2.9546573405, 2.1023796042, 1.4866068747, 1.0770329614, 1.2041594579, 1.7492855685, 2.3086792761, 3.0016662040, 0.7810249676, 0.0000000000, 0.8544003745, 1.7262676502, 0.7280109889, 0.5099019514, 1.3892443989, 1.5811388301, 2.2090722034, 1.3341664064, 0.8544003745, 0.0000000000, 1.2727922061, 0.4123105626, 1.3928388277, 0.4123105626, 0.9486832981, 1.7088007491, 2.1954498400, 1.4866068747, 1.0000000000], (9, 9)))


def test_trajectory_distances_dspd(x):
    # ecotraj::trajectoryDistances(x) and symmetrization=NULL
    close(trajectory_distances(x).values, F([0.0000000000, 1.6252554955, 1.3843735514, 1.6252554955, 0.0000000000, 1.9234707948, 1.3843735514, 1.9234707948, 0.0000000000], (3, 3)))
    close(trajectory_distances(x, symmetrization=None).values, F([0.0000000000, 1.6534931198, 1.3985663839, 1.5970178712, 0.0000000000, 1.8720555690, 1.3701807188, 1.9748860206, 0.0000000000], (3, 3)))


def test_trajectory_distances_spd(x):
    # ecotraj::trajectoryDistances(x, "SPD"); SPD ignores max/min (always mean)
    close(trajectory_distances(x, "SPD").values, F([0.0000000000, 1.1120362830, 0.8425579201, 1.1120362830, 0.0000000000, 1.3887508220, 0.8425579201, 1.3887508220, 0.0000000000], (3, 3)))
    close(trajectory_distances(x, "SPD", symmetrization=None).values, F([0.0000000000, 1.1476101435, 0.7321471505, 1.0764624226, 0.0000000000, 1.4950529916, 0.9529686897, 1.2824486524, 0.0000000000], (3, 3)))
    close(trajectory_distances(x, "SPD", symmetrization="max").values, F([0.0000000000, 1.1120362830, 0.8425579201, 1.1120362830, 0.0000000000, 1.3887508220, 0.8425579201, 1.3887508220, 0.0000000000], (3, 3)))
    close(trajectory_distances(x, "SPD", symmetrization="min").values, F([0.0000000000, 1.1120362830, 0.8425579201, 1.1120362830, 0.0000000000, 1.3887508220, 0.8425579201, 1.3887508220, 0.0000000000], (3, 3)))


def test_trajectory_distances_hausdorff(x):
    # ecotraj::trajectoryDistances(x, "Hausdorff"); diagonal is non-zero without symmetrization
    close(trajectory_distances(x, "Hausdorff").values, F([0.0000000000, 2.9154759474, 2.5612496950, 2.9154759474, 0.0000000000, 3.2557641192, 2.5612496950, 3.2557641192, 0.0000000000], (3, 3)))
    close(trajectory_distances(x, "Hausdorff", symmetrization=None).values, F([2.2360679775, 2.9154759474, 2.5612496950, 2.9154759474, 1.8027756377, 3.2557641192, 2.5612496950, 3.2557641192, 1.4866068747], (3, 3)))


def test_convergence(x,xs):
    # ecotraj::trajectoryConvergence(xs, type=...)   (xs: same data without times)
    for t, tag in [("pairwise.asymmetric", "pairwise.asymmetric"), ("pairwise.symmetric", "pairwise.symmetric")]:
        r = trajectory_convergence(xs, type=t)
        tau = F({"pairwise.asymmetric": [nan, 0.3333333333, -0.6666666667, 0.0000000000, nan, -1.0000000000, 0.3333333333, 1.0000000000, nan], "pairwise.symmetric": [nan, 0.0000000000, -0.3333333333, 0.0000000000, nan, -1.0000000000, -0.3333333333, -1.0000000000, nan]}[tag], (3, 3))
        p = F({"pairwise.asymmetric": [nan, 0.7500000000, 0.3333333333, 1.0000000000, nan, 0.0833333333, 0.7500000000, 0.0833333333, nan], "pairwise.symmetric": [nan, 1.0000000000, 0.7500000000, 1.0000000000, nan, 0.0833333333, 0.7500000000, 0.0833333333, nan]}[tag], (3, 3))
        close(r["tau"].values, tau)
        close(r["p_value"].values, p)
    r = trajectory_convergence(xs, type="multiple")
    close(r["tau"], -0.3333333333)
    close(r["p_value"], 0.7500000000)


def test_triangle_correction_angles_and_directionality():
    # d[1,3] set to 4.5 (violates triangle inequality) in R:
    #   d <- as.matrix(dist(xy)); d[1,3] <- d[3,1] <- 4.5; xn <- defineTrajectories(as.dist(d), ent, sv, tm)
    from scipy.spatial.distance import pdist, squareform
    d = squareform(pdist(XY))
    d[0, 2] = d[2, 0] = 4.5
    xn = define_trajectories(d, ENT, SV, TM)
    # trajectoryAngles(xn, stats=FALSE)  (first angle ~0 after correction: float noise ~1e-6)
    close(trajectory_angles(xn, stats=False).values, F([0.0000008538, 17.9873346521, 70.7504741273, 58.7362683056, 82.0565281894, 73.6861475737], (3, 2)), 1e-5)
    # add=FALSE gives NaN for the impossible triplet
    close(trajectory_angles(xn, stats=False, add=False).values, F([nan, 17.9873346521, 70.7504741273, 58.7362683056, 82.0565281894, 73.6861475737], (3, 2)), 1e-5)
    close(trajectory_directionality(xn).values, [0.5887513148, 0.7083979224, 0.7148708000])
    # add=FALSE: NaN triplets are excluded from numerator and denominator
    close(trajectory_directionality(xn, add=False).values, [0.5076718490, 0.7083979224, 0.7148708000])
    assert not is_metric(d)
    assert is_metric(squareform(pdist(XY)))


def test_is_metric_relative_tolerance():
    # R: d <- abs(outer(c(0,1,10,20), c(0,1,10,20), "-")); d[1,3] <- d[3,1] <- 0.5;
    #    bisection on is.metric(as.dist(d), tol=) gives threshold 0.8333333 = 7.5/9... (violation / d_ij)
    v = np.array([0, 1, 10, 20.0])
    d = np.abs(v[:, None] - v[None, :])
    d[0, 2] = d[2, 0] = 0.5
    assert not is_metric(d, tol=0.83)
    assert is_metric(d, tol=0.84)


def test_directionality_permutation_pvalue(x):
    r = trajectory_directionality(x, nperm=99, rng=1)
    assert list(r.columns) == ["directionality", "p.value"]
    close(r["directionality"].values, [0.8214573464, 0.7083979224, 0.7148708000])
    assert np.all((r["p.value"] >= 1 / 100) & (r["p.value"] <= 1))
    # p-values are multiples of 1/(1+nperm)
    close(r["p.value"].values * 100, np.round(r["p.value"].values * 100))


def test_directionality_straight_line_and_reversal():
    t = np.arange(6.0)[:, None] * np.array([[1.0, 0.0]])
    assert directionality_coords(t) == pytest.approx(1.0)
    zig = np.array([[0, 0], [1, 0], [0, 0], [1, 0.0]])
    assert directionality_coords(zig) < 0.6


def test_clr():
    a = np.array([[1.0, 2.0, 4.0], [3.0, 3.0, 3.0]])
    c = clr(a)
    close(c.sum(axis=1), [0, 0])
    close(c[1], [0, 0, 0])
    close(c[0], np.log([1, 2, 4]) - np.log(8) / 3 * 1)


def test_signflip_and_split():
    C = np.cumsum(np.random.default_rng(0).normal(size=(8, 3)), axis=0)
    ones = np.ones(7)
    np.testing.assert_allclose(signflip_path(C, ones), C)
    flipped = signflip_path(C, -ones)
    np.testing.assert_allclose(flipped[1:] - flipped[:-1], -(C[1:] - C[:-1]))
    signs = np.random.default_rng(1).choice([-1.0, 1.0], size=(50, 7))
    r = signflip_test(C, signs=signs)
    assert len(r["null"]) == 50 and r["p_value"] == (1 + np.sum(r["null"] >= r["stat"])) / 51
    X = np.cumsum(np.random.default_rng(2).normal(size=(8, 2)), axis=0)
    s = increment_split(C, X)
    dC = np.diff(C, axis=0)
    np.testing.assert_allclose(np.diff(s["market"], axis=0) + np.diff(s["residual"], axis=0), dC)
    np.testing.assert_allclose(s["residual"][0], C[0])
    # exactly explained increments -> energy 1 and zero residual
    B = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, -1.0]])
    C2 = np.vstack([[0, 0, 0], np.cumsum(np.diff(X, axis=0) @ B, axis=0)])
    s2 = increment_split(C2, X)
    assert s2["step_energy_market"] == pytest.approx(1.0)
