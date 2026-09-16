import numpy as np

from pgvb.prototypes import cosine_distances, make_prototype, nearest


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def test_make_prototype_is_unit_norm_mean():
    examples = np.array(
        [
            _unit(np.array([1.0, 0.0, 0.0])),
            _unit(np.array([0.9, 0.1, 0.0])),
        ]
    )
    proto = make_prototype(examples)
    assert proto.shape == (3,)
    np.testing.assert_allclose(np.linalg.norm(proto), 1.0, atol=1e-6)


def test_cosine_distance_zero_for_identical_vector():
    proto = _unit(np.array([1.0, 1.0, 0.0]))
    distances = cosine_distances(proto, proto[None, :])
    np.testing.assert_allclose(distances, [0.0], atol=1e-6)


def test_cosine_distance_two_for_opposite_vectors():
    proto = _unit(np.array([1.0, 0.0, 0.0]))
    distances = cosine_distances(-proto, proto[None, :])
    np.testing.assert_allclose(distances, [2.0], atol=1e-6)


def test_nearest_picks_closest_and_reports_margin():
    protos = np.stack(
        [
            _unit(np.array([1.0, 0.0, 0.0])),
            _unit(np.array([0.0, 1.0, 0.0])),
            _unit(np.array([0.0, 0.0, 1.0])),
        ]
    )
    e = _unit(np.array([0.9, 0.1, 0.0]))
    idx, d1, d2 = nearest(e, protos)
    assert idx == 0
    assert d1 < d2


def test_nearest_second_distance_is_inf_with_one_prototype():
    protos = _unit(np.array([1.0, 0.0, 0.0]))[None, :]
    e = _unit(np.array([0.9, 0.1, 0.0]))
    idx, d1, d2 = nearest(e, protos)
    assert idx == 0
    assert d2 == float("inf")
