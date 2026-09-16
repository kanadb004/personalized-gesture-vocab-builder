import numpy as np

from pgvb.features import feature_dim, normalize_landmarks


def _random_landmarks(rng: np.random.Generator) -> np.ndarray:
    lm = rng.uniform(-1.0, 1.0, size=(21, 3)).astype(np.float64)
    lm[9] += 0.5  # keep the middle-MCP away from the wrist to avoid a degenerate scale
    return lm


def test_output_shape_and_dtype():
    rng = np.random.default_rng(0)
    lm = _random_landmarks(rng)
    out = normalize_landmarks(lm, "Right")
    assert out.shape == (feature_dim(),)
    assert out.dtype == np.float32


def test_translation_invariance():
    rng = np.random.default_rng(1)
    lm = _random_landmarks(rng)
    shift = np.array([1.5, -2.0, 0.3])
    out1 = normalize_landmarks(lm, "Right")
    out2 = normalize_landmarks(lm + shift, "Right")
    np.testing.assert_allclose(out1, out2, atol=1e-5)


def test_scale_invariance():
    rng = np.random.default_rng(2)
    lm = _random_landmarks(rng)
    out1 = normalize_landmarks(lm, "Right")
    out2 = normalize_landmarks(lm * 3.0, "Right")
    np.testing.assert_allclose(out1, out2, atol=1e-5)


def test_left_right_mirror_to_same_vector():
    rng = np.random.default_rng(3)
    lm = _random_landmarks(rng)
    mirrored = lm.copy()
    mirrored[:, 0] *= -1.0
    out_right = normalize_landmarks(lm, "Right")
    out_left = normalize_landmarks(mirrored, "Left")
    np.testing.assert_allclose(out_right, out_left, atol=1e-5)
