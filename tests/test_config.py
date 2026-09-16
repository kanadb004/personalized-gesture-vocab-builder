from pgvb import config


def test_load_default_returns_config():
    cfg = config.load()
    assert isinstance(cfg, config.Config)


def test_every_section_present_with_default_values():
    cfg = config.load()

    assert cfg.camera.index == 0
    assert cfg.camera.width == 640
    assert cfg.camera.height == 480
    assert cfg.camera.mirror is True

    assert cfg.landmarks.num_hands == 1
    assert cfg.landmarks.min_hand_score == 0.5
    assert cfg.landmarks.min_hand_detection_confidence == 0.5
    assert cfg.landmarks.min_tracking_confidence == 0.5

    assert cfg.features.rotate is False
    assert cfg.features.dim == 63

    assert cfg.backbone.input_dim == 63
    assert cfg.backbone.embed_dim == 64
    assert cfg.backbone.hidden_dims == [128, 128]
    assert cfg.backbone.dropout == 0.2
    assert cfg.backbone.id == "backbone_v1"

    assert cfg.recognize.max_distance == 0.15
    assert cfg.recognize.min_margin == 0.0

    assert cfg.enroll.min_samples == 5
    assert cfg.enroll.max_samples == 10
    assert cfg.enroll.stability_frames == 5
    assert cfg.enroll.stability_max_motion == 0.02
    assert cfg.enroll.collision_distance == 0.2

    assert cfg.smoothing.window == 12
    assert cfg.smoothing.min_votes == 8
    assert cfg.smoothing.release_frames == 6
    assert cfg.smoothing.cooldown_ms == 800

    assert cfg.output.tts_backend == "pyttsx3"
    assert cfg.output.tts_rate == 175
    assert cfg.output.tts_voice == "default"
    assert cfg.output.board_size == 5
    assert cfg.output.muted is False

    assert cfg.profile.schema_version == 1
