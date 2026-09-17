from pgvb.smoothing import TriggerStateMachine, VoteWindow


def test_vote_window_confirms_after_min_votes():
    vw = VoteWindow(window=5, min_votes=3)
    confirmed = [vw.push(label) for label in [None, "a", "a", None, "a"]]
    assert confirmed[-1] == "a"


def test_vote_window_needs_min_votes_within_the_window():
    vw = VoteWindow(window=5, min_votes=4)
    confirmed = [vw.push(label) for label in [None, "a", "a", None, "a"]]
    assert confirmed[-1] is None


def test_vote_window_flicker_never_confirms():
    vw = VoteWindow(window=10, min_votes=8)
    labels = ["a", None, "a", None, "a", None, "a", None, "a", None]
    confirmed = [vw.push(label) for label in labels]
    assert all(c is None for c in confirmed)


def test_trigger_fires_once_while_held():
    sm = TriggerStateMachine(release_frames=3, cooldown_ms=1000)
    triggers = [sm.update("a", ts) for ts in [0, 100, 200, 300, 400]]
    fired = [t for t in triggers if t is not None]
    assert len(fired) == 1
    assert fired[0].label == "a"


def test_trigger_brief_flicker_of_none_does_not_refire():
    sm = TriggerStateMachine(release_frames=5, cooldown_ms=0)
    seq = ["a", "a", None, "a", "a"]
    triggers = [sm.update(label, ts * 100) for ts, label in enumerate(seq)]
    fired = [t for t in triggers if t is not None]
    assert len(fired) == 1


def test_trigger_release_then_repeat_fires_again():
    sm = TriggerStateMachine(release_frames=2, cooldown_ms=0)
    seq = ["a", "a", None, None, "a"]
    triggers = [sm.update(label, ts * 1000) for ts, label in enumerate(seq)]
    fired = [t for t in triggers if t is not None]
    assert len(fired) == 2
    assert [t.label for t in fired] == ["a", "a"]


def test_trigger_cooldown_blocks_quick_refire_after_release():
    sm = TriggerStateMachine(release_frames=1, cooldown_ms=1000)
    seq = ["a", None, "a"]
    triggers = [sm.update(label, ts) for ts, label in zip([0, 100, 200], seq)]
    fired = [t for t in triggers if t is not None]
    assert len(fired) == 1


def test_trigger_different_labels_do_not_interfere():
    sm = TriggerStateMachine(release_frames=2, cooldown_ms=0)
    triggers = [
        sm.update("a", 0),
        sm.update("b", 100),
        sm.update("a", 200),
    ]
    fired = [t.label for t in triggers if t is not None]
    assert fired == ["a", "b", "a"]
