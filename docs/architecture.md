# Architecture

One frozen embedding space, built once offline from ASL landmarks (Phase 2), and two things a
caregiver does with it at runtime: enroll a personal gesture from a few examples, and have the
system recognize it live. Neither path ever updates the embedding network's weights.

## Pipeline

```mermaid
flowchart LR
    subgraph capture [Capture]
        cam[Camera frame] --> track[HandTracker\nMediaPipe HandLandmarker]
    end
    track -->|HandFrame or None| feat[normalize_landmarks\n63-d feature]
    feat --> embed[Backbone.embed\n64-d unit vector]
    embed --> rec[Recognizer.decide\nnearest prototype + threshold]
    rec -->|per-frame label or None| vote[VoteWindow\nmajority vote]
    vote -->|confirmed label| trig[TriggerStateMachine\nrising edge]
    trig -->|Trigger| speak[Speaker]
    trig -->|Trigger| board[MessageBoard]
```

`Pipeline.process` runs the whole left-to-right chain on a live camera frame;
`Pipeline.process_hand_frame` starts from an already-tracked `HandFrame` (or `None`), which is
how session replay and every evaluation script in Phase 5 exercise the same code path offline.

## Enrollment vs. recognition data flow

```mermaid
flowchart TD
    subgraph offline [Offline, once, Phase 2]
        asl[ASL landmark dataset] --> train[Prototypical-loss training]
        train --> backbone[backbone_v1.npz\nfrozen weights]
    end

    subgraph enroll [Enrollment, per gesture, training-free]
        demo[5 to 10 stable demonstrations] --> embedE[Backbone.embed each]
        embedE --> examples[Gesture.examples]
        examples --> proto[make_prototype\nunit-mean]
        proto --> profileJson[profiles/user.json]
    end

    subgraph recognize [Recognition, live]
        frame[Live frame] --> embedR[Backbone.embed]
        profileJson --> nearest[nearest prototype\ncosine distance]
        embedR --> nearest
        nearest --> decision[accept if d <= max_distance]
    end

    subgraph refine [Refinement, caregiver flags a miss]
        miss[One more stable sample] --> embedF[Backbone.embed]
        embedF --> append[append to that gesture's\nexamples only]
        append --> recompute[recompute that\nprototype only]
        recompute --> profileJson
    end

    backbone --> embedE
    backbone --> embedR
    backbone --> embedF
```

The backbone is shared and frozen across both paths. Enrollment and refinement only ever append
to, or recompute the prototype of, the one gesture being taught; every other gesture's examples
and prototype are untouched (`pgvb.enroll.Refiner`, verified by `tests/test_stability.py` and,
per participant, by `scripts/eval_stability.py`), which is what O4 requires.

## Module map

See `docs/PLAN.md` Section 3 for the full repository layout. In short: `pgvb/landmarks.py` and
`pgvb/features.py` are the capture-to-feature stage; `pgvb/embedding.py` wraps the frozen
backbone; `pgvb/profile.py` and `pgvb/prototypes.py` hold the per-user gesture store;
`pgvb/enroll.py`, `pgvb/recognize.py`, `pgvb/smoothing.py`, and `pgvb/output.py` implement
enrollment, recognition, smoothing, and speech; `pgvb/pipeline.py` wires all of it together;
`pgvb/gui/` is the Tkinter front end built in Phase 4.
