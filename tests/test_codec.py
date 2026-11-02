import numpy as np

from surgical_tokens.codec import CodecFrame, patch_importance


def test_patch_importance_localizes_residual():
    rgb = np.zeros((32, 32, 3), np.uint8)
    residual = np.zeros((32, 32), np.float32)
    residual[:16, 16:] = 100
    frame = CodecFrame(rgb, 0.0, False, np.zeros((32, 32, 2), np.float32), residual)
    score = patch_importance(frame, 16, motion_weight=0)
    assert score.shape == (2, 2)
    assert score[0, 1] > score[0, 0]
