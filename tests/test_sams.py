import numpy as np

from sams_clustering import SAMS, SAMSConfig


def test_sams_separates_two_modes():
    rng = np.random.default_rng(0)
    cluster1 = rng.normal(loc=0.0, scale=0.2, size=(10, 2))
    cluster2 = rng.normal(loc=5.0, scale=0.2, size=(10, 2))
    X = np.vstack([cluster1, cluster2])

    config = SAMSConfig(
        bandwidth=0.6,
        sample_fraction=1.0,
        max_iter=100,
        random_state=123,
        merge_radius=0.5,
    )
    sams = SAMS(config)
    modes, labels = sams.fit(X)

    assert modes.shape[1] == 2
    assert len(set(labels[:10])) == 1
    assert len(set(labels[10:])) == 1
    assert labels[0] != labels[-1]
