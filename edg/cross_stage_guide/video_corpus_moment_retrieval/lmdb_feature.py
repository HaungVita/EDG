"""Read TVR's msgpack-numpy LMDB as a small h5py-like mapping."""
from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy


class LmdbFeatureStore:
    def __init__(self, path):
        path = Path(path)
        if path.is_file() and path.name == "data.mdb":
            path = path.parent
        self.path = str(path)
        self.env = lmdb.open(
            self.path, readonly=True, lock=False, readahead=False,
            meminit=False, max_readers=256,
        )

    def __getitem__(self, key):
        key = key.encode("utf-8") if isinstance(key, str) else key
        with self.env.begin(buffers=True) as txn:
            value = txn.get(key)
            if value is None:
                raise KeyError(key)
            item = msgpack.unpackb(
                bytes(value), raw=False, object_hook=msgpack_numpy.decode
            )
        return item["features"]

    def close(self):
        self.env.close()


def open_video_features(path, h5driver=None):
    if Path(path).is_dir() or Path(path).name == "data.mdb":
        return LmdbFeatureStore(path)
    import h5py
    return h5py.File(path, "r", driver=h5driver)
