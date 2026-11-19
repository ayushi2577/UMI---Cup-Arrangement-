"""Merge the base demo set with handoff-DAgger episodes into a new zarr.zip. The base file is only read.
  python merge_dagger.py --base ~/umi_mujoco/sim_demos/cup_sim_v2.zarr.zip --extra ~/umi_mujoco/sim_demos/dagger_a.zarr --out ~/umi_mujoco/sim_demos/cup_sim_v2_dagger.zarr
Writes OUT (directory) and OUT.zip. Train on OUT.zip."""
import os, sys, argparse
sys.path.insert(0, os.path.expanduser('~/umi_mujoco/universal_manipulation_interface'))
import numpy as np, zarr
try:
    import diffusion_policy.codecs.imagecodecs_numcodecs as codecs_mod
except ImportError:
    import umi.codecs.imagecodecs_numcodecs as codecs_mod
codecs_mod.register_codecs()

ap = argparse.ArgumentParser()
ap.add_argument('--base', required=True)
ap.add_argument('--extra', required=True, help='directory .zarr from record_dagger.py')
ap.add_argument('--out', required=True, help='directory ending in .zarr (must not exist)')
args = ap.parse_args()
base, extra, out = [os.path.expanduser(p) for p in (args.base, args.extra, args.out)]
if os.path.exists(out) or os.path.exists(out + '.zip'):
    sys.exit('refusing to overwrite %s' % out)

src = zarr.ZipStore(base, mode='r')
dst = zarr.DirectoryStore(out)
zarr.copy_store(src, dst)
src.close()
root = zarr.open_group(dst, mode='a')
ex = zarr.open_group(zarr.DirectoryStore(extra), mode='r')
ends_ex = ex['meta/episode_ends'][:]
n_ex = int(ends_ex[-1])
off = int(root['meta/episode_ends'][-1])
for k in root['data'].array_keys():
    if root['data'][k].shape[0] != off:
        sys.exit('base array %s has %d frames, expected %d' % (k, root['data'][k].shape[0], off))
    if ex['data'][k].shape[0] != n_ex:
        sys.exit('extra array %s has %d frames, expected %d' % (k, ex['data'][k].shape[0], n_ex))
for k in root['data'].array_keys():
    for i in range(0, n_ex, 2000):
        root['data'][k].append(ex['data'][k][i:i + 2000])
root['meta/episode_ends'].append((ends_ex + off).astype(np.int64))

zpath = out + '.zip'
with zarr.ZipStore(zpath, mode='w') as z:
    zarr.copy_store(zarr.DirectoryStore(out), z)
st = zarr.ZipStore(zpath, mode='r')
r = zarr.group(store=st)
e = r['meta/episode_ends'][:]
print('base episodes %d + extra %d = %d | frames %d' % (len(e) - len(ends_ex), len(ends_ex), len(e), int(e[-1])))
for k, v in r['data'].arrays():
    assert v.shape[0] == int(e[-1]), k
    print('  %-28s shape=%s' % (k, v.shape))
st.close()
print('wrote', zpath, '| %.2f GB' % (os.path.getsize(zpath) / 1e9))
