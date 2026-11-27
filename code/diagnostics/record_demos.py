"""Record scripted-expert demos in UMI's zarr format.
Usage: python record_demos.py --out OUT.zarr --n 5
Resumable: re-run the same command and it continues until --n successful demos exist."""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.expanduser('~/umi_mujoco/universal_manipulation_interface'))

import numpy as np, zarr, numcodecs, imageio
try:
    import diffusion_policy.codecs.imagecodecs_numcodecs as codecs_mod
except ImportError:
    import umi.codecs.imagecodecs_numcodecs as codecs_mod
codecs_mod.register_codecs()

from umi_sim.scene import build_scene
from umi_sim.env import UmiSimEnv
from umi_sim.expert import ScriptedExpert

ap = argparse.ArgumentParser()
ap.add_argument('--menagerie', default=os.path.expanduser('~/umi_mujoco/sim/mujoco_menagerie'))
ap.add_argument('--out', required=True, help='directory ending in .zarr')
ap.add_argument('--n', type=int, default=5, help='number of successful demos wanted')
ap.add_argument('--start_seed', type=int, default=1000)
ap.add_argument('--max_attempts', type=int, default=100000)
ap.add_argument('--ref', default=os.path.expanduser('~/umi_mujoco/data/cup_in_the_wild.zarr.zip'))
args = ap.parse_args()
args.out = os.path.expanduser(args.out)

LOWDIM = {
    'robot0_eef_pos': (np.float32, (3,)),
    'robot0_eef_rot_axis_angle': (np.float32, (3,)),
    'robot0_gripper_width': (np.float32, (1,)),
    'robot0_demo_start_pose': (np.float64, (6,)),
    'robot0_demo_end_pose': (np.float64, (6,)),
}


def img_compressor(ref):
    try:
        st = zarr.ZipStore(ref, mode='r')
        cfg = zarr.group(store=st)['data/camera0_rgb'].compressor.get_config()
        st.close()
        print("using the reference dataset's image compressor:", cfg)
        return numcodecs.get_codec(cfg)
    except Exception as e:
        print('could not copy the reference compressor (%s: %s); falling back to JpegXl(level=99)' % (type(e).__name__, e))
        return codecs_mod.JpegXl(level=99)


def open_out(path, comp):
    root = zarr.open_group(zarr.DirectoryStore(path), mode='a')
    data = root.require_group('data')
    meta = root.require_group('meta')
    if 'camera0_rgb' not in data:
        data.create_dataset('camera0_rgb', shape=(0, 224, 224, 3), chunks=(1, 224, 224, 3), dtype=np.uint8, compressor=comp)
        for k, (dt, tail) in LOWDIM.items():
            data.create_dataset(k, shape=(0,) + tail, chunks=(400,) + tail, dtype=dt, compressor=None)
        meta.create_dataset('episode_ends', shape=(0,), chunks=(1596,), dtype=np.int64, compressor=None)
    ends = meta['episode_ends']
    total = int(ends[-1]) if ends.shape[0] else 0
    for k in data.array_keys():          # repair if a previous run died mid-write
        a = data[k]
        if a.shape[0] != total:
            print('repairing %s: %d -> %d frames' % (k, a.shape[0], total))
            a.resize((total,) + a.shape[1:])
    return root


def run_episode(env, seed):
    frames = [env.reset(seed=seed)]
    ex = ScriptedExpert(env, render=True)
    ex.on_step = lambda obs, x, g: frames.append(obs)
    ex.run()
    ok, dxy, cz = ex.success()
    return frames, ok, dxy, cz


def unwrap_rotvec(rv):
    """Keep an axis-angle sequence continuous (same rotations, no +pi/-pi flips); start on the negative-x side."""
    out = rv.astype(np.float64).copy()
    def alt(r):
        th = np.linalg.norm(r)
        return r if th < 1e-6 else r - 2 * np.pi * r / th
    if out[0][0] > 0:
        out[0] = alt(out[0])
    for i in range(1, len(out)):
        r = out[i]
        a2 = alt(r)
        if np.linalg.norm(a2 - out[i - 1]) < np.linalg.norm(r - out[i - 1]):
            out[i] = a2
    return out


def to_arrays(frames):
    n = len(frames)
    pos = np.stack([f['robot0_eef_pos'] for f in frames]).astype(np.float64)
    rot = unwrap_rotvec(np.stack([f['robot0_eef_rot_axis_angle'] for f in frames]))
    return {
        'camera0_rgb': np.stack([f['camera0_rgb'] for f in frames]).astype(np.uint8),
        'robot0_eef_pos': pos.astype(np.float32),
        'robot0_eef_rot_axis_angle': rot.astype(np.float32),
        'robot0_gripper_width': np.stack([f['robot0_gripper_width'] for f in frames]).astype(np.float32),
        'robot0_demo_start_pose': np.tile(np.concatenate([pos[0], rot[0]]), (n, 1)),
        'robot0_demo_end_pose': np.tile(np.concatenate([pos[-1], rot[-1]]), (n, 1)),
    }


def append_episode(root, arrs):
    n = len(arrs['camera0_rgb'])
    for k, v in arrs.items():
        root['data'][k].append(v)
    ends = root['meta/episode_ends']
    prev = int(ends[-1]) if ends.shape[0] else 0
    ends.append(np.array([prev + n], dtype=np.int64))   # written last, so it is the commit marker


os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
comp = img_compressor(args.ref)
root = open_out(args.out, comp)
log_path = args.out + '.attempts.jsonl'
attempted = []
if os.path.exists(log_path):
    attempted = [json.loads(l)['seed'] for l in open(log_path) if l.strip()]
seed = max(attempted) + 1 if attempted else args.start_seed
n_ok = int(root['meta/episode_ends'].shape[0])
print('starting with %d demos already stored, want %d, next seed %d' % (n_ok, args.n, seed))

env = UmiSimEnv(build_scene(args.menagerie))
tries = ok_now = 0
t_all = time.time()
while n_ok < args.n and tries < args.max_attempts:
    t0 = time.time()
    try:
        frames, ok, dxy, cz = run_episode(env, seed)
    except Exception as e:
        print('seed %d crashed: %s: %s' % (seed, type(e).__name__, e))
        frames, ok, dxy, cz = [], False, float('nan'), float('nan')
    if ok:
        append_episode(root, to_arrays(frames))
        n_ok += 1
        ok_now += 1
    with open(log_path, 'a') as f:
        f.write(json.dumps({'seed': seed, 'ok': bool(ok), 'frames': len(frames), 'dxy': dxy}) + '\n')
    print('seed %5d | %s | %3d frames | xy dist %.3f | %.1fs | stored %d/%d'
          % (seed, 'OK  ' if ok else 'FAIL', len(frames), dxy, time.time() - t0, n_ok, args.n))
    seed += 1
    tries += 1
env.close()
print('\nthis run: %d successes in %d attempts, %.1f min' % (ok_now, tries, (time.time() - t_all) / 60))

# ---- zip it (UMI loads a .zarr.zip) and read it back
zpath = args.out + '.zip'
if os.path.exists(zpath):
    os.remove(zpath)
with zarr.ZipStore(zpath, mode='w') as dst:
    zarr.copy_store(zarr.DirectoryStore(args.out), dst)
print('wrote', zpath, '| %.2f GB' % (os.path.getsize(zpath) / 1e9))

st = zarr.ZipStore(zpath, mode='r')
r = zarr.group(store=st)
for k, v in r['data'].arrays():
    print('  %-28s shape=%s dtype=%s' % (k, v.shape, v.dtype))
ends = r['meta/episode_ends'][:]
lens = np.diff(np.concatenate([[0], ends]))
print('episodes:', len(ends), '| frames:', int(ends[-1]), '| length min/mean/max:', lens.min(), round(lens.mean(), 1), lens.max())
for k in ('robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width'):
    a = r['data'][k][:]
    print('  %-28s min %s max %s' % (k, np.round(a.min(0), 3), np.round(a.max(0), 3)))
idx = np.linspace(0, int(ends[0]) - 1, 6).astype(int)
imageio.imwrite(args.out + '_preview.png', np.concatenate([r['data']['camera0_rgb'][int(i)] for i in idx], axis=1))
print('preview of episode 0 (6 wrist frames):', args.out + '_preview.png')
st.close()
