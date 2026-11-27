"""Builds umi_sim/eval_sim10.py from eval_sim9.py.

Adds a DIAGNOSTIC ONLY ground-truth log (never used for control):
  - at grasp detection (g0) and at first lift above 6 cm (g1): cup-minus-TCP offset
    in world frame and in the TCP (hand) frame, cup tilt (z-axis of cup, 1.0 = upright),
    measured gripper width
  - max cup-TCP separation while carried (maxsep)
  - at the end: a GT table of means per outcome group, plus one GTSEED line per episode.
Controller behaviour is identical to eval_sim9.py. Refuses to write on anchor mismatch.
"""
import os, sys

SRC = os.path.expanduser('~/umi_mujoco/sim/umi_sim/eval_sim9.py')
DST = os.path.expanduser('~/umi_mujoco/sim/umi_sim/eval_sim10.py')

s = open(SRC).read()


def once(text, anchor):
    n = text.count(anchor)
    if n != 1:
        sys.exit('ANCHOR MISMATCH (%d matches), nothing written:\n%s' % (n, anchor))


# 1) helper before rollout()
A1 = "def rollout(env, policy, cfg, ik, seed, device, record=False):\n"
once(s, A1)
H1 = """def gt_snap(env, w):
    # ground truth, diagnosis only
    d = env.data
    cup = d.xpos[env.cup_id]
    tcp = d.site_xpos[env.tcp_id]
    R = d.site_xmat[env.tcp_id].reshape(3, 3)
    off = cup - tcp
    loc = R.T @ off
    return dict(w=tuple(float(v) for v in off), l=tuple(float(v) for v in loc),
                tilt=float(d.xmat[env.cup_id].reshape(3, 3)[2, 2]), gw=float(w),
                sep=float(np.linalg.norm(off)), z=float(cup[2]))


"""
s = s.replace(A1, H1 + A1)

# 2) grasp-detection snapshot
A2 = "            if st['holding_n'] >= 15:\n                st['grasped'] = True\n"
once(s, A2)
s = s.replace(A2, A2 + "                if 'g0' not in st:\n                    st['g0'] = gt_snap(env, w_m)\n")

# 3) lift-off snapshot and max separation
A3 = "            max_cup_z = max(max_cup_z, float(env.get_object_state()['cup_pos'][2]))\n"
once(s, A3)
H3 = """            if 'g1' not in st and float(env.get_object_state()['cup_pos'][2]) > 0.06:
                st['g1'] = gt_snap(env, float(obs['robot0_gripper_width'][0]))
            if 'g1' in st and not st.get('release'):
                st['maxsep'] = max(st.get('maxsep', 0.0), gt_snap(env, 0.0)['sep'])
"""
s = s.replace(A3, A3 + H3)

# 4) store in the result dict
A4 = "max_cup_z=max_cup_z, steps=steps)"
once(s, A4)
s = s.replace(A4, "max_cup_z=max_cup_z, steps=steps, gt=dict(g0=st.get('g0'), g1=st.get('g1'), maxsep=st.get('maxsep')))")

# 5) table at the end
A5 = "print('cup lifted above 6 cm in %d/%d episodes' % (sum(r['max_cup_z'] > 0.06 for r in results), len(results)))"
once(s, A5)
H5 = """def _grp(r):
    if r['ok']:
        return 'success'
    return 'never_lifted' if r['max_cup_z'] < 0.06 else 'failed_after_lift'
for r in results:
    g = r.get('gt', {})
    f = lambda x: ('l(mm)=%+.1f,%+.1f,%+.1f tilt=%.2f w=%.3f' % (x['l'][0]*1000, x['l'][1]*1000, x['l'][2]*1000, x['tilt'], x['gw'])) if x else 'none'
    print('GTSEED %3d %-17s g0 %s | g1 %s | maxsep %.3f' % (r['seed'], _grp(r), f(g.get('g0')), f(g.get('g1')), g.get('maxsep') or 0.0))
for _g in ('success', 'failed_after_lift', 'never_lifted'):
    for _k, _lab in (('g0', 'grasp'), ('g1', 'liftoff')):
        _rows = [r['gt'][_k] for r in results if _grp(r) == _g and r.get('gt', {}).get(_k)]
        if not _rows:
            print('GT %-17s %-7s n=0' % (_g, _lab))
            continue
        _L = np.array([x['l'] for x in _rows]) * 1000.0
        _W = np.array([x['w'] for x in _rows]) * 1000.0
        _T = np.array([x['tilt'] for x in _rows])
        _G = np.array([x['gw'] for x in _rows])
        print('GT %-17s %-7s n=%2d | hand-frame mm mean x %+.1f y %+.1f z %+.1f sd %.1f %.1f %.1f | world mm dx %+.1f dy %+.1f dz %+.1f | tilt %.3f | width %.3f' % (
            _g, _lab, len(_rows), _L[:, 0].mean(), _L[:, 1].mean(), _L[:, 2].mean(), _L[:, 0].std(), _L[:, 1].std(), _L[:, 2].std(),
            _W[:, 0].mean(), _W[:, 1].mean(), _W[:, 2].mean(), _T.mean(), _G.mean()))
    _ms = [r['gt']['maxsep'] for r in results if _grp(r) == _g and r.get('gt', {}).get('maxsep') is not None]
    print('GT %-17s maxsep mean %.3f n=%d' % (_g, (sum(_ms) / len(_ms)) if _ms else 0.0, len(_ms)))
"""
s = s.replace(A5, H5 + A5)

open(DST, 'w').write(s)
print('wrote', DST)
