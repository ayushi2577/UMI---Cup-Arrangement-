import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, 'eval_sim.py')).read().split("device = torch.device('cuda:0')")[0]
ckpt = sys.argv[1]
nep = int(sys.argv[2]) if len(sys.argv) > 2 else 40
sys.argv = ['eval_sim.py', '--ckpt', ckpt, '--lead', '6']
exec(compile(SRC, 'eval_sim.py', 'exec'), globals())
device = torch.device('cuda:0')
policy, cfg = load_policy(os.path.expanduser(ckpt), device)
env = UmiSimEnv(build_scene(args.menagerie))
ik = PoseIK(env)
m = env.model
g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'cup_geom')
gain0, bias0 = m.actuator_gainprm[7].copy(), m.actuator_biasprm[7].copy()
print('gripper actuator: gain', np.round(gain0[:3], 4), '| bias', np.round(bias0[:3], 3), '| force range', m.actuator_forcerange[7])


def configure(torsion, kscale):
    m.geom_condim[g] = 4
    m.geom_friction[g] = [1.5, torsion, 0.001]
    m.actuator_gainprm[7] = gain0 * kscale
    m.actuator_biasprm[7] = bias0 * kscale


configs = [
    ('A current (torsion 0.05, grip x1)', 0.05, 1.0),
    ('B torsion 0.2', 0.2, 1.0),
    ('C grip force x4', 0.05, 4.0),
    ('D torsion 0.2 + grip x4', 0.2, 4.0),
]
print('config                             | policy success | lifted >6cm | median cup-saucer dist of lifted (cm) | expert')
for name, tor, k in configs:
    configure(tor, k)
    res = [rollout(env, policy, cfg, ik, s, device)[0] for s in range(nep)]
    lifted = [r for r in res if r['max_cup_z'] > 0.06]
    med = 100 * np.median([r['dxy'] for r in lifted]) if lifted else float('nan')
    ex_ok = 0
    for s in range(10):
        env.reset(seed=s); ex = ScriptedExpert(env); ex.run(); ex_ok += int(ex.success()[0])
    print('%-34s |     %2d/%d     |   %2d/%d   |                %5.1f                |  %2d/10' % (
        name, sum(r['ok'] for r in res), nep, len(lifted), nep, med, ex_ok))
env.close()
