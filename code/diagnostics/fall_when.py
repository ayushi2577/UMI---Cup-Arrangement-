import csv, os, re, sys
H = os.path.expanduser('~/umi_mujoco')
log = sys.argv[1]
out = {}
for l in open(H + '/exp/' + log):
    m = re.match(r'GTX (\d+) (\S+)', l)
    if m: out[int(m.group(1))] = m.group(2)
print('seed outcome | stage when cup first falls/tips | step | cup_z mm, tilt, cup speed | gripper gap mm | hand-to-cup dx dy dz mm')
for s in sorted(out):
    if out[s] == 'success':
        continue
    p = '%s/eval_results/reltrace_seed%d.csv' % (H, s)
    if not os.path.exists(p):
        continue
    R = list(csv.DictReader(open(p)))
    for r in R:
        for k in list(r):
            if k != 'phase':
                r[k] = float(r[k])
    hi = [i for i, r in enumerate(R) if r['cup_z'] > 0.08]
    if not hi:
        print('%d %-18s | never lifted' % (s, out[s])); continue
    start = hi[-1]
    ev = next((r for r in R[start:] if r['cup_z'] < 0.04 or r['cup_up'] < 0.95), None)
    if ev is None:
        print('%d %-18s | no fall found' % (s, out[s])); continue
    print('%d %-18s | %-12s | %3d | %4.0f, %.3f, %.3f | %4.1f | %5.0f %5.0f %5.0f' % (
        s, out[s], ev['phase'], ev['n'], 1000*ev['cup_z'], ev['cup_up'], ev['cup_v'],
        1000*ev['width'], 1000*ev['dx'], 1000*ev['dy'], 1000*ev['dz']))
