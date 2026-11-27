import csv, os, re, sys
H = os.path.expanduser('~/umi_mujoco')
out = {}
for l in open(H + '/exp/' + sys.argv[1]):
    m = re.match(r'GTX (\d+) (\S+)', l)
    if m: out[int(m.group(1))] = m.group(2)
TH = 0.005
print('seed outcome | saucer first moves (step, stage) | cup first falls (step, stage) | peak saucer speed')
ok_move = ok_n = 0
for s in sorted(out):
    p = '%s/eval_results/reltrace_seed%d.csv' % (H, s)
    if not os.path.exists(p): continue
    R = list(csv.DictReader(open(p)))
    for r in R:
        for k in list(r):
            if k != 'phase': r[k] = float(r[k])
    hi = [i for i, r in enumerate(R) if r['cup_z'] > 0.08]
    if not hi: continue
    st = hi[-1]
    sm = next((r for r in R[st:] if r['sau_v'] > TH), None)
    pk = max(r['sau_v'] for r in R[st:])
    if out[s] == 'success':
        ok_n += 1; ok_move += sm is not None
        continue
    cf = next((r for r in R[st:] if r['cup_z'] < 0.04 or r['cup_up'] < 0.95), None)
    a = '%d %s' % (sm['n'], sm['phase']) if sm else 'never'
    b = '%d %s' % (cf['n'], cf['phase']) if cf else 'none'
    print('%d %-18s | %-22s | %-22s | %.3f' % (s, out[s], a, b, pk))
print('successes where the saucer moved above %.3f: %d of %d' % (TH, ok_move, ok_n))
