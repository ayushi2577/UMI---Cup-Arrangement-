import csv, os, re, sys
H = os.path.expanduser('~/umi_mujoco')
log = sys.argv[1]
out = {}
for l in open(H + '/exp/' + log):
    m = re.match(r'GTX (\d+) (\S+)', l)
    if m: out[int(m.group(1))] = m.group(2)
print('seed outcome | grip offset dx dy dz (mm) | carry: gap start->end (mm), min contacts, last force, min tilt | end cup_z (mm)')
for s in sorted(out):
    p = '%s/eval_results/reltrace_seed%d.csv' % (H, s)
    if not os.path.exists(p):
        continue
    R = list(csv.DictReader(open(p)))
    for r in R:
        for k in list(r):
            if k != 'phase':
                r[k] = float(r[k])
    pol = [r for r in R if r['phase'] == 'policy']
    g = next((r for r in pol if r['nc'] >= 2), None)
    if g is None:
        print(s, out[s], '| no grip found'); continue
    carry = [r for r in pol[pol.index(g):] if r['cup_z'] > 0.08 and r['width'] < 0.06]
    head = '%d %-18s | %5.1f %5.1f %5.1f' % (s, out[s], 1000*g['dx'], 1000*g['dy'], 1000*g['dz'])
    if not carry:
        print(head, '| never carried | end %.0f' % (1000*R[-1]['cup_z'])); continue
    print(head + ' | %4.1f -> %4.1f, nc_min %d, force %.2f, tilt %.4f | end %.0f' % (
        1000*carry[0]['width'], 1000*carry[-1]['width'], min(r['nc'] for r in carry),
        carry[-1]['fn'], min(r['cup_up'] for r in carry), 1000*R[-1]['cup_z']))
