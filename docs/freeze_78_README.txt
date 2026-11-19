Policy: final_policy.ckpt (original, no DAgger)
Script: eval_sim18.py
Fresh seeds 1000-1119, 120 episodes:
  old setting (--zbias 0.012 only): 72/120 = 60%
  new setting (--zbias_grasp 0.0):  94/120 = 78%
Tuning seeds 160-279: 68/120 = 57% old, 100/120 = 83% new
Command for new setting:
python umi_sim/eval_sim18.py --ckpt $CK --lead 6 --xbias 0.012 --zbias 0.012 --zbias_grasp 0.0 --trace --retreat --xbias_mode pregrasp --glead 0 --rt_trig 0.055 --rt_full 0.078 --rt_back -0.10 --n 120 --start_seed 1000
