# Backup manifest — 2026-06-24 22:55 (written 2026-06-24T21:47:22.781712)
# Session: M1.b₂ stage 1+2 completed. See NEXT_AI_HANDOFF_2026-06-25.md.

  [DATA] /apdcephfs/private_shayladeng/tokenrl_autoVLA/data/navtrain_nocot
        files=19125  total_size=125779218B (120.0 MB)
  [DATA] /apdcephfs/private_shayladeng/tokenrl_autoVLA/data/navtrain_nocot_probe100
        files=100  total_size=657646B (0.6 MB)
  [EXP] /apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1b2_d0_smoke_probe100_alllayers
        files=100  total_size=129780600B (123.8 MB)
  [EXP] /apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1b2_navtrain_full_window_tokens.txt
        size=326825B  md5_head4k=8dad654a3e51fdac
  [EXP] /apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1b2_navtrain_available_tokens.txt
        size=2592415B  md5_head4k=ca6099db95a674f9
  [EXP] /apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1b2_navtrain_available_intersect.txt
        size=1755896B  md5_head4k=373bbfe1547e184e
  [EXP] /apdcephfs/private_shayladeng/tokenrl_autoVLA/exp/m1a_navtrain_probeA_setup/navtrain_window_clean_tokens.txt
        size=326825B  md5_head4k=8dad654a3e51fdac
  [CODE] /apdcephfs/private_shayladeng/tokenrl_autoVLA/code/third_party/AutoVLA/config/dataset/qwen2.5-vl-3B-navtrain_full.yaml
        size=1553B  md5_head4k=81f27cf9e7ac8cd7
  [CODE] /apdcephfs/private_shayladeng/tokenrl_autoVLA/code/third_party/AutoVLA/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain_avail19k.yaml
        size=496424B  md5_head4k=d2e53c09d8f02773
  [CODE] /apdcephfs/private_shayladeng/tokenrl_autoVLA/tools/scan_available_tokens.py
        size=3529B  md5_head4k=792720835840c5a0
  [CODE] /apdcephfs/private_shayladeng/tokenrl_autoVLA/tools/scan_navtrain_full_window.py
        size=4644B  md5_head4k=8fd508bad849defe
  [LOG] /apdcephfs/private_shayladeng/tokenrl_autoVLA/logs/m1b2_d0/stage1.log
        size=3629B  md5_head4k=879e3bd311f384a4
  [LOG] /apdcephfs/private_shayladeng/tokenrl_autoVLA/logs/m1b2_d0/stage2.log
        size=201534B  md5_head4k=a5770e25c3200b09
  [LOG] /apdcephfs/private_shayladeng/tokenrl_autoVLA/logs/m1b2_d0/stage2_attempt1_FAILED.log
        size=52449B  md5_head4k=d9c6ce5d8f9bfd2a
  [DOC] /apdcephfs/private_shayladeng/tokenrl_autoVLA/docs/journal/2026-06-24_m1b2_stage1_2_full_journey.md
        size=9795B  md5_head4k=0099ff05f6c06e06
  [DOC] /apdcephfs/private_shayladeng/tokenrl_autoVLA/docs/PATHS.md
        size=9055B  md5_head4k=01ac0428866d9b45
  [DOC] /apdcephfs/private_shayladeng/tokenrl_autoVLA/docs/_internal/NEXT_AI_HANDOFF_2026-06-25.md
        size=8257B  md5_head4k=fe4ccbbfb9e20e57

Notes:
  - This is an INDEX, not a copy. Large data (.pt, .json) stays in-place.
  - md5_head4k = MD5 of first 4096 bytes only (for quick file-identity check).
  - To 'restore' an experiment: paths point into the live workspace; nothing to copy back.
  - If workspace is cloned/migrated, ensure these paths are preserved.