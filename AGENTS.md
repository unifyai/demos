# Agents in this repo

Read `README.md` first: its rules bind every experiment here.

- Work only in your own `concepts/<name>/` folder. Change `common/` only by
  saying what you need; whoever leads the round makes the change, so every
  concept keeps sharing one tested client.
- Keep what is worth remembering in the concept's README, in the same change
  as the work: the plan, every result, every bug found and how it was fixed,
  and what a reader needs to re-run it. Nothing lives only in a transcript or
  on the cloud machine.
- Never print, log or commit a secret. The OpenRouter key reaches a process
  only through `bin/with-openrouter-key`.
- `runs/` stays out of git (`.gitignore`); the concept's `runs.tar.zst` is the
  committed copy of it. Rebuild the archive whenever the runs change:
  `tar -cf - runs | zstd -19 -T0 -o runs.tar.zst`. Over 45 MB, commit it in
  parts instead (`split -b 45m runs.tar.zst runs.tar.zst.part-`); the
  README says how.
