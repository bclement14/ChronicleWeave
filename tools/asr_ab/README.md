# asr_ab — blind comparison of two transcriptions (local only)

1. Build: `python -m tools.asr_ab.ab build --a <A>/json_files --b <B>/json_files --a-name large-v3 --b-name dec16 --chunks <A>/chunked_tracks --out <A>/../ab_page`
2. Open `ab_page/index.html` in a browser, listen, choose X / Y / same, click "Export answers" (saves `answers.json`).
3. Score: `python -m tools.asr_ab.ab score --answers answers.json --mapping ab_page/mapping.json --a-script <A>/final_outputs/final_script.txt --b-script <B>/final_outputs/final_script.txt --candidate dec16`

Rule (spec 4.9): ≥ 20 non-tied answers, candidate wins ≥ 2/3 of them, and no more repetition loops than the baseline.
