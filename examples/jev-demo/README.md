# Live Jev demo recording

Three authored code candidates solve the same small Python task. The demo runs
four actual checks per candidate, sends their code/diff/evidence to Jev, and
records the live terminal output. These are illustrative candidates, not outputs
from a live coding agent or a representative benchmark.

Set `TYPESAFE_API_KEY` in your environment using your normal credential mechanism.
Then run from the repository root:

```sh
python examples/jev-demo/run_demo.py --output /tmp/jev-live-demo
```

The output directory must be new. This makes three live Jev requests (with the
adapter's bounded retries on temporary failures). No key appears in the output.
To prepare the examples and execute only local checks, add `--prepare-only`.

Render the recorded run with Pillow and imageio-ffmpeg installed:

```sh
python examples/jev-demo/render_recording.py /tmp/jev-live-demo
```

The 63-second MP4 is an edited playback paced for readability. `demo.cast`
preserves original terminal output and timestamps; `recording.json` contains the
captured events. Raw results, rubric, original candidates, and the report are
saved alongside the video. No narration or fabricated scores are added.
