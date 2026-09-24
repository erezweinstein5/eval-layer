#!/usr/bin/env python3
"""Render a paced MP4 from the recorded live demo and its saved results.

Requires Pillow and imageio-ffmpeg for rendering only. The evaluation itself
uses stdlib. This is an edited playback, not a real-time desktop screen capture;
demo.cast preserves the unedited terminal output and timestamps.
"""
import argparse
import json
from pathlib import Path
import subprocess
import textwrap
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

W, H, FPS = 1600, 900, 15
BG, PANEL, WHITE, MUTED = '#101620', '#192332', '#EDF2F8', '#A6B4C9'
CYAN, GREEN, RED, AMBER = '#67D7ED', '#7BE0AC', '#FF969E', '#F8D18A'
FONT = '/System/Library/Fonts/SFNS.ttf'
MONO = '/System/Library/Fonts/Menlo.ttc'


def font(size, mono=False):
    return ImageFont.truetype(MONO if mono else FONT, size)


def text(draw, xy, value, size=28, fill=WHITE, mono=False, width=None):
    lines = textwrap.wrap(value, width=width) if width else [value]
    for i, line in enumerate(lines):
        draw.text((xy[0], xy[1] + i * (size + 11)), line, fill=fill, font=font(size, mono))
    return len(lines) * (size + 11)


def base(stage, progress):
    image = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((62, 48, 228, 90), radius=12, fill=PANEL)
    text(draw, (80, 53), 'eval-layer', 24, CYAN)
    text(draw, (W - 440, 56), 'LIVE JEV / DEMO CANDIDATES', 19, MUTED, mono=True)
    draw.line((64, 817, 1536, 817), fill='#2F3B4B', width=2)
    text(draw, (64, 844), stage, 21, MUTED)
    text(draw, (1040, 844), 'github.com/erezweinstein5/eval-layer', 18, MUTED)
    draw.rectangle((0, H-5, int(W * progress), H), fill=CYAN)
    return image, draw


def render(directory, output):
    recording = json.loads((directory / 'recording.json').read_text())
    rows = [json.loads(line) for line in (directory / 'results.jsonl').read_text().splitlines()]
    rubric = json.loads((directory / 'rubric.json').read_text())
    scales = {dim['name']: dim['scale'] for dim in rubric['dimensions']}
    if len(rows) != 3 or any(not r.get('judge') or 'error' in r['judge'] for r in rows):
        raise ValueError('The shareable video requires all three successful live judge responses')
    if not (directory / 'demo.cast').is_file():
        raise ValueError('Missing original terminal recording')
    task = rows[0]['input']
    timeline = [('intro', 4), ('task', 8), ('candidate0', 12), ('candidate1', 12),
                ('candidate2', 12), ('summary', 10), ('close', 5)]
    total = sum(duration for _, duration in timeline)
    frame_index = 0
    args = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo',
            '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
            '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    with subprocess.Popen(args, stdin=subprocess.PIPE) as proc:
        for stage, duration in timeline:
            for frame in range(duration * FPS):
                elapsed = frame / FPS
                image, draw = base('Live evaluation • playback paced for readability', frame_index / (total * FPS))
                if stage == 'intro':
                    text(draw, (80, 215), 'Three fixes.', 92)
                    text(draw, (80, 326), 'One rubric.', 92, CYAN)
                    text(draw, (85, 505), 'Watch Jev evaluate code — with independent checks.', 34)
                    text(draw, (85, 612), 'Authored demo candidates, not a coding-agent benchmark.', 25, MUTED)
                    text(draw, (85, 656), 'Actual TypeSafe API responses. No invented explanations.', 25, MUTED)
                elif stage == 'task':
                    text(draw, (80, 160), 'THE TASK', 23, CYAN, mono=True)
                    text(draw, (80, 225), task, 47, width=56)
                    text(draw, (80, 470), 'Four independent checks', 34)
                    text(draw, (80, 527), 'empty list  /  ordinary values  /  negatives  /  fractions', 26, MUTED)
                    text(draw, (80, 605), 'Rubric: correctness 50% · instructions 30% · maintainability 20%', 26)
                    text(draw, (80, 666), 'Pass = score ≥ 0.80 AND every required check passes.', 29, AMBER)
                elif stage.startswith('candidate'):
                    index = int(stage[-1]); row = rows[index]; judge = row['judge']
                    text(draw, (80, 150), f'0{index+1}  /  {row["label"]}', 50)
                    draw.rounded_rectangle((80, 251, 817, 707), radius=18, fill=PANEL)
                    text(draw, (108, 277), 'stats.py', 23, CYAN, mono=True)
                    for n, line in enumerate(row['agent_output']['code'].splitlines()):
                        text(draw, (108, 339+n*44), line, 25, mono=True)
                    checks = row['evidence']['checks']; passes = sum(c['passed'] for c in checks)
                    if elapsed >= 2:
                        text(draw, (108, 580), f'Independent tests: {passes}/4 pass', 25, GREEN if passes==4 else RED)
                        scope = row['gates']['allowed_paths']
                        text(draw, (108, 627), 'Change-scope check: PASS' if scope else 'Scope FAIL: settings.py also changed', 23, GREEN if scope else RED)
                    if elapsed < 4:
                        text(draw, (882, 319), 'Evaluating with Jev…', 32, CYAN)
                        text(draw, (882, 377), recording['model'], 25, MUTED, mono=True)
                    else:
                        score = row['rejudge']['weighted_score']
                        text(draw, (883, 260), f'{score:.3f}', 88, CYAN)
                        text(draw, (1198, 312), '/ 1.000', 28, MUTED)
                        verdict = 'PASS' if row['passed'] else 'FAIL'
                        text(draw, (885, 375), verdict, 44, GREEN if row['passed'] else RED)
                        for n, (name, val) in enumerate(judge['scores'].items()):
                            label = name.replace('_', ' ')
                            confidence = judge['details'][name]['confidence']
                            text(draw, (885, 466 + n*65), f'{label}: {val:.2f}/{scales[name]}', 24)
                            text(draw, (885, 497 + n*65), f'confidence {confidence:.2f}', 19, MUTED)
                        text(draw, (885, 721), f'{judge["latency_ms"]} ms · {judge["resolved_model_id"]}', 21, MUTED)
                elif stage == 'summary':
                    text(draw, (80, 153), 'The results', 62)
                    text(draw, (83, 249), 'CANDIDATE', 21, MUTED, mono=True)
                    text(draw, (755, 249), 'JEV SCORE', 21, MUTED, mono=True)
                    text(draw, (1055, 249), 'GATES', 21, MUTED, mono=True)
                    text(draw, (1320, 249), 'VERDICT', 21, MUTED, mono=True)
                    for n, row in enumerate(rows):
                        y = 320 + n*104
                        text(draw, (83, y), row['label'], 33)
                        text(draw, (772, y), f'{row["rejudge"]["weighted_score"]:.3f}', 34, CYAN, mono=True)
                        gates = all(row['gates'].values())
                        text(draw, (1070, y), 'PASS' if gates else 'FAIL', 29, GREEN if gates else RED)
                        text(draw, (1340, y), 'PASS' if row['passed'] else 'FAIL', 29, GREEN if row['passed'] else RED)
                    text(draw, (83, 691), 'A failed required check always blocks a pass.', 35, AMBER)
                else:
                    text(draw, (80, 197), 'Judge quality.', 76)
                    text(draw, (80, 294), 'Verify behavior.', 76, CYAN)
                    text(draw, (84, 457), 'Fractional scores + numeric confidence + retained evidence.', 31)
                    text(draw, (84, 528), 'Confidence is not a probability of correctness.', 27, MUTED)
                    text(draw, (84, 584), 'Jev explanations: unavailable. Nothing fabricated.', 27, MUTED)
                    text(draw, (84, 692), 'MP4 + original terminal recording + raw results', 26, AMBER)
                proc.stdin.write(image.tobytes())
                if stage == 'summary' and frame == FPS:
                    image.save(directory / 'poster.png')
                frame_index += 1
        proc.stdin.close()
        code = proc.wait()
        if code:
            raise RuntimeError(f'ffmpeg failed with exit {code}')
    print(f'Rendered {total}s MP4: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    render(args.directory, args.output or args.directory / 'jev-demo.mp4')
