#!/usr/bin/env python3
"""Create the single-line public launch command after a pinned publication."""
import argparse
import json
from pathlib import Path
import shlex


def download_command(final, expected, urls, *, ca=None, after='true', retry_seconds=3):
    # Explicit separators also work in terminals that collapse pasted newlines.
    quoted = ' '.join(shlex.quote(str(x)) for x in urls)
    q = shlex.quote
    ca_assignment = 'C=()' if ca is None else f'C=(); [ ! -f {q(str(ca))} ] || C=(--cacert {q(str(ca))})'
    return (f'( set -e; F={q(str(final))}; H={q(expected)}; mkdir -p "$(dirname "$F")"; '
        f'{ca_assignment}; T="$F.part"; '
        'while ! printf \'%s  %s\\n\' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; do '
        f'for U in {quoted}; do '
        'RC=0; curl -4 "${C[@]}" -fL -C - --connect-timeout 10 --max-time 180 '
        '-H \'Accept: application/vnd.github.raw+json\' "$U" -o "$T" || RC=$?; '
        'if [ "$RC" -eq 33 ] || [ "$RC" -eq 36 ] || [ "$RC" -eq 22 ]; then rm -f "$T"; fi; '
        'if [ "$RC" -eq 0 ]; then '
        'if printf \'%s  %s\\n\' "$H" "$T" | sha256sum -c -; then mv -f "$T" "$F"; break; '
        'else rm -f "$T"; fi; fi; done; '
        'if ! printf \'%s  %s\\n\' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then '
        f'printf \'DOWNLOAD_RETRY: partial download retained; retrying\\n\'; sleep {int(retry_seconds)}; fi; '
        'done; printf \'%s  %s\\n\' "$H" "$F" | sha256sum -c -; '+after+' )')


def school_command(commit, manifest):
    path = 'experiments/followup_20261025_r8/online/START-PAPER3-R8-ONLINE.py'
    urls = [f'https://raw.githubusercontent.com/jackzhu119/research3/{commit}/{path}',
            f'https://api.github.com/repos/jackzhu119/research3/contents/{path}?ref={commit}']
    python = '/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python'
    final = '/public/home/zhuziyuan/paper3-round8-downloads/START-PAPER3-R8-ONLINE.py'
    ca = '/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/lib/python3.12/site-packages/botocore/cacert.pem'
    return download_command(final, manifest['wrapper_sha256'], urls, ca=ca,
        after=f'{shlex.quote(python)} -I -S -B -u "$F" --submit')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args=parser.parse_args()
    command=school_command(args.commit,json.loads(Path(args.manifest).read_text()))
    Path(args.output).write_text(command+'\n')
    print(args.output)
