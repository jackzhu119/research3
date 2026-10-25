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
        f'{ca_assignment}; '
        'while ! printf \'%s  %s\\n\' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; do '
        f'I=0; for U in {quoted}; do I=$((I+1)); T="$F.part.$I"; '
        'RC=0; curl -4 "${C[@]}" -fL -C - --connect-timeout 10 --max-time 180 '
        '-H \'Accept: application/vnd.github.raw+json\' "$U" -o "$T" || RC=$?; '
        'if [ "$RC" -eq 33 ] || [ "$RC" -eq 36 ] || [ "$RC" -eq 22 ]; then rm -f "$T"; fi; '
        'if [ "$RC" -eq 0 ]; then '
        'if printf \'%s  %s\\n\' "$H" "$T" | sha256sum -c -; then mv -f "$T" "$F"; break; '
        'else rm -f "$T"; fi; fi; done; '
        'if ! printf \'%s  %s\\n\' "$H" "$F" | sha256sum -c - >/dev/null 2>&1; then '
        f'printf \'DOWNLOAD_RETRY: partial download retained; retrying\\n\'; sleep {int(retry_seconds)}; fi; '
        'done; printf \'%s  %s\\n\' "$H" "$F" | sha256sum -c -; '+after+' )')


def school_command(commit, manifest, *, submit=False):
    path = 'experiments/followup_20261025_r9/online/START-PAPER3-R9-ONLINE.py'
    urls = [f'https://raw.githubusercontent.com/jackzhu119/research3/{commit}/{path}',
            f'https://api.github.com/repos/jackzhu119/research3/contents/{path}?ref={commit}']
    python = '/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python'
    final = '/public/home/zhuziyuan/paper3-round9-downloads/START-PAPER3-R9-ONLINE.py'
    ca = '/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/lib/python3.12/site-packages/botocore/cacert.pem'
    return download_command(final, manifest['wrapper_sha256'], urls, ca=ca,
        after=f'{shlex.quote(python)} -I -S -B -u "$F" ' + ('--submit' if submit else '--prepare-only'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--submit', action='store_true', help='Emit explicit submission command; default only prepares')
    args=parser.parse_args()
    command=school_command(args.commit,json.loads(Path(args.manifest).read_text()), submit=args.submit)
    Path(args.output).write_text(command+'\n')
    print(args.output)
