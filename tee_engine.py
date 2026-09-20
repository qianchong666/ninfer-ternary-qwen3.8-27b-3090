#!/usr/bin/env python3
"""Run a child process, echoing its merged stdout/stderr to this console AND to a log file.

Every line is flushed to BOTH sinks immediately, so the console copy stays live even though
PowerShell's tee / Get-Content -Wait buffering proved unreliable under Windows Terminal
(Get-Content -Wait stopped tracking appends at ~1.5 KB while the file kept growing).

Usage:  python tee_engine.py <logfile> -- <command> [args...]
"""
import subprocess
import sys


def main():
    if len(sys.argv) < 4 or sys.argv[2] != '--':
        sys.stderr.write(__doc__)
        return 2
    logpath = sys.argv[1]
    cmd = sys.argv[3:]

    out = sys.stdout
    try:
        out.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    except Exception:
        pass

    log = open(logpath, 'w', encoding='utf-8', errors='replace', buffering=1)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    try:
        for raw in iter(proc.stdout.readline, b''):
            line = raw.decode('utf-8', 'replace')
            log.write(line)
            log.flush()
            out.write(line)
            out.flush()
    except KeyboardInterrupt:
        proc.terminate()
    proc.wait()

    tail = '\n[tee] engine exited with code %s\n' % proc.returncode
    log.write(tail)
    log.flush()
    out.write(tail)
    out.flush()
    log.close()
    return proc.returncode


if __name__ == '__main__':
    sys.exit(main())
