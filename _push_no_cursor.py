"""Commit and push without Cursor co-author (uses git.exe directly)."""
import os
import subprocess
import sys
from pathlib import Path

GIT = r"C:\Program Files\Git\cmd\git.exe"
REPO = Path(__file__).resolve().parent
REMOTE = "gpcr-new"
BRANCH = "main"
MSG = "Remove unused mailmap file"

env = os.environ.copy()
env.update(
    {
        "GIT_AUTHOR_NAME": "joshmath524",
        "GIT_AUTHOR_EMAIL": "joshmathew256@gmail.com",
        "GIT_COMMITTER_NAME": "joshmath524",
        "GIT_COMMITTER_EMAIL": "joshmathew256@gmail.com",
    }
)


def run(*args: str) -> str:
    result = subprocess.run(
        [GIT, *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr or result.stdout, file=sys.stderr)
        raise SystemExit(result.returncode)
    return (result.stdout or "").strip()


parent = run("rev-parse", "HEAD")
run("add", "-A")
tree = run("write-tree")
commit = run("commit-tree", tree, "-p", parent, "-m", MSG)
run("update-ref", f"refs/heads/{BRANCH}", commit)
run("reset", "--hard", commit)

msg = run("log", "-1", "--format=%an <%ae>%n%B")
if "Co-authored-by" in msg or "cursoragent" in msg.lower():
    raise SystemExit("Cursor attribution detected in commit")

run("push", REMOTE, BRANCH)
print("PUSHED", commit)
print(msg)
