"""Push a single clean commit to a new GitHub repo (no Cursor co-author)."""
import os
import subprocess
import sys
from pathlib import Path

GIT = r"C:\Program Files\Git\cmd\git.exe"
REPO = Path(__file__).resolve().parent
REMOTE = "https://github.com/joshmath524/GPCR-FAP.git"
MSG = "Initial public release of GPCR-FAP Streamlit GUI"

env = os.environ.copy()
env["GIT_AUTHOR_NAME"] = "joshmath524"
env["GIT_AUTHOR_EMAIL"] = "joshmathew256@gmail.com"
env["GIT_COMMITTER_NAME"] = "joshmath524"
env["GIT_COMMITTER_EMAIL"] = "joshmathew256@gmail.com"


def run(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        [GIT, *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        print(result.stderr or result.stdout, file=sys.stderr)
        raise SystemExit(result.returncode)
    return (result.stdout or "").strip()


run("add", "-A")
tree = run("write-tree")
commit = run("commit-tree", tree, "-m", MSG)
run("update-ref", "refs/heads/main", commit)
run("reset", "--hard", commit)
msg = run("log", "-1", "--format=%B")
if "Co-authored-by" in msg or "cursoragent" in msg.lower():
    raise SystemExit("Commit still contains Cursor attribution")

run("remote", "remove", "gpcr-new", check=False)
run("remote", "add", "gpcr-new", REMOTE, check=False)
print("Pushing to", REMOTE)
run("push", "-u", "gpcr-new", "main", "--force")
print("DONE", commit)
