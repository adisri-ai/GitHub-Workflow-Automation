import subprocess
import os

def run_git_command(command, cwd=None):
    """
    Executes a git/gh command.
    Accepts command as a list (preferred) or string.
    """
    # Convert string to list if necessary for safety, unless it's a raw shell command
    if isinstance(command, str):
        command = command.split()
    
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False # We handle return codes manually
        )
        if result.returncode != 0:
            print(f"Command failed: {command}\nError: {result.stderr}")
        return result.stdout, result.stderr
    except Exception as e:
        return "", str(e)

def git_commit_push(repo_path, message):
    """
    STAGES, COMMITS, and PUSHES changes.
    """
    # 1. Add all changes
    run_git_command(["git", "add", "."], cwd=repo_path)
    
    # 2. Commit (only if there are changes)
    # We use a list to handle spaces/quotes in the message correctly
    commit_cmd = ["git", "commit", "-m", message]
    out, err = run_git_command(commit_cmd, cwd=repo_path)
    
    if "nothing to commit" in out or "nothing to commit" in err:
        print("Nothing to commit.")
        return

    # 3. Push
    out, err = run_git_command(["git", "push"], cwd=repo_path)
    if err:
        print(f"Push output: {err}")