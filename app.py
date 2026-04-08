import json
import os
import subprocess
import re
import requests
from collections import deque
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from backend.agent.planner import plan_task, EXTERNAL_API_URL
from backend.github import github_api

app = Flask(__name__)
CORS(app)

# --- GLOBAL STATE ---
CONTEXT_FILE = "agent_context.json"
UNDO_STACK = deque(maxlen=5) 

# --- CONTEXT MANAGERS ---
def load_context():
    if os.path.exists(CONTEXT_FILE):
        try:
            with open(CONTEXT_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_context(ctx):
    try:
        with open(CONTEXT_FILE, 'w') as f:
            json.dump(ctx, f)
    except:
        pass

def update_context_from_params(params):
    ctx = load_context()
    for k, v in params.items():
        if v is not None:
            ctx[k] = v
    save_context(ctx)

def fill_params_from_context(params):
    ctx = load_context()
    for k, v in params.items():
        if v is None and k in ctx:
            params[k] = ctx[k]
    return params

# --- UNDO LOGIC ---
def push_undo_action(action, params):
    reverse_map = {
        "create_repo": "delete_repo",
        "make_repo_private": "make_repo_public",
        "make_repo_public": "make_repo_private",
        "add_file": "delete_file",
        "create_branch": "delete_branch",
        "add_collaborator": "remove_collaborator",
    }
    if action in reverse_map:
        UNDO_STACK.append({"action": reverse_map[action], "params": params.copy()})

# --- AUTH & SETUP ---
def check_gh_auth():
    code = subprocess.call(["gh", "auth", "status"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return code == 0

def setup_environment():
    try:
        subprocess.run(["gh", "config", "set", "prompt", "disabled"], check=False)
        subprocess.run(["gh", "auth", "setup-git"], check=False)
        subprocess.run(["git", "config", "--global", "user.email", "agent@gitpilot.ai"], check=False)
        subprocess.run(["git", "config", "--global", "user.name", "GitPilot Agent"], check=False)
        os.environ["EDITOR"] = "true"
        
        res = subprocess.run(["gh", "api", "user", "--jq", ".login"], capture_output=True, text=True)
        if res.returncode == 0:
            owner = res.stdout.strip()
            ctx = load_context()
            ctx['_owner'] = owner
            save_context(ctx)
    except Exception:
        pass

# --- ENDPOINTS ---

@app.route('/api/auth/status', methods=['GET'])
def get_auth_status():
    is_logged_in = check_gh_auth()
    if is_logged_in: setup_environment()
    return jsonify({"authenticated": is_logged_in, "undo_available": len(UNDO_STACK) > 0})

@app.route('/api/auth/login', methods=['POST'])
def start_auth_flow():
    try:
        cmd = ["gh", "auth", "login", "--hostname", "github.com", "--git-protocol", "https", "--device", "-s", "delete_repo,repo,workflow,user"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        user_code = None
        while True:
            line = proc.stdout.readline() or proc.stderr.readline()
            if not line: break
            match = re.search(r"[A-Z0-9]{4}-[A-Z0-9]{4}", line)
            if match:
                user_code = match.group(0)
                break 
        if user_code:
            return jsonify({"user_code": user_code, "verification_uri": "https://github.com/login/device"})
        return jsonify({"error": "Could not generate login code."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/undo', methods=['POST'])
def perform_undo():
    if not UNDO_STACK: return jsonify({"error": "Nothing to undo"}), 400
    undo_task = UNDO_STACK.pop()
    action = undo_task['action']
    params = undo_task['params']
    
    repo = params.get('repo_name')
    ctx = load_context()
    owner = ctx.get('_owner')
    if repo and "/" not in repo and owner:
        repo = f"{owner}/{repo}"

    try:
        result = {"message": "Undo executed"}
        if action == "delete_repo": result = github_api.delete_repo(repo)
        elif action == "make_repo_private": result = github_api.make_repo_private(repo)
        elif action == "make_repo_public": result = github_api.make_repo_public(repo)
        elif action == "delete_file": result = github_api.delete_file(repo, params.get('file_path'), "Undo add file")
        elif action == "delete_branch": result = github_api.delete_branch(repo, params.get('branch_name'))
        elif action == "remove_collaborator": result = github_api.remove_collaborator(repo, params.get('username'))
        
        return jsonify({"status": "success", "action": action, "output": result, "remaining_undo": len(UNDO_STACK)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- FEEDBACK ENDPOINT (NEW) ---
@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """
    Receives feedback from frontend and forwards to LLM API.
    Expected payload: { prompt: str, response: str/object, is_good: bool }
    """
    try:
        data = request.json
        prompt = data.get('prompt')
        response = data.get('response')
        is_good = data.get('is_good')
        
        if prompt is None or response is None or is_good is None:
            return jsonify({"error": "Missing required fields: prompt, response, is_good"}), 400
        
        # Convert response to string if it's an object
        if isinstance(response, (dict, list)):
            response = json.dumps(response)
        
        # Forward to LLM API's /feedback endpoint
        # Construct the feedback endpoint URL from the main API URL
        feedback_url = f"{EXTERNAL_API_URL}/feedback"
        
        feedback_payload = {
            "prompt": prompt,
            "response": response,
            "is_good": is_good
        }
        
        llm_response = requests.post(feedback_url, json=feedback_payload, timeout=10)
        llm_response.raise_for_status()
        
        return jsonify({"status": "success", "message": "Feedback submitted successfully"})
    
    except requests.RequestException as e:
        return jsonify({"error": f"Failed to submit feedback to LLM: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- AGENT ENDPOINT ---
@app.route('/api/agent', methods=['POST'])
def run_agent():
    data = request.json
    user_input = data.get('prompt', '').strip()
    
    def generate():
        yield f"data: {json.dumps({'type': 'log', 'message': '🧠 Querying AI...'})}\n\n"
        
        try:
            tasks = plan_task(user_input)
            
            if isinstance(tasks, dict) and "error" in tasks:
                yield f"data: {json.dumps({'type': 'error', 'message': tasks['error']})}\n\n"
                return

            # Send plan along with original prompt and raw LLM response for feedback
            yield f"data: {json.dumps({'type': 'plan', 'tasks': tasks, 'original_prompt': user_input, 'llm_response': tasks})}\n\n"

            for task in tasks:
                action = task.get('action')
                raw_params = task.get('params', {})
                
                params = fill_params_from_context(raw_params.copy())

                repo = params.get('repo_name')
                
                ctx = load_context()
                owner = ctx.get('_owner')
                if repo and "/" not in repo and owner:
                    repo = f"{owner}/{repo}"
                    params['repo_name'] = repo

                visibility = params.get('visibility')
                file_path = params.get('file_path')
                content = params.get('content') or ""
                commit_msg = f"GitPilot: {action}"
                branch = params.get('branch_name')
                src_branch = params.get('source_branch')
                tgt_branch = params.get('target_branch')
                username = params.get('username')
                issue_num = params.get('issue_number')
                title = params.get('title') or "Created by GitPilot"
                body = params.get('body') or ""
                
                # Validation
                missing_args = []
                if action in ["create_repo", "create" , "delete_repo", "update_visibility", "fork_repo", "star_repo"] and not repo:
                    missing_args.append("repo_name")
                if action in ["add_file", "update_file", "delete_file"] and (not repo or not file_path):
                    if not repo: missing_args.append("repo_name")
                    if not file_path: missing_args.append("file_path")
                if action in ["create_branch", "delete_branch"] and (not repo or not branch):
                    if not repo: missing_args.append("repo_name")
                    if not branch: missing_args.append("branch_name")
                if action in ["add_collaborator", "remove_collaborator"] and (not repo or not username):
                    if not repo: missing_args.append("repo_name")
                    if not username: missing_args.append("username")

                if missing_args:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Missing: {', '.join(missing_args)} for {action}'})}\n\n"
                    continue 

                yield f"data: {json.dumps({'type': 'log', 'message': f'▶ Executing: {action} on {repo}'})}\n\n"
                
                result = {"error": f"Action {action} not implemented"}
                
                # --- DISPATCHER ---
                if action == "create_repo" or action=="create":
                    is_private = (visibility == "private")
                    result = github_api.create_repo(repo, private=is_private)
                    if "error" not in result: push_undo_action("create_repo", params)

                elif action == "update_visibility" or action=="make_public" or action=="make_private" or action=="changerepoprivacy" or action=="set_repo_visibility":
                    if(action=="make_public"): visibility = "public"
                    if(action=="make_private"): visibility="private"
                    if visibility == "private":
                        result = github_api.make_repo_private(repo)
                        if "error" not in result: push_undo_action("make_repo_private", params)
                    else:
                        result = github_api.make_repo_public(repo)
                        if "error" not in result: push_undo_action("make_repo_public", params)

                elif action == "delete_repo": 
                    result = github_api.delete_repo(repo)
                elif action == "fork_repo": 
                    result = github_api.fork_repo(repo)
                elif action == "star_repo": 
                    result = github_api.star_repo(repo)

                elif action == "add_file" or "init_file":
                    result = github_api.add_file(repo, file_path, content, commit_msg)
                    if "error" not in result: push_undo_action("add_file", params)
                elif action == "update_file":
                    result = github_api.update_file(repo, file_path, content, commit_msg)
                elif action == "delete_file" or action=="deletefile": 
                    result = github_api.delete_file(repo, file_path, commit_msg)

                elif action == "create_branch":
                    result = github_api.create_branch(repo, branch)
                    if "error" not in result: push_undo_action("create_branch", params)
                elif action == "delete_branch": 
                    result = github_api.delete_branch(repo, branch)

                elif action == "add_collaborator" or action=="add_collaborators":
                    result = github_api.add_collaborator(repo, username)
                    if "error" not in result: push_undo_action("add_collaborator", params)
                elif action == "remove_collaborator": 
                    result = github_api.remove_collaborator(repo, username)

                elif action == "create_issue": 
                    result = github_api.create_issue(repo, title, body)
                elif action == "close_issue": 
                    result = github_api.close_issue(repo, issue_num)

                elif action == "create_pr":
                    result = github_api.create_pr(repo, title, body, src_branch or branch, tgt_branch or "main")
                
                update_context_from_params(params)

                if result and "error" in result:
                    yield f"data: {json.dumps({'type': 'error', 'message': result['error']})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'result', 'output': result})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'message': 'Sequence Completed.', 'undo_available': len(UNDO_STACK) > 0})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')