import json
import requests
import os
EXTERNAL_API_URL = os.environ.get("EXTERNAL_API_URL" , "*****************")
def normalize_task(task):
    """
    Normalizes the inconsistent API response into a standard format.
    """
    raw_action = task.get("action", "").lower().replace("-", "_").replace(" ", "_")
    
    action_map = {
        "create_repo": "create_repo",
        "delete": "delete_repo",
        "remove": "delete_repo",
        "change_visibility": "update_visibility",
        "updaterepoprivacy": "update_visibility",
        "fork": "fork_repo",
        "star": "star_repo",
        "add_star": "star_repo",
        "add_file": "add_file",
        "create_file": "add_file",
        "update": "update_file",
        "modify": "update_file",
        "create_branch": "create_branch",
        "branch": "create_branch",
        "delete_branch": "delete_branch",
        "invite_collaborator": "add_collaborator",
        "add": "add_collaborator",
        "revoke": "remove_collaborator",
        "create_issue": "create_issue",
        "open_issue": "create_issue",
        "close": "close_issue",
        "close_issue": "close_issue",
        "create_pull_request": "create_pr",
        "open_pr": "create_pr",
    }
    
    if raw_action in ["delete", "remove"]:
        if task.get("resource") == "repository" or task.get("name"):
            raw_action = "delete_repo"
        elif task.get("branch"):
            raw_action = "delete_branch"
        elif task.get("type") == "collaborator" or task.get("user"):
            raw_action = "remove_collaborator"
    
    if raw_action == "add" and (task.get("username") or task.get("permission")):
        raw_action = "add_collaborator"

    action = action_map.get(raw_action, raw_action)
    
    repo_name = task.get("repo") or task.get("repository") or task.get("name") or None
    
    file_path = task.get("file") or task.get("file_path")
    if file_path and "/" in file_path and not repo_name:
        parts = file_path.split("/", 1)
        if len(parts) == 2:
            repo_name = parts[0]
            file_path = parts[1]
    
    content = task.get("content") or task.get("value") or task.get("description")
    branch_name = task.get("branch") or task.get("branch_name") or task.get("to")
    source_branch = task.get("source") or task.get("from") or task.get("branch_from")
    target_branch = task.get("merge_into") or task.get("branch_to") or task.get("target")
    
    visibility = task.get("visibility") or task.get("privacy")
    if task.get("private") == True:
        visibility = "private"
    elif task.get("private") == False:
        visibility = "public"
    
    username = task.get("username") or task.get("user") or task.get("add_collaborator") or task.get("assignee")
    title = task.get("title")
    body = task.get("body") or task.get("description")
    issue_number = task.get("issue") or task.get("issue_number")
    if isinstance(issue_number, str):
        try:
            issue_number = int(issue_number)
        except:
            issue_number = None
    
    params = {
        "repo_name": repo_name,
        "file_path": file_path,
        "content": content,
        "branch_name": branch_name,
        "source_branch": source_branch,
        "target_branch": target_branch,
        "username": username,
        "visibility": visibility,
        "title": title,
        "body": body,
        "issue_number": issue_number,
    }
    
    return {"action": action, "params": params}


def plan_task(user_input: str):
    try:
        response = requests.post(f"{EXTERNAL_API_URL}/generate", json={"text": user_input})
        response.raise_for_status()
        
        data = response.json()
        
        if isinstance(data, list):
            tasks = data
        elif isinstance(data, dict):
            if 'response' in data:
                resp = data['response']
                tasks = json.loads(resp) if isinstance(resp, str) else resp
            else:
                tasks = [data]
        else:
            return {"error": "Unexpected API response format."}

        if not isinstance(tasks, list):
            tasks = [tasks]
        
        normalized_tasks = []
        for i, task in enumerate(tasks):
            normalized = normalize_task(task)
            normalized["step"] = i + 1
            normalized_tasks.append(normalized)
        
        return normalized_tasks

    except json.JSONDecodeError:
        return {"error": "Failed to parse API response."}
    except Exception as e:
        return {"error": f"Planning failed: {str(e)}"}
