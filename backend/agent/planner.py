import json
import requests
def plan_task(user_input: str):
    """
    Sends user input to external API and parses the specific response format.
    Expected format: {'response': 'res : [ ...JSON... ]'}
    """
    try:
        # 1. Call External API
        # Adjust the payload key ('prompt', 'text', 'query') based on what your API expects
        headers=  {
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true'
        }
        payload = {
                    "text" : user_input
                }        
        url = "*********"
        response = requests.post(url , json = payload , headers = headers,timeout = 60)
        response.raise_for_status()
        
        data = response.json()
        if isinstance(data, list):
            tasks = data
            return tasks
        # 2. Extract the string containing the list
        # based on user provided example: {'response': 'res : [...]'}
        raw_string = data.get("response", "")
        
        if not raw_string:
            return {"error": "Empty response from external API"}

        # 3. Clean the prefix "res : " to get valid JSON
        if "res :" in raw_string:
            json_part = raw_string.split("res :", 1)[1].strip()
        else:
            json_part = raw_string.strip()

        # 4. Parse JSON
        tasks = json.loads(json_part)
        
        # Sort by step just in case
        tasks.sort(key=lambda x: x.get('step', 0))
        
        return tasks

    except json.JSONDecodeError:
        return {"error": "Failed to parse external API response. Invalid JSON format."}
    except Exception as e:
        return {"error": f"Planning failed: {str(e)}"}
