# Project Overview
GitPilot is an AI Agent that automates the task of performing GitHub workflows by user prompt and the user no longer needs to remember the hard syntax of GitHub CLI commands.   
You can access the final Docker Repository from [here](https://hub.docker.com/repository/docker/adisrinitw/gitpilot30-gitpilot/general)   
# Project Features  
1. **Use Of Natural Language** : The user makes use of natural language prompt to explain the task to be perfromed.
2. **Feedback using RLHF** : The project uses RLHF to train further upon human feedback of whether the task was performed rightly.
3. **Voice Assistance** : The user may also give prompts using voice assistance.
4. **Direct Performance of Action** : The Agent directly performs the action that reflect on your GitHub repository. 
# Tech Stack  
**Frontend**          : ReactJS  
**Backend**           : Flask  
**HuggingFace Model** : [Qwen 7b](https://huggingface.co/Qwen/Qwen2-7B-Instruct)   
**Voice assistance**  : WebAPI  
**Containerization**  : Docker
# Project Architecture   
**Frontend**  : The frontend made using ReactJs framework takes the user input either through prompt or through voice assistance.  
                In case of voice input, WebAPI converts the voice input into text input and finally the frontend sends its request to the Backend.  
**Trained-LLM** : The trained LLM model is hosted on a seperate API.  
**Backend**   : The backend makes a POST request to another API that hosts our trained LLM Model. The API responds with the set of actions to be taken and the                      corresponding parameters. The backend then processes these actions and executes GitHub CLI commands on it's own. A detailed [Backend Documentation](https://github.com/adisri-ai/GitPilot/blob/main/BACKEND.md) can be viewed from here
# How to train the LLM Model  
1. Open *training.py* file and change paths according to current working directory. 
Note: change model_path="Qwen/Qwen2-7B" is for the first time use and change it to it stored path for later use.
3. Run *training.py* for completing training it takes approximately 2 to 3 hours.
4. Run *api.py* to host the trained LLM model on a seperate API.
5. Save the link generated
(Note that for this project I have used free hosting from google colab and hence the link to LLM API is dynamic and keeps changing on every run)
# How to access the application  
1. Run the following commands from project **root directory**:   
   1. ***docker pull adisrinitw/gitpilot30-gitpilot:latest***
   2. ***docker run -it -e EXTERNAL_API_URL=<YOUR_API_URL> -p 5173:5173 -p 5000:5000 adisrinitw/gitpilot30-gitpilot:latest***
3. Click on the vite link to open project frontend. You can now use the agent after authenticating your GitHub. 
# References   
The architecture of the implemented LLM Model draws inspiration fromn [Section 2.2 of this Research Paper](https://github.com/adisri-ai/GitPilot/blob/be57f1c5c69c89c41df52945bbf871540d5d9a67/Referenced_Paper.pdf)
