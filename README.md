# Project Overview
GitPilot is an AI Agent that automates the task of performing GitHub workflows by user prompt and the user no longer needs to remember the hard syntax of GitHub CLI commands. 
# Tech Stack  
**Frontend**          : ReactJS  
**Backend**           : Flask  
**HuggingFace Model** : [Qwen 7b](https://huggingface.co/Qwen/Qwen2-7B-Instruct)   
**Voice assistance**  : WebAPI  
**Containerization**  : Docker
# Project Architecture   
**Frontend**  : The frontend made using ReactJs framework takes the user input either through prompt or through voice assistance.  
                In case of voice input, WebAPI converts the voice input into text input and finally the frontend sends its request to the Backend.  
**Trained-LLM API** : The trained LLM model is hosted on a seperate API   
**Backend**   : The backend makes a POST request to another API that hosts our trained LLM Model. The API responds with the set of actions to be taken and the                      corresponding parameters. The backend then processes these actions and executes GitHub CLI commands on it's own.
