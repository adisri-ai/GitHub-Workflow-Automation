# Project Backend  
The purpose of the project backend is to extract the set of tasks for a given user prompt and execute the relevant GitHub CLI commands in the terminal to perform the user task.  
# Project Backend Directory  
This is the structure of the project backend:  
*/backend*
1. *__init__.py* : The *__init__* file requried for intialization of package so that it can be used as a package for the main app.
2. */agent*      : This module deals with agent actions such as extracting the set of tasks and directing the code to execute the tasks.
                    The files present in this directory include:
                     1. *github_api.py* : Maps various github actions to corresponding commands based on argument values
                     2. *github_cli.py* : Contains the mechanism to exectute GitHub CLI commands in terminal. 
4. */github*     : Contains the logical code for running github_cli commands.
