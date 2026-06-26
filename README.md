<div align="center">

```
 ██████╗ ██╗████████╗██████╗ ██╗██╗      ██████╗ ████████╗
██╔════╝ ██║╚══██╔══╝██╔══██╗██║██║     ██╔═══██╗╚══██╔══╝
██║  ███╗██║   ██║   ██████╔╝██║██║     ██║   ██║   ██║   
██║   ██║██║   ██║   ██╔═══╝ ██║██║     ██║   ██║   ██║   
╚██████╔╝██║   ██║   ██║     ██║███████╗╚██████╔╝   ██║   
 ╚═════╝ ╚═╝   ╚═╝   ╚═╝     ╚═╝╚══════╝ ╚═════╝    ╚═╝   
```

**Your GitHub copilot. Speak the task — GitPilot handles the rest.**

[![Docker](https://img.shields.io/badge/Docker-Available-2496ED?style=flat-square&logo=docker&logoColor=white)](https://hub.docker.com)
[![Model](https://img.shields.io/badge/🤗%20Model-Qwen%202%207B-FFD21E?style=flat-square)](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
[![Backend](https://img.shields.io/badge/Backend-Flask-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Frontend](https://img.shields.io/badge/Frontend-React-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)

</div>

---

## 〉 Overview

GitPilot is an AI Agent that automates the task of performing GitHub workflows by user prompt — the user no longer needs to remember the hard syntax of GitHub CLI commands.

You can access the final Docker Repository from here.

---

## 〉 Features

| # | Feature | Description |
|---|---------|-------------|
| 01 | **Natural Language** | Make use of natural language prompts to explain the task to be performed |
| 02 | **Feedback via RLHF** | Uses RLHF to train further upon human feedback of whether the task was performed rightly |
| 03 | **Voice Assistance** | Give prompts using voice assistance — no typing required |
| 04 | **Direct Action** | The agent directly performs actions that reflect on your GitHub repository |

---

## 〉 Tech Stack

```
┌─────────────────┬─────────────────────────────────────────────┐
│  Frontend       │  ReactJS                                    │
│  Backend        │  Flask                                      │
│  LLM Model      │  Qwen 2 7B (HuggingFace)                   │
│  Voice          │  WebAPI                                     │
│  Container      │  Docker                                     │
└─────────────────┴─────────────────────────────────────────────┘
```

---

## 〉 Architecture

```
  User Input
  ┌─────────────────────────────┐
  │  Voice  ──► WebAPI ──► Text │
  │  Text   ──────────────────► │
  └──────────────┬──────────────┘
                 │
         ┌───── ▼ ──────┐
         │   ReactJS    │  Frontend
         │   Frontend   │
         └───── │ ──────┘
                │  POST Request
         ┌───── ▼ ──────┐
         │    Flask     │  Backend
         │   Backend    │
         └──┬───────────┘
            │  POST Request
    ┌─────── ▼ ─────────┐
    │   Trained LLM     │  Hosted API
    │   (Qwen 2 7B)     │ ──► Actions + Params
    └───────────────────┘
            │
    ┌─────── ▼ ─────────┐
    │    GitHub CLI     │  Execution
    │    Commands       │
    └───────────────────┘
```

- **Frontend** — Built with ReactJS, takes user input via prompt or voice. Voice input is converted to text by WebAPI before being sent to the backend.
- **Trained LLM** — The trained model is hosted on a separate API, responding with the set of actions and corresponding parameters.
- **Backend** — Makes a POST request to the LLM API, processes the returned actions, and executes GitHub CLI commands autonomously.

📄 View the full [Backend Documentation](https://github.com/adisri-ai/GitPilot/blob/main/BACKEND.md)

---

## 〉 Training the LLM Model

```bash
# Step 1 — Configure paths
# Open training.py and update paths to your current working directory
# First run:  model_path = "Qwen/Qwen2-7B"
# Later runs: model_path = "<your-stored-local-path>"

# Step 2 — Run training (~2–3 hours)
python training.py

# Step 3 — Host the model on a separate API
python api.py

# Step 4 — Save the generated link
# Note: if using Google Colab free tier, the LLM API link is dynamic
# and will change on every run.
```

---

## 〉 Running the Application

From the project **root directory**, run:

```bash
# Pull the image
docker pull adisrinitw/gitpilot30-gitpilot:latest

# Run the container
docker run -it \
  -e EXTERNAL_API_URL=<YOUR_API_URL> \
  -p 5173:5173 \
  -p 5000:5000 \
  adisrinitw/gitpilot30-gitpilot:latest
```

Then click the **Vite link** to open the project frontend. Authenticate with your GitHub account to start using the agent.

---

## 〉 References

The architecture of the implemented LLM Model draws inspiration from **[Section 2.2](https://github.com/adisri-ai/GitHub-Workflow-Automation/blob/main/Referenced_Paper.pdf)** of the referenced research paper.

---

<div align="center">
<sub>Built with 🤖 and too many GitHub CLI man pages.</sub>
</div>
