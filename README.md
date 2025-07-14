## REQUIREMENTS
- Python 3.11
- Download the latest version of [Ollama](https://ollama.com/download)

## INSTALLATION
1. Clone the repository:
    ```bash
    git clone https://github.com/lilmaxie/MultiAgentLLM.git
    ```
2. Navigate to the project directory:
    ```bash
    cd MultiAgentLLM
    ```
3. Install the required packages: 
   ```bash
   pip install -r requirements.txt
   ```
4. Pull ollama models:
   ```bash
    python -m utils.pull_ollama_model
   ```
   OR 
4. You can just pull the specific model you need:
   ```bash
   ollama pull qwen3:1.7b
   ```
5. Run the multi-agent system:
   ```bash
   python -m multiagent_system.py
   ```
**NOTE**
- You can modify the query in `multiagent_system.py` to test different scenarios. (line 488-497)