#!/bin/bash

# Check if HF_TOKEN is set, if not prompt for it
if [ -z "$HF_API_KEY" ]; then
    read -p "Enter your Hugging Face token: " HF_API_KEY
fi

# Export the token as an environment variable
export HF_API_KEY

# Run the Streamlit application
exec "streamlit run --server.port 8501 app.py"