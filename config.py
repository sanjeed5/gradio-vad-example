"""
Configuration file for API providers and models.
Can be switched between OpenAI and Groq as needed.
"""

# API Provider: 'openai' or 'groq'
API_PROVIDER = 'openai'

# API Keys - set as environment variables
# OPENAI_API_KEY or GROQ_API_KEY

# Models configuration
MODELS = {
    'openai': {
        'chat': 'gpt-4o-mini',
        'audio': 'whisper-1'
    },
    'groq': {
        'chat': 'llama-3.2-11b-vision-preview',
        'audio': 'whisper-large-v3-turbo'
    }
}

# System prompt for nutrition assistant
SYSTEM_PROMPT = """In conversation with the user, ask questions to estimate and provide 
(1) total calories, (2) protein, carbs, and fat in grams, (3) fiber and sugar content. 
Only ask *one question at a time*. Be conversational and natural.""" 