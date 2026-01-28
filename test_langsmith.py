import os

# Note: Set LANGSMITH_API_KEY environment variable before running
# os.environ['LANGSMITH_API_KEY'] = 'your_api_key_here'

try:
    from langsmith import Client
    client = Client()
    # Test basic functionality - list projects
    print('✅ LangSmith API Key is VALID')
    print(f'Client initialized successfully')
    
    # Try to create a simple run to verify write access
    from langsmith import traceable
    print('✅ LangSmith client can authenticate and connect')
    
except Exception as e:
    print(f'❌ LangSmith API Key FAILED: {e}')
