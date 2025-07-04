"""
Quick test to verify OpenAI API key is working.
"""

import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

async def test_openai():
    """Test OpenAI API with a simple request."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ No OPENAI_API_KEY found in environment")
        return
    
    print(f"🔑 Using API key: {api_key[:10]}...{api_key[-4:]}")
    
    client = AsyncOpenAI(api_key=api_key)
    
    try:
        print("📡 Testing OpenAI API...")
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using 3.5 for faster test
            messages=[
                {"role": "user", "content": "Say 'API is working!' in 5 words or less."}
            ],
            max_tokens=20,
            temperature=0
        )
        
        print(f"✅ Response: {response.choices[0].message.content}")
        print(f"✅ Model used: {response.model}")
        print(f"✅ API is working correctly!")
        
    except Exception as e:
        print(f"❌ API Error: {type(e).__name__}: {str(e)}")
        if "api_key" in str(e).lower():
            print("🔧 The API key appears to be invalid or expired")
            print("   Please update the OPENAI_API_KEY in .env file")

if __name__ == "__main__":
    asyncio.run(test_openai())