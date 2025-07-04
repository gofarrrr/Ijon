#!/usr/bin/env python3
"""
Setup Google Drive authentication for the Ijon PDF RAG system.
"""

import asyncio
import json
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

print("=" * 70)
print("🔐 Google Drive Authentication Setup")
print("=" * 70)

# Manual environment loading to avoid config issues
def load_env():
    env_vars = {}
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.strip().split('=', 1)
                    if value:
                        env_vars[key] = value
                        os.environ[key] = value
    return env_vars

env = load_env()


async def setup_drive_auth():
    """Setup Google Drive authentication."""
    
    print("\n📋 Prerequisites:")
    print("-" * 60)
    print("1. Go to Google Cloud Console: https://console.cloud.google.com/")
    print("2. Create a new project or select existing one")
    print("3. Enable Google Drive API")
    print("4. Create OAuth2 credentials (Desktop application)")
    print("5. Download the credentials JSON file")
    print("-" * 60)
    
    # Check for credentials file
    creds_path = Path("credentials.json")
    if not creds_path.exists():
        print("\n❌ credentials.json not found!")
        print("\n📁 Please place your OAuth2 credentials file as 'credentials.json'")
        print("   in the current directory: " + str(Path.cwd()))
        return False
    
    print("\n✅ Found credentials.json")
    
    # Try to authenticate
    try:
        from src.google_drive.auth import GoogleDriveAuth
        
        print("\n🔑 Starting OAuth2 flow...")
        print("   A browser window will open for authentication.")
        print("   Please authorize access to your Google Drive.")
        
        auth = GoogleDriveAuth(
            credentials_path=creds_path,
            token_path=Path("token.json")
        )
        
        # Authenticate
        credentials = await auth.authenticate()
        
        print("\n✅ Authentication successful!")
        
        # Get user info
        user_info = await auth.get_user_info()
        print(f"\n👤 Authenticated as: {user_info.get('email', 'Unknown')}")
        
        # Test Drive API access
        from googleapiclient.discovery import build
        
        service = build('drive', 'v3', credentials=credentials)
        
        # List some files to verify access
        print("\n📂 Testing Drive access - listing recent files:")
        print("-" * 60)
        
        results = service.files().list(
            pageSize=5,
            fields="files(id, name, mimeType, modifiedTime)",
            orderBy="modifiedTime desc"
        ).execute()
        
        files = results.get('files', [])
        if files:
            for file in files:
                print(f"   • {file['name']} ({file['mimeType']})")
        else:
            print("   No files found (this is OK if Drive is empty)")
        
        # Update .env with Drive folder IDs if provided
        print("\n📁 Google Drive Folder Setup")
        print("-" * 60)
        print("Would you like to specify Google Drive folders to sync?")
        print("You can find folder IDs in the URL when viewing a folder:")
        print("https://drive.google.com/drive/folders/[FOLDER_ID_HERE]")
        
        folder_ids = input("\nEnter folder IDs (comma-separated) or press Enter to skip: ").strip()
        
        if folder_ids:
            # Convert to JSON array format
            folder_list = [fid.strip() for fid in folder_ids.split(',') if fid.strip()]
            folder_json = json.dumps(folder_list)
            
            # Update .env file
            env_lines = []
            if Path('.env').exists():
                with open('.env', 'r') as f:
                    env_lines = f.readlines()
            
            # Update or add DRIVE_FOLDER_IDS
            updated = False
            for i, line in enumerate(env_lines):
                if line.startswith('DRIVE_FOLDER_IDS='):
                    env_lines[i] = f'DRIVE_FOLDER_IDS={folder_json}\n'
                    updated = True
                    break
            
            if not updated:
                env_lines.append(f'\n# Google Drive folders to sync (JSON array)\n')
                env_lines.append(f'DRIVE_FOLDER_IDS={folder_json}\n')
            
            with open('.env', 'w') as f:
                f.writelines(env_lines)
            
            print(f"\n✅ Updated .env with folder IDs: {folder_json}")
        
        print("\n" + "=" * 70)
        print("🎉 Google Drive setup complete!")
        print("=" * 70)
        print("\nYou can now:")
        print("1. Run the PDF sync: python sync_drive_pdfs.py")
        print("2. Process PDFs: python test_real_pipeline.py")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("\nPlease install Google API dependencies:")
        print("pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")
        return False
        
    except Exception as e:
        print(f"\n❌ Error during authentication: {e}")
        return False


async def main():
    """Run the setup."""
    success = await setup_drive_auth()
    
    if not success:
        print("\n⚠️  Setup incomplete. Please resolve the issues and try again.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())