# Google Drive Integration Setup

This guide will help you connect Google Drive to the Ijon PDF RAG system.

## Prerequisites

1. Google account with access to Drive
2. Google Cloud Console access
3. Python environment with dependencies installed

## Setup Steps

### 1. Enable Google Drive API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Navigate to "APIs & Services" > "Library"
4. Search for "Google Drive API"
5. Click on it and press "Enable"

### 2. Create OAuth2 Credentials

1. In Google Cloud Console, go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth client ID"
3. If prompted, configure the OAuth consent screen:
   - Choose "External" for testing
   - Fill in required fields (app name, email)
   - Add your email to test users
4. For Application type, select "Desktop app"
5. Give it a name (e.g., "Ijon PDF RAG")
6. Click "Create"
7. Download the credentials JSON file

### 3. Set Up Authentication

1. Place the downloaded credentials file as `credentials.json` in the project root:
   ```bash
   mv ~/Downloads/client_secret_*.json ./credentials.json
   ```

2. Run the setup script:
   ```bash
   source venv_ijon/bin/activate
   python setup_google_drive.py
   ```

3. Follow the browser prompts to authorize access
4. Optionally enter Google Drive folder IDs to sync specific folders

### 4. Sync PDFs from Drive

Once authenticated, sync PDFs from your Drive:

```bash
python sync_drive_pdfs.py
```

This will:
- Download PDFs from specified folders (or all PDFs if no folders specified)
- Process them through the PDF extraction pipeline
- Create embeddings and store in Pinecone
- Track processed files to avoid reprocessing

### 5. Query Your PDFs

Query the processed PDFs:

```bash
# Interactive mode
python query_drive_pdfs.py

# Single query
python query_drive_pdfs.py "What is machine learning?"
```

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# Google Drive folder IDs (JSON array format)
DRIVE_FOLDER_IDS=["folder_id_1","folder_id_2"]

# Or single folder
DRIVE_FOLDER_IDS=["single_folder_id"]

# Legacy comma-separated format also supported
# DRIVE_FOLDER_IDS=folder_id_1,folder_id_2

# Credentials path (optional, defaults to credentials.json)
DRIVE_CREDENTIALS_PATH=credentials.json
```

### Finding Folder IDs

1. Open Google Drive in your browser
2. Navigate to the folder you want to sync
3. Look at the URL: `https://drive.google.com/drive/folders/[FOLDER_ID]`
4. Copy the FOLDER_ID part

## File Management

### Processed Files Tracking

The system tracks processed files in `processed_files.json`:
- Prevents redownloading unchanged files
- Updates files when modified in Drive
- Stores metadata about processing

### Storage Locations

- Downloaded PDFs: `drive_pdfs/`
- OAuth token: `token.json`
- Processed files list: `processed_files.json`

## Troubleshooting

### Authentication Issues

If authentication fails:
1. Delete `token.json` and re-authenticate
2. Check that Drive API is enabled in Google Cloud Console
3. Verify credentials.json is valid

### API Quota Limits

Google Drive API has quotas:
- 1,000,000,000 requests per day
- 1,000 requests per 100 seconds per user

The sync script respects these limits automatically.

### Large Files

For PDFs larger than 100MB:
- Download may take longer
- Processing uses streaming to avoid memory issues
- Consider chunking very large documents

## Security Notes

1. **Never commit credentials.json or token.json to git**
   - These files are already in .gitignore
   
2. **Token expiration**
   - Tokens auto-refresh when possible
   - Re-authenticate if refresh fails

3. **Scope limitations**
   - We request read-only access by default
   - Minimal permissions for security

## Advanced Usage

### Service Account Authentication

For server deployments without user interaction:

1. Create a service account in Google Cloud Console
2. Download the service account key
3. Share Drive folders with the service account email
4. Use `ServiceAccountAuth` class in the code

### Selective Sync

Sync specific file types or patterns:
```python
# In sync_drive_pdfs.py, modify the query
query = "mimeType='application/pdf' and name contains 'research'"
```

### Incremental Updates

The system automatically handles incremental updates:
- Only downloads new or modified files
- Preserves existing embeddings
- Updates only changed content