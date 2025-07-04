#!/usr/bin/env python3
"""
Sync PDFs from Google Drive folders and process them into the RAG system.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import json

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Load environment manually
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

print("=" * 70)
print("🔄 Google Drive PDF Sync & Processing")
print("=" * 70)


class GoogleDriveSync:
    """Sync PDFs from Google Drive."""
    
    def __init__(self):
        self.download_dir = Path("drive_pdfs")
        self.download_dir.mkdir(exist_ok=True)
        self.processed_file = Path("processed_files.json")
        self.processed_files = self._load_processed_files()
    
    def _load_processed_files(self) -> Dict[str, Any]:
        """Load list of already processed files."""
        if self.processed_file.exists():
            with open(self.processed_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_processed_files(self):
        """Save processed files list."""
        with open(self.processed_file, 'w') as f:
            json.dump(self.processed_files, f, indent=2)
    
    async def authenticate(self):
        """Authenticate with Google Drive."""
        from src.google_drive.auth import GoogleDriveAuth
        
        self.auth = GoogleDriveAuth(
            credentials_path=Path("credentials.json"),
            token_path=Path("token.json")
        )
        
        # Check if already authenticated
        if Path("token.json").exists():
            print("✅ Using existing authentication token")
        else:
            print("🔑 Starting authentication flow...")
        
        self.credentials = await self.auth.authenticate()
        
        # Get user info
        user_info = await self.auth.get_user_info()
        print(f"👤 Authenticated as: {user_info.get('email', 'Unknown')}")
        
        # Build service
        from googleapiclient.discovery import build
        self.service = build('drive', 'v3', credentials=self.credentials)
    
    async def list_pdf_files(self, folder_id: str = None) -> List[Dict[str, Any]]:
        """List PDF files in Drive or specific folder."""
        print(f"\n📂 Listing PDF files{f' in folder {folder_id}' if folder_id else ''}...")
        
        query_parts = ["mimeType='application/pdf'"]
        if folder_id:
            query_parts.append(f"'{folder_id}' in parents")
        query_parts.append("trashed=false")
        
        query = " and ".join(query_parts)
        
        pdf_files = []
        page_token = None
        
        while True:
            try:
                results = self.service.files().list(
                    q=query,
                    pageSize=100,
                    fields="nextPageToken, files(id, name, size, modifiedTime, parents)",
                    pageToken=page_token
                ).execute()
                
                files = results.get('files', [])
                pdf_files.extend(files)
                
                page_token = results.get('nextPageToken')
                if not page_token:
                    break
                    
            except Exception as e:
                print(f"❌ Error listing files: {e}")
                break
        
        print(f"✅ Found {len(pdf_files)} PDF files")
        return pdf_files
    
    async def download_pdf(self, file_info: Dict[str, Any]) -> Path:
        """Download a PDF file from Drive."""
        file_id = file_info['id']
        file_name = file_info['name']
        file_path = self.download_dir / file_name
        
        # Check if already downloaded and up to date
        if file_path.exists():
            if file_id in self.processed_files:
                stored_time = self.processed_files[file_id].get('modifiedTime')
                if stored_time == file_info.get('modifiedTime'):
                    print(f"   ⏭️  Skipping {file_name} (already up to date)")
                    return None
        
        print(f"   ⬇️  Downloading {file_name}...")
        
        try:
            from googleapiclient.http import MediaIoBaseDownload
            import io
            
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"      Download {int(status.progress() * 100)}%")
            
            # Write to file
            fh.seek(0)
            with open(file_path, 'wb') as f:
                f.write(fh.read())
            
            print(f"   ✅ Downloaded to {file_path}")
            
            # Update processed files
            self.processed_files[file_id] = {
                'name': file_name,
                'modifiedTime': file_info.get('modifiedTime'),
                'downloadedAt': datetime.now().isoformat(),
                'path': str(file_path)
            }
            self._save_processed_files()
            
            return file_path
            
        except Exception as e:
            print(f"   ❌ Error downloading {file_name}: {e}")
            return None
    
    async def sync_folders(self, folder_ids: List[str] = None):
        """Sync PDFs from specified folders or entire Drive."""
        if not folder_ids:
            # List all PDFs if no folders specified
            pdf_files = await self.list_pdf_files()
        else:
            # List PDFs from each folder
            pdf_files = []
            for folder_id in folder_ids:
                files = await self.list_pdf_files(folder_id)
                pdf_files.extend(files)
        
        if not pdf_files:
            print("\n❌ No PDF files found")
            return []
        
        # Download PDFs
        print(f"\n📥 Downloading PDFs...")
        downloaded_files = []
        
        for file_info in pdf_files[:10]:  # Limit to 10 files for testing
            file_path = await self.download_pdf(file_info)
            if file_path:
                downloaded_files.append(file_path)
        
        return downloaded_files


async def process_pdfs(pdf_paths: List[Path]):
    """Process downloaded PDFs into the RAG system."""
    if not pdf_paths:
        print("\n⏭️  No new PDFs to process")
        return
    
    print(f"\n🔄 Processing {len(pdf_paths)} PDFs...")
    
    # Import processing components
    from src.pdf_processor.extractor import PDFExtractor
    from src.text_processing.preprocessor import TextPreprocessor
    from src.text_processing.semantic_chunker import SemanticChunker
    from src.vector_db.pinecone_adapter import PineconeVectorDB
    from src.models import Document
    import openai
    
    # Initialize components
    extractor = PDFExtractor(enable_ocr=False)
    preprocessor = TextPreprocessor()
    chunker = SemanticChunker(
        chunk_size=int(env.get('CHUNK_SIZE', '1000')),
        chunk_overlap=int(env.get('CHUNK_OVERLAP', '200'))
    )
    
    # Initialize vector DB
    openai_client = openai.Client(api_key=env.get('OPENAI_API_KEY'))
    
    async def generate_embedding(texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = openai_client.embeddings.create(
                input=text,
                model=env.get('EMBEDDING_MODEL', 'text-embedding-ada-002')
            )
            embeddings.append(response.data[0].embedding)
        return embeddings
    
    vector_db = PineconeVectorDB(
        api_key=env.get('PINECONE_API_KEY'),
        environment=env.get('PINECONE_ENVIRONMENT'),
        index_name='ijon-drive-pdfs',
        dimension=1536,
        embedding_function=generate_embedding
    )
    
    try:
        await vector_db.initialize()
        print("✅ Vector database initialized")
    except Exception as e:
        print(f"⚠️  Vector DB initialization warning: {e}")
        print("   Continuing without vector storage")
        vector_db = None
    
    # Process each PDF
    for pdf_path in pdf_paths:
        try:
            print(f"\n📄 Processing: {pdf_path.name}")
            
            # Extract metadata
            metadata = await extractor.extract_metadata(pdf_path)
            print(f"   ✓ Pages: {metadata.total_pages}")
            
            # Extract pages
            pages = await extractor.extract_pages(pdf_path)
            
            # Process and chunk
            all_chunks = []
            for page_num, page in enumerate(pages, 1):
                if page.text.strip():
                    # Clean text
                    page.text = preprocessor.clean_text(page.text)
                    
                    # Create chunks
                    chunks = chunker.chunk_text(
                        page.text,
                        metadata={
                            'page_number': page_num,
                            'pdf_id': pdf_path.stem,
                            'filename': pdf_path.name,
                            'source': 'google_drive'
                        }
                    )
                    all_chunks.extend(chunks)
            
            print(f"   ✓ Created {len(all_chunks)} chunks")
            
            # Store in vector DB if available
            if vector_db and all_chunks:
                documents = []
                for i, chunk in enumerate(all_chunks):
                    doc = Document(
                        id=f"{pdf_path.stem}_chunk_{i}",
                        content=chunk.content,
                        metadata={
                            'pdf_id': pdf_path.stem,
                            'filename': pdf_path.name,
                            'page_numbers': chunk.page_numbers,
                            'chunk_index': i,
                            'source': 'google_drive',
                            'processed_at': datetime.now().isoformat()
                        }
                    )
                    documents.append(doc)
                
                # Generate embeddings
                texts = [doc.content for doc in documents]
                embeddings = await generate_embedding(texts)
                
                for doc, embedding in zip(documents, embeddings):
                    doc.embedding = embedding
                
                # Store
                await vector_db.upsert_documents(documents)
                print(f"   ✓ Stored {len(documents)} chunks in vector DB")
            
        except Exception as e:
            print(f"   ❌ Error processing {pdf_path.name}: {e}")


async def main():
    """Main sync and process workflow."""
    # Initialize Drive sync
    sync = GoogleDriveSync()
    
    try:
        # Authenticate
        await sync.authenticate()
        
        # Get folder IDs from env (JSON format)
        folder_ids_str = env.get('DRIVE_FOLDER_IDS', '')
        folder_ids = []
        if folder_ids_str:
            try:
                folder_ids = json.loads(folder_ids_str)
                if isinstance(folder_ids, str):
                    # Fallback for old comma-separated format
                    folder_ids = [f.strip() for f in folder_ids_str.split(',') if f.strip()]
            except json.JSONDecodeError:
                # Fallback for old comma-separated format
                folder_ids = [f.strip() for f in folder_ids_str.split(',') if f.strip()]
        
        if folder_ids:
            print(f"\n📁 Syncing from {len(folder_ids)} folders")
        else:
            print("\n📁 Syncing all PDFs from Drive")
        
        # Sync PDFs
        downloaded_files = await sync.sync_folders(folder_ids)
        
        # Process PDFs
        await process_pdfs(downloaded_files)
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 Sync Summary")
        print("=" * 70)
        print(f"✅ Downloaded: {len(downloaded_files)} new/updated PDFs")
        print(f"✅ Processed: {len(downloaded_files)} PDFs")
        print(f"✅ Total tracked: {len(sync.processed_files)} files")
        
        if downloaded_files:
            print("\n📚 Processed files:")
            for path in downloaded_files:
                print(f"   • {path.name}")
        
        print("\n🎉 Google Drive sync complete!")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\n⚠️  Please run setup_google_drive.py first to authenticate")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())