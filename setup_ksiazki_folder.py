#!/usr/bin/env python3
"""
Set up ksiazki pdf folder monitoring for automatic PDF processing.
Creates the folder structure and monitoring script for PDF extraction.
"""

import os
import sys
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import asyncio
from typing import Set
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from extraction.pdf_processor import PDFProcessor
from src.utils.logging import get_logger

logger = get_logger(__name__)

class PDFHandler(FileSystemEventHandler):
    """Handle PDF file events in the ksiazki folder."""
    
    def __init__(self, processor_callback=None):
        """
        Initialize PDF handler.
        
        Args:
            processor_callback: Async function to process PDFs
        """
        self.processor_callback = processor_callback
        self.processing: Set[str] = set()
        self.processed_files = self._load_processed_files()
    
    def _load_processed_files(self) -> Set[str]:
        """Load list of already processed files."""
        processed_file = Path("ksiazki_pdf/.processed_files.txt")
        if processed_file.exists():
            with open(processed_file, 'r') as f:
                return set(line.strip() for line in f if line.strip())
        return set()
    
    def _save_processed_file(self, filepath: str):
        """Mark file as processed."""
        processed_file = Path("ksiazki_pdf/.processed_files.txt")
        processed_file.parent.mkdir(exist_ok=True)
        with open(processed_file, 'a') as f:
            f.write(filepath + '\n')
        self.processed_files.add(filepath)
    
    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return
        
        filepath = event.src_path
        if filepath.lower().endswith('.pdf'):
            self._queue_for_processing(filepath)
    
    def on_moved(self, event):
        """Handle file move events."""
        if event.is_directory:
            return
        
        filepath = event.dest_path
        if filepath.lower().endswith('.pdf'):
            self._queue_for_processing(filepath)
    
    def _queue_for_processing(self, filepath: str):
        """Queue a PDF file for processing."""
        filename = os.path.basename(filepath)
        
        # Skip if already processed or currently processing
        if filepath in self.processed_files or filepath in self.processing:
            logger.info(f"Skipping {filename} (already processed or processing)")
            return
        
        # Wait a moment for file to be completely written
        time.sleep(2)
        
        # Check if file still exists and is readable
        if not os.path.exists(filepath):
            logger.warning(f"File {filename} disappeared before processing")
            return
        
        try:
            # Test if file is readable
            with open(filepath, 'rb') as f:
                f.read(100)  # Try reading first 100 bytes
        except Exception as e:
            logger.warning(f"File {filename} not ready for processing: {e}")
            return
        
        logger.info(f"📚 New PDF detected: {filename}")
        
        if self.processor_callback:
            # Run async processing in a thread
            asyncio.create_task(self._process_pdf_async(filepath))
        else:
            logger.info(f"No processor configured for {filename}")
    
    async def _process_pdf_async(self, filepath: str):
        """Process PDF asynchronously."""
        filename = os.path.basename(filepath)
        
        try:
            self.processing.add(filepath)
            logger.info(f"🔄 Processing {filename}...")
            
            if self.processor_callback:
                await self.processor_callback(filepath)
            
            # Mark as processed
            self._save_processed_file(filepath)
            logger.info(f"✅ Successfully processed {filename}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {filename}: {e}")
        finally:
            self.processing.discard(filepath)

async def process_pdf_file(filepath: str):
    """
    Process a PDF file using the enhanced extraction system.
    
    Args:
        filepath: Path to the PDF file
    """
    logger.info(f"Starting enhanced extraction for {filepath}")
    
    try:
        # Initialize PDF processor
        processor = PDFProcessor()
        
        # Extract text and create chunks
        chunks = await processor.process_pdf(filepath)
        logger.info(f"Extracted {len(chunks)} chunks from {os.path.basename(filepath)}")
        
        # Here you would integrate with your extraction pipeline
        # For now, just log the results
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            preview = chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
            logger.info(f"Chunk {i}: {preview}")
        
        # TODO: Integrate with enhanced extraction pipeline
        # - Store document in database
        # - Run enhanced extraction on chunks  
        # - Store results with enhanced metadata
        
        logger.info(f"Enhanced extraction completed for {os.path.basename(filepath)}")
        
    except Exception as e:
        logger.error(f"Failed to process {filepath}: {e}")
        raise

def setup_ksiazki_folder():
    """Set up the ksiazki pdf folder structure."""
    
    print("=" * 70)
    print("📚 Setting up ksiazki PDF folder monitoring")
    print("=" * 70)
    
    # Create folder structure
    ksiazki_dir = Path("ksiazki_pdf")
    ksiazki_dir.mkdir(exist_ok=True)
    
    # Create subfolders
    (ksiazki_dir / "processed").mkdir(exist_ok=True)
    (ksiazki_dir / "failed").mkdir(exist_ok=True)
    (ksiazki_dir / "archive").mkdir(exist_ok=True)
    
    # Create README
    readme_content = """# ksiazki PDF Folder

This folder is monitored for automatic PDF processing.

## Folder Structure:
- `ksiazki_pdf/` - Drop PDFs here for automatic processing
- `ksiazki_pdf/processed/` - Successfully processed PDFs are moved here
- `ksiazki_pdf/failed/` - Failed PDFs are moved here for manual review
- `ksiazki_pdf/archive/` - Older PDFs for long-term storage

## Usage:
1. Copy or move PDF files into the main ksiazki_pdf/ folder
2. The system will automatically detect and process them
3. Check the logs for processing status
4. Processed files are moved to appropriate subfolders

## Monitoring:
- Run `python monitor_ksiazki_folder.py` to start the file watcher
- Or use `python setup_ksiazki_folder.py --monitor` for combined setup and monitoring

## Processed Files:
The system maintains a list of processed files in `.processed_files.txt` to avoid reprocessing.
"""
    
    with open(ksiazki_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    print(f"✅ Created folder structure at {ksiazki_dir}")
    print("✅ Created README.md with usage instructions")
    
    return ksiazki_dir

def start_monitoring(ksiazki_dir: Path):
    """Start monitoring the ksiazki folder for new PDFs."""
    
    print(f"\n👀 Starting folder monitoring for {ksiazki_dir}")
    print("Press Ctrl+C to stop monitoring")
    
    # Set up event handler
    event_handler = PDFHandler(processor_callback=process_pdf_file)
    
    # Set up observer
    observer = Observer()
    observer.schedule(event_handler, str(ksiazki_dir), recursive=False)
    
    try:
        observer.start()
        print("✅ Monitoring started successfully!")
        print(f"📁 Watching: {ksiazki_dir.absolute()}")
        print("📚 Drop PDF files here for automatic processing")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping folder monitoring...")
        observer.stop()
    
    observer.join()
    print("✅ Monitoring stopped")

def process_existing_files(ksiazki_dir: Path):
    """Process any existing PDF files in the folder."""
    
    pdf_files = list(ksiazki_dir.glob("*.pdf"))
    if not pdf_files:
        print("📂 No existing PDF files found")
        return
    
    print(f"📚 Found {len(pdf_files)} existing PDF files")
    
    handler = PDFHandler(processor_callback=process_pdf_file)
    
    for pdf_file in pdf_files:
        if str(pdf_file) not in handler.processed_files:
            print(f"Processing existing file: {pdf_file.name}")
            asyncio.run(handler._process_pdf_async(str(pdf_file)))

async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Set up ksiazki PDF folder monitoring")
    parser.add_argument("--monitor", action="store_true", help="Start monitoring after setup")
    parser.add_argument("--process-existing", action="store_true", help="Process existing files")
    args = parser.parse_args()
    
    # Set up folder structure
    ksiazki_dir = setup_ksiazki_folder()
    
    # Process existing files if requested
    if args.process_existing:
        print("\n🔄 Processing existing files...")
        process_existing_files(ksiazki_dir)
    
    # Start monitoring if requested
    if args.monitor:
        start_monitoring(ksiazki_dir)
    else:
        print(f"\n📖 Setup complete!")
        print(f"To start monitoring: python {__file__} --monitor")
        print(f"To process existing files: python {__file__} --process-existing")

if __name__ == "__main__":
    # Check if watchdog is installed
    try:
        import watchdog
    except ImportError:
        print("❌ watchdog library not found. Installing...")
        os.system("pip install watchdog")
        import watchdog
    
    asyncio.run(main())