# ksiazki PDF Folder

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
