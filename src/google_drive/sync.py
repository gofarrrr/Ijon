"""
Google Drive folder synchronization module.

This module handles syncing PDFs from Google Drive folders,
tracking processed/unprocessed files, and managing sync state.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.config import get_settings
from src.models import PDFMetadata, ProcessingStatus
from src.utils.errors import GoogleDriveError
from src.utils.logging import get_logger, log_performance
from src.google_drive.client import GoogleDriveClient

logger = get_logger(__name__)


class DriveSyncManager:
    """Manage synchronization of PDFs from Google Drive folders."""

    def __init__(
        self,
        drive_client: Optional[GoogleDriveClient] = None,
        sync_state_file: Optional[Path] = None,
    ) -> None:
        """
        Initialize sync manager.

        Args:
            drive_client: Google Drive client instance
            sync_state_file: Path to store sync state
        """
        self.settings = get_settings()
        self.drive_client = drive_client or GoogleDriveClient()
        self.sync_state_file = sync_state_file or Path(".sync_state.json")
        
        self._sync_state: Dict[str, Any] = self._load_sync_state()

    def _load_sync_state(self) -> Dict[str, Any]:
        """Load sync state from file."""
        if self.sync_state_file.exists():
            try:
                import json
                with open(self.sync_state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load sync state: {e}")
        
        return {
            "last_sync": None,
            "processed_files": {},
            "failed_files": {},
            "sync_history": [],
        }

    def _save_sync_state(self) -> None:
        """Save sync state to file."""
        try:
            import json
            with open(self.sync_state_file, "w") as f:
                json.dump(self._sync_state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save sync state: {e}")

    @log_performance
    async def sync_folders(
        self,
        folder_ids: Optional[List[str]] = None,
        force_resync: bool = False,
    ) -> Dict[str, Any]:
        """
        Sync PDFs from configured Google Drive folders.

        Args:
            folder_ids: Specific folder IDs to sync (None for all configured)
            force_resync: Force re-sync of all files

        Returns:
            Sync results dictionary

        Raises:
            GoogleDriveError: If sync fails
        """
        # Ensure connected
        if not self.drive_client._service:
            await self.drive_client.connect()
        
        folder_ids = folder_ids or self.settings.drive_folder_ids
        if not folder_ids:
            raise GoogleDriveError("No folder IDs configured for sync")
        
        logger.info(f"Starting sync for {len(folder_ids)} folders")
        
        sync_start = datetime.utcnow()
        results = {
            "folders_synced": len(folder_ids),
            "new_files": 0,
            "updated_files": 0,
            "failed_files": 0,
            "total_files": 0,
            "files": [],
        }
        
        try:
            # Get all PDFs from folders
            all_pdfs = await self.drive_client.list_pdfs(
                folder_id=None,  # Use all configured folders
                include_subfolders=True,
            )
            results["total_files"] = len(all_pdfs)
            
            # Process each PDF
            for pdf in all_pdfs:
                file_id = pdf["id"]
                
                # Check if already processed
                if not force_resync and self.is_file_processed(file_id):
                    logger.debug(f"Skipping already processed file: {pdf['name']}")
                    continue
                
                # Check if file was modified since last sync
                if not force_resync and self.is_file_unchanged(file_id, pdf):
                    logger.debug(f"Skipping unchanged file: {pdf['name']}")
                    continue
                
                # Mark file for processing
                file_info = {
                    "id": file_id,
                    "name": pdf["name"],
                    "size": int(pdf.get("size", 0)),
                    "modified_time": pdf.get("modifiedTime"),
                    "status": "pending",
                }
                
                # Determine if new or updated
                if file_id in self._sync_state["processed_files"]:
                    results["updated_files"] += 1
                    file_info["status"] = "updated"
                else:
                    results["new_files"] += 1
                    file_info["status"] = "new"
                
                results["files"].append(file_info)
            
            # Update sync state
            self._sync_state["last_sync"] = sync_start.isoformat()
            self._sync_state["sync_history"].append({
                "timestamp": sync_start.isoformat(),
                "folders": folder_ids,
                "results": {
                    "total": results["total_files"],
                    "new": results["new_files"],
                    "updated": results["updated_files"],
                },
            })
            
            # Keep only last 100 sync history entries
            self._sync_state["sync_history"] = self._sync_state["sync_history"][-100:]
            
            self._save_sync_state()
            
            logger.info(
                f"Sync completed: {results['new_files']} new, "
                f"{results['updated_files']} updated, "
                f"{results['total_files']} total files"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Sync failed: {str(e)}")
            raise GoogleDriveError(f"Sync failed: {str(e)}")

    def is_file_processed(self, file_id: str) -> bool:
        """Check if a file has been successfully processed."""
        return file_id in self._sync_state["processed_files"]

    def is_file_unchanged(self, file_id: str, file_metadata: Dict[str, Any]) -> bool:
        """Check if a file hasn't changed since last sync."""
        if file_id not in self._sync_state["processed_files"]:
            return False
        
        last_processed = self._sync_state["processed_files"][file_id]
        current_modified = file_metadata.get("modifiedTime")
        
        return last_processed.get("modified_time") == current_modified

    def mark_file_processed(
        self,
        file_id: str,
        metadata: PDFMetadata,
        chunks_created: int = 0,
    ) -> None:
        """
        Mark a file as successfully processed.

        Args:
            file_id: Google Drive file ID
            metadata: PDF metadata
            chunks_created: Number of chunks created
        """
        self._sync_state["processed_files"][file_id] = {
            "processed_at": datetime.utcnow().isoformat(),
            "filename": metadata.filename,
            "total_pages": metadata.total_pages,
            "file_size": metadata.file_size_bytes,
            "chunks_created": chunks_created,
            "modified_time": metadata.updated_at.isoformat() if metadata.updated_at else None,
        }
        
        # Remove from failed if it was there
        if file_id in self._sync_state["failed_files"]:
            del self._sync_state["failed_files"][file_id]
        
        self._save_sync_state()
        logger.info(f"Marked file {metadata.filename} as processed")

    def mark_file_failed(
        self,
        file_id: str,
        filename: str,
        error: str,
    ) -> None:
        """
        Mark a file as failed to process.

        Args:
            file_id: Google Drive file ID
            filename: File name
            error: Error message
        """
        self._sync_state["failed_files"][file_id] = {
            "failed_at": datetime.utcnow().isoformat(),
            "filename": filename,
            "error": error,
            "retry_count": self._sync_state["failed_files"].get(file_id, {}).get("retry_count", 0) + 1,
        }
        
        self._save_sync_state()
        logger.warning(f"Marked file {filename} as failed: {error}")

    def get_failed_files(self, max_retries: int = 3) -> List[Dict[str, Any]]:
        """
        Get list of failed files that haven't exceeded retry limit.

        Args:
            max_retries: Maximum retry attempts

        Returns:
            List of failed file info
        """
        failed_files = []
        
        for file_id, info in self._sync_state["failed_files"].items():
            if info["retry_count"] < max_retries:
                failed_files.append({
                    "id": file_id,
                    "filename": info["filename"],
                    "error": info["error"],
                    "retry_count": info["retry_count"],
                    "failed_at": info["failed_at"],
                })
        
        return failed_files

    def get_sync_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        total_processed = len(self._sync_state["processed_files"])
        total_failed = len(self._sync_state["failed_files"])
        
        # Calculate total chunks
        total_chunks = sum(
            info.get("chunks_created", 0)
            for info in self._sync_state["processed_files"].values()
        )
        
        # Calculate total size
        total_size = sum(
            info.get("file_size", 0)
            for info in self._sync_state["processed_files"].values()
        )
        
        return {
            "last_sync": self._sync_state["last_sync"],
            "total_processed": total_processed,
            "total_failed": total_failed,
            "total_chunks": total_chunks,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "sync_history_count": len(self._sync_state["sync_history"]),
        }

    async def retry_failed_files(
        self,
        max_retries: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Get list of failed files to retry.

        Args:
            max_retries: Maximum retry attempts

        Returns:
            List of files to retry
        """
        failed_files = self.get_failed_files(max_retries)
        
        if failed_files:
            logger.info(f"Found {len(failed_files)} failed files to retry")
        
        return failed_files

    def reset_sync_state(self) -> None:
        """Reset sync state (use with caution)."""
        logger.warning("Resetting sync state")
        
        self._sync_state = {
            "last_sync": None,
            "processed_files": {},
            "failed_files": {},
            "sync_history": [],
        }
        
        self._save_sync_state()


def create_sync_manager() -> DriveSyncManager:
    """Create a sync manager with default settings."""
    from src.google_drive.client import create_drive_client
    
    drive_client = create_drive_client()
    return DriveSyncManager(drive_client)