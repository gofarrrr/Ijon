"""
Google Drive API client for file operations.

This module provides a high-level interface for interacting with Google Drive,
including listing, downloading, and syncing PDF files.
"""

import asyncio
import io
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

from google.auth.transport.requests import Request
from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import get_settings
from src.models import PDFMetadata, ProcessingStatus
from src.utils.errors import (
    DriveAuthenticationError,
    DriveFileNotFoundError,
    DriveQuotaExceededError,
    GoogleDriveError,
)
from src.utils.logging import LogContext, get_logger, log_performance
from src.google_drive.auth import GoogleDriveAuth

logger = get_logger(__name__)

# MIME types for PDFs
PDF_MIME_TYPE = "application/pdf"
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


class GoogleDriveClient:
    """Client for Google Drive operations."""

    def __init__(
        self,
        auth_manager: Optional[GoogleDriveAuth] = None,
        page_size: int = 100,
    ) -> None:
        """
        Initialize Google Drive client.

        Args:
            auth_manager: Authentication manager
            page_size: Number of files per page when listing
        """
        self.settings = get_settings()
        self.auth_manager = auth_manager or GoogleDriveAuth()
        self.page_size = page_size
        
        self._service: Optional[Resource] = None

    async def connect(self) -> None:
        """
        Connect to Google Drive API.

        Raises:
            DriveAuthenticationError: If authentication fails
        """
        if not self.auth_manager.is_authenticated:
            await self.auth_manager.authenticate()
        
        try:
            self._service = build(
                "drive",
                "v3",
                credentials=self.auth_manager.credentials,
            )
            logger.info("Connected to Google Drive API")
        except Exception as e:
            logger.error(f"Failed to build Drive service: {e}")
            raise GoogleDriveError(f"Failed to connect to Drive API: {str(e)}")

    def ensure_connected(self) -> None:
        """Ensure client is connected to Drive API."""
        if not self._service:
            raise GoogleDriveError("Not connected to Drive API. Call connect() first.")

    @log_performance
    async def list_pdfs(
        self,
        folder_id: Optional[str] = None,
        include_subfolders: bool = True,
        processed_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        List PDF files in Drive folder(s).

        Args:
            folder_id: Specific folder ID (None for all configured folders)
            include_subfolders: Whether to include PDFs in subfolders
            processed_only: Only return processed PDFs

        Returns:
            List of file metadata dictionaries

        Raises:
            GoogleDriveError: If listing fails
        """
        self.ensure_connected()
        
        folder_ids = [folder_id] if folder_id else self.settings.drive_folder_ids
        if not folder_ids:
            logger.warning("No folder IDs configured")
            return []
        
        all_files = []
        
        for fid in folder_ids:
            with LogContext(folder_id=fid):
                files = await self._list_pdfs_in_folder(fid, include_subfolders)
                
                # Filter by processing status if requested
                if processed_only:
                    files = [
                        f for f in files
                        if f.get("properties", {}).get("processed") == "true"
                    ]
                
                all_files.extend(files)
        
        logger.info(f"Found {len(all_files)} PDF files")
        return all_files

    @retry(
        retry=retry_if_exception_type((HttpError,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def _list_pdfs_in_folder(
        self,
        folder_id: str,
        include_subfolders: bool = True,
    ) -> List[Dict[str, Any]]:
        """List PDFs in a specific folder with retry logic."""
        files = []
        
        try:
            # Build query
            query_parts = [
                f"'{folder_id}' in parents",
                "trashed = false",
            ]
            
            if not include_subfolders:
                query_parts.append(f"mimeType = '{PDF_MIME_TYPE}'")
            else:
                query_parts.append(
                    f"(mimeType = '{PDF_MIME_TYPE}' or mimeType = '{FOLDER_MIME_TYPE}')"
                )
            
            query = " and ".join(query_parts)
            
            # List files with pagination
            page_token = None
            while True:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._service.files().list(
                        q=query,
                        pageSize=self.page_size,
                        pageToken=page_token,
                        fields="nextPageToken, files(id, name, mimeType, size, "
                               "createdTime, modifiedTime, parents, properties)",
                    ).execute()
                )
                
                current_files = response.get("files", [])
                
                # Process files
                for file in current_files:
                    if file["mimeType"] == PDF_MIME_TYPE:
                        files.append(file)
                    elif include_subfolders and file["mimeType"] == FOLDER_MIME_TYPE:
                        # Recursively list PDFs in subfolder
                        subfolder_files = await self._list_pdfs_in_folder(
                            file["id"],
                            include_subfolders=True,
                        )
                        files.extend(subfolder_files)
                
                page_token = response.get("nextPageToken")
                if not page_token:
                    break
            
            return files
            
        except HttpError as e:
            if e.resp.status == 429:
                retry_after = e.resp.headers.get("Retry-After", 60)
                raise DriveQuotaExceededError(retry_after=int(retry_after))
            logger.error(f"Failed to list files in folder {folder_id}: {e}")
            raise GoogleDriveError(f"Failed to list files: {str(e)}")

    @log_performance
    async def download_pdf(
        self,
        file_id: str,
        destination: Optional[Path] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Path:
        """
        Download a PDF file from Drive.

        Args:
            file_id: Google Drive file ID
            destination: Local path to save file (auto-generated if None)
            progress_callback: Optional callback for progress updates

        Returns:
            Path to downloaded file

        Raises:
            DriveFileNotFoundError: If file not found
            GoogleDriveError: For other download errors
        """
        self.ensure_connected()
        
        try:
            # Get file metadata
            file_metadata = await self.get_file_metadata(file_id)
            
            if not destination:
                # Create downloads directory
                downloads_dir = Path("downloads")
                downloads_dir.mkdir(exist_ok=True)
                destination = downloads_dir / file_metadata["name"]
            
            logger.info(f"Downloading {file_metadata['name']} to {destination}")
            
            # Download file
            request = self._service.files().get_media(fileId=file_id)
            
            with open(destination, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request, chunksize=1024*1024)
                
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if progress_callback and status:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            progress_callback,
                            status.progress() * 100,
                        )
            
            logger.info(f"Downloaded {destination.name} ({destination.stat().st_size} bytes)")
            return destination
            
        except HttpError as e:
            if e.resp.status == 404:
                raise DriveFileNotFoundError(file_id)
            logger.error(f"Failed to download file {file_id}: {e}")
            raise GoogleDriveError(f"Failed to download file: {str(e)}")

    @log_performance
    async def download_pdf_stream(
        self,
        file_id: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Tuple[io.BytesIO, Dict[str, Any]]:
        """
        Download PDF to memory stream (for large files).

        Args:
            file_id: Google Drive file ID
            progress_callback: Optional progress callback

        Returns:
            Tuple of (file stream, file metadata)

        Raises:
            DriveFileNotFoundError: If file not found
            GoogleDriveError: For other errors
        """
        self.ensure_connected()
        
        try:
            # Get file metadata
            file_metadata = await self.get_file_metadata(file_id)
            
            logger.info(f"Downloading {file_metadata['name']} to stream")
            
            # Download to memory
            request = self._service.files().get_media(fileId=file_id)
            
            file_stream = io.BytesIO()
            downloader = MediaIoBaseDownload(
                file_stream,
                request,
                chunksize=5*1024*1024,  # 5MB chunks for streaming
            )
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if progress_callback and status:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        progress_callback,
                        status.progress() * 100,
                    )
            
            file_stream.seek(0)
            logger.info(f"Downloaded {file_metadata['name']} to stream ({file_stream.tell()} bytes)")
            
            return file_stream, file_metadata
            
        except HttpError as e:
            if e.resp.status == 404:
                raise DriveFileNotFoundError(file_id)
            logger.error(f"Failed to download file {file_id}: {e}")
            raise GoogleDriveError(f"Failed to download file: {str(e)}")

    async def get_file_metadata(self, file_id: str) -> Dict[str, Any]:
        """
        Get file metadata from Drive.

        Args:
            file_id: Google Drive file ID

        Returns:
            File metadata dictionary

        Raises:
            DriveFileNotFoundError: If file not found
        """
        self.ensure_connected()
        
        try:
            metadata = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._service.files().get(
                    fileId=file_id,
                    fields="id, name, mimeType, size, createdTime, modifiedTime, "
                           "parents, properties, description",
                ).execute()
            )
            return metadata
            
        except HttpError as e:
            if e.resp.status == 404:
                raise DriveFileNotFoundError(file_id)
            raise GoogleDriveError(f"Failed to get file metadata: {str(e)}")

    async def update_file_metadata(
        self,
        file_id: str,
        properties: Dict[str, str],
        description: Optional[str] = None,
    ) -> None:
        """
        Update file metadata (properties and description).

        Args:
            file_id: Google Drive file ID
            properties: Key-value pairs to store as properties
            description: Optional file description
        """
        self.ensure_connected()
        
        try:
            body = {"properties": properties}
            if description is not None:
                body["description"] = description
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._service.files().update(
                    fileId=file_id,
                    body=body,
                ).execute()
            )
            
            logger.debug(f"Updated metadata for file {file_id}")
            
        except HttpError as e:
            logger.error(f"Failed to update file metadata: {e}")
            raise GoogleDriveError(f"Failed to update metadata: {str(e)}")

    async def mark_as_processed(
        self,
        file_id: str,
        status: ProcessingStatus,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Mark a PDF as processed in Drive metadata.

        Args:
            file_id: Google Drive file ID
            status: Processing status
            error_message: Optional error message if failed
        """
        properties = {
            "processed": "true" if status == ProcessingStatus.COMPLETED else "false",
            "processing_status": status.value,
            "processed_at": datetime.utcnow().isoformat(),
        }
        
        if error_message:
            properties["error_message"] = error_message[:500]  # Limit length
        
        await self.update_file_metadata(file_id, properties)
        logger.info(f"Marked file {file_id} as {status.value}")

    async def get_unprocessed_pdfs(
        self,
        folder_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get list of unprocessed PDFs.

        Args:
            folder_id: Specific folder ID (None for all configured)

        Returns:
            List of unprocessed PDF metadata
        """
        all_pdfs = await self.list_pdfs(folder_id, include_subfolders=True)
        
        # Filter unprocessed
        unprocessed = [
            pdf for pdf in all_pdfs
            if pdf.get("properties", {}).get("processed") != "true"
        ]
        
        logger.info(f"Found {len(unprocessed)} unprocessed PDFs")
        return unprocessed


def create_drive_client() -> GoogleDriveClient:
    """Create a Google Drive client with default settings."""
    from src.google_drive.auth import create_auth_manager
    
    auth_manager = create_auth_manager()
    return GoogleDriveClient(auth_manager)