"""
Google Drive OAuth2 authentication module.

This module handles OAuth2 authentication flow for Google Drive API access,
including token management and refresh.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from src.config import get_settings
from src.utils.errors import DriveAuthenticationError
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Google Drive API scopes
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
    "https://www.googleapis.com/auth/drive.file",  # For writing metadata
]


class GoogleDriveAuth:
    """Handle Google Drive OAuth2 authentication."""

    def __init__(
        self,
        credentials_path: Optional[Path] = None,
        token_path: Optional[Path] = None,
        scopes: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize Google Drive authentication.

        Args:
            credentials_path: Path to OAuth2 credentials JSON file
            token_path: Path to store/load token
            scopes: OAuth2 scopes (defaults to SCOPES)
        """
        self.settings = get_settings()
        self.credentials_path = credentials_path or self.settings.drive_credentials_path
        self.token_path = token_path or Path("token.json")
        self.scopes = scopes or SCOPES
        
        self._credentials: Optional[Credentials] = None

    @property
    def credentials(self) -> Optional[Credentials]:
        """Get current credentials."""
        return self._credentials

    @property
    def is_authenticated(self) -> bool:
        """Check if authenticated with valid credentials."""
        return self._credentials is not None and self._credentials.valid

    async def authenticate(self, force_reauth: bool = False) -> Credentials:
        """
        Authenticate with Google Drive.

        Args:
            force_reauth: Force re-authentication even if token exists

        Returns:
            Valid credentials

        Raises:
            DriveAuthenticationError: If authentication fails
        """
        try:
            # Try to load existing token
            if not force_reauth and self.token_path.exists():
                self._credentials = self._load_token()
                
                # Refresh if expired
                if self._credentials and self._credentials.expired and self._credentials.refresh_token:
                    logger.info("Refreshing expired token")
                    self._credentials.refresh(Request())
                    self._save_token()
            
            # If no valid credentials, run OAuth flow
            if not self._credentials or not self._credentials.valid or force_reauth:
                logger.info("Running OAuth2 flow")
                self._credentials = await self._run_oauth_flow()
                self._save_token()
            
            logger.info("Successfully authenticated with Google Drive")
            return self._credentials
            
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise DriveAuthenticationError(f"Failed to authenticate: {str(e)}")

    async def _run_oauth_flow(self) -> Credentials:
        """
        Run OAuth2 flow to get credentials.

        Returns:
            Valid credentials

        Raises:
            DriveAuthenticationError: If OAuth flow fails
        """
        if not self.credentials_path.exists():
            raise DriveAuthenticationError(
                f"Credentials file not found: {self.credentials_path}. "
                "Please download OAuth2 credentials from Google Cloud Console."
            )
        
        try:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(self.credentials_path),
                self.scopes,
            )
            
            # Run local server for OAuth callback
            # Note: In production, you might want to use a different method
            credentials = flow.run_local_server(
                port=0,  # Use any available port
                authorization_prompt_message="Opening browser for Google Drive authentication...",
                success_message="Authentication successful! You can close this window.",
                open_browser=True,
            )
            
            return credentials
            
        except Exception as e:
            raise DriveAuthenticationError(f"OAuth flow failed: {str(e)}")

    def _load_token(self) -> Optional[Credentials]:
        """
        Load token from file.

        Returns:
            Credentials if valid token exists, None otherwise
        """
        try:
            if self.token_path.exists():
                logger.debug(f"Loading token from {self.token_path}")
                return Credentials.from_authorized_user_file(
                    str(self.token_path),
                    self.scopes,
                )
        except Exception as e:
            logger.warning(f"Failed to load token: {e}")
        
        return None

    def _save_token(self) -> None:
        """Save current credentials to token file."""
        if self._credentials:
            logger.debug(f"Saving token to {self.token_path}")
            
            # Ensure directory exists
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save token
            with open(self.token_path, "w") as token_file:
                token_file.write(self._credentials.to_json())

    def revoke(self) -> None:
        """
        Revoke current credentials and delete token.

        Note: This doesn't revoke the token on Google's servers,
        just removes local credentials.
        """
        logger.info("Revoking credentials")
        
        # Clear credentials
        self._credentials = None
        
        # Delete token file
        if self.token_path.exists():
            self.token_path.unlink()
            logger.info(f"Deleted token file: {self.token_path}")

    async def get_user_info(self) -> Dict[str, Any]:
        """
        Get authenticated user information.

        Returns:
            User info dictionary

        Raises:
            DriveAuthenticationError: If not authenticated
        """
        if not self.is_authenticated:
            raise DriveAuthenticationError("Not authenticated")
        
        try:
            # Use Google's API to get user info
            from googleapiclient.discovery import build
            
            service = build("oauth2", "v2", credentials=self._credentials)
            user_info = service.userinfo().get().execute()
            
            logger.info(f"Retrieved user info for: {user_info.get('email', 'Unknown')}")
            return user_info
            
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
            raise DriveAuthenticationError(f"Failed to get user info: {str(e)}")


class ServiceAccountAuth:
    """
    Handle Google Drive service account authentication.
    
    Use this for server-to-server authentication without user interaction.
    """

    def __init__(
        self,
        service_account_path: Path,
        scopes: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize service account authentication.

        Args:
            service_account_path: Path to service account JSON key file
            scopes: OAuth2 scopes
        """
        self.service_account_path = service_account_path
        self.scopes = scopes or SCOPES
        
        if not self.service_account_path.exists():
            raise DriveAuthenticationError(
                f"Service account file not found: {service_account_path}"
            )

    async def authenticate(self) -> Credentials:
        """
        Authenticate using service account.

        Returns:
            Valid credentials

        Raises:
            DriveAuthenticationError: If authentication fails
        """
        try:
            from google.oauth2 import service_account
            
            credentials = service_account.Credentials.from_service_account_file(
                str(self.service_account_path),
                scopes=self.scopes,
            )
            
            logger.info("Successfully authenticated with service account")
            return credentials
            
        except Exception as e:
            logger.error(f"Service account authentication failed: {e}")
            raise DriveAuthenticationError(
                f"Service account authentication failed: {str(e)}"
            )


def create_auth_manager(use_service_account: bool = False) -> GoogleDriveAuth:
    """
    Create appropriate auth manager based on configuration.

    Args:
        use_service_account: Whether to use service account auth

    Returns:
        Auth manager instance
    """
    settings = get_settings()
    
    if use_service_account:
        # Look for service account file
        service_account_path = Path("service_account.json")
        if service_account_path.exists():
            return ServiceAccountAuth(service_account_path)
        else:
            logger.warning("Service account requested but file not found, falling back to OAuth")
    
    return GoogleDriveAuth(
        credentials_path=settings.drive_credentials_path,
        token_path=Path("token.json"),
    )