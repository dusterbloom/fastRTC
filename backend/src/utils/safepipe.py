"""
SafePipe Implementation for Inter-Process Communication

Based on RealtimeSTT's SafePipe implementation for reliable
communication between main process and transcription worker process.
"""

import multiprocessing as mp
import logging
from typing import Any, Tuple

logger = logging.getLogger(__name__)


class SafePipe:
    """
    Safe wrapper around multiprocessing.Pipe() that handles errors gracefully.
    
    This implementation mirrors RealtimeSTT's SafePipe to ensure reliable
    communication between the main process and transcription worker process.
    """
    
    def __init__(self, duplex: bool = True):
        """
        Initialize SafePipe with duplex communication.
        
        Args:
            duplex: Whether the pipe should be duplex (bidirectional)
        """
        try:
            self.parent_conn, self.child_conn = mp.Pipe(duplex=duplex)
            self._closed = False
            logger.debug("SafePipe initialized successfully")
        except Exception as e:
            logger.error(f"Failed to create SafePipe: {e}")
            raise
    
    def __call__(self) -> Tuple[Any, Any]:
        """
        Return the pipe connections for compatibility with RealtimeSTT pattern.
        
        Returns:
            Tuple of (parent_connection, child_connection)
        """
        return self.parent_conn, self.child_conn
    
    def send(self, obj: Any) -> bool:
        """
        Safely send object through parent connection.
        
        Args:
            obj: Object to send
            
        Returns:
            True if successful, False otherwise
        """
        if self._closed:
            logger.warning("Attempted to send on closed SafePipe")
            return False
            
        try:
            self.parent_conn.send(obj)
            return True
        except (BrokenPipeError, EOFError, OSError) as e:
            logger.warning(f"SafePipe send failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected SafePipe send error: {e}")
            return False
    
    def recv(self, timeout: float = None) -> Any:
        """
        Safely receive object from parent connection.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Received object or None if failed
        """
        if self._closed:
            logger.warning("Attempted to recv on closed SafePipe")
            return None
            
        try:
            if timeout is not None:
                if self.parent_conn.poll(timeout):
                    return self.parent_conn.recv()
                else:
                    return None
            else:
                return self.parent_conn.recv()
        except (BrokenPipeError, EOFError, OSError) as e:
            logger.warning(f"SafePipe recv failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected SafePipe recv error: {e}")
            return None
    
    def poll(self, timeout: float = 0) -> bool:
        """
        Check if data is available to read.
        
        Args:
            timeout: Timeout in seconds (0 for non-blocking)
            
        Returns:
            True if data is available, False otherwise
        """
        if self._closed:
            return False
            
        try:
            return self.parent_conn.poll(timeout)
        except (BrokenPipeError, EOFError, OSError) as e:
            logger.warning(f"SafePipe poll failed: {e}")
            self._closed = True
            return False
        except Exception as e:
            logger.warning(f"Unexpected SafePipe poll error: {e}")
            return False
    
    def close(self):
        """Close both connections safely."""
        if not self._closed:
            try:
                self.parent_conn.close()
                self.child_conn.close()
                self._closed = True
                logger.debug("SafePipe closed successfully")
            except Exception as e:
                logger.warning(f"Error closing SafePipe: {e}")
    
    def __del__(self):
        """Ensure connections are closed on deletion."""
        self.close()


def SafePipeFactory() -> Tuple[Any, Any]:
    """
    Factory function that returns SafePipe connections.
    
    This matches the RealtimeSTT SafePipe() function signature.
    
    Returns:
        Tuple of (parent_connection, child_connection)
    """
    pipe = SafePipe()
    return pipe()