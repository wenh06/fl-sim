"""
.. _output_redirector:

fl_sim.utils.output_redirector
-------------------------

This module provides a process output redirecting mechanism with core features:
1. Capture all standard output/error and append process information
2. Automatically handle warnings and exceptions
3. Maintain maximum compatibility

"""
import sys
import os
import time
import warnings
from types import TracebackType
from typing import Optional, Type, List, Dict, Any
from multiprocessing import Queue
from collections import deque

__all__ = [
    "OutputRedirector",
    "generate_output_redirector",
    "release_output",
]

class OutputRedirector:
    """
    Output redirector for multiprocessing.
    
    This class packages standard output/error and warning into a message format, 
    and redirects them to a target message queue.
    A buffer is used to store messages, and flushes them to the queue periodically.
    """
    def __init__(self, queue: Queue, pid: int, task_tag: str, buffer_size: int = 100,
                echo_to_original: bool = False):
        """
        Initialize the redirector.
        
        Parameters:
        - queue: Target message queue for output redirection.
        - pid: Process ID for message tagging.
        - task_tag: Tag for task identification.
        - buffer_size: Maximum number of messages to buffer before flushing.
        - echo_to_original: Whether to echo messages to the original stream (default: False).
        """
        self.queue = queue # target message queue
        self.pid = pid
        self.task_tag = task_tag
        self.orig_stdout = sys.stdout
        self.orig_stderr = sys.stderr
        self.orig_warning = warnings.showwarning
        self.buffer = deque(maxlen=buffer_size)
        self.last_flush_time = time.time()
        self.flush_interval = 0.5  # seconds
        self.echo_to_original = echo_to_original

    def _send(self, msg_type: str, content: str, level: str):
        """Unified message sending with buffer mechanism"""
        message = {
            'type': msg_type,
            'level': level,
            'content': content,
            'pid': self.pid,
            'task_tag': self.task_tag,
            'timestamp': time.time()
        }
        
        # DEBUG
        # if msg_type == 'progress':
        #     print(f"DEBUG: Sending progress update to PID {self.pid}: {content}")
        
        # Add to buffer
        self.buffer.append(message)
        
        # Flush buffer if it's time or buffer is full
        current_time = time.time()
        if current_time - self.last_flush_time > self.flush_interval or len(self.buffer) >= self.buffer.maxlen:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush buffered messages to queue"""
        while self.buffer:
            self.queue.put(self.buffer.popleft())
        self.last_flush_time = time.time()

    def write(self, data: str):
        """Core write method redirecting"""
        if data.strip():
            # Determine current stream type
            if sys.stdout is self:
                self._send('stdout', data, 'INFO')
            elif sys.stderr is self:
                self._send('stderr', data, 'WARNING')
        
        # Maintain original stream writing if enabled
        if self.echo_to_original:
            if sys.stdout is self:
                self.orig_stdout.write(data)
            elif sys.stderr is self:
                self.orig_stderr.write(data)

    def flush(self):
        """Maintain compatibility"""
        self._flush_buffer()
        self.orig_stdout.flush()
        self.orig_stderr.flush()

    def __getattr__(self, name):
        """Transparent forwarding of other attributes"""
        return getattr(self.orig_stdout, name)

def generate_output_redirector(queue: Queue, task_tag: str) -> OutputRedirector:
    """Factory function to start output redirecting"""
    redirector = OutputRedirector(
        queue=queue,
        pid=os.getpid(),
        task_tag=task_tag
    )
    
    # Set standard output/error
    sys.stdout = redirector
    sys.stderr = redirector
    
    # Handle warnings
    def custom_showwarning(message, category, filename, lineno, file=None, line=None):
        # Format the warning message
        warning_msg = warnings.formatwarning(message, category, filename, lineno, line)
        redirector._send('warning', warning_msg, 'WARNING')
        
        # If necessary, you can also call the original showwarning (but this may cause duplicate output)
        # redirector.orig_warning(message, category, filename, lineno, file, line)
    
    warnings.showwarning = custom_showwarning
    
    # In addition, ensure that warnings are not filtered out
    # warnings.simplefilter('always')  # Show all warnings
    
    # Handle uncaught exceptions
    def handle_exception(exc_type: Type[BaseException], 
                        exc_value: BaseException, 
                        traceback: Optional[TracebackType]):
        if traceback:
            tb_text = ''.join(
                traceback.format_exception(exc_type, exc_value, traceback))
            redirector._send('exception', tb_text, 'ERROR')
        sys.__excepthook__(exc_type, exc_value, traceback)
    sys.excepthook = handle_exception
    
    return redirector

def release_output(redirector: OutputRedirector):
    """Release redirecting"""
    # Flush any remaining buffered messages
    redirector._flush_buffer()
    sys.stdout = redirector.orig_stdout
    sys.stderr = redirector.orig_stderr
    warnings.showwarning = redirector.orig_warning
    sys.excepthook = sys.__excepthook__