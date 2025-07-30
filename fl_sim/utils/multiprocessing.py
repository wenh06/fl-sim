"""
fl_sim.utils.multiprocessing
----------------------------
Simplified multiprocessing module with reliable output display.
"""
import os
import sys
import time
import threading
import re
import signal
import atexit
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
from datetime import datetime
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from torch_ecg.cfg import CFG
from ..cli import single_run
from .output_redirector import generate_output_redirector, release_output

__all__ = [
    "Task",
    "Worker",
    "ProcessStatus",
    "OutputManager",
    "MultiprocessManager",
    "run_parallel_tasks"
]

def report_progress(n_iter=None, num_iters=None,
                    current_client_progress=None, selected_clients_count=None,
                    training_phase='idle') -> None:
    """Report the progress of the server."""
    if hasattr(sys.stdout, '_send'):
        # format the progress data
        progress_data = {
            'current_iter': n_iter if n_iter is not None else 0,
            'total_iters': num_iters if num_iters is not None else 0,
            'current_clients': current_client_progress if current_client_progress is not None else 0,
            'total_clients': selected_clients_count if selected_clients_count is not None else 0,
            'phase': training_phase,  # 'idle', 'training', 'evaluating', 'updating'
        }
        
        # DEBUG
        # from .output_redirector import OutputRedirector
        # assert isinstance(sys.stdout, OutputRedirector)
        # print(f"DEBUG: Sending progress update")
        
        # send the progress data to the main process through the stdout redirector
        sys.stdout._send('progress', str(progress_data), 'INFO')

class ProcessStatus:
    """Track and store the status of a federated learning process."""
    
    def __init__(self, pid: int, task_id: int, task_tag: str):
        self.pid = pid
        self.task_id = task_id
        self.task_tag = task_tag
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_clients = 0
        self.total_clients = 0
        self.status = "Running"
        self.last_update = time.time()
        self.start_time = time.time()
        self.end_time = None
        self.message_count = 0
    
    def update(self, message: Dict[str, Any]):
        """Update process status based on received message."""
        content = message.get('content', '')
        self.message_count += 1
        
        # update progress if message from report_progress
        if message.get('type', '') == 'progress':
            # DEBUG info
            # print(f"DEBUG: Received progress update from PID {self.pid}: {content}")
            progress_data = eval(content)
            self.current_epoch = progress_data.get('current_iter', self.current_epoch)
            self.total_epochs = progress_data.get('total_iters', self.total_epochs)
            self.current_clients = progress_data.get('current_clients', self.current_clients)
            self.total_clients = progress_data.get('total_clients', self.total_clients)
        
        self.last_update = time.time()
    
    def mark_completed(self):
        """Mark process as completed."""
        self.status = "Completed"
        self.end_time = time.time()
    
    def get_status_line(self) -> str:
        """Get a single line status string."""
        # Calculate elapsed time
        elapsed = int(time.time() - self.start_time)
        time_str = f"{elapsed//60:02d}:{elapsed%60:02d}"
        
        # Build status components
        epoch_str = f"{self.current_epoch}/{self.total_epochs}" if self.total_epochs > 0 else "-/-"
        client_str = f"{self.current_clients}/{self.total_clients}" if self.total_clients > 0 else "-/-"
        
        # Format: [Tag] PID:xxxx Epoch:x/x Clients:x/x Time:xx:xx Status:xxxxx
        return (f"[{self.task_tag:<15}] PID:{self.pid:<6} "
                f"Epoch:{epoch_str:<7} Clients:{client_str:<7} "
                f"Time:{time_str} Status:{self.status}")


class OutputManager:
    """
    Simplified output manager that prints logs and shows status intelligently.
    """
    
    def __init__(self, status_interval: float = 2.0, min_log_lines: int = 10):
        self.output_queue = Queue()
        self.process_statuses: Dict[int, ProcessStatus] = {}
        self.all_logs: List[Tuple[datetime, str]] = []
        self.status_interval = status_interval
        self.min_log_lines = min_log_lines  # Minimum log lines between status prints
        self.running = False
        self.display_thread = None
        self.lock = threading.Lock()
        self.last_status_print = 0
        self.logs_since_status = 0  # Counter for logs since last status print
        self.show_status = True
        self.total_tasks = 0
        
    def register_process(self, pid: int, task_id: int, task_tag: str) -> None:
        """Register a new process to be monitored."""
        with self.lock:
            self.process_statuses[pid] = ProcessStatus(pid, task_id, task_tag)
    
    def unregister_process(self, pid: int) -> None:
        """Unregister a process from monitoring."""
        with self.lock:
            if pid in self.process_statuses:
                self.process_statuses[pid].mark_completed()
    
    def process_message(self, message: Dict[str, Any]) -> None:
        """Process a message from the queue."""
        pid = message.get('pid')
        msg_type = message.get('type')
        content = message.get('content', '')
        
        # Update process status
        with self.lock:
            if pid in self.process_statuses:
                self.process_statuses[pid].update(message)
        
        # Handle log messages
        if msg_type in ['stdout', 'stderr', 'warning', 'exception']:
            timestamp = datetime.now()
            
            # Format log line
            time_str = timestamp.strftime('%H:%M:%S')
            log_line = f"[{time_str}][P{pid}] {content.strip()}"
            
            with self.lock:
                self.all_logs.append((timestamp, f"[P{pid}] {content.strip()}"))
                self.logs_since_status += 1  # Increment log counter
            
            # Print log immediately
            if self.running:
                # Color coding for different message types
                if msg_type == 'stderr' or msg_type == 'exception':
                    print(f"\033[91m{log_line}\033[0m")  # Red for errors
                elif msg_type == 'warning':
                    print(f"\033[93m{log_line}\033[0m")  # Yellow for warnings
                else:
                    print(log_line)  # Normal for stdout
    
    def _should_print_status(self, current_time: float) -> bool:
        """Determine if status should be printed based on time and log count."""
        # Check if enough time has passed
        time_elapsed = (current_time - self.last_status_print) >= self.status_interval
        
        # Check if enough logs have been printed
        enough_logs = self.logs_since_status >= self.min_log_lines
        
        # Print status if both conditions are met
        # OR if a very long time has passed (5x the normal interval) regardless of log count
        force_print = (current_time - self.last_status_print) >= (self.status_interval * 5)
        
        return time_elapsed and (enough_logs or force_print)
    
    def _print_status_summary(self):
        """Print a status summary."""
        with self.lock:
            if not self.process_statuses:
                return
            
            # Reset log counter
            self.logs_since_status = 0
            
            # Print status header
            print("\n" + "="*80)
            print("PROCESS STATUS UPDATE")
            print("-"*80)
            
            # Print each process status
            for status in self.process_statuses.values():
                print(status.get_status_line())
            
            # Summary statistics
            running = sum(1 for s in self.process_statuses.values() if s.status == "Running")
            completed = sum(1 for s in self.process_statuses.values() if s.status == "Completed")
            total_messages = sum(s.message_count for s in self.process_statuses.values())
            
            print("-"*80)
            print(f"Running: {running} | Task Progress: {completed}/{self.total_tasks} | Total Messages: {total_messages}")
            print("="*80 + "\n")
    
    def _display_loop(self):
        """Main display loop - process messages and show status intelligently."""
        while self.running:
            # Process all pending messages
            messages_processed = 0
            while not self.output_queue.empty() and messages_processed < 50:
                try:
                    message = self.output_queue.get_nowait()
                    self.process_message(message)
                    messages_processed += 1
                except:
                    break
            
            # Check if we should print status
            current_time = time.time()
            if self.show_status and self._should_print_status(current_time):
                self._print_status_summary()
                self.last_status_print = current_time
            
            time.sleep(0.05)  # Small sleep to prevent CPU spinning
    
    def start(self) -> None:
        """Start the output manager."""
        if not self.running:
            self.running = True
            
            # Print header
            print("\n" + "="*80)
            print("MULTIPROCESS FEDERATED LEARNING EXECUTION")
            print("="*80)
            print("Starting execution... Press Ctrl+C to stop")
            print(f"Status updates: every {self.status_interval}s (if >{self.min_log_lines} logs)")
            print("-"*80 + "\n")
            
            # Start display thread
            self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
            self.display_thread.start()
    
    def stop(self) -> None:
        """Stop the output manager and show final summary."""
        self.running = False
        self.show_status = False  # Stop periodic status updates
        
        # Wait for display thread
        if self.display_thread:
            self.display_thread.join(timeout=1.0)
        
        # Print final summary
        self._print_final_summary()
    
    def _print_final_summary(self):
        """Print final execution summary."""
        print("\n" + "="*80)
        print("EXECUTION COMPLETED - FINAL SUMMARY")
        print("="*80)
        
        with self.lock:
            # Process summary
            if self.process_statuses:
                print("\nProcess Details:")
                print("-"*80)
                for status in self.process_statuses.values():
                    duration = int((status.end_time or time.time()) - status.start_time)
                    print(f"  {status.task_tag}:")
                    print(f"    PID: {status.pid}")
                    print(f"    Final Progress: Epoch {status.current_epoch}/{status.total_epochs}")
                    print(f"    Duration: {duration//60:02d}:{duration%60:02d}")
                    print(f"    Messages: {status.message_count}")
                    print(f"    Status: {status.status}")
            else:
                print("No processes were tracked")
            
            # Log summary
            print(f"\nTotal logs collected: {len(self.all_logs)}")
            
            # Option to save logs
            print("\nAll logs have been displayed above and are available in terminal history.")
            
        print("="*80 + "\n")

class Task:
    """Task class for multiprocessing."""
    
    def __init__(self, task_id: int, config: CFG, task_tag: str):
        self.task_id = task_id
        self.config = config
        self.task_tag = task_tag
    
    def run(self):
        """Run the task."""
        single_run(self.config, is_subprocess=True)
        
def _worker_process(task: Task, output_queue: Queue):
    """Worker function to run a task in a subprocess."""
    redirector = None
    if output_queue:
        # mute all tqdm
        import os
        os.environ['TQDM_DISABLE'] = '1'
            
        # redirect output to queue and add task_id to prefix
        redirector = generate_output_redirector(output_queue, f"Task-{task.task_id}")
        
        # monkey patch tqdm
        try:
            import tqdm
            import functools
            
            # save original tqdm init
            _original_tqdm_init = tqdm.tqdm.__init__
            
            # create a disabled tqdm init
            @functools.wraps(_original_tqdm_init)
            def disabled_tqdm_init(self, *args, **kwargs):
                kwargs['disable'] = True  # set disable to True
                _original_tqdm_init(self, *args, **kwargs)
            
            # replace tqdm init
            tqdm.tqdm.__init__ = disabled_tqdm_init
            
            # tqdm.auto
            if hasattr(tqdm, 'auto') and hasattr(tqdm.auto, 'tqdm'):
                tqdm.auto.tqdm.__init__ = disabled_tqdm_init
            
            # DEBUG info
            # print("DEBUG: tqdm has been disabled via monkey patching")
            
        except ImportError:
            print("tqdm not installed")
        except Exception as e:
            print(f"Error disabling tqdm: {e}")
        
        # DEBUG: verify warnings are being redirected
        # print(f"warnings.showwarning type: {type(warnings.showwarning)}")
        # print(f"Is custom showwarning: {warnings.showwarning.__name__ == 'custom_showwarning'}")
    try:
        task.run()
    finally:
        if redirector:
            release_output(redirector)

class Worker:
    """A worker process."""
    
    def __init__(self, worker_id: int, output_queue: Queue = None):
        self.worker_id = worker_id
        self.process = None
        self.task = None
        self.output_queue = output_queue
        self.pid = None
        self.ctx = mp.get_context('spawn') # Get the spawn context
    
    def assign_task(self, task: Task):
        """Assign a task to the worker."""
        self.task = task
    
    def start(self):
        """Start the worker process."""
        if self.task is None:
            raise ValueError("No task assigned to worker")
                    
        # Use spawn context to create subprocess
        self.process = self.ctx.Process(target=_worker_process, args=(self.task, self.output_queue))
        self.process.start()
        self.pid = self.process.pid
        return self.pid
    
    def is_alive(self):
        """Check if the worker process is alive."""
        return self.process is not None and self.process.is_alive()
    
    def terminate(self):
        """Terminate the worker process."""
        if self.process:
            self.process.terminate()
            self.process.join(timeout=5.0)
            self.process = None


class MultiprocessManager:
    """Manager for multiple federated learning processes."""
    
    def __init__(self, num_workers: int = None, status_interval: float = 2.0, 
                 min_log_lines: int = 10):
        # Set spawn method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # if is already set, do nothing
            pass
        
        self.num_workers = num_workers or min(os.cpu_count(), 4)
        self.output_manager = OutputManager(
            status_interval=status_interval,
            min_log_lines=min_log_lines
        )
        self.workers: List[Worker] = []
        self.tasks: List[Task] = []
        self.completed_tasks = 0
    
    def add_task(self, config: CFG, task_tag: str = None) -> int:
        """Add a task to be executed."""
        task_id = len(self.tasks)
        task = Task(task_id, config, task_tag or f"Task-{task_id}")
        self.tasks.append(task)
        return task_id
    
    def start(self) -> None:
        """Start executing all tasks."""
        if not self.tasks:
            print("No tasks to execute")
            return
        
        try:
            # Pass the total number of tasks to the output manager.
            self.output_manager.total_tasks = len(self.tasks)
            
            # Start output manager
            self.output_manager.start()
            
            # Create worker pool
            for i in range(min(self.num_workers, len(self.tasks))):
                worker = Worker(i, self.output_manager.output_queue)
                self.workers.append(worker)
            
            # Initial task assignment
            task_index = 0
            for worker in self.workers:
                if task_index < len(self.tasks):
                    task = self.tasks[task_index]
                    worker.assign_task(task)
                    pid = worker.start()
                    self.output_manager.register_process(pid, task.task_id, task.task_tag)
                    task_index += 1
            
            # Monitor and reassign tasks
            while task_index < len(self.tasks) or any(w.is_alive() for w in self.workers):
                for worker in self.workers:
                    if not worker.is_alive():
                        # Mark previous task as completed
                        if worker.pid:
                            self.output_manager.unregister_process(worker.pid)
                            self.completed_tasks += 1
                            worker.pid = None
                        
                        # Assign new task if available
                        if task_index < len(self.tasks):
                            task = self.tasks[task_index]
                            worker.assign_task(task)
                            pid = worker.start()
                            self.output_manager.register_process(pid, task.task_id, task.task_tag)
                            task_index += 1
                
                time.sleep(0.5)
            
            # Final cleanup - mark remaining tasks as completed
            for worker in self.workers:
                if worker.pid:
                    self.output_manager.unregister_process(worker.pid)
                    self.completed_tasks += 1
                    
        except KeyboardInterrupt:
            print("\n\nInterrupted! Stopping all processes...")
            self._stop_all_workers()
        except Exception as e:
            print(f"\nError during execution: {e}")
            self._stop_all_workers()
            raise
        finally:
            self.output_manager.stop()
            print(f"\nCompleted {self.completed_tasks}/{len(self.tasks)} tasks")
    
    def _stop_all_workers(self):
        """Stop all running workers."""
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()


def run_parallel_tasks(configs: List[CFG], num_workers: int = None, 
                      task_tags: List[str] = None, status_interval: float = 2.0,
                      min_log_lines: int = 10) -> None:
    """
    Run multiple tasks in parallel with output management.
    
    Parameters
    ----------
    configs : List[CFG]
        List of configurations for tasks to run
    num_workers : int, optional
        Number of worker processes to use
    task_tags : List[str], optional
        Custom tags for each task
    status_interval : float, optional
        Interval in seconds between status updates (default: 2.0)
    min_log_lines : int, optional
        Minimum number of log lines between status updates (default: 10)
    """
    manager = MultiprocessManager(num_workers, status_interval, min_log_lines)
    
    # Add all tasks
    for i, config in enumerate(configs):
        tag = task_tags[i] if task_tags and i < len(task_tags) else f"Config-{i}"
        manager.add_task(config, tag)
    
    # Start execution
    manager.start()