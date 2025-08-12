# Multiprocessing Output Management

This module provides a comprehensive solution for managing multiple federated learning processes and their outputs. It includes functionality for:

1. Managing multiple federated learning processes
2. Collecting and displaying output from all processes
3. Tracking and visualizing process status

## Key Features

- **Split-screen display**: Upper section shows process logs with process tags, lower section shows process status
- **Adaptive layout**: Automatically adjusts to terminal window size
- **Real-time updates**: Continuously updates process status and logs
- **Process tagging**: Each log line is tagged with its process ID [p:xxxx]
- **Progress tracking**: Automatically extracts and displays epoch and client progress

## Usage

### Running Multiple Tasks in Parallel

To run multiple tasks in parallel with output management, use the `run_parallel_tasks` function:

```python
from fl_sim.utils.multiprocessing import run_parallel_tasks
from torch_ecg.utils import CFG

# Create a list of task configurations
configs = [CFG(...), CFG(...), ...]

# Run tasks in parallel with output management
run_parallel_tasks(configs, num_workers=4)
```

### Using the Command Line Interface

The command line interface supports parallel execution with output management. Create a YAML configuration file with a parallel section:

```yaml
strategy:
  matrix:
    algorithm:
    - FedAvg
    - FedProx
    clients_sample_ratio:
    - 0.3
    - 0.7
  parallel:
    mode: parallel_task
    num_workers: 2

# ... rest of configuration
```

Then run:

```bash
fl-sim path/to/config.yml
```

### Custom Integration

For more advanced usage, you can directly use the `MultiprocessManager` class:

```python
from fl_sim.utils.multiprocessing import MultiprocessManager

# Create a manager
manager = MultiprocessManager(num_workers=4)

# Add tasks
for config in configs:
    manager.add_task(config, task_tag="Custom Task")

# Start execution with output management
manager.start()
```

## Display Format

The display is divided into two sections:

1. **Process Logs**: Shows the most recent log messages from all processes, each tagged with its process ID [p:xxxx]
2. **Process Status**: Shows a grid of process status information, including:
   - Process ID (pid)
   - Current epoch / total epochs
   - Current clients / total clients

The display automatically adjusts based on terminal size, ensuring optimal visibility on different screen sizes.
