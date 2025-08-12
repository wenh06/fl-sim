"""
Test module for multiprocessing functionality
"""

import io
import sys
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch  # noqa: F401

from torch_ecg.cfg import CFG

from fl_sim.utils.multiprocessing import (  # noqa: F401
    MultiprocessManager,
    OutputManager,
    ProcessStatus,
    Task,
    Worker,
    run_parallel_tasks,
)


class TestProcessStatus(unittest.TestCase):
    """Test process status tracking."""

    def test_basic_functionality(self):
        """Test basic status functionality."""
        status = ProcessStatus(1001, 0, "TestTask")

        # Test initialization
        self.assertEqual(status.pid, 1001)
        self.assertEqual(status.task_tag, "TestTask")
        self.assertEqual(status.status, "Running")

        # Test message parsing
        status.update({"content": "Training epoch 3/10 with clients 2/5"})
        self.assertEqual(status.current_epoch, 3)
        self.assertEqual(status.current_clients, 2)
        self.assertEqual(status.total_epochs, 10)
        self.assertEqual(status.total_clients, 5)

        # Test completion
        status.mark_completed()
        self.assertEqual(status.status, "Completed")
        self.assertIsNotNone(status.end_time)

    def test_status_line_format(self):
        """Test status line formatting."""
        status = ProcessStatus(2001, 0, "LongTaskNameTest")
        status.current_epoch = 5
        status.total_epochs = 10
        status.current_clients = 3
        status.total_clients = 8

        line = status.get_status_line()
        self.assertIn("PID:2001", line)
        self.assertIn("5/10", line)
        self.assertIn("3/8", line)
        self.assertIn("Running", line)


class TestOutputManager(unittest.TestCase):
    """Test output manager functionality."""

    def test_process_management(self):
        """Test process registration and tracking."""
        manager = OutputManager()

        # Register processes
        manager.register_process(3001, 0, "Task1")
        manager.register_process(3002, 1, "Task2")

        self.assertEqual(len(manager.process_statuses), 2)
        self.assertIn(3001, manager.process_statuses)

        # Unregister process
        manager.unregister_process(3001)
        self.assertEqual(manager.process_statuses[3001].status, "Completed")

    def test_message_processing(self):
        """Test message processing and logging."""
        manager = OutputManager()
        manager.register_process(4001, 0, "TestTask")

        # Process messages
        messages = [
            {"pid": 4001, "type": "stdout", "content": "Starting training"},
            {"pid": 4001, "type": "stdout", "content": "epoch: 1/5"},
            {"pid": 4001, "type": "stderr", "content": "Warning: slow"},
        ]

        for msg in messages:
            manager.process_message(msg)

        # Check results
        self.assertEqual(len(manager.all_logs), 3)
        self.assertEqual(manager.logs_since_status, 3)

        status = manager.process_statuses[4001]
        self.assertEqual(status.current_epoch, 1)
        self.assertEqual(status.total_epochs, 5)

    def test_status_print_logic(self):
        """Test intelligent status printing logic."""
        manager = OutputManager(status_interval=2.0, min_log_lines=5)

        # Initial state
        current_time = time.time()
        manager.last_status_print = current_time - 1  # 1 second ago

        # Not enough time passed
        self.assertFalse(manager._should_print_status(current_time))

        # Enough time but not enough logs
        manager.logs_since_status = 3
        self.assertFalse(manager._should_print_status(current_time + 2))

        # Both conditions met
        manager.logs_since_status = 6
        self.assertTrue(manager._should_print_status(current_time + 2))

        # Force print after long time
        self.assertTrue(manager._should_print_status(current_time + 15))


class TestMultiprocessManager(unittest.TestCase):
    """Test multiprocess manager."""

    def test_task_addition(self):
        """Test adding tasks."""
        manager = MultiprocessManager(num_workers=2)

        # Add tasks
        for i in range(3):
            config = CFG({"epochs": i + 1})
            task_id = manager.add_task(config, f"Task-{i}")
            self.assertEqual(task_id, i)

        self.assertEqual(len(manager.tasks), 3)
        self.assertEqual(manager.tasks[0].task_tag, "Task-0")

    def test_empty_task_list(self):
        """Test handling empty task list."""
        manager = MultiprocessManager()

        output = io.StringIO()
        with redirect_stdout(output):
            manager.start()

        self.assertIn("No tasks to execute", output.getvalue())


def run_basic_demo():
    """Run a basic demonstration of the system."""
    print("\n" + "=" * 60)
    print("BASIC DEMO: Output Manager")
    print("=" * 60)

    # Create output manager with short intervals for demo
    manager = OutputManager(status_interval=1.0, min_log_lines=3)

    # Register mock processes
    processes = [
        (5001, 0, "MNIST-Train"),
        (5002, 1, "CIFAR-Train"),
    ]

    for pid, task_id, tag in processes:
        manager.register_process(pid, task_id, tag)

    print("\nSimulating 15 seconds of execution...")
    print("Watch how status updates appear based on log frequency:\n")

    # Start manager (without actual thread for demo)
    manager.running = True
    manager.show_status = True

    # Simulate different phases
    phases = [
        ("Active Phase", 5, 2),  # Many logs per second
        ("Quiet Phase", 5, 0.2),  # Few logs per second
        ("Mixed Phase", 5, 1),  # Moderate logs
    ]

    start_time = time.time()

    for phase_name, duration, logs_per_second in phases:
        print(f"\n--- {phase_name} ({logs_per_second} logs/sec) ---\n")

        phase_start = time.time()
        while time.time() - phase_start < duration:
            # Generate logs based on frequency
            if logs_per_second > 0:
                for pid, _, tag in processes:
                    if time.time() - phase_start < duration:  # Double check
                        elapsed = int(time.time() - start_time)
                        msg = {"pid": pid, "type": "stdout", "content": f"{phase_name} - Progress at {elapsed}s"}
                        manager.process_message(msg)

                # Check if status should be printed
                if manager._should_print_status(time.time()):
                    manager._print_status_summary()
                    manager.last_status_print = time.time()

            # Sleep based on log frequency
            if logs_per_second > 0:
                time.sleep(1.0 / logs_per_second)
            else:
                time.sleep(1.0)

    # Final summary
    print("\nDemo completed!")
    print(f"Total logs: {len(manager.all_logs)}")
    print("Notice how status updates appeared more frequently during active phases.")


def run_visual_test():
    """Run a visual test of the complete system."""
    print("\n" + "=" * 60)
    print("VISUAL TEST: Complete System")
    print("=" * 60)

    # Mock the single_run function
    def mock_single_run(config, is_subprocess=False):
        """Simulate a federated learning task."""
        import random

        # Simulate task execution
        for epoch in range(1, 6):
            print(f"Starting epoch {epoch}/5")
            time.sleep(0.5)

            # Training phase
            for client in range(1, 4):
                print(f"Training with client {client}/3")
                time.sleep(0.2)

            # Evaluation
            accuracy = 0.8 + epoch * 0.03 + random.random() * 0.05
            print(f"Epoch {epoch} completed. Accuracy: {accuracy:.3f}")

            # Occasional warnings
            if random.random() < 0.3:
                print("Warning: Client connection slow", file=sys.stderr)

        print("Task completed successfully!")

    # Patch single_run
    with patch("fl_sim.utils.multiprocessing.single_run", side_effect=mock_single_run):
        # Create test configurations
        configs = [
            CFG({"name": "mnist_test", "epochs": 5}),
            CFG({"name": "cifar_test", "epochs": 5}),
            CFG({"name": "custom_test", "epochs": 5}),
        ]

        # Run tasks
        print("\nRunning 3 parallel tasks with status updates...")
        print("Status will appear when enough logs accumulate.\n")

        run_parallel_tasks(
            configs, num_workers=2, task_tags=["MNIST-FL", "CIFAR-FL", "Custom-FL"], status_interval=2.0, min_log_lines=8
        )


def main():
    """Main test entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test multiprocessing module")
    parser.add_argument("--mode", choices=["unit", "demo", "visual", "all"], default="unit", help="Test mode to run")

    args = parser.parse_args()

    if args.mode == "unit":
        # Run unit tests
        print("Running unit tests...")
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()

        # Add test classes
        for test_class in [TestProcessStatus, TestOutputManager, TestMultiprocessManager]:
            suite.addTests(loader.loadTestsFromTestCase(test_class))

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        return 0 if result.wasSuccessful() else 1

    elif args.mode == "demo":
        # Run basic demonstration
        run_basic_demo()

    elif args.mode == "visual":
        # Run visual test
        run_visual_test()

    elif args.mode == "all":
        # Run everything
        print("1. Running unit tests...")
        print("-" * 40)

        # Run unit tests first
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for test_class in [TestProcessStatus, TestOutputManager, TestMultiprocessManager]:
            suite.addTests(loader.loadTestsFromTestCase(test_class))
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(suite)

        if result.wasSuccessful():
            print("\n2. Running basic demo...")
            print("-" * 40)
            run_basic_demo()

            print("\n3. Running visual test...")
            print("-" * 40)
            run_visual_test()

        return 0 if result.wasSuccessful() else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
