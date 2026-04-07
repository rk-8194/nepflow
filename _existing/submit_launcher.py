#!/usr/bin/env python3
"""
Python wrapper to submit and monitor the VASP launcher as a SLURM job.

This script:
1. Submits the launcher as a SLURM job
2. Optionally monitors the job's progress
3. Reports job status and logs

Usage:
    python submit_launcher.py              # Submit launcher job
    python submit_launcher.py --monitor    # Submit and monitor job
    python submit_launcher.py --status     # Check if launcher is running
"""

import subprocess
import sys
import time
import os
from pathlib import Path
from datetime import datetime


def get_job_status(job_id):
    """Get status of a SLURM job."""
    try:
        result = subprocess.run(
            ['squeue', '-j', str(job_id), '--noheader'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            # Job is in queue
            parts = result.stdout.split()
            status = parts[4] if len(parts) > 4 else "UNKNOWN"
            return status
        else:
            # Job not in queue - check if it completed
            return "COMPLETED"
    except Exception as e:
        print(f"Error checking job status: {e}")
        return "UNKNOWN"


def submit_launcher_job(time_limit="02:00:00", script_name="submit_launcher.sh"):
    """Submit the launcher as a SLURM job.
    
    Args:
        time_limit: Time limit for the job (default: 02:00:00)
        script_name: Name of the launcher script (default: submit_launcher.sh)
    """
    # Use current directory to find script
    script_path = Path.cwd() / script_name
    
    if not script_path.exists():
        print(f"ERROR: Launcher script not found: {script_path}")
        return None
    
    print("=" * 72)
    print("VASP Launcher Job Submission")
    print("=" * 72)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script: {script_path}")
    print(f"Time limit: {time_limit}")
    print()
    
    # Submit the job
    try:
        result = subprocess.run(
            ['sbatch', f'--time={time_limit}', script_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"✗ Job submission failed!")
            print(f"Error: {result.stderr}")
            return None
        
        # Extract job ID from output
        output = result.stdout.strip()
        if "Submitted batch job" in output:
            job_id = output.split()[-1]
            print(f"✓ Job submitted successfully")
            print(f"Job ID: {job_id}")
            print()
            return job_id
        else:
            print(f"✗ Unexpected output: {output}")
            return None
    
    except subprocess.TimeoutExpired:
        print("✗ Job submission timed out")
        return None
    except Exception as e:
        print(f"✗ Error submitting job: {e}")
        return None


def monitor_launcher_job(job_id, check_interval=30, max_duration=7200):
    """Monitor a running launcher job."""
    print("=" * 72)
    print("VASP Launcher Job Monitoring")
    print("=" * 72)
    print(f"Job ID: {job_id}")
    print(f"Check interval: {check_interval} seconds")
    print(f"Max duration: {max_duration} seconds")
    print()
    
    start_time = time.time()
    last_status = None
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_duration:
            print(f"\n✗ Monitoring timeout reached ({max_duration}s)")
            return False
        
        status = get_job_status(job_id)
        
        if status != last_status:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Job status: {status}")
            last_status = status
        
        if status == "COMPLETED":
            print(f"\n✓ Job completed")
            return True
        elif status == "FAILED":
            print(f"\n✗ Job failed")
            return False
        elif status == "UNKNOWN":
            # Job no longer in queue and status unknown
            print(f"\n⚠ Job status unknown (may have completed)")
            return True
        
        # Check every 30 seconds
        time.sleep(check_interval)


def check_launcher_status():
    """Check if launcher is currently running."""
    try:
        result = subprocess.run(
            ['squeue', '-u', os.environ.get('USER', ''), '--noheader', '-n', 'vasp-launcher'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.stdout.strip():
            print("=" * 72)
            print("Active Launcher Jobs")
            print("=" * 72)
            print(result.stdout)
            return True
        else:
            print("No active launcher jobs found")
            return False
    except Exception as e:
        print(f"Error checking status: {e}")
        return False


def main():
    """Main function."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--monitor":
            # Submit and monitor
            job_id = submit_launcher_job()
            if job_id:
                success = monitor_launcher_job(job_id)
                sys.exit(0 if success else 1)
            else:
                sys.exit(1)
        
        elif sys.argv[1] == "--status":
            # Check status only
            check_launcher_status()
            sys.exit(0)
        
        elif sys.argv[1] == "--time":
            # Submit with custom time limit
            if len(sys.argv) > 2:
                time_limit = sys.argv[2]
                job_id = submit_launcher_job(time_limit)
                sys.exit(0 if job_id else 1)
            else:
                print("Usage: python submit_launcher.py --time <HH:MM:SS>")
                sys.exit(1)
        
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print(__doc__)
            sys.exit(1)
    
    else:
        # Just submit
        job_id = submit_launcher_job()
        sys.exit(0 if job_id else 1)


if __name__ == "__main__":
    main()
