#!/usr/bin/env python3
"""
NEPFlow: HPC workflow system for atomistic datasets.

Entry point for the workflow controller.
"""

import sys
import argparse
import logging
from pathlib import Path

# Check required libraries before doing anything else
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "ase": "ase",
    "hiphive": "hiphive",
    "mp_api": "mp-api",
    "pymatgen": "pymatgen",
    "icet": "icet",
    "NepTrainKit": "NepTrainKit",
}

_missing = []
for _mod, _pkg in REQUIRED_PACKAGES.items():
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_pkg)

if _missing:
    print(
        f"Missing required packages: {', '.join(_missing)}\n"
        f"Install with:  pip install {' '.join(_missing)}",
        file=sys.stderr,
    )
    sys.exit(1)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Note: Imports below depend on src/ being in sys.path
# pylint: disable=import-error
from workflow import WorkflowController
from logging_config import setup_logging

logger = logging.getLogger("nepflow")


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="NEPFlow: HPC workflow orchestrator for VASP→NEP→GPUMD pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --project myproject
  python main.py --project myproject --debug
        """
    )
    
    # Global options
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Project name (determines config, state, and output paths)"
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize a new project (required on first run)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (no external dependencies required)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "projects",
        help="Output base directory (default: ./projects)"
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging for the project
    # Log directory is inside each project: projects/project_{name}/logs/
    project_dir = args.output_dir / f"project_{args.project}"
    log_dir = project_dir / "logs"
    
    setup_logging(
        project_name=args.project,
        log_dir=log_dir,
        debug=args.debug
    )
    
    logger.info("NEPFlow started for project: %s", args.project)
    if args.debug:
        logger.debug("Debug mode enabled")
    
    # Create workflow controller
    controller = WorkflowController(
        project_name=args.project,
        output_dir=args.output_dir,
        init_mode=args.init,
        debug=args.debug
    )
    
    # Run workflow: controller determines current stage from .project file
    try:
        controller.run()
        logger.info("Workflow completed successfully")
        print("✓ Workflow completed successfully")
        
    except (ValueError, IOError, RuntimeError) as e:
        # Log the error (message only, not traceback)
        logger.error("Workflow failed: %s", e)
        
        # Print formatted error message
        print("\n" + "=" * 70, file=sys.stderr)
        print("ERROR: Workflow failed", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print("  %s: %s" % (type(e).__name__, e), file=sys.stderr)
        
        if args.debug:
            print("\nTraceback:", file=sys.stderr)
            print("-" * 70, file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            print("-" * 70, file=sys.stderr)
        else:
            print("\n  Use --debug for full traceback", file=sys.stderr)
        
        print("=" * 70 + "\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
