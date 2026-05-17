"""
Main Control Script - Facial Recognition Preprocessing Investigation
"""

import sys
import os
import logging
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.colors import Colors, print_header, print_success, print_error, print_info

# Setup logging
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'main_control.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def print_menu():
    """Print main menu with colors"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'PREPROCESSING INVESTIGATION MENU'.center(80)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    menu_items = [
        (1, "Preprocessing Pipeline", "Capture & compare BEFORE vs AFTER"),
        (2, "Statistical Analysis", "Perform statistical tests on captured data"),
        (3, "View Results", "Display JSON investigation results"),
        (4, "System Information", "Show system stats and project health"),
        (0, "Exit", "Quit application"),
    ]
    
    for num, title, desc in menu_items:
        print(f"{Colors.YELLOW}{num}{Colors.RESET}. {Colors.BOLD}{title:.<35}{Colors.RESET} - {desc}")
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")


def preprocessing_pipeline():
    """Run preprocessing pipeline"""
    try:
        print_header("PREPROCESSING BEFORE/AFTER INVESTIGATION")
        print_info("Capturing your face and comparing preprocessing effects")
        
        # Imports the logic from the script you provided
        from pipelines.run_preprocessing_with_capture import PreprocessingWithCapture
        
        investigation = PreprocessingWithCapture()
        
        if investigation.run_investigation():
            print_success("Preprocessing investigation completed successfully")
            logger.info("Preprocessing pipeline completed")
        else:
            print_error("Preprocessing investigation failed")
            logger.error("Preprocessing pipeline failed")
    
    except Exception as e:
        print_error(f"Error: {e}")
        logger.error(f"Preprocessing pipeline error: {e}")


def statistical_analysis():
    """Run statistical analysis"""
    try:
        print_header("STATISTICAL ANALYSIS")
        print_info("Running comprehensive statistical tests")
        
        from pipelines.main_analysis import ComprehensiveAnalysis
        
        analysis = ComprehensiveAnalysis()
        
        if analysis.run_all_tests():
            print_success("Statistical analysis completed successfully")
            logger.info("Statistical analysis completed")
        else:
            print_error("Statistical analysis failed")
            logger.error("Statistical analysis failed")
    
    except Exception as e:
        print_error(f"Error: {e}")
        logger.error(f"Statistical analysis error: {e}")


def view_results():
    """Directly open and display the most recent investigation results in a readable format"""
    try:
        print_header("INVESTIGATION REPORT")
        
        from core.config import CAPTURED_DIR
        if not CAPTURED_DIR.exists():
            print_info(f"Directory not found: {CAPTURED_DIR}")
            return

        capture_folders = sorted(
            [d for d in CAPTURED_DIR.iterdir() if d.is_dir()],
            key=os.path.getmtime,
            reverse=True
        )

        if not capture_folders:
            print_info("No capture folders found.")
            return

        latest_folder = capture_folders[0]
        results_file = latest_folder / "comparison_results.json"

        if not results_file.exists():
            print_error(f"Results file not found in {latest_folder.name}")
            return

        with open(results_file, 'r') as f:
            data = json.load(f)

        # --- Readable Formatting Starts Here ---
        print(f"\n{Colors.BOLD}{Colors.YELLOW}SESSION SUMMARY: {data.get('timestamp')}{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}Folder:{Colors.RESET} {data.get('folder')}")
        print(f"{Colors.BOLD}Reference Image:{Colors.RESET} {data.get('files', {}).get('reference')}")
        print(f"{Colors.CYAN}{'-'*70}{Colors.RESET}")

        # Define metrics to display
        metrics = [
            ('ssim_percent', 'SSIM Similarity'),
            ('cosine_percent', 'Cosine Similarity'),
            ('euclidean_percent', 'Euclidean Accuracy'),
            ('combined_accuracy', 'COMBINED TOTAL')
        ]

        # Header for the table
        print(f"{'METRIC':<25} | {'BEFORE':<12} | {'AFTER':<12} | {'CHANGE'}")
        print(f"{'-'*70}")

        before_acc = data.get('before', {}).get('accuracy', {})
        after_acc = data.get('after', {}).get('accuracy', {})
        improvement = data.get('improvement', {})

        for key, label in metrics:
            val_before = before_acc.get(key, 0)
            val_after = after_acc.get(key, 0)
            
            # Map metric key to its corresponding change key in JSON
            change_key = key.replace('percent', 'change') if 'percent' in key else f"{key}_change"
            val_change = improvement.get(change_key, 0)

            # Apply colors to the change column
            change_color = Colors.GREEN if val_change > 0 else Colors.RED if val_change < 0 else Colors.GRAY
            
            print(f"{label:<25} | {val_before:>10.2f}% | {val_after:>10.2f}% | {change_color}{val_change:>+9.2f}%{Colors.RESET}")

        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}\n")

    except Exception as e:
        print_error(f"Error formatting results: {e}")
        logger.error(f"View results error: {e}")


def system_information():
    """Show system information and directory check"""
    try:
        print_header("SYSTEM INFORMATION")
        
        from core.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        info = monitor.get_system_info()
        
        if info:
            print(f"{Colors.BOLD}System Stats:{Colors.RESET}")
            for key, val in info.items():
                print(f"  {key:.<40} {Colors.CYAN}{val}{Colors.RESET}")
        
        print(f"\n{Colors.BOLD}Project Structure:{Colors.RESET}")
        project_root = Path(__file__).parent
        
        dirs = ['database/reference_images', 'captured', 'results', 'logs']
        for dir_name in dirs:
            dir_path = project_root / dir_name
            if dir_path.exists():
                file_count = len(list(dir_path.glob("*")))
                print_success(f"{dir_name}: {file_count} items")
            else:
                print_error(f"{dir_name}: not found")
    
    except Exception as e:
        print_error(f"Error: {e}")
        logger.error(f"System info error: {e}")


def main():
    """Main entry point"""
    try:
        print_header("FACIAL RECOGNITION PREPROCESSING ANALYSIS")
        print_info("Testing how preprocessing techniques affect image similarity")
        logger.info("Application started")
        
        while True:
            print_menu()
            
            try:
                choice = input(f"{Colors.BOLD}{Colors.YELLOW}Enter your choice (0-4):{Colors.RESET} ").strip()
                
                if choice == "0":
                    print_success("Thank you for using the Preprocessing Framework!")
                    logger.info("Application closed")
                    break
                elif choice == "1":
                    preprocessing_pipeline()
                elif choice == "2":
                    statistical_analysis()
                elif choice == "3":
                    view_results()
                elif choice == "4":
                    system_information()
                else:
                    print_error("Invalid choice. Please enter 0-4")
                
                input(f"\n{Colors.GRAY}Press Enter to continue...{Colors.RESET}")
                
            except KeyboardInterrupt:
                print_error("\nApplication interrupted")
                logger.info("Application interrupted by user")
                break
            except Exception as e:
                print_error(f"Error: {e}")
                logger.error(f"Menu error: {e}")
    
    except Exception as e:
        print_error(f"Fatal error: {e}")
        logger.error(f"Fatal error: {e}")


if __name__ == "__main__":
    main()