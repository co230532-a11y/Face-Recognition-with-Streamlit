"""
Terminal Colors and Formatting

Provides colored output for terminal display
"""

class Colors:
    """ANSI color codes for terminal output"""
    
    # Basic colors
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[35m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    
    # Formatting
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    
    # Backgrounds
    BG_GREEN = '\033[42m'
    BG_RED = '\033[41m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'


def colored_text(text, color):
    """Return colored text"""
    return f"{color}{text}{Colors.RESET}"


def print_header(title):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title.center(80)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")


def print_subheader(title):
    """Print formatted subheader"""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}{title}{Colors.RESET}")
    print(f"{Colors.YELLOW}{'-'*80}{Colors.RESET}")


def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")


def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")


def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")


def print_info(message):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {message}{Colors.RESET}")


def print_score(label, value, threshold=None, format_str=":.4f"):
    """Print a score with color based on value"""
    formatted_value = f"{value:{format_str}}"
    
    if threshold is None:
        color = Colors.GREEN if value > 0.5 else Colors.YELLOW if value > 0.2 else Colors.RED
    else:
        if value >= threshold:
            color = Colors.GREEN
        elif value >= threshold * 0.8:
            color = Colors.YELLOW
        else:
            color = Colors.RED
    
    print(f"  {label:.<40} {color}{formatted_value}{Colors.RESET}")


def print_table_header(*columns):
    """Print table header"""
    col_width = 20
    header = " | ".join(f"{col:^{col_width}}" for col in columns)
    print(f"{Colors.BOLD}{Colors.CYAN}{header}{Colors.RESET}")
    print(f"{Colors.CYAN}{'-' * (len(columns) * (col_width + 3))}{Colors.RESET}")


def print_table_row(*values):
    """Print table row"""
    col_width = 20
    row = " | ".join(f"{str(val):^{col_width}}" for val in values)
    print(row)
