#!/usr/bin/env python3
"""
Shared Figure Styling for Journal of Public Economic Theory Submission

This module provides consistent figure formatting that follows economics journal standards:
- No in-figure titles (captions handle this in LaTeX)
- No grid lines (cleaner appearance)
- Serif fonts to match LaTeX documents
- Frameless legends
- Sentence case for axis labels
- Appropriate line weights and figure sizes

Usage:
    from figure_style import apply_journal_style, COLORS, save_figure
    
    apply_journal_style()  # Call once at start of script
    # ... create your figure ...
    save_figure('figure_name')  # Saves both PDF and PNG
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# =============================================================================
# COLOR PALETTE (colorblind-friendly)
# =============================================================================

COLORS = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e', 
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'gray': '#7f7f7f',
    'olive': '#bcbd22',
    'cyan': '#17becf'
}

# For sequential data (e.g., different horizons or returns)
COLOR_SEQ = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


# =============================================================================
# JOURNAL STYLE CONFIGURATION
# =============================================================================

def apply_journal_style():
    """
    Apply consistent styling for economics journal figures.
    Call this once at the start of your script.
    """
    plt.rcParams.update({
        # Font settings (serif to match LaTeX)
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 10,
        
        # Axes settings
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'axes.linewidth': 0.8,
        'axes.spines.top': True,
        'axes.spines.right': True,
        
        # Tick settings
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        
        # Legend settings
        'legend.fontsize': 9,
        'legend.frameon': False,  # No frame around legend
        'legend.borderpad': 0.4,
        
        # Line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        
        # Figure settings
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        
        # Grid (disabled by default)
        'axes.grid': False,
        
        # Use LaTeX-style math text
        'mathtext.fontset': 'cm',
    })


def save_figure(filename, formats=['pdf', 'png']):
    """
    Save figure in multiple formats with appropriate settings.
    
    Parameters:
        filename: Base filename without extension
        formats: List of formats to save (default: PDF and PNG)
    """
    for fmt in formats:
        if fmt == 'png':
            plt.savefig(f'{filename}.{fmt}', dpi=150, bbox_inches='tight')
        else:
            plt.savefig(f'{filename}.{fmt}', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {filename}.pdf")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def add_panel_label(ax, label, loc='upper left'):
    """
    Add panel label (a), (b), etc. to subplot.
    
    Parameters:
        ax: Matplotlib axes object
        label: Panel label string, e.g., '(a)'
        loc: Location ('upper left', 'upper right', etc.)
    """
    if loc == 'upper left':
        x, y = 0.02, 0.98
        va, ha = 'top', 'left'
    elif loc == 'upper right':
        x, y = 0.98, 0.98
        va, ha = 'top', 'right'
    else:
        x, y = 0.02, 0.98
        va, ha = 'top', 'left'
    
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11, 
            fontweight='bold', va=va, ha=ha)


def format_percent_axis(ax, axis='y'):
    """Format axis ticks as percentages."""
    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(lambda x, p: f'{x:.0f}%')
    if axis == 'y':
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


# =============================================================================
# STANDARD FIGURE SIZES
# =============================================================================

# Single column figure (for most plots)
FIG_SINGLE = (6.5, 4.5)

# Wide figure (for two-panel plots)
FIG_WIDE = (10, 4.5)

# Two-panel figure
FIG_TWO_PANEL = (10, 4)

# Square figure
FIG_SQUARE = (5.5, 5)
