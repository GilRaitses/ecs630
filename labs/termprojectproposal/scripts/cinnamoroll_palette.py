"""
Cinnamoroll color palette for all visualizations.
Use this module to ensure consistent color scheme across all plots.
"""

# Cinnamoroll Color Palette
CINNAMOROLL_COLORS = {
    'light_blue': '#B4D7EC',
    'lavender': '#E8D7F1',
    'pink': '#F5D4D8',
    'cream': '#FFF9F3',
    'blue': '#8AB4D3',
    'dark_blue': '#2C4A5F',
    'brown': '#5F4A3F',
    'peach': '#FFE5D9',
    'purple': '#C8B5DC',
    'rose': '#E0B8C0',
    'tan': '#E8D8CC'
}

# Color lists for sequential/categorical plots
CINNAMOROLL_PALETTE = [
    CINNAMOROLL_COLORS['light_blue'],
    CINNAMOROLL_COLORS['lavender'],
    CINNAMOROLL_COLORS['pink'],
    CINNAMOROLL_COLORS['peach'],
    CINNAMOROLL_COLORS['purple'],
    CINNAMOROLL_COLORS['rose']
]

# Gradient palette for heatmaps
CINNAMOROLL_GRADIENT = [
    CINNAMOROLL_COLORS['cream'],
    CINNAMOROLL_COLORS['light_blue'],
    CINNAMOROLL_COLORS['blue'],
    CINNAMOROLL_COLORS['dark_blue']
]

def setup_cinnamoroll_style():
    """Configure matplotlib to use Cinnamoroll style."""
    import matplotlib.pyplot as plt
    import matplotlib
    
    # Set style parameters
    plt.rcParams.update({
        'figure.facecolor': CINNAMOROLL_COLORS['cream'],
        'axes.facecolor': 'white',
        'axes.edgecolor': CINNAMOROLL_COLORS['blue'],
        'axes.labelcolor': CINNAMOROLL_COLORS['dark_blue'],
        'text.color': CINNAMOROLL_COLORS['dark_blue'],
        'xtick.color': CINNAMOROLL_COLORS['brown'],
        'ytick.color': CINNAMOROLL_COLORS['brown'],
        'grid.color': CINNAMOROLL_COLORS['tan'],
        'grid.alpha': 0.3,
        'font.family': 'Avenir Next',
        'font.size': 11,
        'axes.linewidth': 1.5,
        'axes.grid': True,
        'grid.linewidth': 0.8,
        'figure.dpi': 300
    })
    
    return plt

def get_cinnamoroll_cmap():
    """Get Cinnamoroll colormap for continuous data."""
    from matplotlib.colors import LinearSegmentedColormap
    
    colors = [
        CINNAMOROLL_COLORS['cream'],
        CINNAMOROLL_COLORS['light_blue'],
        CINNAMOROLL_COLORS['blue'],
        CINNAMOROLL_COLORS['dark_blue']
    ]
    
    return LinearSegmentedColormap.from_list('cinnamoroll', colors)

