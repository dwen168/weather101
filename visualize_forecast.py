import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def generate_premium_forecast_plot(csv_path='hornsby_predicted_rainfall_next_year.csv', threshold=15.0):
    """
    Generates a stunning, premium dark-mode dashboard-quality daily rainfall
    forecast visualization for the next 365 days, highlighting heavy rain events.
    """
    print(f"Loading predictions from {csv_path}...")
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 1. Premium Dark Mode Styling Setup (Cinematic Navy-Black Theme)
    plt.style.use('dark_background')
    
    bg_color = '#070a13'       # Rich Dark Cinematic Blue-Black
    card_color = '#0f172a'     # Card Background Navy-Slate
    accent_blue = '#38bdf8'    # Neon Cyan/Sky Blue
    accent_rose = '#f43f5e'    # Neon Rose/Pink (for heavy rain)
    grid_color = '#1e293b'     # Thin Slate Divider
    text_color = '#f8fafc'     # Pure White-Slate
    text_muted = '#64748b'     # Muted grey-slate
    border_color = '#334155'   # Glassmorphic border slate
    
    fig, ax = plt.subplots(figsize=(18, 9.5), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    # 2. Multi-Pass Neon Glow Daily Rainfall Bars
    # Deep outer glow (widest line, ultra-low opacity)
    ax.vlines(
        df['Date'], 0, df['Predicted_Rainfall'],
        colors='#0284c7', alpha=0.12, linewidth=6.0, zorder=1
    )
    # Mid glow (medium line, low opacity)
    ax.vlines(
        df['Date'], 0, df['Predicted_Rainfall'],
        colors='#0ea5e9', alpha=0.35, linewidth=3.5, zorder=2
    )
    # Core neon bar (thin line, high opacity)
    ax.vlines(
        df['Date'], 0, df['Predicted_Rainfall'],
        colors=accent_blue, alpha=0.90, linewidth=1.5, label='Daily Predicted Rainfall', zorder=3
    )
    
    # Fill the area under the timeline slightly for a modern glow effect
    ax.fill_between(
        df['Date'], 0, df['Predicted_Rainfall'],
        color=accent_blue, alpha=0.04, zorder=1
    )
    
    # 3. Dynamic Banded Weather Risk Zones (Background Gradients)
    y_max = df['Predicted_Rainfall'].max() + 8
    ax.set_ylim(0, y_max)
    
    # Heavy Rain Risk Zone (Above Threshold)
    ax.axhspan(threshold, y_max, color=accent_rose, alpha=0.025, zorder=0)
    ax.text(
        df['Date'].min() + pd.Timedelta(days=5), threshold + 1.2,
        "HEAVY RAIN RISK ZONE (>= 15.0 mm)",
        color=accent_rose, fontsize=10, fontweight='bold', alpha=0.7, zorder=4
    )
    
    # Moderate Rain Zone (5mm - Threshold)
    ax.axhspan(5.0, threshold, color='#0ea5e9', alpha=0.012, zorder=0)
    ax.text(
        df['Date'].min() + pd.Timedelta(days=5), 6.0,
        "MODERATE RAIN ZONE (5.0 - 15.0 mm)",
        color=accent_blue, fontsize=10, fontweight='bold', alpha=0.5, zorder=4
    )
    
    # 4. Identify and Highlight Heavy Rain Events (with 4-Layer Glow Starbursts)
    heavy_rain_df = df[df['Predicted_Rainfall'] >= threshold]
    heavy_count = len(heavy_rain_df)
    
    if heavy_count > 0:
        # Layer 1: Broad Outer Glow
        ax.scatter(
            heavy_rain_df['Date'], heavy_rain_df['Predicted_Rainfall'],
            color=accent_rose, s=500, zorder=4, alpha=0.10
        )
        # Layer 2: Medium Halo
        ax.scatter(
            heavy_rain_df['Date'], heavy_rain_df['Predicted_Rainfall'],
            color=accent_rose, s=220, zorder=5, alpha=0.35
        )
        # Layer 3: Neon Rose point
        ax.scatter(
            heavy_rain_df['Date'], heavy_rain_df['Predicted_Rainfall'],
            color=accent_rose, s=80, zorder=6, alpha=0.9,
            edgecolors='#ffffff', linewidths=0.8, label=f'Heavy Rain (>= {threshold}mm)'
        )
        # Layer 4: Bright Core Center
        ax.scatter(
            heavy_rain_df['Date'], heavy_rain_df['Predicted_Rainfall'],
            color='#ffffff', s=20, zorder=7, alpha=1.0
        )
        
        # Annotate and target the heaviest rainfall event
        heaviest_day = heavy_rain_df.loc[heavy_rain_df['Predicted_Rainfall'].idxmax()]
        
        # Draw target ring on the heaviest day
        ax.scatter(
            [heaviest_day['Date']], [heaviest_day['Predicted_Rainfall']],
            facecolors='none', edgecolors=accent_rose, s=900, linewidths=1.5, zorder=8, alpha=0.8
        )
        
        # Custom glassmorphic popup annotation
        ax.annotate(
            f"  PEAK SYSTEM EVENT\n  Rain: {heaviest_day['Predicted_Rainfall']:.1f} mm\n  Date: {heaviest_day['Date'].strftime('%d %b %Y')}  ",
            xy=(heaviest_day['Date'], heaviest_day['Predicted_Rainfall']),
            xytext=(heaviest_day['Date'] + pd.Timedelta(days=12), heaviest_day['Predicted_Rainfall'] + 2.5),
            color=text_color,
            fontweight='bold',
            fontsize=10.5,
            zorder=10,
            bbox=dict(boxstyle="round,pad=0.6,rounding_size=0.2", facecolor='#1e1b4b', edgecolor=accent_rose, alpha=0.9, linewidth=1.5),
            arrowprops=dict(arrowstyle="->", color=accent_rose, lw=2.0, connectionstyle="arc3,rad=0.2")
        )
        
    # 5. Draw the Threshold Line
    ax.axhline(
        y=threshold, color=accent_rose, linestyle='--', linewidth=1.5, alpha=0.7,
        label=f'Heavy Rain Threshold ({threshold} mm)', zorder=4
    )
    
    # 6. Fine-tune Axes and Grid Lines
    # Date formatting on X-axis (E.g. "Jun 2026", "Jul 2026")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=25, ha='right', color=text_muted, fontsize=11)
    plt.yticks(color=text_muted, fontsize=11)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(grid_color)
    ax.spines['bottom'].set_color(grid_color)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Elegant gridlines
    ax.grid(color=grid_color, linestyle='--', linewidth=0.6, alpha=0.6)
    
    # Title Block
    plt.title(
        'Hornsby Area: 365-Day Daily Rainfall Forecast & Heavy Rain Risk Map',
        fontsize=18, fontweight='bold', color=text_color, pad=25, loc='left'
    )
    
    ax.set_ylabel('Rainfall Intensity (mm)', color=text_color, fontsize=13, labelpad=12)
    ax.set_xlabel('Forecast Timeline', color=text_color, fontsize=13, labelpad=12)
    
    # 7. Premium Glassmorphic Dashboard Summary Panel
    total_annual = df['Predicted_Rainfall'].sum()
    rainy_days = (df['Predicted_Rainfall'] > 0).sum()
    
    summary_text = (
        f"  FORECAST STATS\n"
        f"  ====================\n"
        f"  Range: May 2026 - Apr 2027\n"
        f"  Total Rain: {total_annual:.1f} mm\n"
        f"  Wet Days: {rainy_days} / 365\n"
        f"  Heavy Days: {heavy_count} days"
    )
    
    # Draw a box for the summary panel (Navy slate with bright borders)
    props = dict(boxstyle='round,pad=0.8,rounding_size=0.25', facecolor=card_color, edgecolor=border_color, alpha=0.9, linewidth=1.2)
    ax.text(
        0.02, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props, color=text_color, linespacing=1.65, fontweight='medium', zorder=9
    )
    
    # Legend
    legend = ax.legend(
        loc='upper right', facecolor=card_color, edgecolor=border_color,
        labelcolor=text_color, fontsize=11, framealpha=0.9, borderpad=0.8
    )
    legend.get_frame().set_linewidth(1.2)
    legend.set_zorder(9)
    
    plt.tight_layout()
    output_path = 'hornsby_forecast_premium.png'
    plt.savefig(output_path, dpi=300, facecolor=bg_color)
    plt.close()
    print(f"Successfully generated premium visualization at: {output_path}")

if __name__ == '__main__':
    generate_premium_forecast_plot(threshold=15.0)
