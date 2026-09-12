"""Interactive candlestick chart for causal annotation."""

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QWidget


class CandlestickChart(FigureCanvasQTAgg):
    """Matplotlib chart that can be placed inside a PySide6 window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty candlestick chart.

        Args:
            parent: PySide6 widget responsible for this chart.
        """
        # Create the Matplotlib figure that will contain the candlestick plot.
        figure = Figure(figsize=(12, 7), tight_layout=True)

        # Give this chart one set of axes for prices, candles, and annotations.
        self.axes = figure.add_subplot(1, 1, 1)

        # Turn the Matplotlib figure into a PySide6 widget.
        super().__init__(figure)

        # Attach the chart to its containing window when one is provided.
        self.setParent(parent)
