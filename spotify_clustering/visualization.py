import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def make_histogram_from_dataframe(dataframe: pd.DataFrame) -> go.Figure:
    """Generates a plotly figure with the histogram of each numerical feature.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Numerical dataframe containing the data

    Returns
    -------
    go.Figure
        Hisogram of each dataframe feature.
    """
    # Init plotly figure
    fig = make_subplots(
        rows=4,
        cols=4,
        subplot_titles=dataframe.columns,
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    for i, column_i in enumerate(dataframe.columns):
        # Position of each trace in figure
        row = i // 4 + 1
        col = i % 4 + 1

        trace = go.Histogram(x=dataframe[column_i], name=f"<b>{column_i}</b>")
        fig.add_trace(trace, row=row, col=col)

        # Update axes labels
        fig.update_xaxes(title_text="Value", row=row, col=col)
        fig.update_yaxes(title_text="Frequency", row=row, col=col)

    # Layout settings
    fig.update_layout(
        showlegend=False, margin={"l": 0, "r": 0, "t": 30, "b": 0}, height=800
    )

    return fig


def make_scatter3d_from_dataframe(
    dataframe: pd.DataFrame,
    x_axes: str,
    y_axes: str,
    z_axes: str,
    hoverdata: pd.DataFrame,
    color_by: str = "song_popularity",
) -> go.Figure:
    """_summary_

    Parameters
    ----------
    dataframe : pd.DataFrame
        _description_
    x_axes : str
        _description_
    y_axes : str
        _description_
    z_axes : str
        _description_
    hoverdata : pd.DataFrame
        _description_

    Returns
    -------
    go.Figure
        _description_
    """
    fig = make_subplots(rows=1, cols=1)

    scatter = go.Scatter3d(
        x=dataframe[x_axes],
        y=dataframe[y_axes],
        z=dataframe[z_axes],
        mode="markers",
        marker=dict(
            size=8,
            color=dataframe[color_by],
            colorscale="portland",
            colorbar=dict(title=color_by),
        ),
        customdata=np.stack(
            [hoverdata[i] for i in hoverdata.columns],
            axis=-1,
        ),
        hovertemplate="<b>Artist:</b> %{customdata[0]} <br>"
        + "<b>Genres:</b> %{customdata[1]} <extra></extra>",
        showlegend=False,
    )

    fig.add_trace(scatter)
    fig.update_layout(scene=dict(aspectmode="cube"))

    buttons = [
        dict(
            label=feature_i,
            method="restyle",
            args=[
                {
                    "marker.color": [dataframe[feature_i]],
                    "marker.colorscale": "portland",
                    "marker.colorbar.title": feature_i,
                }
            ],
        )
        for feature_i in dataframe.columns
    ]

    fig.update_layout(
        updatemenus=[dict(type="buttons", showactive=False, buttons=buttons)],
        height=800,
    )

    return fig
