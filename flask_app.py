# /home/<username>/mysite/flask_app.py

import os
from functools import lru_cache

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, html, dcc, Input, Output

app = Dash(__name__)
server = app.server 

DB_PATH = os.path.join(os.path.expanduser("~"), "sports_pipeline.db")
engine = create_engine(f"sqlite:///{DB_PATH}")


@lru_cache(maxsize=1)
def load_favorites_json() -> str:
    """
    Load + preprocess favorites, then return as JSON string.
    Cached so the web worker doesn't hit SQLite on every callback.
    """
    df = pd.read_sql_query(
        """
        SELECT
          l.event_id,
          l.game_id,

          g.season,
          g.game_start_utc,
          g.home_team_name,
          g.away_team_name,
          g.home_score,
          g.away_score,
          g.winner,

          ob.bookmaker_title,
          oq.outcome_name,
          oq.outcome_price,
          oq.implied_prob

        FROM event_game_link l
        JOIN games g
          ON g.game_id = l.game_id
        JOIN odds_quotes oq
          ON oq.event_id = l.event_id
        JOIN odds_bookmakers ob
          ON ob.event_id = oq.event_id
         AND ob.bookmaker_key = oq.bookmaker_key

        WHERE oq.market_key = 'h2h'
        """,
        engine
    )

    # keep only completed games
    df = df[df["winner"].isin(["home", "away"])].copy()

    df["winner_team"] = np.where(
        df["winner"] == "home",
        df["home_team_name"],
        df["away_team_name"]
    )

    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["implied_prob"] = pd.to_numeric(df["implied_prob"], errors="coerce")
    df["outcome_price"] = pd.to_numeric(df["outcome_price"], errors="coerce")

    df = df.dropna(subset=["implied_prob", "season"]).copy()

    df = df.sort_values(
        ["event_id", "bookmaker_title", "implied_prob"],
        ascending=[True, True, False]
    )
    favorites = df.groupby(["event_id", "bookmaker_title"], as_index=False).first()

    favorites["favorite_won"] = (favorites["outcome_name"] == favorites["winner_team"]).astype(int)

    def profit_per_1_unit_bet(american_odds, won: int):
        """
        Profit (excluding stake) on a 1-unit bet.
        Win:
          +odds => odds/100
          -odds => 100/abs(odds)
        Loss: -1
        """
        if pd.isna(american_odds):
            return np.nan
        o = float(american_odds)
        if won == 1:
            return (o / 100.0) if o > 0 else (100.0 / abs(o))
        return -1.0

    favorites["profit_1u"] = favorites.apply(
        lambda r: profit_per_1_unit_bet(r["outcome_price"], int(r["favorite_won"])),
        axis=1
    )

    return favorites.to_json(date_format="iso", orient="split")


def get_favorites_df() -> pd.DataFrame:
    return pd.read_json(load_favorites_json(), orient="split")


def wilson_interval(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + (z**2) / n
    center = (phat + (z**2) / (2 * n)) / denom
    half = (z * np.sqrt((phat * (1 - phat) + (z**2) / (4 * n)) / n)) / denom
    return (max(0, center - half), min(1, center + half))


def make_figs(fav_df):
    bins = np.linspace(0, 1, 21)
    tmp = fav_df.dropna(subset=["implied_prob"]).copy()
    tmp["prob_bin"] = pd.cut(tmp["implied_prob"], bins=bins, include_lowest=True)

    calib = (
        tmp.groupby("prob_bin", as_index=False)
        .agg(
            implied_prob_mean=("implied_prob", "mean"),
            wins=("favorite_won", "sum"),
            n=("favorite_won", "size"),
            actual_win_rate=("favorite_won", "mean"),
        )
    )
    calib = calib[calib["n"] >= 10].copy()

    ci = calib.apply(lambda r: wilson_interval(int(r["wins"]), int(r["n"])), axis=1)
    calib["ci_low"] = [c[0] for c in ci]
    calib["ci_high"] = [c[1] for c in ci]
    calib["err_low"] = calib["actual_win_rate"] - calib["ci_low"]
    calib["err_high"] = calib["ci_high"] - calib["actual_win_rate"]

    fig_calib = go.Figure()
    fig_calib.add_trace(go.Scatter(
        x=calib["implied_prob_mean"],
        y=calib["actual_win_rate"],
        mode="markers",
        marker=dict(size=np.clip(calib["n"], 10, 120), opacity=0.85),
        error_y=dict(
            type="data",
            symmetric=False,
            array=calib["err_high"],
            arrayminus=calib["err_low"]
        ),
        customdata=np.stack([calib["n"], calib["wins"], calib["ci_low"], calib["ci_high"]], axis=1),
        hovertemplate=(
            "<b>Bin</b><br>"
            "Mean implied: %{x:.3f}<br>"
            "Actual win rate: %{y:.3f}<br>"
            "95% CI: [%{customdata[2]:.3f}, %{customdata[3]:.3f}]<br>"
            "n=%{customdata[0]:.0f}, wins=%{customdata[1]:.0f}"
            "<extra></extra>"
        ),
        name="Binned calibration",
    ))
    fig_calib.add_shape(type="line", x0=0, y0=0, x1=1, y1=1)
    fig_calib.update_layout(
        title="Calibration: Implied Probability vs Actual Win Rate (with 95% CI)",
        xaxis_title="Mean implied probability (favorite)",
        yaxis_title="Actual win rate (favorite)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=50, r=30, t=60, b=50),
    )

    tmp2 = fav_df.dropna(subset=["implied_prob"]).copy()
    tmp2 = tmp2[tmp2["implied_prob"] >= 0.5].copy()
    tmp2["prob_bin"] = pd.cut(tmp2["implied_prob"], bins=np.linspace(0.5, 1.0, 11), include_lowest=True)

    bias = (
        tmp2.groupby("prob_bin", as_index=False)
        .agg(
            mean_implied=("implied_prob", "mean"),
            actual_win_rate=("favorite_won", "mean"),
            n_games=("favorite_won", "size"),
        )
    )
    bias["bias"] = bias["actual_win_rate"] - bias["mean_implied"]

    fig_bias = go.Figure()
    fig_bias.add_trace(go.Bar(
        x=bias["mean_implied"],
        y=bias["bias"],
        customdata=np.stack([bias["n_games"], bias["actual_win_rate"], bias["mean_implied"]], axis=1),
        hovertemplate=(
            "<b>Favorite strength bin</b><br>"
            "Mean implied: %{customdata[2]:.3f}<br>"
            "Actual win rate: %{customdata[1]:.3f}<br>"
            "<b>Bias (actual - implied): %{y:.3f}</b><br>"
            "Games: %{customdata[0]:.0f}"
            "<extra></extra>"
        ),
        name="Bias"
    ))
    fig_bias.add_hline(y=0)
    fig_bias.update_layout(
        title="Bias by Favorite Strength (Actual − Implied)",
        xaxis_title="Mean implied probability (favorite)",
        yaxis_title="Bias (Actual − Implied)",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=50, r=30, t=60, b=50),
    )

    fig_dist = px.histogram(
        fav_df.dropna(subset=["implied_prob"]),
        x="implied_prob",
        color="favorite_won",
        nbins=20,
        title="Distribution of Implied Probabilities by Outcome",
        labels={"implied_prob": "Implied probability", "favorite_won": "Favorite won"},
    )
    fig_dist.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=50, r=30, t=60, b=50),
    )

    tmp3 = fav_df.dropna(subset=["implied_prob", "profit_1u"]).copy()
    tmp3 = tmp3[tmp3["implied_prob"] >= 0.5].copy()
    tmp3["prob_bin"] = pd.cut(tmp3["implied_prob"], bins=np.linspace(0.5, 1.0, 11), include_lowest=True)

    roi = (
        tmp3.groupby("prob_bin", as_index=False)
        .agg(
            mean_implied=("implied_prob", "mean"),
            avg_profit=("profit_1u", "mean"),
            n=("profit_1u", "size"),
        )
    )

    fig_roi = go.Figure()
    fig_roi.add_trace(go.Bar(
        x=roi["mean_implied"],
        y=roi["avg_profit"],
        customdata=np.stack([roi["n"]], axis=1),
        hovertemplate=(
            "Mean implied: %{x:.3f}<br>"
            "Avg profit per bet: %{y:.3f}<br>"
            "n=%{customdata[0]:.0f}"
            "<extra></extra>"
        ),
        name="ROI"
    ))
    fig_roi.add_hline(y=0)
    fig_roi.update_layout(
        title="ROI by Favorite Strength (1 unit bets on favorites)",
        xaxis_title="Mean implied probability (favorite)",
        yaxis_title="Avg profit per bet",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=50, r=30, t=60, b=50),
    )

    return fig_calib, fig_bias, fig_dist, fig_roi


def make_conclusion(fav_df):
    n = len(fav_df)
    if n == 0:
        return "No data for the selected filters."

    win_rate = float(fav_df["favorite_won"].mean())
    mean_p = float(fav_df["implied_prob"].mean())
    roi = float(fav_df["profit_1u"].mean())
    brier = float(np.mean((fav_df["implied_prob"] - fav_df["favorite_won"])**2))

    tmp = fav_df.dropna(subset=["implied_prob"]).copy()
    tmp = tmp[tmp["implied_prob"] >= 0.5].copy()
    tmp["prob_bin"] = pd.cut(tmp["implied_prob"], bins=np.linspace(0.5, 1.0, 11), include_lowest=True)

    bias = (
        tmp.groupby("prob_bin")
        .agg(mean_implied=("implied_prob", "mean"),
             actual=("favorite_won", "mean"),
             n=("favorite_won", "size"))
        .reset_index()
    )
    bias["bias"] = bias["actual"] - bias["mean_implied"]
    bias = bias[bias["n"] >= 10].copy()

    if len(bias) > 0:
        worst = bias.iloc[(bias["bias"].abs()).argmax()]
        worst_txt = (
            f"Biggest deviation shows up around **implied ≈ {worst['mean_implied']:.2f}** "
            f"(bias **{worst['bias']:+.3f}**, n={int(worst['n'])})."
        )
    else:
        worst_txt = "Not enough games per bin to reliably identify the biggest bias bucket (try broader filters)."

    return f"""
### Auto summary (updates with filters)

- You analyzed **{n:,}** favorite bets.
- Favorites won **{win_rate:.1%}** of the time vs an average implied probability of **{mean_p:.1%}**.
- **Calibration** looks {'close' if abs(win_rate-mean_p) < 0.02 else 'off'} overall (difference **{(win_rate-mean_p):+.2%}**).
- **Brier score** (lower is better): **{brier:.3f}**.
- **ROI** from betting 1 unit on every favorite: **{roi:+.3f}** units per bet.

{worst_txt}
"""


app.title = "NBA Betting Odds vs Outcomes"


def kpi_card(title, value, subtitle):
    return html.Div(
        style={
            "backgroundColor": "white",
            "borderRadius": "16px",
            "padding": "14px 16px",
            "boxShadow": "0 6px 18px rgba(0,0,0,0.06)",
            "border": "1px solid rgba(0,0,0,0.06)",
        },
        children=[
            html.Div(title, style={"fontSize": "13px", "fontWeight": "700", "color": "#374151"}),
            html.Div(value, style={"fontSize": "28px", "fontWeight": "900", "color": "#111827", "marginTop": "6px"}),
            html.Div(subtitle, style={"fontSize": "13px", "color": "#6b7280", "marginTop": "2px"}),
        ],
    )


app.layout = html.Div(
    style={
        "minHeight": "100vh",
        "backgroundColor": "#f6f7fb",
        "padding": "28px 16px",
        "fontFamily": '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
        "color": "#111827",
    },
    children=[
        html.Div(
            style={"maxWidth": "1100px", "margin": "0 auto"},
            children=[
                html.Div(
                    style={
                        "backgroundColor": "white",
                        "borderRadius": "16px",
                        "padding": "22px",
                        "boxShadow": "0 6px 18px rgba(0,0,0,0.06)",
                        "border": "1px solid rgba(0,0,0,0.06)",
                        "marginBottom": "14px",
                    },
                    children=[
                        html.H1(
                            "NBA Betting Odds vs Outcomes",
                            style={"margin": "0 0 6px 0", "fontWeight": "900", "letterSpacing": "-0.5px"},
                        ),
                        html.Div(
                            "Filters update all plots + the written conclusion below.",
                            style={"color": "#6b7280"},
                        ),
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "1fr 2fr",
                                "gap": "12px",
                                "marginTop": "14px",
                            },
                            children=[
                                html.Div([
                                    html.Div("Season", style={"fontWeight": "700", "fontSize": "13px", "color": "#374151", "marginBottom": "6px"}),
                                    dcc.Dropdown(
                                        id="season_dd",
                                        options=[{"label": "Loading...", "value": "ALL"}],
                                        value="ALL",
                                        clearable=False,
                                    ),
                                ]),
                                html.Div([
                                    html.Div("Bookmaker", style={"fontWeight": "700", "fontSize": "13px", "color": "#374151", "marginBottom": "6px"}),
                                    dcc.Dropdown(
                                        id="book_dd",
                                        options=[{"label": "Loading...", "value": "ALL"}],
                                        value="ALL",
                                        clearable=False,
                                    ),
                                ]),
                            ],
                        ),
                    ],
                ),

                html.Div(id="kpi_row", style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(4, minmax(0, 1fr))",
                    "gap": "12px",
                    "marginBottom": "14px",
                }),

                html.Div(
                    style={
                        "backgroundColor": "white",
                        "borderRadius": "16px",
                        "padding": "18px",
                        "boxShadow": "0 6px 18px rgba(0,0,0,0.06)",
                        "border": "1px solid rgba(0,0,0,0.06)",
                        "marginBottom": "14px",
                    },
                    children=[dcc.Markdown(id="conclusion_md")],
                ),

                html.Div(
                    style={
                        "backgroundColor": "white",
                        "borderRadius": "16px",
                        "padding": "16px",
                        "boxShadow": "0 6px 18px rgba(0,0,0,0.06)",
                        "border": "1px solid rgba(0,0,0,0.06)",
                        "marginBottom": "14px",
                    },
                    children=[dcc.Graph(id="fig_calib")],
                ),

                html.Div(
                    style={
                        "backgroundColor": "white",
                        "borderRadius": "16px",
                        "padding": "16px",
                        "boxShadow": "0 6px 18px rgba(0,0,0,0.06)",
                        "border": "1px solid rgba(0,0,0,0.06)",
                        "marginBottom": "14px",
                    },
                    children=[dcc.Graph(id="fig_bias")],
                ),

                html.Div(
                    style={
                        "backgroundColor": "white",
                        "borderRadius": "16px",
                        "padding": "16px",
                        "boxShadow": "0 6px 18px rgba(0,0,0,0.06)",
                        "border": "1px solid rgba(0,0,0,0.06)",
                        "marginBottom": "14px",
                    },
                    children=[dcc.Graph(id="fig_dist")],
                ),

                html.Div(
                    style={
                        "backgroundColor": "white",
                        "borderRadius": "16px",
                        "padding": "16px",
                        "boxShadow": "0 6px 18px rgba(0,0,0,0.06)",
                        "border": "1px solid rgba(0,0,0,0.06)",
                        "marginBottom": "14px",
                    },
                    children=[dcc.Graph(id="fig_roi")],
                ),

                html.Div(
                    style={
                        "backgroundColor": "white",
                        "borderRadius": "16px",
                        "padding": "20px",
                        "boxShadow": "0 6px 18px rgba(0,0,0,0.06)",
                        "border": "1px solid rgba(0,0,0,0.06)",
                        "marginBottom": "14px",
                    },
                    children=[
                        html.H3("Glossary", style={"marginTop": "0", "fontWeight": "900", "color": "#111827"}),
                        dcc.Markdown(
                            """
**Implied Probability**  
The probability of an outcome implied by betting odds.  
For example, -150 odds imply about a **60% chance** of winning.

**Favorite**  
The team with the **higher implied probability** (lower payout, expected to win).

**Calibration**  
How closely implied probabilities match real outcomes.  
Perfect calibration means a 60% favorite wins about **60% of the time**.

**Bias (Actual − Implied)**  
The difference between how often favorites actually win and how often sportsbooks imply they should win.  
- Positive → favorites win **more** than expected  
- Negative → favorites are **overvalued**

**ROI (1 Unit)**  
Average profit or loss from betting **1 unit on every favorite**.  
- Positive ROI → profitable strategy  
- Negative ROI → losing strategy

**Brier Score**  
A measure of prediction accuracy (lower is better).  
It penalizes confident but incorrect predictions more heavily.

**Wilson Confidence Interval**  
A statistically robust range showing uncertainty in win-rate estimates, especially helpful when sample sizes are small.
                            """,
                            style={"fontSize": "14px", "color": "#374151"},
                        ),
                    ],
                ),

                html.Div(
                    "Odds: The Odds API • Results: balldontlie • Built with Dash + SQLite",
                    style={"textAlign": "center", "color": "#6b7280", "fontSize": "13px", "padding": "6px 0"},
                ),
            ],
        )
    ]
)


@app.callback(
    Output("season_dd", "options"),
    Output("book_dd", "options"),
    Output("fig_calib", "figure"),
    Output("fig_bias", "figure"),
    Output("fig_dist", "figure"),
    Output("fig_roi", "figure"),
    Output("conclusion_md", "children"),
    Output("kpi_row", "children"),
    Input("season_dd", "value"),
    Input("book_dd", "value"),
)
def update(season_val, book_val):
    try:
        favorites = get_favorites_df()
    except Exception as e:
        empty_fig = go.Figure()
        msg = f"### Error loading database\n\n`{type(e).__name__}: {e}`\n\nCheck that `sports_pipeline.db` exists in your home directory and has the required tables."
        return (
            [{"label": "All seasons", "value": "ALL"}],
            [{"label": "All bookmakers", "value": "ALL"}],
            empty_fig, empty_fig, empty_fig, empty_fig,
            msg,
            [kpi_card("Status", "DB error", "see message above")]
        )

    # Build dropdown options from the data every time (cheap)
    all_seasons = sorted([int(x) for x in favorites["season"].dropna().unique()])
    all_books = sorted([str(x) for x in favorites["bookmaker_title"].dropna().unique()])
    season_opts = [{"label": "All seasons", "value": "ALL"}] + [{"label": str(s), "value": s} for s in all_seasons]
    book_opts = [{"label": "All bookmakers", "value": "ALL"}] + [{"label": b, "value": b} for b in all_books]

    # Filter
    f = favorites.copy()
    if season_val not in (None, "ALL"):
        f = f[f["season"] == int(season_val)]
    if book_val not in (None, "ALL"):
        f = f[f["bookmaker_title"] == str(book_val)]

    # Make outputs
    fig_calib, fig_bias, fig_dist, fig_roi = make_figs(f)
    conclusion = make_conclusion(f)

    n = len(f)
    kpis = [
        kpi_card("Bets", f"{n:,}", "favorite bets analyzed"),
        kpi_card("Bookmakers", f"{f.bookmaker_title.nunique()}", "books included"),
        kpi_card("Favorite Win %", f"{f.favorite_won.mean():.1%}" if n else "—", "actual win rate"),
        kpi_card("ROI (1u)", f"{f.profit_1u.mean():+.3f}" if n else "—", "avg profit per bet"),
    ]

    return season_opts, book_opts, fig_calib, fig_bias, fig_dist, fig_roi, conclusion, kpis


if __name__ == "__main__":
    app.run_server(debug=True)

server = app.server
