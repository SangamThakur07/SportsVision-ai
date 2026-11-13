"""
Tactical Match Roster & Lineup Normalizer Module
Maps raw multi-object tracking IDs (ByteTrack #1..#N) to official 11 vs 11
tactical football match rosters (White #1..#11 vs Red #1..#11) with positional roles.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


# Standard football tactical positions by depth
ROLE_DEFINITIONS = [
    ("Goalkeeper", "GK"),
    ("Right Back", "RB"),
    ("Center Back", "CB"),
    ("Center Back", "CB"),
    ("Left Back", "LB"),
    ("Defensive Midfielder", "CDM"),
    ("Right Midfielder", "RM"),
    ("Central Midfielder", "CM"),
    ("Center Forward", "ST"),
    ("Attacking Midfielder", "CAM"),
    ("Left Winger", "LW")
]


class TacticalRoster:
    """
    Normalizes raw computer vision track IDs into an official 22-player match lineup
    (11 Team White vs 11 Team Red), categorizing referees and sideline staff separately.
    """

    @staticmethod
    def build_tactical_roster(
        df_kinematics: pd.DataFrame,
        df_tracking: Optional[pd.DataFrame] = None,
        player_teams: Optional[Dict[int, str]] = None,
        scale_m: float = 105.0 / 1280.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build an official 22-player match roster and separate officials/peripheral personnel.

        Args:
            df_kinematics: DataFrame with kinematics metrics (Player ID, Total Distance, Speed, etc.).
            df_tracking: Optional DataFrame with frame-by-frame (x, y) coordinates.
            player_teams: Optional dictionary mapping integer track_id -> 'Team White' or 'Team Red'.
            scale_m: Metric scaling factor (meters per pixel).

        Returns:
            Tuple of (match_roster_df [up to 22 players], officials_df [referees & peripheral]).
        """
        if df_kinematics is None or df_kinematics.empty:
            empty_df = pd.DataFrame(columns=[
                "Tactical Jersey", "Tactical Role", "Role Abbr", "Team", "Jersey Number",
                "Raw Track ID", "Distance (m)", "Top Speed (km/h)", "Avg Speed (km/h)",
                "Tracking Duration (s)", "Frames Tracked", "Workrate Category"
            ])
            return empty_df, empty_df.copy()

        df = df_kinematics.copy()

        # Extract integer track_id from 'Player ID' (e.g. '#5' -> 5)
        if "track_id" not in df.columns:
            if "Player ID" in df.columns:
                df["track_id"] = df["Player ID"].astype(str).str.replace("#", "").str.strip()
                df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce").fillna(0).astype(int)
            else:
                df["track_id"] = range(1, len(df) + 1)

        # Standardize metric conversion
        if "Distance (m)" not in df.columns and "Total Distance (px)" in df.columns:
            df["Distance (m)"] = (df["Total Distance (px)"] * scale_m).round(1)
        elif "Distance (m)" not in df.columns:
            df["Distance (m)"] = 0.0

        if "Top Speed (km/h)" not in df.columns and "Max Speed (px/s)" in df.columns:
            df["Top Speed (km/h)"] = (df["Max Speed (px/s)"] * scale_m * 3.6).round(1)
        elif "Top Speed (km/h)" not in df.columns:
            df["Top Speed (km/h)"] = 0.0

        if "Avg Speed (km/h)" not in df.columns and "Average Speed (px/s)" in df.columns:
            df["Avg Speed (km/h)"] = (df["Average Speed (px/s)"] * scale_m * 3.6).round(1)
        elif "Avg Speed (km/h)" not in df.columns:
            df["Avg Speed (km/h)"] = 0.0

        if "Frames Tracked" not in df.columns:
            df["Frames Tracked"] = 10

        if "Tracking Duration (s)" not in df.columns:
            df["Tracking Duration (s)"] = (df["Frames Tracked"] / 25.0).round(2)

        # Workrate classification
        df["Workrate Category"] = df["Top Speed (km/h)"].apply(
            lambda s: "⚡ Sprinting (>20 km/h)" if s >= 20.0 else (
                "🏃 Running (12-20 km/h)" if s >= 12.0 else "🚶 Positional Zone (<12 km/h)"
            )
        )

        # Pitch coordinate extraction for tactical spatial sorting
        if df_tracking is not None and not df_tracking.empty:
            mean_coords = df_tracking.groupby("track_id")[["x", "y"]].mean().reset_index()
            df = pd.merge(df, mean_coords, on="track_id", how="left")
        if "x" not in df.columns:
            df["x"] = 640.0
        if "y" not in df.columns:
            df["y"] = 360.0

        # Team assignments
        if not player_teams:
            player_teams = {tid: ("Team White" if tid % 2 == 1 else "Team Red") for tid in df["track_id"]}

        df["Team"] = df["track_id"].map(player_teams).fillna("Team White")

        match_roster_list: List[pd.DataFrame] = []
        officials_list: List[pd.DataFrame] = []

        team_configs = [
            ("Team White", "White", True),   # Ascending x (defending left)
            ("Team Red", "Red", False)       # Descending x (defending right)
        ]

        for team_name, prefix, asc_x in team_configs:
            team_df = df[df["Team"] == team_name].copy()
            if team_df.empty:
                continue

            # Prioritize players by tracking continuity & presence
            team_df = team_df.sort_values(by=["Frames Tracked", "Distance (m)"], ascending=[False, False])

            # Select the top 11 active pitch players for the match roster
            main_11 = team_df.head(11).copy()
            overflow = team_df.iloc[11:].copy()

            # Sort the 11 players by pitch depth (x coordinate) to assign natural football positions
            main_11 = main_11.sort_values(by="x", ascending=asc_x).reset_index(drop=True)

            tactical_jerseys = []
            tactical_roles = []
            role_abbrs = []
            jersey_nums = []

            for i in range(len(main_11)):
                j_num = i + 1
                role_desc, role_abbr = ROLE_DEFINITIONS[min(i, len(ROLE_DEFINITIONS) - 1)]
                tactical_jerseys.append(f"{prefix} #{j_num} ({role_abbr})")
                tactical_roles.append(role_desc)
                role_abbrs.append(role_abbr)
                jersey_nums.append(j_num)

            main_11["Tactical Jersey"] = tactical_jerseys
            main_11["Tactical Role"] = tactical_roles
            main_11["Role Abbr"] = role_abbrs
            main_11["Jersey Number"] = jersey_nums
            main_11["Raw Track ID"] = main_11["track_id"]
            match_roster_list.append(main_11)

            # Categorize overflow detections as peripheral / sideline
            if not overflow.empty:
                overflow = overflow.reset_index(drop=True)
                overflow["Tactical Jersey"] = [f"{prefix} Sub/Staff #{i+12}" for i in range(len(overflow))]
                overflow["Tactical Role"] = "Sideline / Substitute Tracklet"
                overflow["Role Abbr"] = "SUB"
                overflow["Jersey Number"] = list(range(12, 12 + len(overflow)))
                overflow["Raw Track ID"] = overflow["track_id"]
                officials_list.append(overflow)

        match_roster = pd.concat(match_roster_list, ignore_index=True) if match_roster_list else pd.DataFrame()

        # Handle officials (detect tracks that don't belong to either team or extra tracks)
        officials_df = pd.concat(officials_list, ignore_index=True) if officials_list else pd.DataFrame()

        # Format clean columns for match roster
        cols = [
            "Tactical Jersey", "Tactical Role", "Role Abbr", "Team", "Jersey Number",
            "Raw Track ID", "Distance (m)", "Top Speed (km/h)", "Avg Speed (km/h)",
            "Tracking Duration (s)", "Frames Tracked", "Workrate Category"
        ]
        if not match_roster.empty:
            for c in cols:
                if c not in match_roster.columns:
                    match_roster[c] = None
            match_roster = match_roster[cols]

        if not officials_df.empty:
            for c in cols:
                if c not in officials_df.columns:
                    officials_df[c] = None
            officials_df = officials_df[cols]

        return match_roster, officials_df

    @staticmethod
    def get_player_display_mapping(
        match_roster: pd.DataFrame,
        officials_df: Optional[pd.DataFrame] = None
    ) -> Dict[int, str]:
        """
        Generate mapping from integer track_id -> human-friendly tactical display name.
        Example: 5 -> '⚪ White #9 (Center Forward)'
        """
        mapping: Dict[int, str] = {}

        if match_roster is not None and not match_roster.empty:
            for _, r in match_roster.iterrows():
                tid = int(r["Raw Track ID"])
                team_icon = "⚪" if "White" in str(r["Team"]) else "🔴"
                mapping[tid] = f"{team_icon} {r['Tactical Jersey']} — {r['Tactical Role']}"

        if officials_df is not None and not officials_df.empty:
            for _, r in officials_df.iterrows():
                tid = int(r["Raw Track ID"])
                mapping[tid] = f"🏁 Official / Staff (Track #{tid})"

        return mapping
