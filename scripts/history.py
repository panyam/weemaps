"""
Weewar Game JSON Parser

Parses Weewar game data into typed Python objects.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, Any, Tuple
from enum import Enum
import json
import os


# =============================================================================
# Coordinate Conversion
# =============================================================================

# Global offset configuration (set from command line args)
_coord_offset_dc = 0  # delta col (applied to x/col before conversion)
_coord_offset_dr = 0  # delta row (applied to y/row before conversion)
_offset_type = "oddr" # "evenr" or "oddr" offset (default: oddr)

def isOddR() -> bool:
    return _offset_type == "oddr"

def set_coord_offsets(dc: int, dr: int) -> None:
    """Set the coordinate offsets for Weewar to local conversion."""
    global _coord_offset_dc, _coord_offset_dr
    _coord_offset_dc = dc
    _coord_offset_dr = dr


def set_offset_type(offset_type: str) -> None:
    """Set whether to use Odd-R coordinate system for Weewar input."""
    global _offset_type
    _offset_type = offset_type


def weewar_to_axial(col: int, row: int, dr: int, dc: int, debug=False) -> Tuple[int, int]:
    """Convert Weewar offset (col, row) to our Axial (q, r) system.

    By default, assumes Weewar uses Even-R offset coordinates.
    Use --odd flag if Weewar coordinates are in Odd-R format.

    The --drdc option allows translating between Weewar's coordinate origin
    and our game's coordinate origin.
    """
    # Apply offset translation (Weewar coords may be shifted from our game)
    sr,sc = row,col

    col = col + dc
    row = row + dr
    # if row & 1 == 1: col += 1       # should this happen in weewar row or our row?

    xodd = col - (row-(row&1))//2
    xeven = col - (row+(row&1))//2
    x = xodd if isOddR() else xeven
    z = row
    y = -x -z
    q,r = x,z

    if debug:
        print("WRow, WCol, RowAfterDr, ColAfterDc, EvenQ, EvenR, OddQ, OddR", (sr, sc), (row, col), (xeven, z), (xodd, z))

    # Cube to Axial: q=x, r=z
    return (q, r)


def coord_str(col: int, row: int, dr: int, dc: int) -> str:
    """Convert Weewar coordinates to Q,R string format."""
    q, r = weewar_to_axial(col, row, dr, dc)
    return f"{q},{r}"


# =============================================================================
# Mapping for Unit/Terrain Labels
# =============================================================================

# Global mapping data (loaded from mapping.json)
_unit_mapping: dict = {}
_terrain_mapping: dict = {}


def load_mapping(mapping_file: str) -> bool:
    """Load unit and terrain mappings from a JSON file.

    Returns True if loaded successfully, False otherwise.
    """
    global _unit_mapping, _terrain_mapping

    # Expand ~ to home directory
    mapping_path = os.path.expanduser(mapping_file)

    if not os.path.exists(mapping_path):
        return False

    try:
        with open(mapping_path, "r") as f:
            data = json.load(f)

        _unit_mapping = data.get("units", {})
        _terrain_mapping = data.get("terrains", {})
        return True
    except (json.JSONDecodeError, IOError):
        return False


def get_unit_label(unit_id: int) -> str:
    """Get a formatted unit label. Returns 'ID:Name' if mapping exists, else just 'ID'."""
    str_id = str(unit_id)
    if str_id in _unit_mapping:
        name = _unit_mapping[str_id].get("name", "")
        if name:
            return f"{unit_id}:{name}"
    return str(unit_id)


def get_terrain_label(terrain_id: int) -> str:
    """Get a formatted terrain label. Returns 'ID:Name' if mapping exists, else just 'ID'."""
    str_id = str(terrain_id)
    if str_id in _terrain_mapping:
        name = _terrain_mapping[str_id].get("name", "")
        if name:
            return f"{terrain_id}:{name}"
    return str(terrain_id)


# =============================================================================
# Enums
# =============================================================================

class ActionType(Enum):
    BUILD = "b"
    MOVE = "m"
    ATTACK = "a"
    HEAL = "h"


# =============================================================================
# Action Classes
# =============================================================================

@dataclass
class BuildAction:
    """A unit build/creation action."""
    action_type: ActionType = field(default=ActionType.BUILD, init=False)
    x: int  # col in Even-R
    y: int  # row in Even-R
    unit_type: int
    tone: int

    def __str__(self) -> str:
        return f"Build unit type {get_unit_label(self.unit_type)} at ({self.x}, {self.y})"

    def to_ww_command(self) -> str:
        """Generate ww CLI command for this build action."""
        pos = coord_str(self.x, self.y, _coord_offset_dr, _coord_offset_dc)
        return f"ww build t:{pos} {self.unit_type}"

    def to_ww_asserts(self) -> list[str]:
        """Generate ww assert commands to validate this build."""
        pos = coord_str(self.x, self.y, _coord_offset_dr, _coord_offset_dc)
        return [
            f"ww assert exists unit {pos}",
            f"ww assert unit {pos} [type eq {self.unit_type}]",
        ]


@dataclass
class MoveAction:
    """A unit movement action."""
    action_type: ActionType = field(default=ActionType.MOVE, init=False)
    from_x: int  # col in Even-R
    from_y: int  # row in Even-R
    to_x: int    # col in Even-R
    to_y: int    # row in Even-R

    def __str__(self) -> str:
        return f"Move from ({self.from_x}, {self.from_y}) to ({self.to_x}, {self.to_y})"

    def to_ww_command(self) -> str:
        """Generate ww CLI command for this move action."""
        from_pos = coord_str(self.from_x, self.from_y, _coord_offset_dr, _coord_offset_dc)
        to_pos = coord_str(self.to_x, self.to_y, _coord_offset_dr, _coord_offset_dc)
        return f"ww move {from_pos} {to_pos}"

    def to_ww_asserts(self) -> list[str]:
        """Generate ww assert commands to validate this move."""
        from_pos = coord_str(self.from_x, self.from_y, _coord_offset_dr, _coord_offset_dc)
        to_pos = coord_str(self.to_x, self.to_y, _coord_offset_dr, _coord_offset_dc)
        return [
            f"ww assert exists unit {to_pos}",
            f"ww assert notexists unit {from_pos}",
        ]


@dataclass
class AttackAction:
    """A combat/attack action."""
    action_type: ActionType = field(default=ActionType.ATTACK, init=False)
    attacker_x: int  # col in Even-R
    attacker_y: int  # row in Even-R
    target_x: int    # col in Even-R
    target_y: int    # row in Even-R
    damage_dealt: int
    damage_received: int
    combat_details: Optional[list] = None
    extra_info: Optional[Any] = None

    def __str__(self) -> str:
        result = f"Attack from ({self.attacker_x}, {self.attacker_y}) -> ({self.target_x}, {self.target_y})"
        result += f" | Dealt {self.damage_dealt} dmg"
        if self.damage_received > 0:
            result += f", Received {self.damage_received} counter-dmg"
        return result

    def to_ww_command(self) -> str:
        """Generate ww CLI command for this attack action."""
        attacker_pos = coord_str(self.attacker_x, self.attacker_y, _coord_offset_dr, _coord_offset_dc)
        target_pos = coord_str(self.target_x, self.target_y, _coord_offset_dr, _coord_offset_dc)
        return f"ww attack {attacker_pos} {target_pos}"

    def to_ww_asserts(self) -> list[str]:
        """Generate ww assert commands to validate this attack.

        Note: Health assertions use <= because the target may have had less health
        before the attack, or may have been destroyed.
        """
        target_pos = coord_str(self.target_x, self.target_y, _coord_offset_dr, _coord_offset_dc)
        attacker_pos = coord_str(self.attacker_x, self.attacker_y, _coord_offset_dr, _coord_offset_dc)
        asserts = []

        # Check defender took damage (health reduced by damage_dealt)
        if self.damage_dealt > 0:
            max_health_after = 10 - self.damage_dealt
            if max_health_after <= 0:
                # Unit should be destroyed
                asserts.append(f"ww assert notexists unit {target_pos}")
            else:
                asserts.append(f"ww assert unit {target_pos} [health lte {max_health_after}]")

        # Check attacker took counter-damage
        if self.damage_received > 0:
            max_attacker_health = 10 - self.damage_received
            if max_attacker_health <= 0:
                asserts.append(f"ww assert notexists unit {attacker_pos}")
            else:
                asserts.append(f"ww assert unit {attacker_pos} [health lte {max_attacker_health}]")

        return asserts


@dataclass
class HealAction:
    """A heal/hold action."""
    action_type: ActionType = field(default=ActionType.HEAL, init=False)
    x: int  # col in Even-R
    y: int  # row in Even-R

    def __str__(self) -> str:
        return f"Heal/Hold at ({self.x}, {self.y})"

    def to_ww_command(self) -> str:
        """Generate ww CLI command for this heal action.

        Note: Heal is typically implicit (unit heals when not moving/attacking).
        Return a comment instead of a command.
        """
        pos = coord_str(self.x, self.y, _coord_offset_dr, _coord_offset_dc)
        return f"# Heal/Hold at {pos}"

    def to_ww_asserts(self) -> list[str]:
        """No assertions needed for heal action."""
        return []


# Union type for all actions
Action = Union[BuildAction, MoveAction, AttackAction, HealAction]


# =============================================================================
# Game Structure Classes
# =============================================================================

@dataclass
class PlayerTurn:
    """A single player's actions within a round."""
    player_index: int
    actions: list[Action] = field(default_factory=list)

    def __str__(self) -> str:
        if not self.actions:
            return f"  Player {self.player_index}: No actions"
        lines = [f"  Player {self.player_index}: {len(self.actions)} action(s)"]
        for action in self.actions:
            lines.append(f"    - {action}")
        return "\n".join(lines)


@dataclass
class Round:
    """A complete round containing all player turns."""
    round_number: int
    player_turns: list[PlayerTurn] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"Round {self.round_number}:"]
        for turn in self.player_turns:
            lines.append(str(turn))
        return "\n".join(lines)


@dataclass
class GameState:
    """The game metadata and turn history."""
    started: bool
    finished: bool
    ranked: bool
    time_limit: int
    current_round: int
    rounds: list[Round] = field(default_factory=list)

    def __str__(self) -> str:
        status = "Finished" if self.finished else "In Progress"
        return f"Game ({status}) - Round {self.current_round}, Ranked: {self.ranked}"


# =============================================================================
# Unit and Board State Classes
# =============================================================================

@dataclass
class Unit:
    """A unit on the game board."""
    x: int
    y: int
    unit_id: int
    owner_player: int
    tile_owner: Optional[int]
    health: int
    attack_count: int
    available_actions: list[list[str]]
    tone: int

    def __str__(self) -> str:
        return f"Unit {get_unit_label(self.unit_id)} (P{self.owner_player}) at ({self.x}, {self.y}) HP:{self.health}"


@dataclass
class Player:
    """Player information."""
    index: int
    username: str
    player_id: Optional[int]
    is_active: Optional[bool]
    is_defeated: Optional[bool]
    has_accepted: Optional[bool]
    is_online: Optional[bool]
    coins: Optional[int]
    badge_id: Optional[int]
    badge_title: Optional[str]

    def __str__(self) -> str:
        status = ""
        if self.is_active:
            status = " (Active)"
        elif self.is_defeated:
            status = " (Defeated)"
        return f"Player {self.index}: {self.username}{status}"


@dataclass
class ChatMessage:
    """A chat message in the game."""
    timestamp: int
    player_id: Optional[int]
    message: str

    def __str__(self) -> str:
        return f"[{self.timestamp}] {self.message}"


# =============================================================================
# Top-Level Container
# =============================================================================

@dataclass
class WeewarGame:
    """Complete parsed Weewar game data."""
    timestamp: int
    game_state: GameState
    players: list[Player]
    units: list[Unit]
    chat_messages: list[ChatMessage]

    def get_player_by_index(self, index: int) -> Optional[Player]:
        """Get a player by their array index (0-indexed)."""
        for player in self.players:
            if player.index == index:
                return player
        return None

    def get_player_by_unit_owner(self, owner_player: int) -> Optional[Player]:
        """
        Get a player by their unit owner ID (1-indexed).
        
        The coordinate data uses 1-indexed player IDs, while the
        player array is 0-indexed. This helper handles the conversion.
        """
        return self.get_player_by_index(owner_player - 1)

    def print_game_log(self) -> None:
        """Print a human-readable game log."""
        print("=" * 60)
        print("WEEWAR GAME LOG")
        print("=" * 60)
        print(f"\n{self.game_state}\n")

        print("Players:")
        for player in self.players:
            print(f"  {player}")
        print()

        print("Turn History:")
        print("-" * 40)
        for round_data in self.game_state.rounds:
            print(f"\n{round_data}")
        print()

    def print_ww_commands(self, game_id: str = "testgame") -> None:
        """Print ww CLI commands with assertions to replay and validate the game."""
        print("#!/bin/bash")
        print("set -e  # Exit on first error")
        print()
        print("# Error handler - show which command failed")
        print("trap 'echo \"FAILED at line $LINENO: $BASH_COMMAND\" >&2' ERR")
        print()
        print("# Weewar Game Replay Script")
        print(f"# Generated from game history")
        print(f"# Players: {', '.join(p.username for p in self.players)}")
        print()
        print("# Give user the chance to set the world id here")
        print('WORLD_ID="ENTER_YOUR_WORLD_ID_HERE"')
        print("# Look for the line that has \"export WEEWAR_GAME_ID=....\" here so we can extract gameid from it")
        print('gameIdLine=$(ww new $WORLD_ID | grep "export WEEWAR_GAME_ID")')
        print('gameId=$(echo $gameIdLine | sed -e "s/.*export.WEEWAR_GAME_ID=//g")')
        print("export WEEWAR_GAME_ID=$gameId")
        print("export WEEWAR_CONFIRM=false")
        print("echo Created game for testing: $WEEWAR_GAME_ID")
        print()

        # Process each round
        for round_data in self.game_state.rounds:
            print(f"# {'='*60}")
            print(f"# Round {round_data.round_number}")
            print(f"# {'='*60}")
            print()

            for player_turn in round_data.player_turns:
                player = self.get_player_by_index(player_turn.player_index)
                player_name = player.username if player else f"Player {player_turn.player_index}"

                print(f"# --- Player {player_turn.player_index} ({player_name}) ---")

                if not player_turn.actions:
                    print("# No actions")
                else:
                    for action in player_turn.actions:
                        # Print command
                        cmd = action.to_ww_command()
                        print(cmd)

                        # Print assertions
                        for assert_cmd in action.to_ww_asserts():
                            print(assert_cmd)

                # End turn
                print("ww endturn")
                print()

        # Final board state assertions
        print(f"# {'='*60}")
        print("# Final Board State (from Weewar coordinate section)")
        print(f"# {'='*60}")
        print()
        print("# Weewar (col,row) -> Our (Q,R) mapping:")
        for unit in self.units:
            weewar_pos = f"({unit.x},{unit.y})"
            q, r = weewar_to_axial(unit.x, unit.y, _coord_offset_dr, _coord_offset_dc)
            player = self.get_player_by_unit_owner(unit.owner_player)
            player_name = player.username if player else f"P{unit.owner_player}"
            print(f"# Weewar {weewar_pos:8} -> Q,R=({q:3},{r:3}) : Unit type {get_unit_label(unit.unit_id)}, {player_name}")
        print()
        print("# Final board state assertions:")
        for unit in self.units:
            pos = coord_str(unit.x, unit.y, _coord_offset_dr, _coord_offset_dc)
            # Player in assertions uses 1-indexed (matching game state)
            print(f"ww assert unit {pos} [type eq {unit.unit_id}, player eq {unit.owner_player}, health eq {unit.health}]")

        print()
        print("echo 'All assertions passed!'")


# =============================================================================
# Parser Functions
# =============================================================================

def parse_action(raw_action: list) -> Action:
    """Parse a raw action array into a typed Action object."""
    action_code = raw_action[0]

    if action_code == "b":
        return BuildAction(
            x=raw_action[1],
            y=raw_action[2],
            unit_type=raw_action[3],
            tone=raw_action[4]
        )
    elif action_code == "m":
        return MoveAction(
            from_x=raw_action[1],
            from_y=raw_action[2],
            to_x=raw_action[3],
            to_y=raw_action[4]
        )
    elif action_code == "a":
        return AttackAction(
            attacker_x=raw_action[1],
            attacker_y=raw_action[2],
            target_x=raw_action[3],
            target_y=raw_action[4],
            damage_dealt=raw_action[5],
            damage_received=raw_action[6],
            combat_details=raw_action[7] if len(raw_action) > 7 else None,
            extra_info=raw_action[8] if len(raw_action) > 8 else None
        )
    elif action_code == "h":
        return HealAction(
            x=raw_action[1],
            y=raw_action[2]
        )
    else:
        raise ValueError(f"Unknown action type: {action_code}")


def parse_turns(raw_turns: list) -> list[Round]:
    """Parse the nested turn structure into Round objects."""
    rounds = []

    for round_idx, round_data in enumerate(raw_turns, start=1):
        player_turns = []

        for player_idx, player_actions in enumerate(round_data):
            actions = [parse_action(action) for action in player_actions]
            player_turns.append(PlayerTurn(
                player_index=player_idx,
                actions=actions
            ))

        rounds.append(Round(
            round_number=round_idx,
            player_turns=player_turns
        ))

    return rounds


def parse_game_state(game_data: dict) -> GameState:
    """Parse the game section into a GameState object."""
    keys = game_data["key"]
    values = game_data["data"][0]  # First (and only) game entry

    # Map keys to values
    data = dict(zip(keys, values))

    # Parse turns
    rounds = parse_turns(data.get("turn", []))

    return GameState(
        started=data.get("started", False),
        finished=data.get("finished", False),
        ranked=data.get("ranked", False),
        time_limit=data.get("timeLimit", 0),
        current_round=data.get("round", 0),
        rounds=rounds
    )


def parse_unit_data(raw_data: Union[list, dict], keys: list[str]) -> dict:
    """Normalize unit data from either array or object format."""
    if isinstance(raw_data, list):
        return dict(zip(keys, raw_data))
    else:
        # Object format uses string indices
        return {keys[int(k)]: v for k, v in raw_data.items()}


def parse_units(coordinate_data: dict) -> list[Unit]:
    """Parse the coordinate section into Unit objects."""
    keys = coordinate_data["key"]
    units = []

    for raw_unit in coordinate_data["data"]:
        data = parse_unit_data(raw_unit, keys)

        units.append(Unit(
            x=data.get("x", 0),
            y=data.get("y", 0),
            unit_id=data.get("unitId", 0),
            owner_player=data.get("unitPlayer", 0),
            tile_owner=data.get("tilePlayer"),
            health=data.get("unitHealth", 0),
            attack_count=data.get("unitAttackCount", 0),
            available_actions=data.get("unitProgression", []),
            tone=data.get("unitTone", 0)
        ))

    return units


def parse_players(player_data: dict) -> list[Player]:
    """Parse the player section into Player objects."""
    keys = player_data["key"]
    players = []

    for idx, raw_player in enumerate(player_data["data"]):
        data = dict(zip(keys, raw_player))

        players.append(Player(
            index=idx,
            username=data.get("username", "Unknown"),
            player_id=data.get("playerId"),
            is_active=data.get("active"),
            is_defeated=data.get("defeated"),
            has_accepted=data.get("accepted"),
            is_online=data.get("online"),
            coins=data.get("coin"),
            badge_id=data.get("badgeId"),
            badge_title=data.get("badgeTitle")
        ))

    return players


def parse_chat(chat_data: dict) -> list[ChatMessage]:
    """Parse the chat section into ChatMessage objects."""
    keys = chat_data["key"]
    messages = []

    for raw_message in chat_data["data"]:
        data = dict(zip(keys, raw_message))

        messages.append(ChatMessage(
            timestamp=data.get("time", 0),
            player_id=data.get("playerId"),
            message=data.get("message", "")
        ))

    return messages


def parse_weewar_json(json_data: Union[str, dict]) -> WeewarGame:
    """
    Parse a Weewar game JSON into typed Python objects.

    Args:
        json_data: Either a JSON string or already-parsed dict

    Returns:
        WeewarGame: Fully typed game data object
    """
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data

    return WeewarGame(
        timestamp=data.get("time", 0),
        game_state=parse_game_state(data["game"]),
        players=parse_players(data["player"]),
        units=parse_units(data["coordinate"]),
        chat_messages=parse_chat(data["chat"])
    )


# =============================================================================
# Main / Demo
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse and analyze Weewar game JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python history.py game.json              # Print human-readable game log
  python history.py game.json --ww         # Print ww CLI commands
  python history.py game.json --ww --game-id mygame   # With custom game ID
  python history.py game.json --ww --drdc=1,-2        # With coordinate offset (dr=1, dc=-2)
  python history.py game.json --ww --drdc=-3,-4       # Use = for negative values
  python history.py game.json --ww --even             # Assume Weewar uses Even-R coords
  python history.py game.json --mapping ~/my-mapping.json  # Use custom mapping file
"""
    )
    parser.add_argument("file", help="Weewar game JSON file")
    parser.add_argument("--ww", action="store_true",
                        help="Output ww CLI commands instead of human-readable log")
    parser.add_argument("--game-id", default="testgame",
                        help="Game ID to use in ww commands (default: testgame)")
    parser.add_argument("--drdc", type=str, default=None,
                        help="Coordinate offset as 'dr,dc' (use = for negatives: --drdc=-3,-4)")
    parser.add_argument("--even", action="store_true",
                        help="Assume Weewar coordinates are Even-R (default: Odd-R)")
    parser.add_argument("--mapping", type=str, default="~/mapping.json",
                        help="Path to mapping.json for unit/terrain labels (default: ~/mapping.json)")

    args = parser.parse_args()

    # Set Even-R mode if specified (default is Odd-R)
    if args.even:
        set_offset_type("evenr")

    # Load mapping file for unit/terrain labels
    if not load_mapping(args.mapping):
        # Silently continue without labels if mapping file not found
        pass

    # Parse and set coordinate offsets if specified
    if args.drdc:
        try:
            parts = args.drdc.split(",")
            if len(parts) != 2:
                raise ValueError("must have exactly 2 values")
            dr = int(parts[0].strip())
            dc = int(parts[1].strip())
            set_coord_offsets(dc, dr)
        except ValueError as e:
            parser.error(f"--drdc must be 'dr,dc' format (e.g., '1,-2'): {e}")

    with open(args.file, "r") as f:
        json_data = f.read()

    # Parse game data
    game = parse_weewar_json(json_data)

    if args.ww:
        # Output ww CLI commands with assertions
        game.print_ww_commands(game_id=args.game_id)
    else:
        # Human-readable output
        game.print_game_log()

        # Show current board state
        print("\nCurrent Board State:")
        print("-" * 40)
        for unit in game.units:
            player = game.get_player_by_unit_owner(unit.owner_player)
            player_name = player.username if player else f"Player {unit.owner_player}"
            print(f"  {unit} ({player_name})")
