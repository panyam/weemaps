"""
World Shifter

Shifts a world and all units in it by dq/dr
"""

from dataclasses import dataclass, field
from typing import Optional, Union, Any, Tuple
from enum import Enum
import json

# =============================================================================
# Main / Demo
# =============================================================================

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(
        description="Shift a world and all units by dQ,dR.  Ie all Q/R coordinates are shifted",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python shiftworld.py worlddata.json dQ dR     # Shifts all coordinates in worlddata.json by dQ/dR
"""
    )
    parser.add_argument("file", help="Weewar world JSON file to shift.  Will write <file>.shifted.json")
    parser.add_argument("dQ", help="Shift Q by dQ")
    parser.add_argument("dR", help="Shift R by dR")

    args = parser.parse_args()

    with open(args.file, "r") as f:
        json_data = f.read()

        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data

        dQ,dR = int(args.dQ), int(args.dR)
        tiles = data.get("tiles_map")
        units = data.get("units_map")

        shifted_tiles = {}
        shifted_units = {}
        for tile in tiles.values():
            nq = tile["q"] + dQ
            nr = tile["r"] + dR
            newtile = tile.copy()
            tile["q"] = nq
            tile["r"] = nr
            key = f"{nq},{nr}"
            shifted_tiles[key] = newtile

        for unit in units.values():
            nq = unit["q"] + dQ
            nr = unit["r"] + dR
            newunit = unit.copy()
            unit["q"] = nq
            unit["r"] = nr
            key = f"{nq},{nr}"
            shifted_units[key] = newunit

        outpath = args.file + ".shifted"
        with open(outpath, "w") as outfile:
            print("Writing shifted world to: ", outpath)
            outfile.write(json.dumps({"tiles_map": shifted_tiles, "units_map": shifted_units}, indent=2))

