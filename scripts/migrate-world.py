#!/usr/bin/env python3
"""
migrate-world.py - Migrate a world from one service to another

Usage:
    migrate-world.py <source-url> <dest-url> [--token TOKEN]

Examples:
    migrate-world.py http://localhost:6060/api/v1/worlds/Desert http://localhost:8080/api/v1/worlds/Desert
    migrate-world.py http://localhost:6060/api/v1/worlds/Desert https://server/api/v1/worlds/DesertMap --token $TOKEN
"""

import argparse
import re
import sys
import requests


def extract_world_id(url: str) -> str:
    """Extract world ID from URL like .../api/v1/worlds/<id>"""
    match = re.search(r'/api/v1/worlds/([^/]+)/?$', url)
    if not match:
        raise ValueError(f"Could not extract world ID from URL: {url}")
    return match.group(1)


def get_base_url(url: str) -> str:
    """Get base URL (without the world ID) for POST requests"""
    return re.sub(r'/[^/]+/?$', '', url)


def migrate_world(source_url: str, dest_url: str, token: str = None):
    dest_id = extract_world_id(dest_url)
    base_url = get_base_url(dest_url)

    # Fetch from source
    print(f"Fetching world from: {source_url}")
    resp = requests.get(source_url)
    if resp.status_code != 200:
        print(f"Error fetching from source: {resp.status_code} - {resp.text}")
        sys.exit(1)

    data = resp.json()
    if 'error' in data:
        print(f"Error from source: {data.get('message', data['error'])}")
        sys.exit(1)

    # Transform GetWorldResponse to CreateWorldRequest format
    world = data.get('world', {})
    world_data = data.get('worldData', None)
    world['id'] = dest_id

    request_payload = {
        'world': world,
        'worldData': world_data
    }

    # Show what we're migrating
    tile_count = len(world_data.get('tiles', [])) if world_data else 0
    unit_count = len(world_data.get('units', [])) if world_data else 0
    world_name = world.get('name', 'Unknown')

    print(f"World: {world_name}")
    print(f"  Tiles: {tile_count}, Units: {unit_count}")
    print(f"Migrating to: {dest_url} (ID: {dest_id})")

    # Prepare headers
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    # Try POST (create) first
    print("Attempting to create world...")
    resp = requests.post(base_url, json=request_payload, headers=headers)
    result = resp.json()

    if 'error' in result:
        error_msg = result.get('message', result['error'])
        if 'already exists' in error_msg.lower():
            print("World exists, updating instead...")
            resp = requests.put(dest_url, json=request_payload, headers=headers)
            result = resp.json()

            if 'error' in result:
                print(f"Error updating: {result.get('message', result['error'])}")
                sys.exit(1)
            print("World updated successfully!")
        else:
            print(f"Error creating: {error_msg}")
            sys.exit(1)
    else:
        print("World created successfully!")

    print("Migration complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Migrate a world from one service to another'
    )
    parser.add_argument('source_url', help='Source world URL (e.g., http://localhost:6060/api/v1/worlds/Desert)')
    parser.add_argument('dest_url', help='Destination world URL (e.g., http://localhost:8080/api/v1/worlds/Desert)')
    parser.add_argument('--token', '-t', help='Authorization token for destination')

    args = parser.parse_args()

    migrate_world(args.source_url, args.dest_url, args.token)


if __name__ == '__main__':
    main()
