# ✈️ Paris Flights — real-time ADS-B visualisation

Real-time visualisation of aircraft flying over Paris, built in Python from the
[OpenSky Network](https://opensky-network.org/) API.

![Dark map](assets/avions_dark.png)

## Features
- Authenticated access to OpenSky (OAuth2 client credentials, automatic token refresh)
- Live ADS-B states: position, heading (`true_track`), altitude, callsign
- Aircraft icons rotated according to heading, coloured by altitude
- Several basemaps: `satellite`, `light`, `streets`, `topo`, `dark` (with city names)
- Single snapshot (PNG) or animated capture (MP4)
- Handles API rate limits and incomplete data (missing position/heading)

## Install
```bash
git clone https://github.com/<your-username>/paris-flights-opensky.git
cd paris-flights-opensky
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
MP4 export requires [ffmpeg](https://ffmpeg.org/).

## Credentials
Create an API client on your OpenSky account page, then:
```bash
cp .env.example .env        # fill in your values
export OPENSKY_CLIENT_ID=...
export OPENSKY_CLIENT_SECRET=...
```
Never commit your credentials (`.env` and `credentials*.json` are git-ignored).

## Usage
```bash
# One snapshot
python paris_flights.py --snapshot --basemap dark --labels --out flights.png

# 5-minute live capture -> video
python paris_flights.py --duration 300 --interval 10 --labels --basemap dark --out flights.mp4
```

| Option | Description |
|---|---|
| `--basemap` | `satellite`, `light`, `streets`, `topo`, `dark` |
| `--labels` | show callsigns |
| `--duration` / `--interval` | capture length / seconds between API calls (≥ 10 recommended) |
| `--video-fps` | output frame rate |
| `--lamin --lomin --lamax --lomax` | custom bounding box |

## Stack
Python · requests · pandas · numpy · matplotlib · contextily · OpenSky REST API (OAuth2)

## Author
Bahyo Diawara — Master Langue et Informatique (Data & IA), Sorbonne Université
Data © OpenSky Network.

License: MIT
