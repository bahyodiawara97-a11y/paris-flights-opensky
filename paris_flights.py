"""
Affiche/anime les avions en vol AU-DESSUS DE PARIS (ou de toute autre zone)
sur un fond de carte satellite, à partir des positions ADS-B en direct de
l'API OpenSky Network.

DIFFÉRENCE IMPORTANTE AVEC LE SCRIPT MÉTRO (GTFS)
---------------------------------------------------
Pour le métro, on disposait d'horaires THÉORIQUES (GTFS) permettant de
calculer la position d'une rame à n'importe quelle heure passée, instantané-
ment. Pour les avions, il n'existe pas d'équivalent public gratuit aussi
riche : on ne dispose que des positions EN DIRECT (ADS-B), via l'API
OpenSky. Il est donc impossible de "rejouer" une heure passée sans avoir
enregistré les données à ce moment-là. Pour obtenir une animation de 5
minutes, ce script interroge réellement l'API toutes les X secondes PENDANT
5 minutes (le script met donc le même temps à s'exécuter que la durée
simulée).

COMPTE ET CLÉS API REQUIS
--------------------------
1. Crée un compte gratuit sur https://opensky-network.org
2. Va sur https://opensky-network.org/my-opensky/account
3. Crée un "API client" : tu obtiens un client_id et un client_secret
4. Passe-les au script via --client-id / --client-secret, ou via les
   variables d'environnement OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET

LIMITES DU COMPTE GRATUIT (à connaître avant de lancer une longue capture)
----------------------------------------------------------------------------
- Quota de "crédits" quotidien (4000/jour pour un compte standard). Chaque
  appel à /states/all sur une petite zone (comme l'Île-de-France) coûte 1
  crédit. Avec un intervalle de 10s, 5 minutes de capture = ~30 appels =
  30 crédits : largement dans les clous.
- Ne descends pas sous --interval 10 (secondes) pour rester raisonnable et
  respecter les bonnes pratiques d'usage de l'API.

INSTALLATION
------------
pip install requests pandas matplotlib contextily

USAGE
-----
# Instantané unique (rapide, pour tester que tout fonctionne) :
python paris_avions_opensky.py --client-id XXX --client-secret YYY --snapshot --out avions.png

# Capture animée de 5 minutes, une image toutes les 10 secondes :
python paris_avions_opensky.py --client-id XXX --client-secret YYY \
    --duration 300 --interval 10 --out avions_paris.mp4
"""

import argparse
import math
import os
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.path import Path as MplPath
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd
import requests

try:
    import contextily as cx
except ImportError:
    cx = None

TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
STATES_URL = "https://opensky-network.org/api/states/all"

# Zone par défaut : Paris + petite couronne (élargis si tu veux voir plus
# d'avions, par ex. tout le bassin parisien avec CDG et Orly bien visibles)
DEFAULT_BBOX = {
    "lamin": 48.60,
    "lomin": 2.05,
    "lamax": 49.10,
    "lomax": 2.65,
}

STATE_COLUMNS = [
    "icao24", "callsign", "origin_country", "time_position", "last_contact",
    "longitude", "latitude", "baro_altitude", "on_ground", "velocity",
    "true_track", "vertical_rate", "sensors", "geo_altitude", "squawk",
    "spi", "position_source",
]


# ---------------------------------------------------------------------------
# AUTHENTIFICATION OAUTH2
# ---------------------------------------------------------------------------

class TokenManager:
    """Récupère et rafraîchit automatiquement le jeton d'accès OpenSky."""

    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None
        self.expires_at = None

    def get_token(self) -> str:
        if self.token and self.expires_at and datetime.now() < self.expires_at:
            return self.token
        return self._refresh()

    def _refresh(self) -> str:
        r = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        self.token = data["access_token"]
        expires_in = data.get("expires_in", 1800)
        self.expires_at = datetime.now() + timedelta(seconds=expires_in - 30)
        return self.token

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.get_token()}"}


# ---------------------------------------------------------------------------
# RÉCUPÉRATION DES POSITIONS
# ---------------------------------------------------------------------------

def fetch_states(tokens: TokenManager, bbox: dict) -> pd.DataFrame:
    """Récupère un instantané des positions d'avions dans la bbox donnée."""
    params = {
        "lamin": bbox["lamin"], "lomin": bbox["lomin"],
        "lamax": bbox["lamax"], "lomax": bbox["lomax"],
    }
    r = requests.get(STATES_URL, headers=tokens.headers(), params=params, timeout=15)
    if r.status_code == 429:
        wait = int(r.headers.get("X-Rate-Limit-Retry-After-Seconds", 60))
        print(f"Quota API atteint, attente de {wait}s avant de réessayer...")
        time.sleep(wait)
        return fetch_states(tokens, bbox)
    r.raise_for_status()
    data = r.json()
    states = data.get("states") or []
    df = pd.DataFrame(states, columns=STATE_COLUMNS)
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[df["on_ground"] == False]  # noqa: E712 - ne garde que les avions en vol
    return df


# ---------------------------------------------------------------------------
# PROJECTION ET AFFICHAGE
# ---------------------------------------------------------------------------

def lon_to_x(lon):
    return lon * 20037508.34 / 180


def lat_to_y(lat):
    y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
    return y * 20037508.34 / 180


def airplane_marker_path() -> MplPath:
    """
    Construit un petit contour d'avion (vu du dessus, nez vers le haut) fait
    à la main avec quelques points : fuselage, deux ailes, empennage.
    Ce n'est pas une image importée, juste une forme géométrique simple.
    """
    verts = [
        (0.00, 1.00),    # nez
        (0.09, 0.35),
        (0.85, -0.05),   # bout d'aile droite
        (0.12, -0.10),
        (0.28, -0.80),
        (0.08, -0.62),
        (0.00, -0.70),
        (-0.08, -0.62),
        (-0.28, -0.80),
        (-0.12, -0.10),
        (-0.85, -0.05),  # bout d'aile gauche
        (-0.09, 0.35),
        (0.00, 1.00),
    ]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


AIRPLANE_PATH = airplane_marker_path()
ALTITUDE_CMAP = cm.get_cmap("plasma")

BASEMAP_CHOICES = {
    "satellite": "Esri.WorldImagery",
    "light": "Esri.WorldGrayCanvas",
    "streets": "OpenStreetMap.Mapnik",
    "topo": "Esri.WorldTopoMap",
}
# CartoDB (DarkMatter/Positron) exige désormais une clé API payante et n'est
# donc plus utilisé ici. "dark" est simulé en assombrissant artificiellement
# le fond de carte gris (Esri.WorldGrayCanvas), qui lui reste gratuit et sans
# clé.
DARK_OVERLAY_ALPHA = 0.9


def get_basemap_provider(name: str):
    """Résout un nom simple (ex: 'light') vers le provider contextily correspondant.
    'dark' réutilise le fond gris clair, assombri ensuite via un calque (voir add_basemap_with_style)."""
    lookup_name = "light" if name == "dark" else name
    path = BASEMAP_CHOICES.get(lookup_name, lookup_name)
    provider = cx.providers
    for part in path.split("."):
        provider = getattr(provider, part)
    return provider


def get_dark_basemap_image(x_min, y_min, x_max, y_max, zoom=10):
    """
    Récupère un fond de carte clair (Esri World Gray Canvas — gratuit, sans
    clé API, et plus tolérant pour ce type d'usage que les serveurs
    bénévoles d'OpenStreetMap) et inverse ses couleurs pour obtenir un rendu
    sombre avec les routes/noms de villes en clair, plutôt que d'assombrir
    grossièrement le fond avec un calque semi-transparent (qui écrasait les
    étiquettes de villes).
    """
    img, ext = cx.bounds2img(
        x_min, y_min, x_max, y_max, zoom=zoom,
        source=cx.providers.Esri.WorldGrayCanvas, ll=False,
    )
    gray = img[..., :3].astype(float).mean(axis=2)
    inv = 255 - gray
    dark = np.zeros((*inv.shape, 3), dtype=np.uint8)
    # léger ton bleuté, plus agréable qu'un gris neutre
    dark[..., 0] = np.clip(inv * 0.55, 0, 255).astype(np.uint8)
    dark[..., 1] = np.clip(inv * 0.62, 0, 255).astype(np.uint8)
    dark[..., 2] = np.clip(inv * 0.85, 0, 255).astype(np.uint8)
    return dark, ext


def add_basemap_with_style(ax, bbox: dict, basemap: str, zoom: int = 10):
    """Ajoute le fond de carte. 'dark' utilise un fond Esri inversé (voir get_dark_basemap_image)."""
    if cx is None:
        return
    try:
        if basemap == "dark":
            x_min, x_max = lon_to_x(bbox["lomin"]), lon_to_x(bbox["lomax"])
            y_min, y_max = lat_to_y(bbox["lamin"]), lat_to_y(bbox["lamax"])
            dark_img, ext = get_dark_basemap_image(x_min, y_min, x_max, y_max, zoom=zoom)
            ax.imshow(dark_img, extent=ext, origin="upper", zorder=1, aspect="auto")
        else:
            cx.add_basemap(ax, source=get_basemap_provider(basemap), zoom=zoom)
    except Exception as e:
        print(f"Fond de carte indisponible ({e}), affichage sans basemap.")


def altitude_to_color(alt_m: float, vmin=0, vmax=12000):
    alt_m = 0 if alt_m is None or (isinstance(alt_m, float) and math.isnan(alt_m)) else alt_m
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    return ALTITUDE_CMAP(norm(alt_m))


def draw_airplane(ax, x, y, heading_deg, color, size=340, zorder=3):
    """Dessine un avion orienté selon son cap (heading, degrés depuis le nord, sens horaire)."""
    heading_deg = 0 if heading_deg is None or (isinstance(heading_deg, float) and math.isnan(heading_deg)) else heading_deg
    marker = MarkerStyle(AIRPLANE_PATH, transform=Affine2D().rotate_deg(-heading_deg))
    return ax.scatter(
        [x], [y], marker=marker, s=size, c=[color],
        edgecolors="white", linewidths=0.6, zorder=zorder,
    )


def plot_snapshot(df: pd.DataFrame, bbox: dict, title: str, out_path: str, show_labels: bool, basemap: str = "satellite"):
    fig, ax = plt.subplots(figsize=(10, 13), facecolor="black")
    fig.patch.set_facecolor("black")

    if len(df) > 0:
        x_vals = df["longitude"].apply(lon_to_x).to_numpy()
        y_vals = df["latitude"].apply(lat_to_y).to_numpy()
        for xi, yi, alt, heading, callsign in zip(
            x_vals, y_vals, df["baro_altitude"], df["true_track"], df["callsign"].fillna("")
        ):
            color = altitude_to_color(alt)
            draw_airplane(ax, xi, yi, heading, color)
            if show_labels:
                ax.annotate(callsign.strip(), (xi, yi), ha="center", va="bottom",
                             xytext=(0, 10), textcoords="offset points",
                             fontsize=5, fontweight="bold", color="white", zorder=4)

    x_min, x_max = lon_to_x(bbox["lomin"]), lon_to_x(bbox["lomax"])
    y_min, y_max = lat_to_y(bbox["lamin"]), lat_to_y(bbox["lamax"])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    add_basemap_with_style(ax, bbox, basemap)

    ax.set_axis_off()
    fig.subplots_adjust(top=0.94, bottom=0.01, left=0.01, right=0.99)
    ax.set_title(title, color="white", fontsize=18, fontweight="bold")
    plt.savefig(out_path, dpi=180, facecolor="black")
    print(f"Image enregistrée : {out_path}")


def make_live_animation(tokens: TokenManager, bbox: dict, duration_s: int, interval_s: int,
                         out_path: str, video_fps: int, show_labels: bool, basemap: str = "satellite"):
    """
    Interroge l'API toutes les `interval_s` secondes pendant `duration_s`
    secondes au total (durée réelle), et assemble les instantanés en vidéo.
    """
    import matplotlib.animation as animation

    n_polls = max(1, duration_s // interval_s)
    print(f"Capture de {n_polls} instantanés, un toutes les {interval_s}s "
          f"(durée réelle totale : ~{duration_s}s)...")

    frames = []
    for i in range(n_polls):
        t0 = time.time()
        df = fetch_states(tokens, bbox)
        stamp = datetime.now().strftime("%H:%M:%S")
        frames.append((stamp, df))
        print(f"  [{i+1}/{n_polls}] {stamp} : {len(df)} avions en vol")
        elapsed = time.time() - t0
        sleep_left = interval_s - elapsed
        if sleep_left > 0 and i < n_polls - 1:
            time.sleep(sleep_left)

    fig, ax = plt.subplots(figsize=(10, 13), facecolor="black")
    fig.patch.set_facecolor("black")

    x_min, x_max = lon_to_x(bbox["lomin"]), lon_to_x(bbox["lomax"])
    y_min, y_max = lat_to_y(bbox["lamin"]), lat_to_y(bbox["lamax"])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    add_basemap_with_style(ax, bbox, basemap)

    ax.set_axis_off()
    fig.subplots_adjust(top=0.94, bottom=0.01, left=0.01, right=0.99)
    title = ax.set_title("", color="white", fontsize=18, fontweight="bold")

    # Les avions ont une orientation propre (cap), donc contrairement aux
    # points du métro, on ne peut pas les mettre à jour via un seul
    # PathCollection : on recrée les marqueurs à chaque image.
    plane_artists = []
    label_texts = []

    def update(i):
        nonlocal label_texts, plane_artists
        stamp, df = frames[i]

        for artist in plane_artists:
            artist.remove()
        plane_artists = []
        for t in label_texts:
            t.remove()
        label_texts = []

        if len(df) > 0:
            x_vals = df["longitude"].apply(lon_to_x).to_numpy()
            y_vals = df["latitude"].apply(lat_to_y).to_numpy()
            for xi, yi, alt, heading, callsign in zip(
                x_vals, y_vals, df["baro_altitude"], df["true_track"], df["callsign"].fillna("")
            ):
                color = altitude_to_color(alt)
                artist = draw_airplane(ax, xi, yi, heading, color)
                plane_artists.append(artist)
                if show_labels:
                    txt = ax.annotate(
                        callsign.strip(), (xi, yi), ha="center", va="bottom",
                        xytext=(0, 10), textcoords="offset points",
                        fontsize=5, fontweight="bold", color="white", zorder=4,
                    )
                    label_texts.append(txt)

        title.set_text(f"Avions au-dessus de Paris — {len(df)} vols à {stamp}")
        return plane_artists + label_texts + [title]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)

    if out_path.lower().endswith(".mp4"):
        try:
            writer = animation.FFMpegWriter(fps=video_fps, bitrate=4000)
        except Exception:
            raise RuntimeError(
                "ffmpeg est introuvable. Installe-le avec 'brew install ffmpeg', "
                "ou utilise un fichier de sortie en .gif à la place."
            )
        anim.save(out_path, writer=writer, dpi=180)
    else:
        anim.save(out_path, writer="pillow", fps=video_fps)

    print(f"Animation enregistrée : {out_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", default=os.environ.get("OPENSKY_CLIENT_ID"))
    parser.add_argument("--client-secret", default=os.environ.get("OPENSKY_CLIENT_SECRET"))
    parser.add_argument("--snapshot", action="store_true", help="Un seul instantané au lieu d'une animation")
    parser.add_argument("--duration", type=int, default=300, help="Durée réelle de capture, en secondes")
    parser.add_argument("--interval", type=int, default=10, help="Intervalle entre deux appels API, en secondes (min. conseillé: 10)")
    parser.add_argument("--video-fps", type=int, default=2, help="Images par seconde de la vidéo/gif final")
    parser.add_argument("--labels", action="store_true", help="Affiche l'indicatif (callsign) de chaque avion")
    parser.add_argument(
        "--basemap",
        default="satellite",
        choices=list(BASEMAP_CHOICES.keys()) + ["dark"],
        help="Style de fond de carte : satellite, dark, light, streets, topo",
    )
    parser.add_argument("--out", default="avions_paris.png")
    parser.add_argument("--lamin", type=float, default=DEFAULT_BBOX["lamin"])
    parser.add_argument("--lomin", type=float, default=DEFAULT_BBOX["lomin"])
    parser.add_argument("--lamax", type=float, default=DEFAULT_BBOX["lamax"])
    parser.add_argument("--lomax", type=float, default=DEFAULT_BBOX["lomax"])
    args = parser.parse_args()

    if not args.client_id or not args.client_secret:
        raise SystemExit(
            "Il manque --client-id / --client-secret (ou les variables d'environnement "
            "OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET). Crée-les sur "
            "https://opensky-network.org/my-opensky/account"
        )

    bbox = {"lamin": args.lamin, "lomin": args.lomin, "lamax": args.lamax, "lomax": args.lomax}
    tokens = TokenManager(args.client_id, args.client_secret)

    if args.snapshot:
        df = fetch_states(tokens, bbox)
        stamp = datetime.now().strftime("%H:%M:%S")
        print(f"{len(df)} avions en vol détectés à {stamp}")
        plot_snapshot(df, bbox, f"Avions au-dessus de Paris — {len(df)} vols à {stamp}", args.out, args.labels, args.basemap)
    else:
        make_live_animation(tokens, bbox, args.duration, args.interval, args.out, args.video_fps, args.labels, args.basemap)


if __name__ == "__main__":
    main()
