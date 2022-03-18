import requests

from overtrack_cv.games.valorant.data import agents


def main():
    for a in agents:
        r = requests.get(f"https://cdn.mobalytics.gg/assets/valorant/images/agents/icons/{a.lower()}.png")
        r.raise_for_status()
        with open(a.lower() + ".png", "wb") as f:
            f.write(r.content)


if __name__ == "__main__":
    main()
