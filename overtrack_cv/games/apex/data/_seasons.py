import datetime
from dataclasses import dataclass
from typing import Optional


import requests
import pandas as pd
import re

url = 'https://apexlegends.fandom.com/wiki/Season'
html = requests.get(url).content
table_list = pd.read_html(html)
SeasonData = table_list[0]
#print(SeasonData)
#df.to_csv('SeasonData.csv') # cache?


@dataclass
class Season:
    index: int
    start: float
    end: float
    has_ranked: bool = True
    has_duos: bool = True
    has_arenas: bool = True
    season_name: Optional[str] = None

    @property
    def name(self) -> str:
        return self.season_name or f"Season {self.index}"

_PDT = datetime.timezone(datetime.timedelta(hours=-7))
_PST = datetime.timezone(datetime.timedelta(hours=-8))
seasons=[None] * len(SeasonData)

for i in range(len(SeasonData)):
	# TODO Use loc with column names
	SName = str(SeasonData.loc[i][0]).replace('(', ' (')
	SDuration = str(SeasonData.loc[i][1])
	
	#got: Feb. 04, 2019 -Mar. 19, 2019(43 days)
	#old: "Mar 19 2019 10:00AM",
	#oldformat: "%b %d %Y %I:%M%p",
	try:
		season_start = datetime.datetime.strptime(re.findall('^.*(?= -.*)', SDuration)[0], "%b. %d, %Y").replace(tzinfo=_PDT, hour=10, minute=0)
	except ValueError:
		season_start = datetime.datetime.strptime(re.findall('^.*(?= -.*)', SDuration)[0], "%b %d, %Y").replace(tzinfo=_PDT, hour=10, minute=0)
	try:
		season_end = datetime.datetime.strptime(re.findall('(?<= -).*(?=\(.*\))', SDuration)[0], "%b. %d, %Y").replace(tzinfo=_PDT, hour=10, minute=0)
	except ValueError:
		season_end = datetime.datetime.strptime(re.findall('(?<= -).*(?=\(.*\))', SDuration)[0], "%b %d, %Y").replace(tzinfo=_PDT, hour=10, minute=0)
	
	if i > 2: ranked = True 
	else: ranked = False
	if i > 2: duos = True
	else: duos = False
	if i > 8: arenas = True
	else: arenas = False
	seasons[i] = Season(i, season_start, season_end,  ranked, duos, arenas, SName)


current_season = sorted([s for s in seasons if s.index < 100], key=lambda s: s.end)[-1]


def main():
    from pprint import pprint

    pprint(seasons)


if __name__ == "__main__":
    main()
