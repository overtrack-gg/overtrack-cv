[![All Contributors](https://img.shields.io/badge/dynamic/json?color=orange&label=all%20contributors&query=%24.contributors.length&url=https%3A%2F%2Fraw.githubusercontent.com%2Fovertrack-gg%2Fovertrack-cv%2Fmaster%2F.all-contributorsrc&style=flat-square)](#contributors-)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Discord chat](https://img.shields.io/badge/chat-on_discord-008080.svg?style=flat-square&logo=discord)](https://discord.gg/JywstAB)

# overtrack-cv
This is the local client code that handles Computer Vision/OCR for OverTrack, plus the game-specific processors that collect data for Apex Legends.

For more info on using OverTrack, visit [overtrack.gg](https://overtrack.gg/)
# overtrack-cv
This is the local client code that handles Computer Vision/OCR for OverTrack, plus the game-specific processors that collect data for Apex Legends.

For more info on using OverTrack, visit [overtrack.gg](https://overtrack.gg/)
## Development
### Pre-reqs
* Windows
* Python 3.7 -- you can use [chocolatey](https://chocolatey.org/) or another package manager, or go to [python.org](https://www.python.org/downloads/windows/)
    ```
    > choco install python --version=3.7.9
    ```
* [Poetry](https://python-poetry.org) is used to manage the project's venv and dependencies. See [installation instructions](https://python-poetry.org/docs/#installation) to install it via PowerShell.
* Install dependencies and venv:
    ```
    > poetry install
    ```
* That's it. As long as there were no errors, you're ready to go.

### Local execution/testing
You can run the processors individually, and they will run against all the screenshots in their respective `/samples` directory.
* open venv shell, or run via poetry
    ```
    > # To open a shell and then execute things
    > poetry shell
    > cd .\overtrack_cv\games\apex\processors\squad_summary
    > python squad_summary_processor.py
    > exit
    > # Use `poetry run` instead of launching a shell
    > cd .\overtrack_cv\games\apex\processors\squad_summary
    > poetry run python squad_summary_processor.py
    ```
* when executing a processor, place any images you want processed into the `samples` directory in that processor directory. After executing, a popup will appear for each screenshot in order, showing you details of all the data identified on that image.

## How it works
_(shamelessly stolen from Muon on Discord, starting [here](https://discord.com/channels/274351102906859521/341810870910713857/822970261358772254))_

The key entrypoint for the apex processing `pipeline` is [/overtrack_cv/games/apex/default_pipeline.py#L32](https://github.com/overtrack-gg/overtrack-cv/blob/master/overtrack_cv/games/apex/default_pipeline.py#L32)

A `pipeline` is a series of `processors` that take `frames`. Each processor is responsible for extracting something from the image and putting it into structured data on the `frame`.

The entire `pipeline` is run on each and every `frame`. We build a list of all frames for a game with their rich data added to them, and this is sent to the server where the data is processed into a game.

The `pipeline` concept captures the fact that often `processors` are exclusive, as they are tracking only during a specific game state (ex: There is no need to track player coordinates while on the menu screen) so these `processors` are children of a `shortcircuit processor`.

A good starting point for understanding how this all works is the `map_loading` processor, found [here](https://github.com/overtrack-gg/overtrack-cv/blob/master/overtrack_cv/games/apex/processors/map_loading/map_loading_processor.py#L24). All this one does is locate the map name on the map loading screen to determine which map the player is about to play on.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://overtrack.gg"><img src="https://avatars0.githubusercontent.com/u/2515062?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Simon Pinfold</b></sub></a><br /><a href="https://github.com/overtrack-gg/overtrack-cv/commits?author=synap5e" title="Code">💻</a> <a href="#design-synap5e" title="Design">🎨</a></td>
    <td align="center"><a href="https://github.com/brahaney"><img src="https://avatars.githubusercontent.com/u/4593316?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Dan Brahaney</b></sub></a><br /><a href="https://github.com/overtrack-gg/overtrack-cv/commits?author=brahaney" title="Documentation">📖</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
