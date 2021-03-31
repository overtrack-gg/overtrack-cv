# overtrack-cv

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
* open venv shell, or run via poetry
    ```
    > # To open a shell and then execute things
    > poetry shell
    > cd .\overtrack_cv\games\apex\processors\squad_summary
    > python squad_summary_processor.py
    > exit
    > # Use `poetry run` instead of launching a shell
    > cd python squad_summary_processor.py
    > poetry run python squad_summary_processor.py
    ```
* when executing a processor, place any images you want processed into the `samples` directory in that processor directory. After executing, a popup will appear for each screenshot in order, showing you details of all the data identified on that image.
