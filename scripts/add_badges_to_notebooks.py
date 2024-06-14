import glob
import json

BADGE = """<a target="_blank" href="https://colab.research.google.com/github/ArtificialIntelligenceToolkit/aitk/blob/master/{filename}"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>"""

for filename in glob.glob("notebooks/*.ipynb"):
    with open(filename) as fp:
        json_data = json.loads(fp.read())

    badge_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": []
    }
    badge_cell["source"] = BADGE.format(filename=filename)
    json_data["cells"].insert(0, badge_cell)

    pip_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "id": "AQMTpw5e_0cx"
        },
        "outputs": [],
        "source": [
            "%pip install aitk --upgrade --quiet"
        ]
    }
    json_data["cells"].insert(1, pip_cell)

    with open(filename, "w") as fp:
        fp.write(json.dumps(json_data, indent=1))

