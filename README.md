# TCN-BiLSTM Magnetic Design Website

[中文说明](README_zh-CN.md)

A local web application for magnetic-field trajectory prediction and magnetic-component design. The website packages trained TCN-BiLSTM models behind a browser interface, so it can be deployed and used on a personal computer without building a separate frontend or API service.

## Main features

- **Design Center** — simulate a single-phase full-bridge operating condition, configure toroidal-core dimensions, predict the `H` trajectory, and calculate the displayed magnetic-energy/loss result.
- **Test Center** — upload a single `B` sequence, run long-sequence cold-start prediction, compare trajectories, and export the result.
- **Model Libraries** — view the supported material models and configure model-related parameters.
- **Prediction and API pages** — run window-level prediction and call the same backend through JSON endpoints.

The browser workflow currently supports four material grades: **3C92, 3C95, FEC014, and T37**.

## Quick local deployment

The easiest option is to download the full deployment ZIP from the [latest release](https://github.com/ABCdada/TCN-BiLSTM-Magnetic-Design/releases/latest). It already contains the website and the required model checkpoints.

Requirements:

- Python 3.10 or newer;
- internet access during the first start to install Python packages;
- Windows, Linux, or macOS. CPU inference is supported.

### Windows

1. Download and extract `TCN-BiLSTM-Magnetic-Design-v1.3.2-full-deployment.zip`.
2. Double-click `start_windows.bat`, or run it in Command Prompt.
3. Wait for the first-time dependency installation and model verification.
4. Open <http://127.0.0.1:8000> in a browser.

### Linux or macOS

```bash
unzip TCN-BiLSTM-Magnetic-Design-v1.3.2-full-deployment.zip
cd TCN-BiLSTM-Magnetic-Design-v1.3.2-full
sh start_linux.sh
```

Then open <http://127.0.0.1:8000>.

The startup scripts create a local `.venv`, install dependencies, verify the model files, and start the website. To use another address, set `MAGNET_LISTEN` before starting, for example `0.0.0.0:8000` on a trusted local network.

## Source-code installation

```bash
git clone https://github.com/ABCdada/TCN-BiLSTM-Magnetic-Design.git
cd TCN-BiLSTM-Magnetic-Design
python -m venv .venv
```

Activate the virtual environment, install dependencies, and start the server:

```bash
python -m pip install -r requirements.txt
python -m waitress --listen=127.0.0.1:8000 app:app
```

The Git repository does not store model binaries. For a ready-to-use installation, use the full deployment ZIP from Releases. Advanced deployment and API options are documented in [DEPLOYMENT.md](DEPLOYMENT.md) and [API.md](API.md).

## Project layout

```text
app.py              Flask website and API
magnetic_model/     model loading and inference
templates/          website pages
static/             styles, scripts, images, and examples
weights/            model checkpoints and manifests
start_windows.bat   Windows startup script
start_linux.sh      Linux/macOS startup script
```

## Notes

- The first startup can take several minutes while PyTorch and other dependencies are installed.
- Keep the files under `weights/` together; the server verifies them before loading.
- Model Libraries provides model information and parameter configuration. Model training itself remains an offline workflow.
- License information is available in [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
