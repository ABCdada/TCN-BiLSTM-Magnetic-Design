# TCN-BiLSTM Magnetic Design Website v1.3.2

This release provides a complete local deployment package for the magnetic-field prediction and magnetic-component design website.

## Website functions

- Single-phase full-bridge Design Center with configurable toroidal-core dimensions;
- long-sequence `H` prediction in Test Center;
- material and parameter display in Model Libraries;
- browser charts, result export, and JSON prediction APIs;
- support for 3C92, 3C95, FEC014, and T37 in the browser workflow.

## Local deployment

1. Download `TCN-BiLSTM-Magnetic-Design-v1.3.2-full-deployment.zip` from this release.
2. Extract the ZIP.
3. Run `start_windows.bat` on Windows or `sh start_linux.sh` on Linux/macOS.
4. Open <http://127.0.0.1:8000>.

The full deployment package contains the required model checkpoints. The first start creates a virtual environment, installs dependencies, verifies the model files, and starts the website.
