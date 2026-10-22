# API examples

Start the server at `http://127.0.0.1:8000`, then generate deterministic
requests:

```bash
python scripts/generate_sample_inputs.py --output-dir examples/generated
```

For the browser's **Upload CSV** tab, use the tracked
`examples/data/synthetic_window_A_1000.csv` fixture or regenerate it with:

```bash
python scripts/generate_upload_example.py
```

Submit one request with curl:

```bash
curl -X POST http://127.0.0.1:8000/api/predict-window \
  -H "Content-Type: application/json" \
  --data-binary @examples/generated/predict_window_A.json
```

Or use the Python clients:

```bash
python examples/python_client.py \
  --input examples/generated/predict_window_A.json \
  --output examples/generated/predict_window_response.json

python examples/circuit_demo_client.py \
  --input examples/generated/circuit_demo_A.json \
  --output examples/generated/circuit_demo_response.json
```

The generated waveforms are synthetic smoke-test inputs. They do not reproduce
the manuscript's datasets or accuracy results. Complete tracked input/output
pairs and pointwise CSV exports are under `results/reproducible_examples/`.
