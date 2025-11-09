import os, io, base64, secrets, time, json
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless
import matplotlib.pyplot as plt
from flask import (Flask, request, render_template, flash,
                   redirect, url_for, send_file, Response)

# Use tqdm if available, otherwise provide a minimal no-op fallback so the app
# runs without the external dependency.
try:
    import importlib
    _tqdm_mod = importlib.import_module("tqdm")
    tqdm = _tqdm_mod.tqdm
except Exception:
    class tqdm:
        def __init__(self, total=0, desc=None, unit=None):
            self.total = total
            self.n = 0
            self.desc = desc
            self.unit = unit

        def update(self, n=1):
            # no-op progress; keep internal counter for possible debugging
            self.n += (n or 1)

        def close(self):
            pass

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024   # 2 MB uploads

# --------------------------------------------------
# Core simulation (unchanged logic, only tqdm adapted)
# --------------------------------------------------
def run_simulation(rows: int, cols: int, n_mines: int, n_iter: int,
                   seed=None, chunk_size=1000, progress_obj=None):
    rng = np.random.default_rng(seed)
    total_counts = np.zeros((rows, cols), dtype=np.int64)
    n_cells = rows * cols
    flat_indices = np.arange(n_cells)

    chunks = [(i, min(i+chunk_size, n_iter)) for i in range(0, n_iter, chunk_size)]

    for start, end in chunks:
        n_chunk = end - start
        for _ in range(n_chunk):
            chosen = rng.choice(flat_indices, size=n_mines, replace=False)
            r_idx = chosen // cols
            c_idx = chosen % cols
            np.add.at(total_counts, (r_idx, c_idx), 1)
            if progress_obj is not None:
                progress_obj.update(1)
        # tiny sleep keeps tqdm smooth on web
        time.sleep(0.0)

    return total_counts.astype(float) / float(n_iter)


def plot_to_b64(prob_matrix, annotate):
    """Return PNG as base64 string."""
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(prob_matrix, vmin=0, vmax=1, cmap="viridis", aspect="equal")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    ax.set_xticks(np.arange(prob_matrix.shape[1]))
    ax.set_yticks(np.arange(prob_matrix.shape[0]))
    if annotate:
        cmap_func = plt.get_cmap("viridis")
        for i in range(prob_matrix.shape[0]):
            for j in range(prob_matrix.shape[1]):
                val = prob_matrix[i, j]
                rgba = cmap_func(val)
                lum = 0.2126*rgba[0] + 0.7152*rgba[1] + 0.0722*rgba[2]
                color = "black" if lum > 0.5 else "white"
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        color=color, fontsize=8)
    fig.colorbar(im, ax=ax, label="Probability")
    fig.tight_layout()
    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def summary_stats(prob_matrix):
    mean_prob = float(np.mean(prob_matrix))
    min_val = float(np.min(prob_matrix))
    max_val = float(np.max(prob_matrix))
    min_pos = np.argwhere(prob_matrix == min_val)[0].tolist()
    max_pos = np.argwhere(prob_matrix == max_val)[0].tolist()
    return {
        "mean": mean_prob,
        "min": min_val,
        "min_pos": (int(min_pos[0]) + 1, int(min_pos[1]) + 1),
        "max": max_val,
        "max_pos": (int(max_pos[0]) + 1, int(max_pos[1]) + 1)
    }


# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            rows   = int(request.form["rows"])
            cols   = int(request.form["cols"])
            mines  = int(request.form["mines"])
            iters  = int(request.form["iters"])
            seed   = request.form.get("seed", "").strip()
            seed   = int(seed) if seed else None
            annotate = bool(request.form.get("annotate"))
            if not (1 <= mines < rows*cols):
                raise ValueError("Mines must be ≥1 and < total cells.")
            if iters <= 0 or iters > 1_000_000:
                raise ValueError("Iterations must be 1-1 000 000.")
        except Exception as e:
            flash(f"Input error: {e}")
            return redirect(url_for("index"))

        # Progress bar via server-sent events (simple polling fallback)
        job_id = secrets.token_urlsafe(8)
        PROGRESS[job_id] = {"done": 0, "total": iters}

        # Run simulation in background thread
        from threading import Thread
        def task():
            pbar = tqdm(total=iters, desc="sim", unit="iter")
            prob = run_simulation(rows, cols, mines, iters, seed,
                                  progress_obj=pbar)
            PROGRESS[job_id]["prob"] = prob
            PROGRESS[job_id]["rows"] = rows
            PROGRESS[job_id]["cols"] = cols
            PROGRESS[job_id]["mines"] = mines
            PROGRESS[job_id]["iters"] = iters
            PROGRESS[job_id]["annotate"] = annotate
            pbar.close()
            PROGRESS[job_id]["finished"] = True
        Thread(target=task, daemon=True).start()

        return redirect(url_for("result", job_id=job_id))
    return render_template("index.html")


PROGRESS = {}   # in-memory store; ok for single worker


@app.route("/result/<job_id>")
def result(job_id):
    if job_id not in PROGRESS:
        flash("Invalid or expired session.")
        return redirect(url_for("index"))
    if not PROGRESS[job_id].get("finished"):
        return render_template("result.html", job_id=job_id, progress=PROGRESS[job_id])
    # Finished – deliver full results
    prob = PROGRESS[job_id]["prob"]
    annotate = PROGRESS[job_id]["annotate"]
    img = plot_to_b64(prob, annotate)
    stats = summary_stats(prob)
    df = pd.DataFrame(prob,
                      columns=[f"C{j+1}" for j in range(prob.shape[1])],
                      index=[f"R{i+1}" for i in range(prob.shape[0])])
    csv_buf = io.BytesIO()
    df.to_csv(csv_buf, index=True)
    csv_buf.seek(0)
    # Store csv in memory for download
    PROGRESS[job_id]["csv"] = csv_buf.read()
    return render_template("result.html",
                           job_id=job_id,
                           finished=True,
                           img=img,
                           stats=stats,
                           tables=df.round(4).to_html(classes="table table-sm table-striped"))


@app.route("/download/<job_id>")
def download(job_id):
    if job_id not in PROGRESS or "csv" not in PROGRESS[job_id]:
        flash("File not ready.")
        return redirect(url_for("index"))
    return send_file(io.BytesIO(PROGRESS[job_id]["csv"]),
                     mimetype="text/csv",
                     as_attachment=True,
                     download_name="probability_matrix.csv")


# --------------------------------------------------
# Optional: simple SSE endpoint for progress
# --------------------------------------------------
@app.route("/progress/<job_id>")
def progress_sse(job_id):
    def gen():
        import json
        while True:
            if job_id not in PROGRESS:
                break
            obj = PROGRESS[job_id]
            yield f"data: {json.dumps({'done': obj['done'], 'total': obj['total']})}\n\n"
            if obj.get("finished"):
                break
            time.sleep(0.5)
    return Response(gen(), mimetype="text/event-stream")


# --------------------------------------------------
if __name__ == "__main__":
    # Local dev
    app.run(debug=True, threaded=True)