import json
from pathlib import Path


NB_PATH = Path("MULDE_Training_GMM.ipynb")


def set_source(nb, idx, source):
    cell = nb["cells"][idx]
    cell["source"] = source.splitlines(keepends=True)
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# Clear stale outputs from all code cells.
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

cell9 = "".join(nb["cells"][9]["source"])
cell9 = cell9.replace(
    "CHECKPOINT_EVERY = 50\n",
    "# Periodic checkpoints are disabled; only the final checkpoint is saved.\n"
    "CHECKPOINT_EVERY = None\n"
    "# Run train/test GMM evaluation every N epochs during training.\n"
    "PERIODIC_EVAL_EVERY = 5\n"
    "EVAL_BATCH_SIZE = BATCH_SIZE\n",
)
cell9 = cell9.replace(
    "SMOOTH_SIGMA_FRAMES = None\n",
    "SMOOTH_SIGMA_FRAMES = 7\n",
)
cell9 = cell9.replace(
    '    "gmm_covariance": GMM_COVARIANCE,\n'
    '    "smooth_sigma_frames": SMOOTH_SIGMA_FRAMES,\n',
    '    "gmm_covariance": GMM_COVARIANCE,\n'
    '    "checkpoint_every": CHECKPOINT_EVERY,\n'
    '    "periodic_eval_every": PERIODIC_EVAL_EVERY,\n'
    '    "eval_batch_size": EVAL_BATCH_SIZE,\n'
    '    "smooth_sigma_frames": SMOOTH_SIGMA_FRAMES,\n',
)
set_source(nb, 9, cell9)

set_source(nb, 19, r'''TRAINING_LOG_PATH = LOG_DIR / "training_log.csv"
PERIODIC_EVAL_LOG_PATH = LOG_DIR / "periodic_eval_log.csv"
PERIODIC_EVAL_JSONL_PATH = LOG_DIR / "periodic_eval_metrics.jsonl"

SIGMA_LEVELS = np.linspace(SIGMA_LOW, SIGMA_HIGH, L, dtype=np.float32)
print("Evaluation sigmas:", SIGMA_LEVELS)


def append_training_log(row: dict) -> None:
    df = pd.DataFrame([row])
    df.to_csv(TRAINING_LOG_PATH, mode="a", header=not TRAINING_LOG_PATH.exists(), index=False)


def append_periodic_eval_log(rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(PERIODIC_EVAL_LOG_PATH, mode="a", header=not PERIODIC_EVAL_LOG_PATH.exists(), index=False)
    with open(PERIODIC_EVAL_JSONL_PATH, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def save_checkpoint(epoch: int, final: bool = False) -> Path:
    name = "mulde_final.pt" if final else f"mulde_epoch_{epoch:04d}.pt"
    path = CHECKPOINT_DIR / name
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": CONFIG,
            "train_mean": train_mean,
            "train_std": train_std,
        },
        path,
    )
    return path


def sample_log_uniform_sigma(batch_size: int, device: torch.device) -> torch.Tensor:
    log_low = math.log(SIGMA_LOW)
    log_high = math.log(SIGMA_HIGH)
    return torch.exp(torch.empty(batch_size, 1, device=device).uniform_(log_low, log_high))


def train_one_epoch(epoch: int) -> dict:
    model.train()
    loss_values = []
    dsm_values = []
    reg_values = []
    score_norm_values = []
    log_density_values = []

    progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
    for (x_cpu,) in progress:
        x = x_cpu.to(DEVICE, non_blocking=True)
        sigma = sample_log_uniform_sigma(x.shape[0], DEVICE)
        noise = torch.randn_like(x, device=DEVICE) * sigma
        x_noisy = x + noise

        # Official MULDE score convention: score() returns grad(-f_theta).
        score, log_density_noisy = model.score(
            torch.cat([x_noisy, sigma], dim=1),
            return_log_density=True,
        )

        score_x = score[:, :-1]  # exclude gradient w.r.t. sigma conditioning dimension
        dsm_per_sample = torch.linalg.vector_norm(score_x + noise / (sigma ** 2), dim=1) ** 2
        lambda_factor = (sigma ** 2).reshape(-1)
        loss_dsm = (lambda_factor * dsm_per_sample).mean() / 2.0

        # Same regularizer as the paper and author code, computed by direct
        # forward pass to avoid an unused clean-input score gradient.
        log_density_clean = model(torch.cat([x, sigma], dim=1))
        loss_reg = BETA * (log_density_clean.reshape(-1) ** 2).mean() / 2.0
        loss = loss_dsm + loss_reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_values.append(float(loss.detach().cpu()))
        dsm_values.append(float(loss_dsm.detach().cpu()))
        reg_values.append(float(loss_reg.detach().cpu()))
        score_norm_values.append(float(torch.linalg.vector_norm(score_x.detach(), dim=1).pow(2).mean().cpu()))
        log_density_values.append(float(log_density_noisy.detach().mean().cpu()))
        progress.set_postfix(loss=np.mean(loss_values), dsm=np.mean(dsm_values), reg=np.mean(reg_values))

    row = {
        "epoch": epoch,
        "loss_total": float(np.mean(loss_values)),
        "loss_dsm": float(np.mean(dsm_values)),
        "loss_regularizer": float(np.mean(reg_values)),
        "score_norm_mean": float(np.mean(score_norm_values)),
        "log_density_noisy_mean": float(np.mean(log_density_values)),
        "lr": float(optimizer.param_groups[0]["lr"]),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    append_training_log(row)
    return row


def compute_multiscale_log_density_for_eval(
    features_std: np.ndarray,
    name: str,
    batch_size: int = EVAL_BATCH_SIZE,
) -> np.ndarray:
    model.eval()
    n = features_std.shape[0]
    out = np.empty((n, len(SIGMA_LEVELS)), dtype=np.float32)

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc=f"Periodic scoring {name}", leave=False):
            end = min(start + batch_size, n)
            x = torch.from_numpy(features_std[start:end]).to(DEVICE, non_blocking=True)
            cols = []
            for sigma_value in SIGMA_LEVELS:
                sigma_col = torch.full((x.shape[0], 1), float(sigma_value), device=DEVICE)
                log_density = model(torch.cat([x, sigma_col], dim=1)).reshape(-1)
                cols.append(log_density.detach().cpu().numpy().astype(np.float32))
            out[start:end] = np.stack(cols, axis=1)
    return out


def apply_optional_video_smoothing_for_eval(df: pd.DataFrame, score_col: str) -> str:
    eval_col = f"{score_col}_eval"
    if SMOOTH_SIGMA_FRAMES is None:
        df[eval_col] = df[score_col].to_numpy(dtype=np.float32)
        return eval_col

    smoothed = np.empty(len(df), dtype=np.float32)
    for _, indices in df.groupby("video_id", sort=False).groups.items():
        idx = np.asarray(list(indices))
        values = df.loc[idx, score_col].to_numpy(dtype=np.float32)
        smoothed[idx] = gaussian_filter1d(values, sigma=float(SMOOTH_SIGMA_FRAMES), mode="nearest")
    df[eval_col] = smoothed
    return eval_col


def safe_auc_for_eval(labels: np.ndarray, scores: np.ndarray):
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def compute_micro_macro_auc_for_eval(df: pd.DataFrame, score_col: str) -> dict:
    y = df["label"].to_numpy(dtype=np.uint8)
    s = df[score_col].to_numpy(dtype=np.float32)
    micro_auc = safe_auc_for_eval(y, s)

    video_aucs = []
    skipped = []
    for video_id, sub in df.groupby("video_id", sort=False):
        auc = safe_auc_for_eval(sub["label"].to_numpy(dtype=np.uint8), sub[score_col].to_numpy(dtype=np.float32))
        if auc is None:
            skipped.append(str(video_id))
        else:
            video_aucs.append(auc)

    macro_auc = float(np.mean(video_aucs)) if video_aucs else None
    return {
        "micro_auc": micro_auc,
        "macro_auc": macro_auc,
        "num_macro_videos": int(len(video_aucs)),
        "skipped_macro_videos_one_class": skipped,
    }


def evaluate_split_with_gmm(
    split: str,
    epoch: int,
    components: int,
    gmm: GaussianMixture,
    log_density: np.ndarray,
    video_ids: np.ndarray,
    frame_indices: np.ndarray,
    labels: np.ndarray,
) -> dict:
    score_col = f"gmm_{components}_nll"
    df = pd.DataFrame(
        {
            "video_id": video_ids,
            "frame_index": frame_indices.astype(np.int64),
            "label": labels.astype(np.uint8),
            score_col: (-gmm.score_samples(log_density)).astype(np.float32),
        }
    )
    eval_score_col = apply_optional_video_smoothing_for_eval(df, score_col)
    auc_info = compute_micro_macro_auc_for_eval(df, eval_score_col)
    scores = df[eval_score_col].to_numpy(dtype=np.float32)

    return {
        "epoch": int(epoch),
        "split": split,
        "components": int(components),
        "score_col": eval_score_col,
        "micro_auc": auc_info["micro_auc"],
        "macro_auc": auc_info["macro_auc"],
        "num_macro_videos": auc_info["num_macro_videos"],
        "skipped_macro_videos_one_class": "|".join(auc_info["skipped_macro_videos_one_class"]),
        "nll_mean": float(np.mean(scores)),
        "nll_std": float(np.std(scores)),
        "nll_min": float(np.min(scores)),
        "nll_max": float(np.max(scores)),
        "smooth_sigma_frames": SMOOTH_SIGMA_FRAMES,
        "lr_used_this_epoch": None,
        "lr_after_scheduler": float(optimizer.param_groups[0]["lr"]),
        "timestamp": datetime.datetime.now().isoformat(),
    }


def format_auc(value) -> str:
    return "NA" if value is None else f"{value:.6f}"


def run_periodic_gmm_evaluation(epoch: int, lr_used_this_epoch: float) -> list[dict]:
    eval_start = time.time()
    print(f"\n--- Periodic GMM evaluation at epoch {epoch} ---")
    train_log_density_eval = compute_multiscale_log_density_for_eval(train_features, f"train epoch {epoch}")
    test_log_density_eval = compute_multiscale_log_density_for_eval(test_features, f"test epoch {epoch}")

    # MULDE training splits are normal-only, so train AUC is undefined unless
    # a future dataset supplies train labels with both classes.
    train_eval_labels = np.zeros(train_log_density_eval.shape[0], dtype=np.uint8)

    rows = []
    best_test_row = None
    for components in GMM_COMPONENTS:
        gmm = GaussianMixture(
            n_components=components,
            covariance_type=GMM_COVARIANCE,
            random_state=SEED,
        )
        gmm.fit(train_log_density_eval)

        train_row = evaluate_split_with_gmm(
            "train",
            epoch,
            components,
            gmm,
            train_log_density_eval,
            train_video_ids,
            train_frame_indices,
            train_eval_labels,
        )
        test_row = evaluate_split_with_gmm(
            "test",
            epoch,
            components,
            gmm,
            test_log_density_eval,
            test_video_ids,
            test_frame_indices,
            test_labels,
        )
        train_row["lr_used_this_epoch"] = float(lr_used_this_epoch)
        test_row["lr_used_this_epoch"] = float(lr_used_this_epoch)
        rows.extend([train_row, test_row])

        if test_row["micro_auc"] is not None and (
            best_test_row is None or test_row["micro_auc"] > best_test_row["micro_auc"]
        ):
            best_test_row = test_row

        print(
            f"Epoch {epoch:04d} | GMM({components}) | "
            f"train micro={format_auc(train_row['micro_auc'])} macro={format_auc(train_row['macro_auc'])} | "
            f"test micro={format_auc(test_row['micro_auc'])} macro={format_auc(test_row['macro_auc'])}"
        )

    elapsed_min = (time.time() - eval_start) / 60.0
    for row in rows:
        row["eval_elapsed_min"] = float(elapsed_min)
    append_periodic_eval_log(rows)

    if best_test_row is not None:
        print(
            f"Best periodic test AUC at epoch {epoch}: "
            f"GMM({best_test_row['components']}) micro={best_test_row['micro_auc']:.6f} "
            f"macro={format_auc(best_test_row['macro_auc'])}"
        )
    print(f"Saved periodic eval log: {PERIODIC_EVAL_LOG_PATH}")
    return rows


training_start = time.time()
for epoch in range(start_epoch, EPOCHS + 1):
    row = train_one_epoch(epoch)

    scheduler.step()

    print(
        f"Epoch {epoch:04d}/{EPOCHS} | "
        f"loss={row['loss_total']:.6f} | dsm={row['loss_dsm']:.6f} | reg={row['loss_regularizer']:.6f}"
    )

    if PERIODIC_EVAL_EVERY and epoch % PERIODIC_EVAL_EVERY == 0:
        run_periodic_gmm_evaluation(epoch, lr_used_this_epoch=row["lr"])

    if CHECKPOINT_EVERY and epoch % CHECKPOINT_EVERY == 0:
        ckpt_path = save_checkpoint(epoch, final=False)
        print(f"Saved checkpoint: {ckpt_path}")

final_checkpoint = save_checkpoint(EPOCHS, final=True)
state_dict_path = CHECKPOINT_DIR / "mulde_final_state_dict.pt"
model.save(str(state_dict_path))

elapsed_min = (time.time() - training_start) / 60.0
print(f"Training complete in {elapsed_min:.2f} minutes")
print(f"Final checkpoint: {final_checkpoint}")
print(f"Final state dict via official model.save(): {state_dict_path}")
print(f"Periodic eval CSV: {PERIODIC_EVAL_LOG_PATH}")
print(f"Periodic eval JSONL: {PERIODIC_EVAL_JSONL_PATH}")
''')

set_source(nb, 20, r'''## 10. Compute Final Multiscale Log-Density Vectors

During training, this notebook now evaluates train/test GMM metrics every
`PERIODIC_EVAL_EVERY` epochs. After training finishes, this cell recomputes the
final multiscale log-density vectors and saves them as artifacts for the final
GMM evaluation/export stage.

''')

set_source(nb, 23, r'''# Smoothing is configured in Step 4 and used by both periodic and final evaluation.
print(f"SMOOTH_SIGMA_FRAMES = {SMOOTH_SIGMA_FRAMES}")
''')

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Patched {NB_PATH}")
