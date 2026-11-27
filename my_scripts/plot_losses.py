#!/usr/bin/env python3
"""
Plot training and test losses from a crystalsizer3d train_synthetic .out log.

Usage:
    python plot_losses.py <logfile.out> [-o output.png]

The weights, batch size and number of batches per epoch are read from the log.  
It also works for a run that is still in progress: epochs that have not finished are left out.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg") # Agg draws into a file instead of onto a screen.
import matplotlib.pyplot as plt


# ==============================================================================
#  HOW THE LOG REPORTS LOSSES, AND WHY TRAINING AND TEST ARE HANDLED DIFFERENTLY
# ==============================================================================
#
#  The three curves in each panel come from three different ways in which losses
#  are reported in the log.
#
#  (1) PER-BATCH TRAINING LINES  ->  the grey curve
#      Written every --log-every-n-batches batches:
#
#      E.g., [1/5][7/25]  Loss: 1.7566E+00  losses/distances: 3.0471E-01  etc.
#
#      "[1/5]"   = epoch 1 of 5           
#      "[7/25]"  = batch 7 of 25
#      "Loss:"   = the total loss (weighed) averaged over the batch, i.e. the number the
#                  optimiser actuallwy minimises
#      "losses/<term>" = each loss term averaged over the batch. 
#
#  (2) END-OF-EPOCH TRAINING LINES  ->  the blue curve
#      One per loss term, after the last batch of the epoch:
#
#      E.g., Train losses/distances: 6.2223E+00
#
#      These are SUMS, accumulated over every batch of the epoch (including
#      the batches that are not printed in the log). To get the average over the epoch, 
#      I divide by the number of batches in the epoch.
#
#  (3) END-OF-EPOCH TEST LINES  ->  the orange curve
#      One per loss term, after the test pass:
#
#      E.g., Test losses/distances: 1.1520E-01
#
#      These are already MEANS, so they are used as they are.
#
#  Also, the training losses are accumulated WHILE the weights are still changing during an epoch, 
#  whereas the test losses are measured ONCE, at the end of the epoch, with the weights fixed 
#  after the last update of that epoch.  
#
#  Finally, the end-of-epoch lines do not print a total loss. This is calculated here 
#  from the weighted sum: L = Sum_k  w_k * L_k. Only the grey curve is
#  read directly from the log, because the total loss is recorded.
#
# ===========================================================================


# --------------------------------------------------------------- appearance
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#c3c2b7"

# The five loss terms, in the order their panels appear after "Total loss".
# Each entry is (key as it appears in the log, name shown in the title).
# Editing this list changes the figure.
PANELS = [
    ("distances", "Distances"),
    ("transformation", "Transformation"),
    ("material", "Material"),
    ("light", "Light"),
    ("3d", "3D vertices"),
]


# ------------------------------------------------------------------ parsing

# "-------- Train epoch: 3 --------"
RE_EPOCH_HEADER = re.compile(r"Train epoch:\s*(\d+)")

# "[1/5][7/25]\tLoss: 1.7566E+00"  -> epoch, n_epochs, batch, n_batches, total
RE_BATCH = re.compile(r"\[(\d+)/(\d+)\]\[(\d+)/(\d+)\]\s+Loss:\s+([-\d.Ee+]+)")

# "losses/distances: 3.0471E-01" anywhere on a batch line -> term, value
RE_TERM = re.compile(r"losses/(\w+):\s+([-\d.Ee+]+)")

# "Train losses/distances: 6.2223E+00"  -> a SUM over the epoch's batches
RE_TRAIN_SUM = re.compile(r"Train losses/(\w+):\s+([-\d.Ee+]+)")

# "Test losses/distances: 1.1520E-01"   -> already a MEAN over test batches
RE_TEST_MEAN = re.compile(r"Test losses/(\w+):\s+([-\d.Ee+]+)")

# "Time per epoch: 0:01:46.565834"      -> hours, minutes, seconds
RE_EPOCH_TIME = re.compile(r"Time per epoch:\s+(\d+):(\d+):([\d.]+)")


def parse_log(path: Path):
    """
    Read one .out file and return everything the figure needs.

    Returns
        args         dict of the "--- Arguments ---" block, values as strings
        batches      list of (x, weighted total, {term: unweighted value})
        train_sums   {epoch: {term: sum over the epoch}}
        test_means   {epoch: {term: mean over the epoch}}
        n_batches    {epoch: number of training batches in that epoch}
        epoch_times  {epoch: seconds the epoch took}
    """
    text = path.read_text(errors="replace")

    # ---- the "--- Arguments ---" dump that the code prints on start-up ----
    m = re.search(r"--- Arguments ---\n(.*?)\n-{5,}\n", text, re.S)
    if m is None:
        raise SystemExit(f"{path}: no '--- Arguments ---' block found")
    args = {}
    for line in m.group(1).splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            args[k.strip()] = v.strip()

    # ---- containers filled by the single pass below ----------------------
    batches = []                    # (x position, weighted total, {term: value})
    train_sums = defaultdict(dict)  # epoch -> term -> sum over the epoch
    test_means = defaultdict(dict)  # epoch -> term -> mean over the epoch
    n_batches = {}                  # epoch -> training batches in that epoch
    epoch_times = {}                # epoch -> seconds
    epoch = None                    # unknown until the first header is seen

    # One pass over the file, deciding what each line is.
    for line in text.splitlines():

        # --- an epoch header: just note which epoch we are now in ---------
        h = RE_EPOCH_HEADER.search(line)
        if h:
            epoch = int(h.group(1))
            continue            

        # --- a per-batch training line ---------------------
        b = RE_BATCH.search(line)
        if b:
            epoch = int(b.group(1))     # epoch from "[1/5]"
            nb = int(b.group(4))        # batches per epoch from "[7/25]"
            n_batches[epoch] = nb       # needed later to turn sums into means
            # every "losses/<term>: <value>" pair on this line
            terms = {k: float(v) for k, v in RE_TERM.findall(line)}
            # x position: epoch 1 batch 7 of 25 -> 0 + 7/25 = 0.28, so the
            # grey curve runs continuously across the whole run.
            batches.append(((epoch - 1) + int(b.group(3)) / nb,
                            float(b.group(5)),  
                            terms))
            continue

        # --- end-of-epoch summaries -----------------
        # Note that train values are sums, test values are means.  
        # The division happens later, in main().
        for key, value in RE_TRAIN_SUM.findall(line):
            train_sums[epoch][key] = float(value)
        for key, value in RE_TEST_MEAN.findall(line):
            test_means[epoch][key] = float(value)

        # --- "Time per epoch", useful for projecting the full run ---------
        t = RE_EPOCH_TIME.search(line)
        if t and epoch is not None:
            epoch_times[epoch] = (int(t.group(1)) * 3600
                                  + int(t.group(2)) * 60
                                  + float(t.group(3)))

    # dict(...) converts the defaultdicts to plain dicts
    return args, batches, dict(train_sums), dict(test_means), n_batches, epoch_times


def fmt_w(w: float) -> str:
    """
    Format a weight for a panel title.
    """
    s = f"{w:.4f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


# ----------------------------------------------------------------- plotting
def main():
    # description=__doc__ reuses the docstring at the top of this file as the
    # help text, so "python plot_losses.py -h" prints it.
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logfile", type=Path,
                   help="the .out file written by the Slurm job")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="where to write the PNG "
                        "(default: next to the log, <logname>_losses.png)")
    p.add_argument("--dpi", type=int, default=150,
                   help="resolution of the PNG (default: 150)")
    a = p.parse_args()

    args, batches, train_sums, test_means, n_batches, epoch_times = \
        parse_log(a.logfile)

    # ---- the weights, taken from the log so the titles are never wrong ---
    weights = {k: float(args.get(f"w_{k}", 0.0)) for k, _ in PANELS}
    # Defensive only: w_distances and w_transformation are always printed,
    # but if one were missing, 1.0 is the right fallback rather than 0.0.
    weights["distances"] = float(args.get("w_distances", 1.0))
    weights["transformation"] = float(args.get("w_transformation", 1.0))

    # ---- which epochs are complete --------------------------------------
    # An epoch that is still running has batch lines but no summary lines, so
    # it is excluded here. This is what makes a running job plottable.
    tr_epochs = sorted(e for e in train_sums if e in n_batches)
    te_epochs = sorted(test_means)

    # ---- sums -> means ---------------------------------------------------
    train_means = {e: {k: v / n_batches[e] for k, v in train_sums[e].items()}
                   for e in tr_epochs}

    def weighted_total(d):
        """
        L = Sum_k w_k * L_k, for one epoch's dict of per-term values.

        Needed because the log prints no total for an epoch or for the test
        set, only for individual training batches.
        """
        return sum(weights[k] * d[k] for k, _ in PANELS if k in d)

    # ---- the figure ------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    fig.subplots_adjust(hspace=0.45, wspace=0.22,
                        left=0.05, right=0.985, top=0.93, bottom=0.09)

    def panel(ax, title, batch_y, train_y, test_y):
        """
        Draw one of the six panels.
        """
        if batch_y:
            # grey: one point per LOGGED batch, so its density depends on --log-every-n-batches.
            ax.plot([x for x, _ in batch_y], [y for _, y in batch_y],
                    color=GREY, lw=0.8, zorder=1,
                    label="Average train loss per batch")
        if train_y:
            # blue points at the middle of the epoch.
            ax.plot([e - 0.5 for e in tr_epochs], train_y, color=BLUE, lw=1.8,
                    marker="o", ms=4, zorder=3,
                    label="Average train loss per epoch")
        if test_y:
            # orange points at the end of the epoch.
            ax.plot(te_epochs, test_y, color=ORANGE, lw=1.8,
                    marker="s", ms=4, zorder=2,
                    label="Average test loss per epoch")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("epoch")
        ax.grid(True, color="#e6e6e6", lw=0.8)
        ax.set_axisbelow(True)              
        for side in ("top", "right"):       
            ax.spines[side].set_visible(False)

    # ---- panel 1: the total ---------------------------------------------
    panel(axes[0, 0], "Total loss",
          [(x, tot) for x, tot, _ in batches],
          [weighted_total(train_means[e]) for e in tr_epochs],
          [weighted_total(test_means[e]) for e in te_epochs])
    axes[0, 0].legend(fontsize=8, frameon=False, loc="upper right")

    # ---- panels 2-6: one per loss term ----------------------------------
    for ax, (key, name) in zip(axes.flat[1:], PANELS):
        w = weights[key]
        title = f"{name} (w = {fmt_w(w)})"
        if key == "3d" and w == 0:
            # the term is still computed and logged, just not optimised
            title = f"3D vertices, not trained (w = {fmt_w(w)})"
        panel(ax,
              title,
              # these are the UNWEIGHTED values; the weight is stated in the title
              [(x, t[key]) for x, _, t in batches if key in t],
              [train_means[e][key] for e in tr_epochs if key in train_means[e]],
              [test_means[e][key] for e in te_epochs if key in test_means[e]])

    # ---- save ------------------------------------------------------------
    out = a.output or a.logfile.with_name(a.logfile.stem + "_losses.png")
    fig.savefig(out, dpi=a.dpi)
    print(f"wrote {out}")

    # ---- text report -----------------------------------------------------
    print(f"\ndataset      : {args.get('dataset_path')}")
    print(f"batch size   : {args.get('batch_size')}   "
          f"train batches/epoch: {n_batches.get(tr_epochs[0]) if tr_epochs else '?'}")
    print("weights      : " + ", ".join(f"{k}={fmt_w(weights[k])}"
                                        for k, _ in PANELS))
    print(f"epochs found : {len(tr_epochs)} trained, {len(te_epochs)} tested "
          f"(of {args.get('n_epochs')} requested)")
    if epoch_times:
        mean_t = sum(epoch_times.values()) / len(epoch_times)
        print(f"time/epoch   : {mean_t:7.1f} s mean   "
              f"-> {mean_t * int(args.get('n_epochs', 0)) / 3600:.1f} h "
              f"for the full run")
    if te_epochs:
        # the last tested epoch
        last = te_epochs[-1]
        print(f"\nfinal test losses (epoch {last}):")
        for key, name in PANELS:
            if key in test_means[last]:
                print(f"  {name:<16} {test_means[last][key]:.4f}")
        print(f"  {'TOTAL (weighted)':<16} "
              f"{weighted_total(test_means[last]):.4f}")

# Runs only when the file is executed directly, not when it is imported from another script.
if __name__ == "__main__":
    main()
