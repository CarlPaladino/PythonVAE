import ast
import copy
import hashlib
import importlib.util
import json
import statistics
import time
from pathlib import Path

import numpy as np
import src.vae as old
from src.data import load_images

folder = Path("proposals/vae_speedup")
spec = importlib.util.spec_from_file_location("proposed_vae", folder / "vae.py")
new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(new)
rng = np.random.default_rng(53)
max_error = 0.0

# Compare forward and all backward outputs on different channel counts,
# rectangular inputs/kernels, strides, and padding.
for channels, filters, height, width, kh, kw, stride, padding in (
    (1, 2, 5, 6, 3, 3, 1, 0),
    (2, 3, 4, 6, 3, 3, 2, 1),
    (3, 2, 5, 4, 2, 3, 1, 1),
):
    x = rng.normal(size=(channels, height, width))
    kernels = rng.normal(size=(filters, channels, kh, kw))
    biases = rng.normal(size=filters)
    expected = old.convolution(x, kernels, biases, stride, padding)
    actual = new.convolution(x, kernels, biases, stride, padding)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    upstream = rng.normal(size=actual.shape)
    ga = old.convolution_backward(x, kernels, upstream, stride, padding)
    gb = new.convolution_backward(x, kernels, upstream, stride, padding)
    for a, b in zip(ga, gb):
        max_error = max(max_error, float(np.max(np.abs(a-b))))
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-11)

# Independent numerical derivatives for all inputs, kernel weights, biases.
x = rng.normal(size=(2, 3, 4))
kernels = rng.normal(size=(2, 2, 2, 3))
biases = rng.normal(size=2)
upstream = rng.normal(size=new.convolution(x, kernels, biases, 2, 1).shape)
gradients = new.convolution_backward(x, kernels, upstream, 2, 1)
numerical_count = 0
numerical_error = 0.0
for parameter, gradient in zip((x, kernels, biases), gradients):
    for index in np.ndindex(parameter.shape):
        original = parameter[index]
        try:
            parameter[index] = original + 1e-5
            plus = np.sum(new.convolution(x, kernels, biases, 2, 1) * upstream)
            parameter[index] = original - 1e-5
            minus = np.sum(new.convolution(x, kernels, biases, 2, 1) * upstream)
        finally:
            parameter[index] = original
        measured = (plus-minus) / 2e-5
        numerical_error = max(numerical_error, abs(float(measured-gradient[index])))
        np.testing.assert_allclose(measured, gradient[index], rtol=1e-5, atol=1e-8)
        numerical_count += 1

def compare_tree(a, b):
    if isinstance(a, dict):
        for key in a:
            compare_tree(a[key], b[key])
    elif isinstance(a, list):
        for index in range(len(a)):
            compare_tree(a[index], b[index])
    elif isinstance(a, np.ndarray):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-11)
    else:
        assert a == b

inputs, _ = load_images("data/raw/emnist-balanced-train.csv", limit=5)
rng = np.random.default_rng(9)
encoder = old.initialize_encoder(inputs[0].shape, [8, 16], 16, rng)
decoder = old.initialize_decoder(encoder, 1, 16, rng)
old_e, old_d = copy.deepcopy(encoder), copy.deepcopy(decoder)
new_e, new_d = copy.deepcopy(encoder), copy.deepcopy(decoder)
old_rng, new_rng = copy.deepcopy(rng), copy.deepcopy(rng)
for x in inputs:
    loss_a = old.train_step(x, old_e, old_d, old_rng, 0.001, 1.0)
    loss_b = new.train_step(x, new_e, new_d, new_rng, 0.001, 1.0)
    np.testing.assert_allclose(loss_a, loss_b, rtol=1e-10, atol=1e-10)
compare_tree(old_e, new_e)
compare_tree(old_d, new_d)

# Time the same five-image epoch from identical initial state; excludes I/O.
timings = {"original": [], "proposed": []}
for repeat in range(3):
    for name, module in (("original", old), ("proposed", new)):
        e, d, generator = copy.deepcopy(encoder), copy.deepcopy(decoder), copy.deepcopy(rng)
        start = time.perf_counter()
        for x in inputs:
            module.train_step(x, e, d, generator, 0.001, 1.0)
        timings[name].append(time.perf_counter()-start)
original_seconds = statistics.median(timings["original"])
proposed_seconds = statistics.median(timings["proposed"])

# Test plot cadence and ensure checkpoint cadence is untouched without training.
namespace = {"__name__": "proposal_train_check"}
exec(compile((folder / "train.py").read_text(), "proposed_train.py", "exec"), namespace)
plot_epochs, checkpoint_epochs = [], []
namespace["train_step"] = lambda *args: (1.0, 0.8, 0.2)
namespace["save_checkpoint"] = lambda path,e,d,r,epoch,*args: checkpoint_epochs.append(epoch)
namespace["save_reconstructions"] = lambda path,inputs,e,d,epoch: plot_epochs.append(epoch)
namespace["print"] = lambda *args: None
namespace["train"]([None], {}, {}, np.random.default_rng(1), 23, 0.001, 1.0)
assert checkpoint_epochs == list(range(1,24))
assert plot_epochs == [1,10,20,23]

for name, expected in json.loads((folder / "original_hashes.json").read_text()).items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
report = {
    "convolution_comparison_cases": 3,
    "largest_gradient_difference_from_original": max_error,
    "numerical_derivatives_checked": numerical_count,
    "largest_numerical_gradient_error": numerical_error,
    "full_training_steps_compared": 5,
    "original_five_step_seconds": original_seconds,
    "proposed_five_step_seconds": proposed_seconds,
    "speedup": original_seconds / proposed_seconds,
    "raw_timings": timings,
    "plot_epochs_in_23_epoch_test": plot_epochs,
    "checkpoints_saved_in_23_epoch_test": len(checkpoint_epochs),
    "active_files_unchanged": True
}
(folder / "verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
