# Proposed VAE speedup ? awaiting approval

Active src/vae.py and train.py have NOT been edited.

## Review files
- changes.diff: every proposed production-code change.
- changed_functions.py: just the new/replacement VAE functions, grouped for reading.
- vae.py: complete proposed replacement for src/vae.py.
- train.py: complete proposed replacement for train.py (run from the project root after applying).
- verification.json: measured correctness and benchmark results.
- verify.py: reproducible tests; run from the project root with python -m proposals.vae_speedup.verify.

## Exact scope

1. Add image_to_patches. Explicit loops copy each image window into one matrix row, including channels. Each input channel is padded once per call.
2. Add kernels_to_matrix. Explicit loops copy each filter into one matrix column, in exactly the same channel/row/column order as the patches.
3. Replace convolution. Use np.matmul(patches, filter_weights) for weighted sums, then explicit loops add biases and arrange feature maps.
4. Replace convolution_backward. Use patches.T @ output_gradients for weight gradients, and output_gradients @ filter_weights.T for patch gradients. Explicit loops accumulate overlapping patches into input gradients and sum bias gradients. No full image-sized temporary array per kernel position.
5. Replace dense_backward with np.matmul, np.outer, and a bias-gradient copy.
6. Replace update_parameter with elementwise NumPy subtraction into the existing parameter array. The SGD rule and update timing stay the same.
7. Add optional plot_every=10 to train. Render comparisons at epoch 1, each multiple of 10, and the final epoch. Keep saving a checkpoint EVERY epoch.
8. Remove per-image prints and the redundant reset/increment of sample_count. Keep the dataset count fixed and preserve the epoch loss print.

The loop-based convolve_image, apply_kernel, and their backward helpers remain in the module for reference. Optimized full layers no longer call them.

## What stays the same

Architecture, filter counts, latent size, learning rate, beta, sampling, losses, initialization, checkpoint contents, plot contents, and existing model/caller interfaces are preserved (except the optional train argument). Existing checkpoints retain the same parameter format. Automatic resume is still not implemented.

No slicing is added. Spatial data arrangement and overlapping-gradient accumulation remain explicit loops. The new NumPy operations implement matrix arithmetic and the elementwise parameter update.

To avoid changing every forward/backward cache interface in this proposal, patches are rebuilt during backward rather than cached. Caching could be a later optimization.

## Verification

- Compared convolution outputs and all returned gradients in three configurations, including multiple channels/filters, rectangular kernels, stride and padding.
- Largest gradient difference from the original: 1.78e-15.
- Independently checked 50 numerical derivatives: maximum error 7.20e-11.
- Compared five full training updates with identical random state on the current [8, 16] / latent-size-16 architecture: losses and model parameters matched within floating-point tolerances.
- Tested plot cadence and every-epoch checkpoint cadence.
- Verified active-file hashes stayed unchanged.

## Timing

Median of three five-image training trials (excluding checkpoint/plot I/O):
- Original: 6.44 seconds.
- Proposed: 0.87 seconds.
- Approximately 7.4x faster in this local test.

Individual timings varied, including possible competition from other running work. This is not a guaranteed speedup for every model or run. Floating-point summation order changes slightly; longer stochastic training runs may diverge numerically even though the mathematical operations match.
