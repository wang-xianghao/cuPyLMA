# Changelog

## [0.2.0] - 2025-09-10

### Added

- Add `get_available_gpus` to quickly retrieve available GPUs for the model component, excluding the GPUs allocated to Legate.
<!-- - Add the MNIST example with guidance ([examples/mnist](./examples/mnist)).  -->

### Changed

- **Breaking:** combine `residual_fn` and `loss_fn` into `residual_fn` during constructing `LMA`.
- Flatten the output of `residual_fn` automatically.
- Rewrite the curve example with guidance ([examples/curve](./examples/curve)).

### Removed 

- Remove `benchmarks` folder.