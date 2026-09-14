# Completed WIP handoff

The previously paused branch has been finished as the Observatory update.
See [implementation, validation and scope](observatory-implementation.md).

The user's later steering changed the default: **coordinated physics/render
batches, not a worker thread**. The worker remains optional via `--threaded`.
The update also adds actual GPU/VRAM detection with measured population selection.

Remaining target-machine checks: native Windows execution, actual CUDA graph
replay and same-N/dt performance. Startup automatically checks the GPU numerical
path. GPU FMM, higher-order integration, and CUDA/OpenGL zero-copy rendering are
outside this completed batch; no performance claims rely on them.
