# Horn–Schunck Optical Flow (Serial + OpenMP, OpenCV)

## Overview
This project implements the Horn–Schunck (HS) dense optical flow algorithm:
- Serial version (single-threaded)
- Parallel version using OpenMP (Jacobi double buffering)

Given two consecutive grayscale frames, HS estimates a per-pixel motion vector field (u, v).

## Algorithm (intuitive)
HS assumes two things:
1) Brightness constancy: the same point keeps the same intensity between frames.
2) Smooth flow: neighboring pixels move similarly.

It balances these by minimizing an energy made of:
- A data term (enforces brightness constancy using image derivatives Ix, Iy, It)
- A smoothness term (enforces spatial smoothness, controlled by α)

Iterative update:
- Compute spatial derivatives Ix, Iy and temporal derivative It.
- At each iteration, average neighbors of u and v (smoothing), then correct using Ix, Iy, It.
- Repeat for a fixed number of iterations.

Update schemes:
- Serial uses in-place Gauss–Seidel (faster per iteration).
- Parallel uses Jacobi double buffering (u_next/v_next from previous u/v, then swap) to avoid races and be deterministic.

## Requirements
- C++17
- CMake 3.16+
- OpenCV (core, imgproc, highgui)
- OpenMP (for the parallel build)

## Build
Linux:
- mkdir -p build
- cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
- cmake --build build -j

Note: If CMake target name differs, adjust the run command below accordingly.

## Run
- Ensure you have two grayscale images, e.g., data/rm1.jpg and data/rm2.jpg
- mkdir -p output
- ./build/optical_flow <frame1> <frame2>
  - Example: ./build/optical_flow ./data/rm1.jpg ./data/rm2.jpg

The program will:
- Print serial and parallel timing
- Save visualizations to:
  - output/flow_serial.png
  - output/flow_parallel.png
- Open windows showing both flows (requires a GUI/X11 session)

Headless tip: If imshow fails on a server, comment out the imshow/waitKey lines or run with a virtual display.

## Tuning
- alpha (regularization): higher = smoother flow, lower = more detail but noisier.
  - Typical range: 0.1 to 1.5 (default here: 0.5)
- iterations: more = better convergence but slower.
  - Typical range: 50–500 (default here: 100)

Matching results:
- If you want the serial and parallel outputs to be identical, use Jacobi in serial as well (double buffering). Gauss–Seidel may converge faster per iteration but will differ slightly.

## Performance tips
- Set OMP_NUM_THREADS to control threads, e.g.:
  - export OMP_NUM_THREADS=$(nproc)
- Build Release for speed.
- For very large images, consider 8-neighbor averaging or a separable blur for the smoothing step.

## Files
- main.cpp: I/O, timing, visualization
- include/opt_flow_serial.hpp, include/opt_flow_parallel.hpp: APIs
- src/opt_flow_serial.cpp: serial HS
- src/opt_flow_parallel.cpp: OpenMP HS (Jacobi double buffering)

## Reference
- B. K. P. Horn and B. G. Schunck, “Determining Optical